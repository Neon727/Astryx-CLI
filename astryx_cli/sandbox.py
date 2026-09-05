"""
Docker-backed sandbox for Astryx's shell commands. In sandbox mode, commands
run inside a throwaway container instead of your host shell -- no network
by default, memory/CPU capped, and only a dedicated workspace directory is
mounted (not your home directory or the rest of the filesystem). This is
what makes it reasonable to auto-approve commands instead of confirming
every single one.

This is real isolation, not just a permission prompt -- but it's still not
a hard security boundary against a truly adversarial payload (container
escapes exist). Treat it as "safe enough to let it run unattended on a
scratch directory," not "safe enough to point at anything."

Requires Docker installed and running (`docker info` should succeed).
"""

import atexit
import subprocess
import uuid
import os

SANDBOX_IMAGE = "python:3.11-slim"
CONTAINER_PREFIX = "astryx-sandbox-"


class SandboxError(Exception):
    pass


def _gc_orphaned_containers():
    """Removes any leftover astryx-sandbox-* containers from a previous
    run that didn't get cleaned up -- e.g. the process was SIGKILL'd, the
    machine lost power, or an unhandled exception fired before stop() ran.
    Runs once at Sandbox.start(), best-effort: a Docker error here just
    means we skip cleanup rather than blocking the new sandbox from
    starting."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name={CONTAINER_PREFIX}"],
            capture_output=True, text=True, timeout=5,
        )
        ids = [line for line in result.stdout.splitlines() if line.strip()]
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=15)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # best-effort -- a fresh sandbox can still start even if GC fails


class Sandbox:
    def __init__(self, workdir: str, allow_network: bool = False,
                 memory_limit: str = "512m", cpu_limit: str = "1",
                 run_as_nonroot: bool = False):
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.allow_network = allow_network
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        # Off by default: python:3.11-slim's default user (root) owns
        # site-packages, so `pip install` inside the sandbox works out of
        # the box. Running as a non-root uid is more defense-in-depth but
        # can break pip installs unless the image/workdir permissions are
        # set up for it -- opt in only if you know your commands don't
        # need to install anything as part of the run.
        self.run_as_nonroot = run_as_nonroot
        self.container_name = f"{CONTAINER_PREFIX}{uuid.uuid4().hex[:8]}"
        self._started = False

    def check_docker_available(self):
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
            if result.returncode != 0:
                raise SandboxError(
                    "Docker is installed but not running/accessible. "
                    "Start Docker Desktop (or the docker daemon under WSL2) and try again."
                )
        except FileNotFoundError:
            raise SandboxError(
                "Docker isn't installed. Sandbox mode requires it -- install Docker Desktop "
                "(with WSL2 integration enabled) or run without --sandbox to use the "
                "confirm-before-running mode instead."
            )
        except subprocess.TimeoutExpired:
            raise SandboxError("Docker didn't respond -- is the daemon running?")

    def start(self):
        self.check_docker_available()
        _gc_orphaned_containers()

        network_flag = [] if self.allow_network else ["--network", "none"]
        user_flag = ["--user", "1000:1000"] if self.run_as_nonroot else []
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "--memory", self.memory_limit,
            "--cpus", self.cpu_limit,
            *network_flag,
            *user_flag,
            "-v", f"{self.workdir}:/workspace",
            "-w", "/workspace",
            SANDBOX_IMAGE,
            "sleep", "infinity",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SandboxError(f"Failed to start sandbox container: {result.stderr.strip()}")
        self._started = True
        # Safety net for the SIGKILL/crash case -- normal exits already go
        # through the CLI's own try/finally, this just covers the gap.
        atexit.register(self.stop)

    def run(self, command: str, timeout: int = 30) -> tuple[bool, str]:
        if not self._started:
            raise SandboxError("Sandbox not started -- call start() first.")
        try:
            # sh, not bash -- python:3.11-slim has bash, but if SANDBOX_IMAGE
            # is ever pointed at something minimal (e.g. an alpine-based
            # image), bash won't exist there and every command would fail.
            # POSIX sh is present on effectively every base image.
            result = subprocess.run(
                ["docker", "exec", self.container_name, "sh", "-c", command],
                capture_output=True, text=True, timeout=timeout,
            )
            output = (result.stdout + result.stderr).strip()
            return result.returncode == 0, output[:4000]
        except subprocess.TimeoutExpired:
            return False, "(command timed out inside sandbox)"

    def stop(self):
        if self._started:
            subprocess.run(["docker", "rm", "-f", self.container_name],
                            capture_output=True, timeout=10)
            self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
