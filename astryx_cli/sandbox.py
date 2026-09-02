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

import subprocess
import uuid
import os

SANDBOX_IMAGE = "python:3.11-slim"


class SandboxError(Exception):
    pass


class Sandbox:
    def __init__(self, workdir: str, allow_network: bool = False,
                 memory_limit: str = "512m", cpu_limit: str = "1"):
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.allow_network = allow_network
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.container_name = f"astryx-sandbox-{uuid.uuid4().hex[:8]}"
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
        network_flag = [] if self.allow_network else ["--network", "none"]
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "--memory", self.memory_limit,
            "--cpus", self.cpu_limit,
            *network_flag,
            "-v", f"{self.workdir}:/workspace",
            "-w", "/workspace",
            SANDBOX_IMAGE,
            "sleep", "infinity",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SandboxError(f"Failed to start sandbox container: {result.stderr.strip()}")
        self._started = True

    def run(self, command: str, timeout: int = 30) -> tuple[bool, str]:
        if not self._started:
            raise SandboxError("Sandbox not started -- call start() first.")
        try:
            result = subprocess.run(
                ["docker", "exec", self.container_name, "bash", "-c", command],
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