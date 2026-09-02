"""
A general-purpose agentic coding CLI -- runs a coder/critic/reviser loop
with visible reasoning traces (Hermes-style <think> blocks), a live
context-window meter, real test execution, and shell command access, on
top of whichever local model you point it at.

Optimized for Astryx by default (it was trained specifically on this
loop's <think>/<shell> format), but not locked to it -- swap in any local
Hugging Face model with --model. (Cloud APIs -- Gemini/Claude/OpenAI/
DeepSeek -- are used for generating training data in generate_trajectories.py
and generate_multifile_trajectories.py, not here; this CLI runs local models.)

Usage:
  astryx run "write a function that reverses a linked list"    # one-shot task
  astryx run "fix the bug" --files utils.py main.py --tests test_main.py
  astryx run "..." --max-iters 3          # cap coder<->critic<->reviser loops
  astryx run "..." --no-think             # hide reasoning traces, just show output

  astryx chat                             # persistent session -- talk, get code,
                                           # then just keep telling it what to do next
  astryx chat --no-think

  Inside chat:
    /load <file>   add a file's contents to context
    /test <file>   run the last code block against a test file
    /reset         clear conversation history (starts a new session id)
    /exit          quit

  astryx sessions                    # list saved chat sessions
  astryx sessions show <id>          # print a saved session's transcript
  astryx chat --resume               # resume the most recent session
  astryx chat --resume <id>          # resume a specific session

  Every chat turn is auto-saved to ../sessions/<id>.json, not just on
  exit, so a crash or closed terminal doesn't lose the conversation.

Choosing a model with --model (default: astryx):
  astryx run "..." --model astryx                    # default: Astryx (local, merged/adapter)
  astryx run "..." --model /path/to/local-model       # any local Hugging Face model
  astryx run "..." --model Qwen/Qwen2.5-7B-Instruct   # ...including straight from the Hub

  Non-Astryx models weren't trained on the <think>/<shell> format -- they
  get the same prompted instructions, and most current models follow them
  reasonably well, but adherence isn't guaranteed the way it is for Astryx.

Auto-compaction: once chat history crosses ~80% of the context window, older
messages are automatically summarized into a single turn (code preserved in
full, small talk dropped) so the conversation can keep going instead of
hitting the ceiling. A notice is printed each time this happens.

Shell access: the model can request commands via <shell>command</shell>
tags in its output.
  - Default mode: every command is shown to you and requires a y/N
    confirmation before it runs.
  - --sandbox mode: commands run inside an isolated Docker container (no
    network by default, memory/CPU capped, only a dedicated workspace
    directory mounted) and are auto-approved without a per-command prompt,
    since the container is the safety boundary instead of you reading each
    one. Requires Docker installed and running. See sandbox.py.
"""

import argparse
import re
import subprocess
import tempfile
import os
import json
from datetime import datetime
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.markdown import Markdown
from rich.rule import Rule
from rich.align import Align
from rich.text import Text

from .sandbox import Sandbox, SandboxError

console = Console()

ASTRYX_BASE_MODEL = "Qwen/Qwen3.5-9B"

# Once this is a global `astryx` command, relative paths like "../out/astryx-merged"
# would resolve against whatever directory you happen to be standing in when you
# run it -- meaningless outside the repo. Everything user-specific lives under a
# fixed home directory instead, overridable per-piece via env vars if you'd rather
# point at the repo's ../out/ directly (e.g. during development).
ASTRYX_HOME = os.environ.get("ASTRYX_HOME", os.path.expanduser("~/.astryx"))
ASTRYX_ADAPTER_PATH = os.environ.get("ASTRYX_ADAPTER_PATH", os.path.join(ASTRYX_HOME, "astryx-lora"))
ASTRYX_MERGED_PATH = os.environ.get("ASTRYX_MERGED_PATH", os.path.join(ASTRYX_HOME, "astryx-merged"))
ASTRYX_NATIVE_CONTEXT = 262_144
SESSIONS_DIR = os.environ.get("ASTRYX_SESSIONS_DIR", os.path.join(ASTRYX_HOME, "sessions"))

MAX_TOOL_ITERS = 4  # cap how many shell round-trips happen before forcing a final answer
COMPACT_THRESHOLD = 0.80  # auto-summarize chat history once usage crosses this fraction
COMPACT_KEEP_LAST = 4     # how many recent messages to leave untouched, verbatim

# Default UI accent -- dark blue/purple, matching the logo's palette. Status
# colors (green/yellow/red for pass/fail, context meter) are left as-is
# since those carry meaning rather than being purely thematic.
ACCENT = "slate_blue3"

# Commands that never run without being typed out in full by you, even if
# confirmed -- these are the ones where a single wrong flag is destructive.
# This is a speed bump, not a sandbox: treat this whole feature as "trusts
# you to read the confirmation prompt," not as a security boundary.
DANGEROUS_PATTERNS = [r"\brm\s+-rf\b", r"\bmkfs\b", r"\bdd\s+if=", r">\s*/dev/sd"]

THINK_SYSTEM_SUFFIX = (
    "\n\nBefore answering, think through the problem inside <think>...</think> "
    "tags, then give your final answer after the closing tag."
)

SHELL_SYSTEM_SUFFIX = (
    "\n\nYou can run shell commands to inspect the filesystem (e.g. ls, cat, grep, "
    "find, pwd) by writing <shell>command</shell>. You'll see the output and can "
    "use it before giving your final answer. Only use this when you actually need "
    "to look at something -- don't run commands speculatively."
)


def role_systems(identity: str) -> dict:
    return {
        "coder": f"You are {identity}, a coding agent. Write correct, clean Python solutions.",
        "critic": f"You are {identity}, acting as a code critic. Diagnose bugs precisely and concisely.",
        "reviser": f"You are {identity}, acting as a code reviser. Fix bugs based on the critique given.",
    }


def chat_system(identity: str) -> str:
    return (
        f"You are {identity}, a coding agent. Help with whatever the user asks -- write code, "
        "explain it, modify it, or answer questions. When you write or modify code, keep "
        "track of what you've already written earlier in this conversation so follow-up "
        "requests like 'now add error handling' apply to that same code, not a new example."
    )


# ---------------------------------------------------------------------------
# Engine abstraction -- run_agent_turn and friends talk to an engine
# through a small generate_raw / count_tokens / max_context surface,
# rather than assuming Astryx specifically. This is what makes the CLI
# general-purpose: LocalEngine works with any local Hugging Face model.
# ---------------------------------------------------------------------------

class LocalEngine:
    """Wraps a local Hugging Face causal LM (+ optional LoRA adapter).
    Exact token counts via the real tokenizer."""

    approx_tokens = False

    def __init__(self, model_path: str, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        self.model = model
        self.max_context = getattr(model.config, "max_position_embeddings", None) or ASTRYX_NATIVE_CONTEXT

    def count_tokens(self, messages: list) -> int:
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return len(self.tokenizer(prompt_text)["input_ids"])

    def generate_raw(self, messages: list, max_new_tokens: int = 1024, deterministic: bool = False) -> tuple[str, int, int]:
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=not deterministic,
                temperature=0.7 if not deterministic else None,
                top_p=0.9 if not deterministic else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        completion_tokens = output.shape[1] - prompt_tokens
        raw = self.tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)
        return raw, prompt_tokens, completion_tokens


def build_engine(model_arg: str) -> tuple[object, str]:
    """Resolves --model into (engine, identity_name). Both branches load a
    local Hugging Face model -- this CLI runs local models only; the cloud
    providers in llm_providers.py are for the training-data generators,
    not for inference here.
    - "astryx" (default): local merged Astryx model, or base+adapter if
      the merge step hasn't been run yet
    - anything else: treated as a local HF model path/repo id -- this is
      the "general purpose" path, works with any causal LM
    """
    if model_arg == "astryx":
        if os.path.isdir(ASTRYX_MERGED_PATH):
            console.print(f"[dim]Loading merged Astryx model from {ASTRYX_MERGED_PATH}...[/dim]")
            return LocalEngine(ASTRYX_MERGED_PATH), "Astryx"
        console.print(f"[dim]No merged model at {ASTRYX_MERGED_PATH} -- loading "
                       f"{ASTRYX_BASE_MODEL} + Astryx adapter instead. Run merge_adapter.py "
                       f"to skip this step next time.[/dim]")
        return LocalEngine(ASTRYX_BASE_MODEL, adapter_path=ASTRYX_ADAPTER_PATH), "Astryx"

    console.print(f"[dim]Loading local model {model_arg}...[/dim]")
    identity = os.path.basename(model_arg.rstrip("/")) or "Assistant"
    return LocalEngine(model_arg), identity


@dataclass
class SessionState:
    tokens_used: int = 0
    max_context: int = ASTRYX_NATIVE_CONTEXT
    approx: bool = False
    history: list = field(default_factory=list)


# ANSI Shadow-style block lettering -- hand-set, not generated from the logo
# image pixel-by-pixel, but echoing its palette (cyan star tips -> blue/violet
# core -> the metallic text reads as bright white here since terminals don't
# do gradients on individual glyphs the way the logo's chrome text does).
BANNER_ROWS = [
    " █████╗ ███████╗████████╗██████╗ ██╗   ██╗██╗  ██╗",
    "██╔══██╗██╔════╝╚══██╔══╝██╔══██╗╚██╗ ██╔╝╚██╗██╔╝",
    "███████║███████╗   ██║   ██████╔╝ ╚████╔╝  ╚███╔╝ ",
    "██╔══██║╚════██║   ██║   ██╔══██╗  ╚██╔╝   ██╔██╗ ",
    "██║  ██║███████║   ██║   ██║  ██║   ██║   ██╔╝ ██╗",
    "╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝",
]
BANNER_GRADIENT = ["bright_cyan", "cyan", "blue", "blue", "medium_purple3", "magenta"]


def print_banner():
    console.print()
    console.print(Align.center(Text("╾──────────✦ ASTRYX ✦──────────╼", style=f"bold {ACCENT}")))
    console.print()
    for row, color in zip(BANNER_ROWS, BANNER_GRADIENT):
        console.print(Align.center(Text(row, style=f"bold {color}")))
    console.print()
    console.print(Align.center(Text("a general agentic CLI, tuned for Astryx", style="dim italic")))
    console.print()


def render_context_bar(state: SessionState):
    pct = state.tokens_used / state.max_context
    color = "green" if pct < 0.5 else "yellow" if pct < 0.85 else "red"
    prefix = "~" if state.approx else ""
    with Progress(
        TextColumn("[bold]Context[/bold]"),
        BarColumn(bar_width=40, complete_style=color),
        TextColumn(f"{prefix}{state.tokens_used:,} / {state.max_context:,} tokens ({pct:.1%})"),
        console=console,
    ) as progress:
        progress.add_task("", total=state.max_context, completed=min(state.tokens_used, state.max_context))


def compact_history(engine, state: SessionState, messages: list) -> list:
    """Hermes-style auto-compaction: once context usage crosses COMPACT_THRESHOLD,
    summarize everything except the system prompt and the last few messages into
    a single summary turn, so the conversation can keep going instead of hitting
    the ceiling."""
    system_msg = messages[0]
    if len(messages) <= COMPACT_KEEP_LAST + 1:
        return messages  # not enough history yet to be worth compacting

    to_summarize = messages[1:-COMPACT_KEEP_LAST]
    recent = messages[-COMPACT_KEEP_LAST:]

    transcript = "\n\n".join(
        f"[{m['role']}]: {strip_shell_tags(split_think(m['content'])[1]) if m['role'] == 'assistant' else m['content']}"
        for m in to_summarize
    )
    summarize_prompt = [
        {"role": "system", "content": "Summarize this conversation concisely. Preserve: any code "
                                       "written (in full, don't paraphrase code), key decisions made, "
                                       "and unresolved questions. Drop small talk and redundant back-and-forth."},
        {"role": "user", "content": transcript},
    ]
    summary, _, _ = engine.generate_raw(summarize_prompt, max_new_tokens=800, deterministic=True)
    summary = summary.strip()

    new_messages = [
        system_msg,
        {"role": "user", "content": f"[Summary of earlier conversation, compacted to save context]\n{summary}"},
        {"role": "assistant", "content": "Got it, continuing from that summary."},
        *recent,
    ]

    old_tokens = engine.count_tokens(messages)
    new_tokens = engine.count_tokens(new_messages)
    state.tokens_used = new_tokens

    prefix = "~" if state.approx else ""
    console.print(Panel(
        f"Compacted {len(to_summarize)} older messages into a summary "
        f"({prefix}{old_tokens:,} → {prefix}{new_tokens:,} tokens).",
        title="[dim]context compacted[/dim]", border_style="dim",
    ))
    return new_messages


def split_think(raw: str) -> tuple[str, str]:
    match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if not match:
        return "", raw.strip()
    reasoning = match.group(1).strip()
    answer = raw[match.end():].strip()
    return reasoning, answer


def extract_code(text: str) -> str:
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def run_tests_against(code: str, test_code: str, timeout: int = 10) -> tuple[bool, str]:
    """Actually execute code + tests and report real pass/fail -- used by
    both run_loop's coder<->critic<->reviser cycle and chat's /test command."""
    full_source = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_source)
        path = f.name
    try:
        result = subprocess.run(
            ["python3", path], capture_output=True, text=True, timeout=timeout
        )
        passed = result.returncode == 0
        error = result.stderr.strip() if not passed else ""
        return passed, error
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Shell tool-calling
# ---------------------------------------------------------------------------

def extract_shell_commands(text: str) -> list[str]:
    return [m.strip() for m in re.findall(r"<shell>(.*?)</shell>", text, re.DOTALL)]


def strip_shell_tags(text: str) -> str:
    return re.sub(r"<shell>.*?</shell>", "", text, flags=re.DOTALL).strip()


def is_dangerous(cmd: str) -> bool:
    return any(re.search(p, cmd) for p in DANGEROUS_PATTERNS)


def run_in_sandbox(cmd: str, sandbox: Sandbox) -> str:
    """Sandbox path: no confirmation needed, the container is the boundary.
    Still shown to you for transparency, just not gated on your approval."""
    console.print(f"[{ACCENT}][sandbox][/{ACCENT}] [bold]{cmd}[/bold]")
    passed, output = sandbox.run(cmd)
    console.print(Panel(output or "(no output)", title=f"[dim]$ {cmd}[/dim]", border_style="dim"))
    return output or "(no output)"


def confirm_and_run_host(cmd: str, timeout: int = 15) -> str:
    """Host path: show the command, require explicit y/N, run it directly
    on your machine, return the output (or a note that it was skipped)."""
    flag = " [red bold](flagged as potentially destructive)[/red bold]" if is_dangerous(cmd) else ""
    console.print(f"[yellow]Model wants to run:[/yellow] [bold]{cmd}[/bold]{flag}")
    try:
        answer = console.input("  allow? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        answer = "n"

    if answer != "y":
        console.print("[dim]  skipped.[/dim]\n")
        return "(command not executed -- user declined)"

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=os.getcwd()
        )
        output = (result.stdout + result.stderr).strip()
        output = output[:4000]  # cap so one noisy command can't blow the context budget
        console.print(Panel(output or "(no output)", title=f"[dim]$ {cmd}[/dim]", border_style="dim"))
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        console.print("[red]  timed out.[/red]\n")
        return "(command timed out)"


def execute_command(cmd: str, sandbox: Sandbox | None) -> str:
    return run_in_sandbox(cmd, sandbox) if sandbox else confirm_and_run_host(cmd)


def run_agent_turn(engine, state: SessionState, messages: list,
                    show_think: bool, label: str, color: str,
                    sandbox: Sandbox | None = None,
                    max_new_tokens: int = 1024) -> str:
    """Generates a response through `engine`, and if it contains <shell>
    tags, runs them (via sandbox if provided, else host confirmation),
    feeds the output back in, and lets the model continue -- up to
    MAX_TOOL_ITERS round trips before forcing a final answer. Mutates
    `messages` in place."""
    console.print(Rule(f"[{color}]{label}[/{color}]", style=color))

    for tool_iter in range(MAX_TOOL_ITERS):
        raw, prompt_tokens, completion_tokens = engine.generate_raw(messages, max_new_tokens=max_new_tokens)
        state.tokens_used = prompt_tokens + completion_tokens
        reasoning, answer = split_think(raw)

        if show_think and reasoning and tool_iter == 0:
            console.print(Panel(reasoning, title=f"[dim]{label.lower()} reasoning[/dim]", border_style="dim", expand=False))

        commands = extract_shell_commands(answer)
        if not commands or tool_iter == MAX_TOOL_ITERS - 1:
            clean_answer = strip_shell_tags(answer)
            console.print(Markdown(clean_answer))
            render_context_bar(state)
            console.print()
            messages.append({"role": "assistant", "content": raw})
            return clean_answer

        # Model wants to run something -- append what it said so far, run
        # the commands, feed results back as a new user turn, then loop
        # to let it respond with the actual answer.
        messages.append({"role": "assistant", "content": raw})
        results = []
        for cmd in commands:
            out = execute_command(cmd, sandbox)
            results.append(f"$ {cmd}\n{out}")
        messages.append({"role": "user", "content": "Command output:\n" + "\n\n".join(results)})

    return ""  # unreachable given the loop structure above, but keeps type-checkers happy


def generate(engine, state: SessionState, identity: str, role: str, user_prompt: str,
             show_think: bool = True, sandbox: Sandbox | None = None) -> str:
    """One-off role call (coder/critic/reviser) with fresh context each time --
    used by run_loop, which doesn't need cross-call conversation history."""
    system = role_systems(identity)[role] + THINK_SYSTEM_SUFFIX + SHELL_SYSTEM_SUFFIX
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    role_color = {"coder": ACCENT, "critic": "yellow", "reviser": "magenta"}[role]
    answer = run_agent_turn(engine, state, messages, show_think, role.upper(), role_color, sandbox)
    state.history.append({"role": role, "answer": answer})
    return answer


def chat_turn(engine, state: SessionState, identity: str, messages: list, show_think: bool,
              sandbox: Sandbox | None = None) -> str:
    """One turn of a persistent conversation -- `messages` carries the full
    history in and out, so shell tool calls and follow-ups both see it."""
    return run_agent_turn(engine, state, messages, show_think, identity.upper(), ACCENT, sandbox)


# ---------------------------------------------------------------------------
# One-shot loop: astryx run "..."
# ---------------------------------------------------------------------------

def run_loop(engine, identity: str, task: str, files: list[str] | None,
             tests: str | None, max_iters: int, show_think: bool,
             sandbox: Sandbox | None = None):
    state = SessionState(max_context=engine.max_context, approx=engine.approx_tokens)

    file_context = ""
    if files:
        blocks = []
        for fpath in files:
            try:
                with open(fpath) as f:
                    blocks.append(f"### {fpath}\n```python\n{f.read()}\n```")
            except OSError as e:
                console.print(f"[red]Couldn't read {fpath}: {e}[/red]")
        file_context = "\n\n".join(blocks)

    test_code = None
    if tests:
        try:
            with open(tests) as f:
                test_code = f.read()
        except OSError as e:
            console.print(f"[red]Couldn't read tests file {tests}: {e}[/red]")

    coder_prompt = task if not file_context else f"{task}\n\nRelevant files:\n{file_context}"
    code = extract_code(generate(engine, state, identity, "coder", coder_prompt, show_think, sandbox))

    if test_code is None:
        console.print("[dim]No --tests file given -- skipping verification, returning first draft.[/dim]")
        return code, state

    for _ in range(max_iters):
        passed, error = run_tests_against(code, test_code)
        if passed:
            console.print("[bold green]✓ Tests passed -- loop complete[/bold green]")
            break

        console.print(f"[red]✗ Tests failed:[/red]\n{error}\n")
        critique = generate(
            engine, state, identity, "critic",
            f"This code failed with: {error}\n```python\n{code}\n```", show_think, sandbox
        )
        code = extract_code(generate(
            engine, state, identity, "reviser",
            f"Code:\n```python\n{code}\n```\nCritique: {critique}\nFix it.", show_think, sandbox
        ))
    else:
        console.print(f"[yellow]Hit max_iters ({max_iters}) without passing -- returning last attempt.[/yellow]")

    return code, state


# ---------------------------------------------------------------------------
# Session persistence -- chat transcripts saved to disk, resumable, listable
# ---------------------------------------------------------------------------

def _sessions_path() -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    return SESSIONS_DIR


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def session_file(session_id: str) -> str:
    return os.path.join(_sessions_path(), f"{session_id}.json")


def save_session(session_id: str, identity: str, model_arg: str, messages: list):
    """Called after every turn, not just on exit -- a crash or Ctrl-C
    shouldn't lose the conversation. Last-write-wins, cheap enough to do
    every turn for a chat-sized message list."""
    first_user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    existing_created = None
    path = session_file(session_id)
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing_created = json.load(f).get("created")
        except (json.JSONDecodeError, OSError):
            pass

    payload = {
        "id": session_id,
        "identity": identity,
        "model": model_arg,
        "created": existing_created or datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "preview": first_user_msg[:80],
        "messages": messages,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_session(session_id: str) -> dict:
    path = session_file(session_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No session found with id '{session_id}' at {path}")
    with open(path) as f:
        return json.load(f)


def list_sessions() -> list:
    path = _sessions_path()
    sessions = []
    for fname in os.listdir(path):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(path, fname)) as f:
                sessions.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(sessions, key=lambda s: s.get("updated", ""), reverse=True)


def most_recent_session_id() -> str | None:
    sessions = list_sessions()
    return sessions[0]["id"] if sessions else None


def print_sessions_table():
    sessions = list_sessions()
    if not sessions:
        console.print(f"[dim]No saved sessions yet in {os.path.abspath(SESSIONS_DIR)}/[/dim]")
        return
    from rich.table import Table
    table = Table(show_header=True, header_style=f"bold {ACCENT}")
    table.add_column("ID")
    table.add_column("Model")
    table.add_column("Updated")
    table.add_column("Turns")
    table.add_column("Preview")
    for s in sessions:
        turns = sum(1 for m in s["messages"] if m["role"] == "user")
        table.add_row(
            s["id"], s.get("model", "?"),
            s.get("updated", "?")[:19].replace("T", " "),
            str(turns), s.get("preview", "")[:60],
        )
    console.print(table)
    console.print(f"\n[dim]Resume with: astryx chat --resume <ID>  (or --resume last for the most recent)[/dim]")


def print_session_transcript(session_id: str):
    session = load_session(session_id)
    console.print(Panel(
        f"Session {session['id']}  ·  model: {session.get('model', '?')}  ·  "
        f"created: {session.get('created', '?')[:19].replace('T', ' ')}",
        border_style=ACCENT,
    ))
    for m in session["messages"]:
        if m["role"] == "system":
            continue
        elif m["role"] == "user":
            console.print(Rule(f"[{ACCENT}]YOU[/{ACCENT}]", style=ACCENT))
            console.print(m["content"])
        elif m["role"] == "assistant":
            console.print(Rule(f"[{ACCENT}]{session.get('identity', 'ASSISTANT').upper()}[/{ACCENT}]", style=ACCENT))
            reasoning, answer = split_think(m["content"])
            console.print(Markdown(strip_shell_tags(answer)))
        console.print()


# ---------------------------------------------------------------------------
# Persistent session: astryx chat
# ---------------------------------------------------------------------------

def run_chat(engine, identity: str, model_arg: str, show_think: bool,
             sandbox: Sandbox | None = None, resume_id: str | None = None):
    state = SessionState(max_context=engine.max_context, approx=engine.approx_tokens)
    system = chat_system(identity) + THINK_SYSTEM_SUFFIX + SHELL_SYSTEM_SUFFIX
    last_code = ""

    if resume_id:
        try:
            session = load_session(resume_id)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return
        messages = session["messages"]
        session_id = session["id"]
        state.tokens_used = engine.count_tokens(messages)
        console.print(f"[dim]Resumed session {session_id} ({sum(1 for m in messages if m['role'] == 'user')} "
                       f"prior turns). Type normally to continue.[/dim]\n")
    else:
        messages = [{"role": "system", "content": system}]
        session_id = new_session_id()

    shell_note = (
        "it can run shell commands inside an isolated sandbox (auto-approved, no network)."
        if sandbox else
        "it can run shell commands (with your confirmation) to look around."
    )
    console.print(Panel(
        f"Chatting with {identity} -- same agent as `run`, just conversational, and {shell_note}\n\n"
        f"Session: {session_id} (auto-saved every turn, see 'astryx sessions')\n\n"
        "[bold]/load <file>[/bold]   add a file's contents to context\n"
        "[bold]/test <file>[/bold]   run the last code block against a test file\n"
        "[bold]/reset[/bold]         clear conversation history (starts a new session id)\n"
        "[bold]/exit[/bold]          quit",
        border_style=ACCENT,
    ))

    while True:
        try:
            user_input = console.input(f"[bold {ACCENT}]you>[/bold {ACCENT}] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye. Session saved -- resume any time with "
                           f"'astryx chat --resume {session_id}'.[/dim]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped == "/exit":
            console.print(f"[dim]Session saved -- resume any time with 'astryx chat --resume {session_id}'.[/dim]")
            break
        if stripped == "/reset":
            messages = [{"role": "system", "content": system}]
            state.tokens_used = 0
            last_code = ""
            session_id = new_session_id()
            console.print(f"[dim]History cleared. New session: {session_id}[/dim]\n")
            continue

        if stripped.startswith("/load "):
            fpath = stripped[len("/load "):].strip()
            try:
                with open(fpath) as f:
                    content = f.read()
                messages.append({
                    "role": "user",
                    "content": f"Here is the contents of {fpath}, for context:\n```python\n{content}\n```",
                })
                console.print(f"[dim]Loaded {fpath} into context.[/dim]\n")
                save_session(session_id, identity, model_arg, messages)
            except OSError as e:
                console.print(f"[red]Couldn't read {fpath}: {e}[/red]\n")
            continue

        if stripped.startswith("/test "):
            fpath = stripped[len("/test "):].strip()
            if not last_code:
                console.print("[yellow]No code generated yet this session -- nothing to test.[/yellow]\n")
                continue
            try:
                with open(fpath) as f:
                    test_code = f.read()
            except OSError as e:
                console.print(f"[red]Couldn't read {fpath}: {e}[/red]\n")
                continue

            passed, error = run_tests_against(last_code, test_code)
            if passed:
                console.print("[bold green]✓ Tests passed[/bold green]\n")
                continue

            console.print(f"[red]✗ Tests failed:[/red]\n{error}\n")
            messages.append({
                "role": "user",
                "content": f"Running that against {fpath} failed:\n{error}\n\nDiagnose it, then fix it.",
            })
            if state.tokens_used / state.max_context > COMPACT_THRESHOLD:
                messages = compact_history(engine, state, messages)
            answer = chat_turn(engine, state, identity, messages, show_think, sandbox)
            new_code = extract_code(answer)
            if new_code:
                last_code = new_code
            save_session(session_id, identity, model_arg, messages)
            continue

        messages.append({"role": "user", "content": user_input})
        if state.tokens_used / state.max_context > COMPACT_THRESHOLD:
            messages = compact_history(engine, state, messages)
        answer = chat_turn(engine, state, identity, messages, show_think, sandbox)
        new_code = extract_code(answer)
        if new_code:
            last_code = new_code

        if state.tokens_used / state.max_context > 0.9:
            console.print("[yellow]⚠ Approaching context limit -- attempting to compact.[/yellow]")
            messages = compact_history(engine, state, messages)

        save_session(session_id, identity, model_arg, messages)


def main():
    print_banner()
    parser = argparse.ArgumentParser(prog="astryx")
    sub = parser.add_subparsers(dest="command", required=True)

    model_help = (
        "Which local model to run on. 'astryx' (default) uses the local "
        "merged/adapter Astryx model. A filesystem path or Hub repo id loads "
        "any other local Hugging Face model."
    )

    run_p = sub.add_parser("run", help="Run the coder/critic/reviser loop on a task")
    run_p.add_argument("task", type=str)
    run_p.add_argument("--model", type=str, default="astryx", help=model_help)
    run_p.add_argument("--files", nargs="*", default=None, help="Files to include as context")
    run_p.add_argument("--tests", type=str, default=None, help="Test file to verify against")
    run_p.add_argument("--max-iters", type=int, default=3)
    run_p.add_argument("--no-think", action="store_true", help="Hide reasoning traces")
    run_p.add_argument("--sandbox", action="store_true",
                        help="Run shell commands in an isolated Docker container instead of "
                             "confirming each one on your host")
    run_p.add_argument("--sandbox-dir", type=str, default="./astryx_sandbox",
                        help="Directory mounted into the sandbox container (default: ./astryx_sandbox)")
    run_p.add_argument("--sandbox-network", action="store_true",
                        help="Allow network access inside the sandbox (off by default)")

    chat_p = sub.add_parser("chat", help="Interactive multi-turn chat")
    chat_p.add_argument("--model", type=str, default="astryx", help=model_help)
    chat_p.add_argument("--no-think", action="store_true", help="Hide reasoning traces")
    chat_p.add_argument("--sandbox", action="store_true",
                         help="Run shell commands in an isolated Docker container instead of "
                              "confirming each one on your host")
    chat_p.add_argument("--sandbox-dir", type=str, default="./astryx_sandbox",
                         help="Directory mounted into the sandbox container (default: ./astryx_sandbox)")
    chat_p.add_argument("--sandbox-network", action="store_true",
                         help="Allow network access inside the sandbox (off by default)")
    chat_p.add_argument("--resume", type=str, nargs="?", const="last", default=None,
                         help="Resume a previous session. Give a session ID, or omit the value "
                              "(just '--resume') to resume the most recent one. See 'astryx sessions'.")

    sessions_p = sub.add_parser("sessions", help="List or view saved chat sessions")
    sessions_p.add_argument("action", nargs="?", choices=["list", "show"], default="list")
    sessions_p.add_argument("session_id", nargs="?", default=None,
                             help="Required for 'show', e.g. astryx sessions show 20260831-143022")

    args = parser.parse_args()

    if args.command == "sessions":
        if args.action == "list":
            print_sessions_table()
        elif args.action == "show":
            if not args.session_id:
                console.print("[red]Usage: astryx sessions show <session_id>[/red]")
                return
            try:
                print_session_transcript(args.session_id)
            except FileNotFoundError as e:
                console.print(f"[red]{e}[/red]")
        return

    sandbox = None
    if getattr(args, "sandbox", False):
        sandbox = Sandbox(args.sandbox_dir, allow_network=args.sandbox_network)
        try:
            sandbox.start()
            console.print(
                f"[green]Sandbox running[/green] -- workspace: {os.path.abspath(args.sandbox_dir)}, "
                f"network: {'on' if args.sandbox_network else 'off'}\n"
            )
        except SandboxError as e:
            console.print(f"[red]{e}[/red]")
            return

    resume_id = None
    if args.command == "chat" and args.resume:
        resume_id = args.resume
        if resume_id == "last":
            resume_id = most_recent_session_id()
            if resume_id is None:
                console.print("[yellow]No saved sessions to resume from -- starting a new one instead.[/yellow]")
        elif not os.path.exists(session_file(resume_id)):
            console.print(f"[red]No session found with id '{resume_id}'. Run 'astryx sessions' to see what's saved.[/red]")
            return

    try:
        engine, identity = build_engine(args.model)
    except Exception as e:
        console.print(f"[red]Couldn't load model '{args.model}': {e}[/red]")
        if sandbox is not None:
            sandbox.stop()
        return

    try:
        if args.command == "run":
            run_loop(engine, identity, args.task, args.files, args.tests,
                      args.max_iters, not args.no_think, sandbox)
        elif args.command == "chat":
            run_chat(engine, identity, args.model, not args.no_think, sandbox, resume_id=resume_id)
    finally:
        if sandbox is not None:
            sandbox.stop()
            console.print("[dim]Sandbox container removed.[/dim]")


if __name__ == "__main__":
    main()
