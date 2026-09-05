# Astryx CLI

A general-purpose agentic coding CLI — runs a coder → critic → reviser
loop with visible reasoning traces, real test execution, and shell command
access, on top of a local model. Defaults to Astryx (optimized for this
loop specifically), but works with any local Hugging Face model.

## Install

```bash
git clone https://github.com/Neon727/Astryx-CLI
cd Astryx-CLI
pip install -e .
```

That gives you a global `astryx` command, usable from any directory.

Optional, only if you use `--sandbox`: [Docker](https://docs.docker.com/get-docker/), installed and running.

Optional, only if you want to run a GGUF model (e.g. one pulled via
Ollama — see below):

```bash
pip install -e ".[gguf]"
```

## Pointing it at an Astryx model

`--model astryx` (the default) searches a handful of common locations for
a model rather than only checking one fixed path, in this order:

1. `ASTRYX_MERGED_PATH` (defaults to `~/.astryx/astryx-merged`) — the
   intentional, documented location; set this if you want to be explicit
2. `./out/astryx-merged`, `../out/astryx-merged`, `../../out/astryx-merged`
   relative to wherever you're running the command from — a common place
   for a build/output step to have left one
3. `~/models/astryx-merged`, `~/Downloads/astryx-merged`,
   `~/Desktop/astryx-merged` — common places people just leave things
   after downloading them
4. The Hugging Face cache (`~/.cache/huggingface/hub`) — covers the case
   where it was downloaded via a Hub repo id (`from_pretrained(...)`)
   rather than placed locally
5. Same search again for an adapter (`astryx-lora` instead of
   `astryx-merged`) if no merged model turned up anywhere — loaded on top
   of the Qwen3.5-9B base
6. Ollama's local model store (`~/.ollama/models`) — covers `ollama pull
   astryx`, if a GGUF build is available that way. A match here loads via
   `GGUFEngine` (`llama-cpp-python`) instead of the usual `LocalEngine`

If a merged model and an adapter both exist, the merged one wins (it's
what you'd actually want — one self-contained model instead of a base +
patch). If nothing is found anywhere, it errors out listing every
location it checked, so you know exactly what to fix rather than getting
a mysterious failure.

**The reliable option is still to just put it where it's expected:**

```bash
# Symlink to wherever the model actually is
mkdir -p ~/.astryx
ln -s /path/to/astryx-merged ~/.astryx/astryx-merged

# Or point the env var at it directly
export ASTRYX_MERGED_PATH=/path/to/astryx-merged
```

The fallback search is there to save you a step in the common cases, not
something to depend on long-term — if you're scripting this or setting up
a fresh machine, be explicit with the env var or the symlink rather than
relying on it happening to be in one of the searched spots.

## Quick start

```bash
# One-shot: give it a task, it does the loop, exits
astryx run "write a function that reverses a linked list"

# One-shot with real verification against a test file
astryx run "fix the bug" --files utils.py main.py --tests test_main.py

# Persistent session: talk, get code, keep talking — no relaunching needed
astryx chat
```

## `run` vs `chat`

**`run`** is one-shot. Give it a task; it loops coder → critic → reviser
until it passes or hits `--max-iters`, then exits. Two ways to verify:

- **`--tests <file>`** — real execution against a test file you wrote.
  This is the reliable option: it checks something external to the model.
- **No `--tests` given** — a "tester" role writes assert-based tests from
  the task description itself, and those get run instead. This is a
  **heuristic, not a guarantee**: the same model wrote the code and is now
  judging what "correct" looks like, so it can share the code's
  misunderstanding of the task, or lean toward confirming the
  implementation rather than independently checking it. The CLI prints a
  clear warning when tests are self-generated so a pass doesn't get
  mistaken for a real verification. Use `--tests` when correctness
  actually matters.

**`chat`** is persistent. Same agent, same loop, but it keeps the whole
conversation in context — so after it writes something, you just keep
talking ("now add error handling", "extract that into its own function")
instead of relaunching the CLI for every follow-up.

Inside `chat`:

| Command | What it does |
|---|---|
| *(plain message)* | continues the conversation normally |
| `/load <file>` | pulls a file's contents into context (capped at 200KB, rejects binaries) |
| `/test <file>` | runs the last code block against a real test file; on failure, feeds the result back into the conversation and has it fix it |
| `/reset` | clears history and starts a new session id (useful near the context ceiling) |
| `/exit` | quit |

## Sessions: resuming and reviewing past chats

Every `chat` session is auto-saved to disk after each turn — not just on
exit — so a crash or a closed terminal doesn't lose the conversation.

```bash
astryx sessions              # list saved sessions
astryx sessions show <id>    # print a past session's transcript
astryx chat --resume         # resume the most recent session
astryx chat --resume <id>    # resume a specific one
```

`sessions` shows a table: session ID (a timestamp, e.g.
`20260831-143022`), which model it used, when it was last updated, how
many turns, and a preview of the first message. `sessions show <id>`
prints the full back-and-forth without loading a model or starting a
session — useful for just checking what happened in an old conversation.

Resuming picks the conversation back up exactly where it left off — full
message history restored, context meter recalculated against it. If it's
close to the context ceiling already, the usual auto-compaction kicks in
on the next turn rather than immediately on resume.

Sessions are stored as JSON files in `~/.astryx/sessions/` (or wherever
`ASTRYX_SESSIONS_DIR` points), one per session. `/reset` inside a chat
starts a fresh session id rather than overwriting the current one — so
resetting mid-conversation doesn't destroy what came before, it's still
resumable by its old id.

## Choosing a model

```bash
astryx chat --model astryx                    # default
astryx chat --model qwen2.5-coder             # searched by name (see below)
astryx chat --model /path/to/local-model      # a local directory, used directly
astryx chat --model Qwen/Qwen2.5-7B-Instruct  # a Hub repo id
```

Local models only — this CLI doesn't call out to cloud APIs for
inference.

Any `--model` value that isn't `astryx` and isn't already a valid local
directory gets **searched by name**, in this order:

1. The Hugging Face cache (`~/.cache/huggingface/hub`) — covers a model
   you've already downloaded via `from_pretrained` under a Hub repo id
2. Ollama's local model store (`~/.ollama/models`) — covers a model
   you've pulled with `ollama pull <name>`. Ollama stores models as GGUF
   files, a different format from what `transformers` reads, so a match
   here loads through **`GGUFEngine`** (via `llama-cpp-python`) instead of
   `LocalEngine` — install that piece with `pip install -e ".[gguf]"` if
   you hit an error saying it's missing
3. If neither has a match, it falls through to treating the value as a
   literal local path or Hub repo id and lets `from_pretrained` do its own
   resolution — which includes a download if it's a valid repo id you
   have network access to and it isn't cached yet

This search runs for `--model astryx` too (in `find_astryx_model`,
alongside the fixed `~/.astryx/` locations) — so a trained Astryx model
sitting in the Hugging Face cache or in Ollama gets picked up the same
way a custom name would.

Non-Astryx models weren't trained on the `<think>`/`<shell>` tag format —
they get the exact same prompted instructions, and most current models
follow them reasonably well, but adherence isn't guaranteed the way it is
for Astryx. If a model never emits `<think>` tags, the CLI just shows no
reasoning panel and carries on rather than breaking.

## Flags

| Flag | Applies to | Default | What it does |
|---|---|---|---|
| `--model` | both | `astryx` | Which local model to load |
| `--files` | `run` | none | Files to include as context |
| `--tests` | `run` | none | Test file to verify against; if omitted, tests are self-generated from the task instead (see above) |
| `--max-iters` | `run` | `3` | Cap on coder↔critic↔reviser loop iterations |
| `--no-think` | both | off | Hide reasoning-trace panels |
| `--sandbox` | both | off | Run shell commands in an isolated Docker container instead of confirming each on your host |
| `--sandbox-dir` | both | `./astryx_sandbox` | Directory mounted into the sandbox container (cwd-relative on purpose — it mounts near whatever project you're currently in) |
| `--sandbox-network` | both | off | Allow network access inside the sandbox |
| `--sandbox-nonroot` | both | off | Run sandbox commands as uid 1000 instead of root — more defense-in-depth, but can break commands that need to `pip install` or write to system paths inside the container |
| `--resume [id]` | `chat` | none | Resume a saved session; omit the id to resume the most recent one |

## What the CLI shows you

- **Streamed generation** — tokens appear live as the model generates
  them, in a dim "generating..." panel, instead of the terminal sitting
  silent until the whole response is ready. Ctrl+C during generation
  actually interrupts it (via a stopping criteria checked each step for
  local models), not just detaches from something that keeps running in
  the background. The raw `<think>`/`<shell>` tags are visible during this
  live preview — there's no clean way to hide them before generation
  finishes — and the panel disappears once done, replaced by the normal
  polished rendering below.
- **Role-labeled reasoning traces** — coder, critic, tester, and reviser
  each emit a `<think>...</think>` block before acting, shown in a dim
  panel titled with whichever role produced it, so it's clear which step
  you're looking at.
- **Live context meter** — a progress bar tracking cumulative tokens used
  against the model's context window, counted exactly via its tokenizer.
- **Shell command output** — shown in a bordered panel as it happens, so
  you see exactly what the model saw before it answers.

## Shell access

The model can request commands via `<shell>command</shell>` tags in its
output — `ls`, `cat`, `grep`, `find`, `pwd`, that kind of thing, so it can
actually look at your project instead of only working from what you paste
in. Two modes:

### Default: confirm on host

```bash
astryx chat
```

Every command is shown to you and requires an explicit `y` before it
runs, directly on your machine. No auto-approve. This is a speed bump,
not a sandbox — it makes you read what it's about to do before it does
it, since a model can propose a wrong or destructive command with total
confidence. A few especially destructive patterns (`rm -rf`, `dd if=`,
`mkfs`) get an extra warning label, but nothing is blocked outright — the
decision is always yours.

### `--sandbox`: isolated, auto-approved

```bash
astryx chat --sandbox
astryx run "task" --sandbox --sandbox-network   # allow network too
```

Requires Docker. Commands execute inside a throwaway container instead of
your host shell:
- Only a dedicated workspace directory is mounted (`./astryx_sandbox` by
  default) — not your home directory or anything else on the machine
- No network access by default (`--sandbox-network` to opt in)
- Memory and CPU capped (512MB / 1 CPU by default — edit `Sandbox.__init__`
  in `astryx_cli/sandbox.py` to change)
- Runs as root inside the container by default (needed for `pip install`
  and similar to work without extra setup); `--sandbox-nonroot` switches
  to uid 1000 for more defense-in-depth, at the cost of breaking commands
  that need root inside the container
- Commands run via `sh -c`, not `bash -c` — works on minimal/Alpine-style
  base images too, not just the default `python:3.11-slim`, in case you
  ever change `SANDBOX_IMAGE`
- Container is created fresh at session start and removed on exit — and
  if a previous run's container got orphaned (crash, `SIGKILL`, a closed
  laptop lid), the next `--sandbox` invocation cleans it up automatically
  before starting a new one, rather than leaving it running indefinitely

Because the container is the actual safety boundary here, commands run
without a per-command confirmation — you'll still see every command and
its output printed as it happens, just not gated on your approval.

**Be clear-eyed about what this is and isn't:** container isolation is
real and meaningfully better than running arbitrary shell on your host,
but it's not an airtight security boundary — container escapes exist, and
the mounted workspace directory can still be overwritten by a bad command
since it needs read-write access to be useful. Treat sandbox mode as
"safe enough to let run less-supervised on a scratch directory," not
"safe enough to point at anything important." If Docker isn't installed
or running, `--sandbox` fails with a clear message rather than silently
falling back to unsandboxed execution.

## Auto-compaction

In `chat` mode, once token usage crosses ~80% of the context window,
older messages are automatically summarized into a single turn — code
that was written is preserved in full (not paraphrased), small talk and
resolved back-and-forth is dropped, and the last few messages are left
untouched so nothing you just said gets swept into the summary. A
`context compacted` notice prints the before/after token count each time
it happens. This runs automatically; `/reset` is still there if you'd
rather start clean instead.

## Package layout

```
astryx-cli/
├── pyproject.toml
├── README.md
└── astryx_cli/
    ├── __init__.py
    ├── cli.py        # everything: engine abstraction, loop, chat, sessions
    └── sandbox.py     # Docker-backed isolated execution for --sandbox
```

## Development

```bash
pip install -e .        # editable install -- code changes take effect immediately
astryx run "..."        # test it out
```

Since it's an editable install, there's no reinstall step while iterating
— just edit `astryx_cli/cli.py` and run `astryx` again.

## Example session

```
$ astryx chat

        ╾──────────✦ ASTRYX ✦──────────╼
        [banner]
        a general agentic CLI, tuned for Astryx

Chatting with Astryx -- same agent as `run`, just conversational...

you> write a function that checks if a string is a palindrome

── ASTRYX ──
┌─ astryx reasoning ──────────────────────────┐
│ Strip non-alphanumeric characters, lowercase │
│ everything, then compare to its reverse.     │
└───────────────────────────────────────────────┘
Here's a palindrome checker:

​```python
def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
​```

Context  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1,204 / 262,144 tokens (0.5%)

you> now add a docstring and a couple of test cases

── ASTRYX ──
...
```
