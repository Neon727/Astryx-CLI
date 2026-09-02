# Astryx CLI

A general-purpose agentic coding CLI — runs a coder → critic → reviser
loop with visible reasoning traces, real test execution, and shell command
access, on top of a local model. Defaults to Astryx (optimized for this
loop specifically), but works with any local Hugging Face model.

## Install

```bash
git clone https://github.com/Neon727/Astryx-CLI
cd astryx-cli
pip install -e
```

That gives you a global `astryx` command — no more `cd`-ing into a scripts
folder or typing `python astryx_cli.py` every time.

Optional, only if you use `--sandbox`: [Docker](https://docs.docker.com/get-docker/), installed and running.

## Pointing it at a trained Astryx model

Since `astryx` now runs from anywhere, it needs a fixed place to look for
things rather than a path relative to wherever you happen to be standing.
That's `~/.astryx/` by default:

| What | Default location | Override |
|---|---|---|
| Merged model | `~/.astryx/astryx-merged` | `ASTRYX_MERGED_PATH` |
| LoRA adapter (if not merged) | `~/.astryx/astryx-lora` | `ASTRYX_ADAPTER_PATH` |
| Chat sessions | `~/.astryx/sessions` | `ASTRYX_SESSIONS_DIR` |
| (base for all of the above) | `~/.astryx` | `ASTRYX_HOME` |

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
using real test execution (if you pass `--tests`) until it passes or hits
`--max-iters`; then it exits. Without `--tests`, it just returns the first
draft with no verification loop.

**`chat`** is persistent. Same agent, same loop, but it keeps the whole
conversation in context — so after it writes something, you just keep
talking ("now add error handling", "extract that into its own function")
instead of relaunching the CLI for every follow-up.

Inside `chat`:

| Command | What it does |
|---|---|
| *(plain message)* | continues the conversation normally |
| `/load <file>` | pulls a file's contents into context |
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
astryx chat --model /path/to/local-model      # any local HF model
astryx chat --model Qwen/Qwen2.5-7B-Instruct  # ...straight from the Hub
```

Local models only, this CLI does not support APIs for external services.

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
| `--tests` | `run` | none | Test file to verify generated code against |
| `--max-iters` | `run` | `3` | Cap on coder↔critic↔reviser loop iterations |
| `--no-think` | both | off | Hide reasoning-trace panels |
| `--sandbox` | both | off | Run shell commands in an isolated Docker container instead of confirming each on your host |
| `--sandbox-dir` | both | `./astryx_sandbox` | Directory mounted into the sandbox container (cwd-relative on purpose — it mounts near whatever project you're currently in) |
| `--sandbox-network` | both | off | Allow network access inside the sandbox |
| `--resume [id]` | `chat` | none | Resume a saved session; omit the id to resume the most recent one |

## What the CLI shows you

- **Role-labeled reasoning traces** — coder, critic, and reviser each emit
  a `<think>...</think>` block before acting, shown in a dim panel titled
  with whichever role produced it, so it's clear which step you're
  looking at.
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
- Container is created fresh at session start and removed on exit

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
