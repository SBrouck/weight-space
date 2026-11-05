#!/bin/sh
# Run the editor agent (Claude Code, headless) in a workspace until its pass budget is spent.
#
# Usage: harness/run_agent.sh ~/ws-runs/run1 [max_rounds] [max_usd_per_round]
# Each round is one `claude -p` call on the same session; later rounds resume it and ask the agent
# to continue from notes.md. Everything is logged to <workspace>/transcript.jsonl.
set -euf  # -f: no globbing, the tool patterns contain *

WS=$(cd "$1" && pwd)
ROUNDS=${2:-6}
USD=${3:-15}
SESSION_FILE="$WS/.session_id"
[ -f "$SESSION_FILE" ] || uuidgen | tr 'A-Z' 'a-z' > "$SESSION_FILE"
SESSION=$(cat "$SESSION_FILE")
cd "$WS"

ALLOWED='Bash(./kpy:*) Bash(./fwd:*) Read Write Edit Glob Grep'
DENIED="WebFetch WebSearch Agent Bash(cd:*) Read(/$HOME/ai-engineering/**) Read(/$HOME/.cache/**) Read(/$HOME/.claude/**)"

# A generation pass takes 10-30 minutes: let the agent's Bash calls run up to 45 minutes.
export BASH_DEFAULT_TIMEOUT_MS=2700000 BASH_MAX_TIMEOUT_MS=2700000

left() { ./fwd status | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['budget']-d['used'])"; }

round=1
while [ "$round" -le "$ROUNDS" ]; do
  if [ "$round" -eq 1 ] && [ ! -s transcript.jsonl ]; then
    PROMPT="$(cat task.md)

Start now. Begin by reading notes.md and running ./fwd status."
    RESUME="--session-id $SESSION"
  else
    [ "$(left)" -gt 0 ] || { echo "budget spent"; break; }
    PROMPT="Continue the task in task.md from where you stopped. Re-read notes.md and run ./fwd status first. $(left) passes are left (the budget may have been raised since your last report: if so, keep working and rewrite final_report.md at the end). When they are all spent, leave your best edit in edit/ and write final_report.md."
    RESUME="--resume $SESSION"
  fi
  echo "== round $round, passes left: $(left)"
  # shellcheck disable=SC2086
  command claude -p "$PROMPT" $RESUME \
    --model claude-opus-5-5 \
    --allowedTools $ALLOWED --disallowedTools $DENIED \
    --strict-mcp-config --max-budget-usd "$USD" \
    --output-format stream-json --verbose >> transcript.jsonl 2>> transcript.stderr || true
  [ -f final_report.md ] && [ "$(left)" -eq 0 ] && { echo "done: final_report.md written"; break; }
  round=$((round + 1))
done
./fwd status
