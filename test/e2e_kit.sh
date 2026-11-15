#!/usr/bin/env bash
# End-to-end local grade: real kit env + real entry + real kit scorer.
# Everything stays on loopback against the kit's simulated providers; no
# provider key is used and nothing leaves the machine.
#
# Usage: test/e2e_kit.sh <trial-index 0|1|2> [entry-dir]
set -u
TRIAL="${1:-0}"
ENTRY="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
KIT="${KIT_DIR:-$HOME/workspace/agent-wallet-sol/free-model-aggregator-01-kit}"
WORK=$(mktemp -d /tmp/agg01-e2e.XXXXXX)
STATE="$WORK/state"
mkdir -p "$STATE" "$WORK/scores"
echo "workdir: $WORK"

MAKEFUN_ENV_DATA_DIR="$KIT/practice/trial-$TRIAL/input/env" \
MAKEFUN_STATE_DIR="$STATE" \
MAKEFUN_ENV_PORT=18101 \
node "$KIT/program/env/server.mjs" > "$WORK/env.log" 2>&1 &
ENV_PID=$!

for i in $(seq 1 50); do [ -f "$STATE/ready" ] && break; sleep 0.1; done
if [ ! -f "$STATE/ready" ]; then echo "ENV FAILED TO START"; cat "$WORK/env.log"; kill $ENV_PID 2>/dev/null; exit 1; fi

MAKEFUN_ENV_URL="http://127.0.0.1:18101" \
MAKEFUN_PROMPT_FILE="$KIT/practice/trial-$TRIAL/input/prompt.txt" \
MAKEFUN_TRIAL_SEED="e2e-local" \
node "$ENTRY/main.ts" > "$WORK/entry.log" 2>&1 &
ENTRY_PID=$!

# The entry exits 0 on its own when the workload is done (60 s cap here).
for i in $(seq 1 600); do kill -0 $ENTRY_PID 2>/dev/null || break; sleep 0.1; done
if kill -0 $ENTRY_PID 2>/dev/null; then echo "ENTRY DID NOT EXIT"; kill $ENTRY_PID; else wait $ENTRY_PID; echo "entry exit code: $?"; fi

kill -TERM $ENV_PID 2>/dev/null
for i in $(seq 1 50); do kill -0 $ENV_PID 2>/dev/null || break; sleep 0.1; done
kill -9 $ENV_PID 2>/dev/null

if [ ! -f "$STATE/export/state.json" ]; then echo "NO STATE EXPORT"; cat "$WORK/env.log"; cat "$WORK/entry.log"; exit 1; fi

MAKEFUN_OUTPUT_DIR="$STATE/export" \
MAKEFUN_REFERENCE_DIR="$KIT/practice/trial-$TRIAL/reference" \
MAKEFUN_SCORES_DIR="$WORK/scores" \
node "$KIT/program/scorer/score.mjs"
echo "scores.json:"
cat "$WORK/scores/scores.json"
echo "--- entry log:"
cat "$WORK/entry.log"
echo "WORKDIR=$WORK"
