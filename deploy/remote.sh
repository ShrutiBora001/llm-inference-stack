#!/usr/bin/env bash
# One-command deploy to a rented GPU box.
#
#   deploy/remote.sh 16094 23.127.144.217 all
#
# Syncs the repo, builds nothing (uses the NGC image directly), runs the chosen
# verb, and pulls results back. No credentials ever touch the remote host --
# rsync over the SSH key you already added to the provider, no tokens, no git.
set -euo pipefail

PORT="${1:?usage: remote.sh PORT HOST [verb]}"
HOST="${2:?usage: remote.sh PORT HOST [verb]}"
VERB="${3:-all}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/lis}"
LOCAL_RESULTS="${LOCAL_RESULTS:-results}"

# Name the key explicitly rather than relying on ssh-agent state. A key whose
# filename is not one of ssh's defaults (id_rsa, id_ed25519) is invisible to a
# bare `ssh` when the agent is empty, which reads as "provider rejected my key"
# and sends you looking in the wrong place.
#
#   SSH_KEY=~/.ssh/id_ed25519_github deploy/remote.sh 40778 88.207.84.248 kernel
SSH_KEY="${SSH_KEY:-}"
KEY_OPT=""
if [ -n "$SSH_KEY" ]; then
  [ -f "$SSH_KEY" ] || { echo "FATAL: SSH_KEY=$SSH_KEY does not exist"; exit 1; }
  KEY_OPT="-i $SSH_KEY -o IdentitiesOnly=yes"
fi

SSH_BASE="ssh -p $PORT $KEY_OPT -o StrictHostKeyChecking=accept-new"
SSH="$SSH_BASE root@$HOST"
here="$(cd "$(dirname "$0")/.." && pwd)"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "sync -> $HOST:$REMOTE_DIR"
rsync -az --delete -e "$SSH_BASE" \
  --exclude .venv --exclude .git --exclude results --exclude __pycache__ \
  --exclude '*.nsys-rep' \
  "$here/" "root@$HOST:$REMOTE_DIR/"

# The upstream finding is a separate repo; vendor it so the box needs no git
# access and no credentials.
if [ -d "$here/../Distributed_Attention00" ]; then
  say "vendor dattn"
  # rsync will not create intermediate directories, and --mkpath is too new to
  # rely on (macOS ships an rsync that lacks it).
  $SSH "mkdir -p $REMOTE_DIR/vendor/dattn"
  rsync -az -e "$SSH_BASE" \
    --exclude .venv --exclude .git --exclude results --exclude __pycache__ \
    "$here/../Distributed_Attention00/" "root@$HOST:$REMOTE_DIR/vendor/dattn/"
fi

say "setup (idempotent)"
$SSH "cd $REMOTE_DIR && bash deploy/setup_remote.sh"

say "run: $VERB"
$SSH "cd $REMOTE_DIR && LIS_WORLD_SIZE=${LIS_WORLD_SIZE:-4} \
      LIS_SEQ=${LIS_SEQ:-32768} LIS_LAYOUT=${LIS_LAYOUT:-striped} \
      LIS_BLOCK=${LIS_BLOCK:-512} LIS_RESULTS=$REMOTE_DIR/results \
      bash deploy/entrypoint.sh $VERB"

say "pull results"
mkdir -p "$here/$LOCAL_RESULTS"
rsync -az -e "$SSH_BASE" "root@$HOST:$REMOTE_DIR/results/" "$here/$LOCAL_RESULTS/"
echo "  -> $here/$LOCAL_RESULTS"

say "DONE. Destroy the instance now -- rented storage vanishes when you stop paying."
