#!/usr/bin/env bash
# Install git hooks into .git/hooks/ as symlinks.
# Run once per clone. Re-run safely re-links existing hooks.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="$REPO_ROOT/scripts/hooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

if ! [[ -d "$HOOKS_DST" ]]; then
    echo "ERROR: .git/hooks not found at $HOOKS_DST. Run inside a git clone." >&2
    exit 1
fi

for hook in pre-push; do
    src="$HOOKS_SRC/$hook"
    dst="$HOOKS_DST/$hook"
    if ! [[ -f "$src" ]]; then
        echo "WARN: $src missing, skipping"
        continue
    fi
    chmod +x "$src"
    ln -sfn "$src" "$dst"
    echo "installed: $dst -> $src"
done

echo "Done. Bypass with 'git push --no-verify' in an emergency."
