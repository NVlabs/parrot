#!/usr/bin/env bash
# sync-upstream.sh
#
# Synchronize branches across three remotes:
#   upstream        = NVlabs/parrot  (fetch; https)
#   upstream-direct = NVlabs/parrot  (push; ssh)
#   origin          = codereport/nvlabs-parrot  (your fork)
#
# What it does:
#   1. Hard-resets `main` to `upstream/main` and force-pushes to `origin`,
#      so `origin/main` mirrors NVlabs exactly (no merge/fork commits).
#   2. Updates `experimental-parrot-python` to include the latest `upstream/main`:
#        - default strategy: merge upstream/main into the branch (one conflict
#          resolution for the whole divergence, preserves history, no force-push
#          needed when the branch only moves forward).
#        - optional --rebase: linearize the branch onto upstream/main (cleaner
#          history, but re-replays every commit; more conflicts on old branches).
#      No-op if upstream/main is already an ancestor of the experimental branch.
#   3. Pushes the experimental branch to `upstream-direct` (and origin).
#
# Any uncommitted work in the working tree is auto-stashed and restored at the
# end. Force pushes use --force-with-lease for safety.
#
# Usage:
#   ./scripts/sync-upstream.sh             # merge strategy (default)
#   ./scripts/sync-upstream.sh --rebase    # rebase strategy
#   ./scripts/sync-upstream.sh --merge     # explicit merge

set -euo pipefail

STRATEGY=merge
for arg in "$@"; do
  case "$arg" in
    --rebase) STRATEGY=rebase ;;
    --merge)  STRATEGY=merge ;;
    -h|--help)
      sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

UPSTREAM_FETCH="${UPSTREAM_FETCH:-upstream}"
UPSTREAM_PUSH="${UPSTREAM_PUSH:-upstream-direct}"
ORIGIN="${ORIGIN:-origin}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"
EXPERIMENTAL_BRANCH="${EXPERIMENTAL_BRANCH:-experimental-parrot-python}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

# --- sanity checks -----------------------------------------------------------

git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "Not inside a git work tree."

for r in "$UPSTREAM_FETCH" "$UPSTREAM_PUSH" "$ORIGIN"; do
  git remote get-url "$r" >/dev/null 2>&1 \
    || die "Remote '$r' is not configured. Run 'git remote -v' to check."
done

START_BRANCH=$(git symbolic-ref --quiet --short HEAD || echo "")
[ -n "$START_BRANCH" ] || die "Detached HEAD; checkout a branch first."

rebase_in_progress() {
  [ -d "$(git rev-parse --git-path rebase-merge)" ] \
    || [ -d "$(git rev-parse --git-path rebase-apply)" ]
}

merge_in_progress() {
  [ -f "$(git rev-parse --git-path MERGE_HEAD)" ] \
    || [ -f "$(git rev-parse --git-path CHERRY_PICK_HEAD)" ]
}

if rebase_in_progress || merge_in_progress; then
  die "A rebase/merge is already in progress. Resolve or abort it first."
fi

# --- stash any local changes -------------------------------------------------

STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet || \
   [ -n "$(git ls-files --others --exclude-standard)" ]; then
  log "Stashing local changes (including untracked)…"
  git stash push --include-untracked -m "sync-upstream autostash $(date +%s)" >/dev/null
  STASHED=1
fi

cleanup() {
  rc=$?
  # If a rebase/merge is in flight, leave everything alone for manual resolution.
  if rebase_in_progress || merge_in_progress; then
    if [ "$STASHED" = "1" ]; then
      warn "Stash kept at stash@{0} (autostash from sync-upstream). After resolving,"
      warn "finish with: git checkout $START_BRANCH && git stash pop"
    fi
    exit $rc
  fi
  if git symbolic-ref --quiet --short HEAD >/dev/null 2>&1; then
    current=$(git symbolic-ref --short HEAD)
    if [ "$current" != "$START_BRANCH" ]; then
      log "Returning to '$START_BRANCH'…"
      git checkout --quiet "$START_BRANCH" || true
    fi
  fi
  if [ "$STASHED" = "1" ]; then
    log "Restoring stashed changes…"
    git stash pop --quiet || warn "Stash pop had conflicts; resolve manually (see 'git stash list')."
  fi
  exit $rc
}
trap cleanup EXIT

# --- fetch everything --------------------------------------------------------

log "Fetching $UPSTREAM_FETCH, $UPSTREAM_PUSH, $ORIGIN…"
git fetch --prune "$UPSTREAM_FETCH"
git fetch --prune "$UPSTREAM_PUSH"
git fetch --prune "$ORIGIN"

UPSTREAM_MAIN="$UPSTREAM_FETCH/$MAIN_BRANCH"
UPSTREAM_EXP="$UPSTREAM_PUSH/$EXPERIMENTAL_BRANCH"

# --- sync main: origin/main := upstream/main (exactly) -----------------------

LOCAL_MAIN_SHA=$(git rev-parse "$MAIN_BRANCH")
UPSTREAM_MAIN_SHA=$(git rev-parse "$UPSTREAM_MAIN")
ORIGIN_MAIN_SHA=$(git rev-parse "$ORIGIN/$MAIN_BRANCH" 2>/dev/null || echo "")

if [ "$LOCAL_MAIN_SHA" = "$UPSTREAM_MAIN_SHA" ] && \
   [ "$ORIGIN_MAIN_SHA" = "$UPSTREAM_MAIN_SHA" ]; then
  log "'$MAIN_BRANCH' already matches '$UPSTREAM_MAIN' — skipping."
else
  log "Syncing '$MAIN_BRANCH' to '$UPSTREAM_MAIN' ($UPSTREAM_MAIN_SHA)…"
  git checkout --quiet "$MAIN_BRANCH"
  git reset --hard "$UPSTREAM_MAIN"
  log "Force-pushing '$MAIN_BRANCH' to '$ORIGIN'…"
  git push --force-with-lease "$ORIGIN" "$MAIN_BRANCH"
fi

# --- sync experimental ------------------------------------------------------

log "Updating local '$EXPERIMENTAL_BRANCH' to '$UPSTREAM_EXP'…"
if git show-ref --quiet --verify "refs/heads/$EXPERIMENTAL_BRANCH"; then
  git checkout --quiet "$EXPERIMENTAL_BRANCH"
  git reset --hard "$UPSTREAM_EXP"
else
  git checkout --quiet -b "$EXPERIMENTAL_BRANCH" "$UPSTREAM_EXP"
fi

# No-op check: is upstream/main already fully contained in the experimental branch?
if git merge-base --is-ancestor "$UPSTREAM_MAIN" HEAD; then
  log "'$EXPERIMENTAL_BRANCH' already contains '$UPSTREAM_MAIN' — no sync needed."
  SYNC_NEEDED=0
else
  SYNC_NEEDED=1
fi

if [ "$SYNC_NEEDED" = "1" ]; then
  case "$STRATEGY" in
    merge)
      log "Merging '$UPSTREAM_MAIN' into '$EXPERIMENTAL_BRANCH'…"
      if ! git merge --no-edit "$UPSTREAM_MAIN"; then
        warn "Merge has conflicts. Resolve them, then run:"
        warn "    git add <files> && git commit"
        warn "    git push --force-with-lease $UPSTREAM_PUSH $EXPERIMENTAL_BRANCH"
        warn "    git push --force-with-lease $ORIGIN $EXPERIMENTAL_BRANCH"
        exit 1
      fi
      ;;
    rebase)
      log "Rebasing '$EXPERIMENTAL_BRANCH' onto '$UPSTREAM_MAIN'…"
      if ! git rebase "$UPSTREAM_MAIN"; then
        warn "Rebase has conflicts. Resolve them, then run:"
        warn "    git rebase --continue"
        warn "    git push --force-with-lease $UPSTREAM_PUSH $EXPERIMENTAL_BRANCH"
        warn "    git push --force-with-lease $ORIGIN $EXPERIMENTAL_BRANCH"
        warn "Tip: if the rebase is too conflict-heavy, abort and rerun without --rebase"
        warn "     to use merge instead (one conflict resolution for the whole divergence)."
        exit 1
      fi
      ;;
  esac

  log "Pushing '$EXPERIMENTAL_BRANCH' to '$UPSTREAM_PUSH'…"
  git push --force-with-lease "$UPSTREAM_PUSH" "$EXPERIMENTAL_BRANCH"

  if git show-ref --quiet --verify "refs/remotes/$ORIGIN/$EXPERIMENTAL_BRANCH"; then
    log "Pushing '$EXPERIMENTAL_BRANCH' to '$ORIGIN'…"
    git push --force-with-lease "$ORIGIN" "$EXPERIMENTAL_BRANCH"
  fi
fi

log "Done. Summary:"
echo "  $ORIGIN/$MAIN_BRANCH                   = $UPSTREAM_MAIN"
if [ "$SYNC_NEEDED" = "0" ]; then
  echo "  $UPSTREAM_PUSH/$EXPERIMENTAL_BRANCH    = already contains $UPSTREAM_MAIN (no change)"
else
  echo "  $UPSTREAM_PUSH/$EXPERIMENTAL_BRANCH    = $STRATEGY-synced with $UPSTREAM_MAIN"
fi
