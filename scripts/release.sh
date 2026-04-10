#!/usr/bin/env bash
# Bump version (optional), commit, tag vX.Y.Z, push branch + tag so .github/workflows/release.yml runs.
# With no version argument: redeploy the current Cargo.toml version (delete remote tag if present, retag HEAD, push).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/release.sh [--dry-run] [VERSION]

  VERSION   New package semver (e.g. 1.4.0 or v1.4.0). Updates Cargo.toml + Cargo.lock, commits,
            pushes the current branch, deletes vVERSION tag on the remote if it exists, retags HEAD,
            and pushes the tag (triggers the Release workflow on tag v*).

  (no arg)  Redeploy binaries for the version already in Cargo.toml: same tag dance without a bump.

  --dry-run Print commands without running them.

Requires a clean git working tree and push access to origin.
EOF
  exit "${1:-0}"
}

DRY_RUN=0
while [[ "${1:-}" == -* ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
done

get_cargo_version() {
  grep '^version = ' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/'
}

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '+'; printf ' %q' "$@"; echo
  else
    "$@"
  fi
}

VERSION_ARG="${1:-}"
TARGET_VER=""

if [[ -n "$VERSION_ARG" ]]; then
  TARGET_VER="${VERSION_ARG#v}"
  if ! [[ "$TARGET_VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo "error: version must look like x.y.z (optional -prerelease suffix)" >&2
    exit 1
  fi
else
  TARGET_VER="$(get_cargo_version)"
fi

TAG="v$TARGET_VER"
CURRENT_IN_TOML="$(get_cargo_version)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

git rev-parse --git-dir >/dev/null 2>&1 || {
  echo "error: not a git repository" >&2
  exit 1
}

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is not clean; commit or stash before releasing." >&2
  exit 1
fi

if [[ -n "$VERSION_ARG" && "$TARGET_VER" != "$CURRENT_IN_TOML" ]]; then
  echo "Bumping Cargo.toml: $CURRENT_IN_TOML -> $TARGET_VER"
  run perl -i -pe "s/^version = \"[^\"]*\"/version = \"$TARGET_VER\"/" Cargo.toml
  run cargo update -p spacetravlr
  run git add Cargo.toml Cargo.lock
  run git commit -m "chore: release $TAG"
elif [[ -n "$VERSION_ARG" && "$TARGET_VER" == "$CURRENT_IN_TOML" ]]; then
  echo "Version $TAG already matches Cargo.toml; redeploying (retag + push only)."
fi

if [[ -z "$VERSION_ARG" ]]; then
  echo "Redeploying current version $TAG (retag HEAD + push)."
fi

run git fetch origin --tags

echo "Branch: $BRANCH  Tag: $TAG"
run git push origin "$BRANCH"

if git rev-parse "$TAG" >/dev/null 2>&1; then
  run git tag -d "$TAG"
fi
if git ls-remote --tags origin "refs/tags/$TAG" | grep -q .; then
  run git push origin ":refs/tags/$TAG"
fi

run git tag -a "$TAG" -m "Release $TAG"
run git push origin "$TAG"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] done (no changes made)."
else
  echo "Pushed $TAG. The Release workflow (on tag push) should run on GitHub Actions."
fi
