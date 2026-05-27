#!/usr/bin/env bash
# Bump version (optional), commit, tag vX.Y.Z, push branch + tag so .github/workflows/release.yml runs.
# With no version argument: redeploy the current Cargo.toml version (delete remote tag if present, retag HEAD, push).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ALL_PLATFORMS=(
  x86_64-unknown-linux-gnu
  x86_64-unknown-linux-gnu-glibc2.31
  x86_64-unknown-linux-gnu-glibc2.28
  aarch64-apple-darwin
)

usage() {
  cat <<'EOF'
Usage: scripts/release.sh [--dry-run] [--platforms LIST] [VERSION]

  VERSION   New package semver (e.g. 1.4.0 or v1.4.0). Updates Cargo.toml + Cargo.lock, commits,
            pushes the current branch, deletes vVERSION tag on the remote if it exists, retags HEAD,
            and pushes the tag (triggers the Release workflow on tag v*).

  (no arg)  Redeploy binaries for the version already in Cargo.toml: same tag dance without a bump.

  --platforms, -p LIST
            Comma-separated build targets (default: all). Names are case-insensitive.

            Canonical IDs:
              x86_64-unknown-linux-gnu
              x86_64-unknown-linux-gnu-glibc2.31
              x86_64-unknown-linux-gnu-glibc2.28
              aarch64-apple-darwin

            Short aliases:
              linux, linux-gnu, linux-standard  -> x86_64-unknown-linux-gnu
              glibc2.31, linux-glibc2.31        -> x86_64-unknown-linux-gnu-glibc2.31
              glibc2.28, linux-glibc2.28        -> x86_64-unknown-linux-gnu-glibc2.28
              macos, mac, darwin                -> aarch64-apple-darwin
              all                               -> all platforms (default)

            Example: scripts/release.sh -p linux,macos 2.9.1

  --dry-run Print commands without running them.

Requires a clean git working tree and push access to origin.
EOF
  exit "${1:-0}"
}

to_lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

SELECTED_PLATFORMS=()

platform_contains() {
  local needle="$1"
  local item
  if [[ ${#SELECTED_PLATFORMS[@]} -eq 0 ]]; then
    return 1
  fi
  for item in "${SELECTED_PLATFORMS[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

add_platform() {
  local id="$1"
  if ! platform_contains "$id"; then
    SELECTED_PLATFORMS+=("$id")
  fi
}

normalize_platform_token() {
  local raw="$1"
  local p
  p="$(to_lower "${raw// /}")"
  case "$p" in
    "" | all) echo "__all__" ;;
    linux | linux-gnu | linux-standard | x86_64-unknown-linux-gnu)
      echo "x86_64-unknown-linux-gnu"
      ;;
    glibc2.31 | linux-glibc2.31 | x86_64-unknown-linux-gnu-glibc2.31)
      echo "x86_64-unknown-linux-gnu-glibc2.31"
      ;;
    glibc2.28 | linux-glibc2.28 | x86_64-unknown-linux-gnu-glibc2.28)
      echo "x86_64-unknown-linux-gnu-glibc2.28"
      ;;
    macos | mac | darwin | aarch64-apple-darwin)
      echo "aarch64-apple-darwin"
      ;;
    *)
      echo "unknown:$p" >&2
      return 1
      ;;
  esac
}

parse_platforms_list() {
  local list="$1"
  local token normalized

  if [[ -z "$list" || "$(to_lower "$list")" == "all" ]]; then
    SELECTED_PLATFORMS=("${ALL_PLATFORMS[@]}")
    return 0
  fi

  SELECTED_PLATFORMS=()
  local IFS=,
  for token in $list; do
    token="${token#"${token%%[![:space:]]*}"}"
    token="${token%"${token##*[![:space:]]}"}"
    [[ -z "$token" ]] && continue
    if ! normalized="$(normalize_platform_token "$token")"; then
      echo "error: unknown platform '$token'" >&2
      echo "Known platforms: ${ALL_PLATFORMS[*]}" >&2
      exit 1
    fi
    if [[ "$normalized" == "__all__" ]]; then
      SELECTED_PLATFORMS=("${ALL_PLATFORMS[@]}")
      return 0
    fi
    add_platform "$normalized"
  done

  if [[ "${#SELECTED_PLATFORMS[@]}" -eq 0 ]]; then
    echo "error: --platforms list is empty" >&2
    exit 1
  fi
}

DRY_RUN=0
PLATFORMS_RAW="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -p | --platforms)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a value" >&2
        usage 2
      fi
      PLATFORMS_RAW="$2"
      shift 2
      ;;
    -h | --help) usage 0 ;;
    -*) echo "unknown option: $1" >&2; usage 2 ;;
    *) break ;;
  esac
done

SELECTED_PLATFORMS=()
parse_platforms_list "$PLATFORMS_RAW"

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

RELEASE_ALL=1
if [[ "${#SELECTED_PLATFORMS[@]}" -ne "${#ALL_PLATFORMS[@]}" ]]; then
  RELEASE_ALL=0
fi

PLATFORMS_CSV="$(IFS=,; echo "${SELECTED_PLATFORMS[*]}")"
if [[ "$RELEASE_ALL" -eq 1 ]]; then
  echo "Platforms: all (${PLATFORMS_CSV})"
else
  echo "Platforms: ${PLATFORMS_CSV} (subset)"
fi

TAG_MESSAGE="Release $TAG"
if [[ "$RELEASE_ALL" -eq 0 ]]; then
  TAG_MESSAGE="${TAG_MESSAGE}

release-platforms: ${PLATFORMS_CSV}"
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

if [[ "$DRY_RUN" -eq 1 ]]; then
  run git tag -a "$TAG" -m "$TAG_MESSAGE"
else
  run git tag -a "$TAG" -F - <<EOF
$TAG_MESSAGE
EOF
fi
run git push origin "$TAG"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] done (no changes made)."
else
  if [[ "$RELEASE_ALL" -eq 1 ]]; then
    echo "Pushed $TAG. The Release workflow (on tag push) should build all platforms on GitHub Actions."
  else
    echo "Pushed $TAG. The Release workflow should build only: ${PLATFORMS_CSV}"
  fi
fi
