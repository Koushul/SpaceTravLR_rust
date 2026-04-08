#!/usr/bin/env sh
# SpaceTravLR CLI installer — keep tarball names / target triples in sync with
# src/self_update.rs (GITHUB_REPO, tarball pattern, host_target_triple).
# Quick install:
#   curl -fsSL https://raw.githubusercontent.com/Koushul/SpaceTravLR_rust/refs/heads/main/scripts/install.sh | sh
set -e

REPO="${SPACETRAVLR_GITHUB_REPO:-Koushul/SpaceTravLR_rust}"
BINARY_NAMES="spacetravlr spacetravlr-perturb spatial_viewer"
INSTALL_DIR="${SPACETRAVLR_INSTALL_DIR:-$HOME/.local/bin}"
UNAME_S="${UNAME_S:-$(uname -s)}"
UNAME_M="${UNAME_M:-$(uname -m)}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

QUIET=0
for arg in "$@"; do
    case "$arg" in
        --quiet) QUIET=1 ;;
    esac
done

is_tty() {
    [ -t 2 ]
}

color_ok=1
if ! is_tty || [ "$QUIET" -eq 1 ]; then
    color_ok=0
fi

info() {
    if [ "$QUIET" -eq 1 ]; then
        return 0
    fi
    if [ "$color_ok" -eq 1 ]; then
        printf "${GREEN}[INFO]${NC} %s\n" "$1"
    else
        printf "[INFO] %s\n" "$1"
    fi
}

warn() {
    if [ "$QUIET" -eq 1 ]; then
        return 0
    fi
    if [ "$color_ok" -eq 1 ]; then
        printf "${YELLOW}[WARN]${NC} %s\n" "$1"
    else
        printf "[WARN] %s\n" "$1"
    fi
}

error() {
    if [ "$color_ok" -eq 1 ]; then
        printf "${RED}[ERROR]${NC} %s\n" "$1" >&2
    else
        printf "[ERROR] %s\n" "$1" >&2
    fi
    exit 1
}

step() {
    if [ "$QUIET" -eq 1 ]; then
        return 0
    fi
    if [ "$color_ok" -eq 1 ]; then
        printf "${BLUE}${BOLD}=>${NC} %s\n" "$1"
    else
        printf "=> %s\n" "$1"
    fi
}

detect_os() {
    case "$UNAME_S" in
        Linux*) OS="linux" ;;
        Darwin*) OS="darwin" ;;
        *) error "Unsupported operating system: $UNAME_S" ;;
    esac
}

detect_arch() {
    case "$UNAME_M" in
        x86_64 | amd64) ARCH="x86_64" ;;
        arm64 | aarch64) ARCH="aarch64" ;;
        *) error "Unsupported architecture: $UNAME_M" ;;
    esac
}

get_target() {
    TARGET=""
    case "$OS" in
        linux)
            case "$ARCH" in
                x86_64) TARGET="x86_64-unknown-linux-gnu" ;;
                aarch64) TARGET="aarch64-unknown-linux-gnu" ;;
            esac
            ;;
        darwin)
            TARGET="${ARCH}-apple-darwin"
            ;;
    esac
    if [ -z "$TARGET" ]; then
        error "Unsupported combination $OS $ARCH"
    fi
}

api_latest_json() {
    API_URL="${SPACETRAVLR_GH_API:-https://api.github.com}/repos/${REPO}/releases/latest"
    curl -fsSL \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        -H "User-Agent: spacetravlr-install-script" \
        "$API_URL"
}

get_latest_version() {
    step "Resolving latest release (GitHub)…"
    _json="$(api_latest_json)" || error "Failed to fetch release metadata"

    if command -v jq >/dev/null 2>&1; then
        VERSION="$(printf '%s' "$_json" | jq -r '.tag_name // empty')"
    else
        VERSION="$(printf '%s' "$_json" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
    fi
    if [ -z "$VERSION" ]; then
        error "Failed to parse tag_name from GitHub API"
    fi
}

tarball_basename() {
    printf 'spacetravlr-%s-%s.tar.gz' "$VERSION" "$TARGET"
}

checksum_for_tarball() {
    printf '%s' "$1" | awk -v t="$2" '
        NF >= 2 {
            h = $1
            f = $2
            sub(/^\*/, "", f)
            if (f == t) { print h; exit 0 }
        }
    '
}

download_url_to() {
    _url="$1"
    _path="$2"
    if is_tty && [ "${CI:-}" != "true" ] && [ "$QUIET" -eq 0 ]; then
        curl -fSL --progress-bar "$_url" -o "$_path"
    else
        curl -fsSL "$_url" -o "$_path"
    fi
}

install_release() {
    BASE="https://github.com/${REPO}/releases/download/${VERSION}/"
    TAR="$(tarball_basename)"
    DOWNLOAD_URL="${BASE}${TAR}"

    step "Downloading ${TAR}…"
    TEMP_DIR="$(mktemp -d)"
    ARCHIVE="${TEMP_DIR}/${TAR}"

    download_url_to "$DOWNLOAD_URL" "$ARCHIVE" || error "Failed to download binary archive"

    if [ "${INSTALL_VERIFY_CHECKSUM:-1}" != "0" ]; then
        SUMS_URL="${BASE}SHA256SUMS"
        step "Verifying SHA256 checksum…"
        _sums="$(curl -fsSL "$SUMS_URL")" || error "Failed to download SHA256SUMS"
        _expect="$(checksum_for_tarball "$_sums" "$TAR")"
        if [ -z "$_expect" ]; then
            error "SHA256SUMS missing entry for ${TAR}"
        fi
        if command -v shasum >/dev/null 2>&1; then
            _got="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
        else
            _got="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
        fi
        if [ "$_expect" != "$_got" ]; then
            error "Checksum mismatch (expected $_expect, got $_got)"
        fi
    fi

    step "Extracting…"
    tar -xzf "$ARCHIVE" -C "$TEMP_DIR"

    step "Installing to ${INSTALL_DIR}…"
    mkdir -p "$INSTALL_DIR"
    for b in $BINARY_NAMES; do
        if [ ! -f "${TEMP_DIR}/${b}" ]; then
            error "Archive missing ${b}"
        fi
        mv "${TEMP_DIR}/${b}" "${INSTALL_DIR}/"
        chmod +x "${INSTALL_DIR}/${b}"
    done

    rm -rf "$TEMP_DIR"
}

verify_install() {
    step "Verifying install…"
    if command -v spacetravlr >/dev/null 2>&1; then
        info "spacetravlr: $(spacetravlr --version 2>&1 | head -1)"
    else
        warn "Installed to ${INSTALL_DIR} but spacetravlr is not on PATH."
        warn "Add to your shell profile: export PATH=\"${INSTALL_DIR}:\$PATH\""
    fi
}

dry_run() {
    step "Dry run (no download) — resolved:"
    printf '  REPO=%s\n' "$REPO"
    printf '  VERSION=%s\n' "$VERSION"
    printf '  TARGET=%s\n' "$TARGET"
    printf '  TAR=%s\n' "$(tarball_basename)"
    printf '  INSTALL_DIR=%s\n' "$INSTALL_DIR"
    info "Unset INSTALL_DRY_RUN to perform install."
}

main() {
    case "${1:-}" in
        -h | --help)
            printf '%s\n' "Usage: install.sh [--quiet]
Environment:
  SPACETRAVLR_INSTALL_DIR   install prefix (default: \$HOME/.local/bin)
  SPACETRAVLR_GITHUB_REPO   owner/repo override
  INSTALL_DRY_RUN=1         print plan only
  INSTALL_TEST_VERSION=v…    skip API; use this tag (for tests / air-gapped preview)
  INSTALL_VERIFY_CHECKSUM=0 skip SHA256SUMS check (not recommended)
  UNAME_S / UNAME_M         override platform (tests)
"
            exit 0
            ;;
    esac

    info "Installing SpaceTravLR CLIs (${REPO})…"

    step "Detecting platform…"
    detect_os
    detect_arch
    get_target
    info "Detected: $OS $ARCH → $TARGET"

    if [ -n "${INSTALL_TEST_VERSION:-}" ]; then
        VERSION="$INSTALL_TEST_VERSION"
        info "Using INSTALL_TEST_VERSION=$VERSION (-- dry-run / test hook)"
    else
        get_latest_version
    fi
    info "Selected release: $VERSION"

    if [ "${INSTALL_DRY_RUN:-}" = "1" ]; then
        dry_run
        exit 0
    fi

    install_release

    info "Successfully installed: $BINARY_NAMES"
    verify_install

    echo ""
    info "Done. Run 'spacetravlr --help' to get started."
}

main "$@"
