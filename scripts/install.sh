#!/usr/bin/env sh
# SpaceTravLR CLI installer — keep tarball names in sync with src/self_update.rs
# (GITHUB_REPO, tarball_name, LINUX_GNU_*, prebuilt_tarball_target).
# After binaries: downloads human_network.parquet + mouse_network.parquet + spaceship_config.toml
# into INSTALL_DIR/data/ from raw.githubusercontent.com (release tag, then main).
# curl -fsSL …/install.sh | sh
set -e

REPO="${SPACETRAVLR_GITHUB_REPO:-Koushul/SpaceTravLR_rust}"
BINARY_NAMES="spacetravlr spacetravlr-perturb spatial_viewer"
INSTALL_DIR="${SPACETRAVLR_INSTALL_DIR:-$HOME/.local/bin}"
UNAME_S="${UNAME_S:-$(uname -s)}"
UNAME_M="${UNAME_M:-$(uname -m)}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

QUIET=0
USE_COLOR=0
for arg in "$@"; do
    case "$arg" in
        --quiet) QUIET=1 ;;
        --color) USE_COLOR=1 ;;
    esac
done

# Plain text by default so terminal scrollback / log files stay readable (no ANSI).
# Colors: pass --color or set SPACETRAVLR_INSTALL_COLOR=1. Honors https://no-color.org/
is_tty() {
    [ -t 1 ] && [ -t 2 ]
}

color_ok=0
if [ "$QUIET" -eq 0 ] && is_tty && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-}" != "dumb" ]; then
    if [ "$USE_COLOR" -eq 1 ] || [ "${SPACETRAVLR_INSTALL_COLOR:-0}" = "1" ]; then
        color_ok=1
    fi
fi

info() {
    [ "$QUIET" -eq 1 ] && return 0
    if [ "$color_ok" -eq 1 ]; then
        printf '%s%s%s\n' "$GREEN" "$1" "$NC"
    else
        printf '%s\n' "$1"
    fi
}

warn() {
    [ "$QUIET" -eq 1 ] && return 0
    if [ "$color_ok" -eq 1 ]; then
        printf '%s%s%s\n' "$YELLOW" "$1" "$NC" >&2
    else
        printf '%s\n' "$1" >&2
    fi
}

error() {
    if [ "$color_ok" -eq 1 ]; then
        printf '%s[ERROR] %s%s\n' "$RED" "$1" "$NC" >&2
    else
        printf '[ERROR] %s\n' "$1" >&2
    fi
    exit 1
}

# $1 = percent 0-100, $2 = label
# Build the bar as two strings so we emit a few ANSI sequences per line (not one per column).
progress_line() {
    [ "$QUIET" -eq 1 ] && return 0
    _pct="$1"
    _label="$2"
    _w=28
    _fill=$((_pct * _w / 100))
    if [ "$_fill" -lt 0 ]; then _fill=0; fi
    if [ "$_fill" -gt "$_w" ]; then _fill="$_w"; fi
    _i=0
    _filled=""
    while [ "$_i" -lt "$_fill" ]; do
        _filled="${_filled}="
        _i=$((_i + 1))
    done
    _i=$_fill
    _empty=""
    while [ "$_i" -lt "$_w" ]; do
        _empty="${_empty}."
        _i=$((_i + 1))
    done
    if [ "$color_ok" -eq 1 ]; then
        printf '  %s[%s' "$DIM" "$NC"
        printf '%s%s%s' "$CYAN" "$_filled" "$NC"
        if [ -n "$_empty" ]; then
            printf '%s%s' "$DIM" "$_empty"
        fi
        printf '%s]%s %3d%%  %s\n' "$DIM" "$NC" "$_pct" "$_label"
    else
        printf '  [%s%s] %3d%%  %s\n' "$_filled" "$_empty" "$_pct" "$_label"
    fi
}

show_banner() {
    [ "$QUIET" -eq 1 ] && return 0
    if [ "$color_ok" -eq 1 ]; then
        printf '%s%s' "$CYAN" "$BOLD"
    fi
    cat << 'EOF'

 ____                             _____                      _      ____
/ ___|  _ __    __ _   ___   ___ |_   _| _ __   __ _ __   __| |    |  _ \
\___ \ | '_ \  / _` | / __| / _ \  | |  | '__| / _` |\ \ / /| |    | |_) |
 ___) || |_) || (_| || (__ |  __/  | |  | |   | (_| | \ V / | |___ |  _ <
|____/ | .__/  \__,_| \___| \___|  |_|  |_|    \__,_|  \_/  |_____||_| \_\
       |_|
EOF
    if [ "$color_ok" -eq 1 ]; then
        printf '%s%s  Characterizing Functional Microniches%s\n' "$NC" "$DIM" "$NC"
    else
        printf '  Characterizing Functional Microniches\n'
    fi
}

show_release_version() {
    _v="$1"
    [ "$QUIET" -eq 1 ] && return 0
    if [ "$color_ok" -eq 1 ]; then
        printf '\n  %sRelease%s  %s%s%s%s\n\n' "$BOLD" "$NC" "$CYAN" "$BOLD" "$_v" "$NC"
    else
        printf '\n  Release  %s\n\n' "$_v"
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
                aarch64)
                    error "No prebuilt Linux ARM64 binaries. Build from source: cargo install spacetravlr --locked --features spatial-viewer"
                    ;;
            esac
            ;;
        darwin)
            case "$ARCH" in
                aarch64) TARGET="aarch64-apple-darwin" ;;
                x86_64)
                    error "No prebuilt Intel Mac binaries. Build from source: cargo install spacetravlr --locked --features spatial-viewer"
                    ;;
            esac
            ;;
    esac
    if [ -z "$TARGET" ]; then
        error "Unsupported combination $OS $ARCH"
    fi
}

linux_gnu_triple="x86_64-unknown-linux-gnu"
linux_gnu_compat_suffix="-glibc2.31"

linux_set_tarball_target_for_glibc() {
    [ "$OS" = linux ] && [ "$ARCH" = x86_64 ] || return 0
    base="$linux_gnu_triple"
    if [ -n "${SPACETRAVLR_LINUX_VARIANT:-}" ]; then
        case "$SPACETRAVLR_LINUX_VARIANT" in
            standard) TARGET="$base" ;;
            compat) TARGET="${base}${linux_gnu_compat_suffix}" ;;
            *) error "Invalid SPACETRAVLR_LINUX_VARIANT; use standard or compat" ;;
        esac
        return 0
    fi
    if ! command -v ldd >/dev/null 2>&1; then
        error "ldd not found (need GNU libc). Set SPACETRAVLR_LINUX_VARIANT=standard or compat to skip detection."
    fi
    _line="$(ldd --version 2>/dev/null | head -1)" || error "ldd --version failed"
    _ver="$(printf '%s' "$_line" | awk '{
        for (i = 1; i <= NF; i++) {
            if (match($i, /^[0-9]+\.[0-9]+/)) v = substr($i, RSTART, RLENGTH)
        }
        print v
    }')"
    if [ -z "$_ver" ]; then
        error "Could not parse glibc from: $_line — set SPACETRAVLR_LINUX_VARIANT=standard or compat"
    fi
    _maj="${_ver%%.*}"
    _rest="${_ver#*.}"
    _min="${_rest%%[^0-9]*}"
    if [ "$_maj" -gt 2 ] 2>/dev/null || { [ "$_maj" -eq 2 ] && [ "${_min:-0}" -ge 35 ] 2>/dev/null; }; then
        TARGET="$base"
    else
        TARGET="${base}${linux_gnu_compat_suffix}"
    fi
}

api_newest_release_json() {
    API_URL="${SPACETRAVLR_GH_API:-https://api.github.com}/repos/${REPO}/releases?per_page=1"
    curl -fsSL \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        -H "User-Agent: spacetravlr-install-script" \
        "$API_URL"
}

get_latest_version() {
    progress_line 15 "Contacting GitHub…"
    _json="$(api_newest_release_json)" || error "Failed to fetch release metadata"

    if command -v jq >/dev/null 2>&1; then
        VERSION="$(printf '%s' "$_json" | jq -r '.[0].tag_name // empty')"
    else
        VERSION="$(printf '%s' "$_json" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
    fi
    if [ -z "$VERSION" ]; then
        error "Failed to parse tag_name from GitHub API"
    fi
    progress_line 35 "Release resolved"
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
    curl -fsSL "$_url" -o "$_path"
}

install_release() {
    BASE="https://github.com/${REPO}/releases/download/${VERSION}/"
    TAR="$(tarball_basename)"
    DOWNLOAD_URL="${BASE}${TAR}"

    progress_line 42 "Downloading release ${TAR}"
    TEMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TEMP_DIR"' EXIT INT HUP
    ARCHIVE="${TEMP_DIR}/${TAR}"

    download_url_to "$DOWNLOAD_URL" "$ARCHIVE" || error "Failed to download binary archive"
    progress_line 62 "Release archive downloaded"

    if [ "${INSTALL_VERIFY_CHECKSUM:-1}" != "0" ]; then
        SUMS_URL="${BASE}SHA256SUMS"
        _sums="$(curl -fsSL "$SUMS_URL")" || error "Failed to download SHA256SUMS"
        _expect="$(checksum_for_tarball "$_sums" "$TAR")"
        if [ -z "$_expect" ]; then
            error "SHA256SUMS missing entry for ${TAR}"
        fi
        if command -v shasum >/dev/null 2>&1; then
            _got="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
        elif command -v sha256sum >/dev/null 2>&1; then
            _got="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
        else
            error "Need shasum or sha256sum to verify checksums"
        fi
        if [ "$_expect" != "$_got" ]; then
            error "Checksum mismatch (expected $_expect, got $_got)"
        fi
        progress_line 75 "Checksum verified"
    fi

    progress_line 82 "Extracting archive"
    tar -xzf "$ARCHIVE" -C "$TEMP_DIR"

    progress_line 88 "Installing binaries to ${INSTALL_DIR}"
    mkdir -p "$INSTALL_DIR"
    for b in $BINARY_NAMES; do
        if [ ! -f "${TEMP_DIR}/${b}" ]; then
            error "Archive missing ${b}"
        fi
        mv "${TEMP_DIR}/${b}" "${INSTALL_DIR}/"
        chmod +x "${INSTALL_DIR}/${b}"
    done

    rm -rf "$TEMP_DIR"
    trap - EXIT INT HUP
    progress_line 90 "Binaries installed"
}

# GRN parquet files for training (same search path as spacetravlr: install_dir/data/).
install_spaceship_config_toml() {
    DATA_DIR="${INSTALL_DIR}/data"
    mkdir -p "$DATA_DIR" || error "Could not create ${DATA_DIR}"
    _dest="${DATA_DIR}/spaceship_config.toml"
    if [ -f "$_dest" ]; then
        _sz=$(wc -c < "$_dest" 2>/dev/null | tr -d '[:space:]' || echo 0)
        if [ "${INSTALL_REFRESH_GRN_DATA:-0}" != "1" ] && [ "${_sz:-0}" -gt 512 ] 2>/dev/null; then
            return 0
        fi
    fi
    _ok=0
    for ref in "$VERSION" main; do
        _url="https://raw.githubusercontent.com/${REPO}/${ref}/spaceship_config.toml"
        if curl -fsSL "$_url" -o "${_dest}.part" 2>/dev/null; then
            mv "${_dest}.part" "$_dest"
            _ok=1
            break
        fi
        rm -f "${_dest}.part"
    done
    if [ "$_ok" != "1" ]; then
        warn "Could not download spaceship_config.toml from GitHub (tried tag ${VERSION} and main). Copy into ${DATA_DIR}/ or pass --config."
    fi
}

install_grn_data() {
    DATA_DIR="${INSTALL_DIR}/data"
    mkdir -p "$DATA_DIR" || error "Could not create ${DATA_DIR}"

    progress_line 90 "Downloading spaceship_config.toml (default CLI config)…"
    install_spaceship_config_toml
    if [ -f "${DATA_DIR}/spaceship_config.toml" ]; then
        progress_line 91 "spaceship_config.toml at ${DATA_DIR}/"
    fi

    if [ "${INSTALL_SKIP_GRN_DATA:-0}" = "1" ]; then
        progress_line 95 "Skipping GRN parquet download (INSTALL_SKIP_GRN_DATA=1)"
        [ "$QUIET" -eq 0 ] && info "  Skipping human_network.parquet / mouse_network.parquet (set SPACETRAVLR_DATA_DIR or copy data/ yourself)"
        progress_line 100 "Complete"
        return 0
    fi

    progress_line 92 "Downloading GRN parquet (human_network, mouse_network) from GitHub…"
    _failed=0
    for f in human_network.parquet mouse_network.parquet; do
        _dest="${DATA_DIR}/${f}"
        if [ -f "$_dest" ]; then
            _sz=$(wc -c < "$_dest" 2>/dev/null | tr -d '[:space:]' || echo 0)
            if [ "${INSTALL_REFRESH_GRN_DATA:-0}" != "1" ] && [ "${_sz:-0}" -gt 100000 ] 2>/dev/null; then
                continue
            fi
        fi
        _ok=0
        for ref in "$VERSION" main; do
            _url="https://raw.githubusercontent.com/${REPO}/${ref}/data/${f}"
            if curl -fsSL "$_url" -o "${_dest}.part" 2>/dev/null; then
                mv "${_dest}.part" "$_dest"
                _ok=1
                break
            fi
            rm -f "${_dest}.part"
        done
        if [ "$_ok" != "1" ]; then
            warn "Could not download ${f} from GitHub (tried tag ${VERSION} and main). Copy into ${DATA_DIR}/ manually or set SPACETRAVLR_DATA_DIR."
            _failed=1
        fi
    done

    if [ "$_failed" -eq 0 ]; then
        progress_line 97 "Bundle ready at ${DATA_DIR} (GRN parquets + spaceship_config.toml next to the binary)"
    else
        progress_line 97 "Install finished (GRN files incomplete — see warning above)"
    fi
    progress_line 100 "Complete"
}

add_install_dir_to_path() {
    export PATH="${INSTALL_DIR}:${PATH}"

    if [ -z "${HOME:-}" ]; then
        warn "HOME is unset; cannot update ~/.bashrc. Add manually: export PATH=\"${INSTALL_DIR}:\$PATH\""
        return 0
    fi

    _rc="${HOME}/.bashrc"
    _marker='# SpaceTravLR: add install dir to PATH (install.sh)'

    if [ -f "$_rc" ] && grep -qF "$_marker" "$_rc" 2>/dev/null; then
        [ "$QUIET" -eq 0 ] && info "  PATH already configured in ${_rc}"
        return 0
    fi

    {
        printf '\n%s\n' "$_marker"
        printf '%s\n' "export PATH=\"${INSTALL_DIR}:\$PATH\""
    } >> "$_rc" || warn "Could not append to ${_rc}; add manually: export PATH=\"${INSTALL_DIR}:\$PATH\""

    [ "$QUIET" -eq 0 ] && info "  Added ${INSTALL_DIR} to PATH for this session and appended to ${_rc}"
}

verify_install() {
    if command -v spacetravlr >/dev/null 2>&1; then
        _ver_out="$(spacetravlr --version 2>&1 | head -n 1)"
        info "  Installed binary: ${_ver_out}"
    elif [ -x "${INSTALL_DIR}/spacetravlr" ]; then
        _ver_out="$("${INSTALL_DIR}/spacetravlr" --version 2>&1 | head -n 1)"
        info "  Installed binary: ${_ver_out}"
    else
        warn "Expected ${INSTALL_DIR}/spacetravlr missing after install."
    fi
}

dry_run() {
    progress_line 100 "Dry run"
    printf '  REPO=%s\n  VERSION=%s\n  TARGET=%s\n  TAR=%s\n  INSTALL_DIR=%s\n' \
        "$REPO" "$VERSION" "$TARGET" "$(tarball_basename)" "$INSTALL_DIR"
    printf '  Bundle → %s/data/ (spaceship_config.toml + human_network.parquet + mouse_network.parquet from GitHub raw, tag then main)\n' \
        "$INSTALL_DIR"
    info "Unset INSTALL_DRY_RUN to install."
}

# Two-column help; colors only when color_ok (opt-in).
show_help() {
    _c=0
    [ "$color_ok" -eq 1 ] && _c=1

    _section() {
        if [ "$_c" -eq 1 ]; then
            printf '\n  %s%s%s\n' "$BOLD" "$1" "$NC"
        else
            printf '\n  %s\n' "$1"
        fi
    }

    _row() {
        _k="$1"
        _d="$2"
        if [ "$_c" -eq 1 ]; then
            printf '    %s%-28s%s  %s%s%s\n' "$CYAN" "$_k" "$NC" "$DIM" "$_d" "$NC"
        else
            printf '    %-28s  %s\n' "$_k" "$_d"
        fi
    }

    if [ "$_c" -eq 1 ]; then
        printf '\n  %s%sSpaceTravLR%s %sinstaller%s\n' "$BLUE" "$BOLD" "$NC" "$BOLD" "$NC"
    else
        printf '\n  SpaceTravLR installer\n'
    fi

    _section "Usage"
    if [ "$_c" -eq 1 ]; then
        printf '    %s%s%s\n' "$GREEN" "install.sh [--quiet] [--color]" "$NC"
    else
        printf '    install.sh [--quiet] [--color]\n'
    fi

    _section "Options"
    _row "--quiet" "Suppress progress and status output"
    _row "--color" "ANSI colors (also SPACETRAVLR_INSTALL_COLOR=1); default is plain text"
    _row "-h, --help" "Show this help and exit"

    _section "Environment"
    if [ "$_c" -eq 1 ]; then
        printf '    %s%-28s%s  %s%s%s\n' "$YELLOW" "Variable" "$NC" "$YELLOW" "Description" "$NC"
        printf '    %s%s%s\n' "$DIM" "──────────────────────────────  ───────────────────────────────────────────" "$NC"
    else
        printf '    %-28s  %s\n' "Variable" "Description"
        printf '    %s\n' "──────────────────────────────  ───────────────────────────────────────────"
    fi
    _row "SPACETRAVLR_INSTALL_DIR" "Install dir; default \$HOME/.local/bin (export + ~/.bashrc)"
    _row "SPACETRAVLR_GITHUB_REPO" "GitHub owner/repo for releases"
    _row "SPACETRAVLR_GH_API" "GitHub API base URL (default https://api.github.com)"
    _row "INSTALL_DRY_RUN=1" "Print plan only; no download or install"
    _row "INSTALL_TEST_VERSION=v..." "Pin release tag (e.g. tests)"
    _row "INSTALL_VERIFY_CHECKSUM=0" "Skip SHA256 verification (unsafe)"
    _row "INSTALL_SKIP_GRN_DATA=1" "Skip downloading human_network.parquet / mouse_network.parquet"
    _row "INSTALL_REFRESH_GRN_DATA=1" "Re-download GRN parquet and spaceship_config.toml even if already present"
    _row "SPACETRAVLR_LINUX_VARIANT" "standard | compat (Linux x86_64; overrides glibc probe)"
    _row "SPACETRAVLR_INSTALL_COLOR=1" "Enable ANSI colors (default off; use with --color)"
    _row "NO_COLOR" "Set (any value) to disable colors per no-color.org"
    _row "UNAME_S / UNAME_M" "Override platform detection (tests)"
    printf '\n'
}

main() {
    case "${1:-}" in
        -h | --help)
            show_help
            exit 0
            ;;
    esac

    show_banner

    detect_os
    detect_arch
    get_target
    linux_set_tarball_target_for_glibc
    progress_line 8 "Platform: ${OS} ${ARCH}  ${TARGET}"

    if [ -n "${INSTALL_TEST_VERSION:-}" ]; then
        VERSION="$INSTALL_TEST_VERSION"
    else
        get_latest_version
    fi

    show_release_version "$VERSION"

    if [ "${INSTALL_DRY_RUN:-}" = "1" ]; then
        dry_run
        exit 0
    fi

    install_release
    install_grn_data
    echo ""
    add_install_dir_to_path
    echo ""
    verify_install
    echo ""
    [ "$QUIET" -eq 0 ] && info "  This terminal may need: source ~/.bashrc   (new shells load PATH from there)"
    info "  Try: spacetravlr --help   ·   spacetravlr --demo"
}

main "$@"