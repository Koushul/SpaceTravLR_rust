#!/usr/bin/env bats
# Bats tests for scripts/install.sh (run from repo root: bats scripts/install.bats)

setup() {
    export SCRIPT_DIR="${BATS_TEST_DIRNAME}"
    export INSTALL_SH="${SCRIPT_DIR}/install.sh"
}

@test "install.sh --help exits 0" {
    run sh "$INSTALL_SH" --help
    [ "$status" -eq 0 ]
    [[ "$output" == *"INSTALL_DRY_RUN"* ]]
}

@test "dry-run linux x86_64 standard tarball" {
    run env UNAME_S=Linux UNAME_M=x86_64 INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v9.8.7 SPACETRAVLR_LINUX_VARIANT=standard sh "$INSTALL_SH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"x86_64-unknown-linux-gnu"* ]]
    [[ "$output" == *"spacetravlr-v9.8.7-x86_64-unknown-linux-gnu.tar.gz"* ]]
}

@test "dry-run linux x86_64 compat tarball" {
    run env UNAME_S=Linux UNAME_M=x86_64 INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v9.8.7 SPACETRAVLR_LINUX_VARIANT=compat sh "$INSTALL_SH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"spacetravlr-v9.8.7-x86_64-unknown-linux-gnu-glibc2.31.tar.gz"* ]]
}

@test "dry-run linux x86_64 compat28 tarball" {
    run env UNAME_S=Linux UNAME_M=x86_64 INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v9.8.7 SPACETRAVLR_LINUX_VARIANT=compat28 sh "$INSTALL_SH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"spacetravlr-v9.8.7-x86_64-unknown-linux-gnu-glibc2.28.tar.gz"* ]]
}

@test "linux aarch64 has no prebuilt installer path" {
    run env UNAME_S=Linux UNAME_M=aarch64 INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v1 sh "$INSTALL_SH"
    [ "$status" -eq 1 ]
    [[ "$output" == *"prebuilt Linux ARM64"* ]]
}

@test "dry-run darwin arm64" {
    run env UNAME_S=Darwin UNAME_M=arm64 INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v1 sh "$INSTALL_SH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"aarch64-apple-darwin"* ]]
}

@test "darwin x86_64 has no prebuilt installer path" {
    run env UNAME_S=Darwin UNAME_M=x86_64 INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v1 sh "$INSTALL_SH"
    [ "$status" -eq 1 ]
    [[ "$output" == *"prebuilt Intel Mac"* ]]
}

@test "unsupported architecture fails" {
    run env UNAME_S=Linux UNAME_M=mips INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v1 sh "$INSTALL_SH"
    [ "$status" -eq 1 ]
    [[ "$output" == *"Unsupported architecture"* ]]
}

@test "quiet suppresses info lines" {
    # SPACETRAVLR_LINUX_VARIANT avoids ldd glibc probe (macOS hosts lack GNU ldd when faking Linux).
    run env UNAME_S=Linux UNAME_M=x86_64 SPACETRAVLR_LINUX_VARIANT=standard INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v1 sh "$INSTALL_SH" --quiet
    [ "$status" -eq 0 ]
    [[ "$output" != *"[INFO]"* ]]
}
