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

@test "dry-run linux x86_64 resolves gnu triple" {
    run env UNAME_S=Linux UNAME_M=x86_64 INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v9.8.7 sh "$INSTALL_SH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"x86_64-unknown-linux-gnu"* ]]
    [[ "$output" == *"spacetravlr-v9.8.7-x86_64-unknown-linux-gnu.tar.gz"* ]]
}

@test "dry-run linux aarch64" {
    run env UNAME_S=Linux UNAME_M=aarch64 INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v1 sh "$INSTALL_SH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"aarch64-unknown-linux-gnu"* ]]
}

@test "dry-run darwin arm64" {
    run env UNAME_S=Darwin UNAME_M=arm64 INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v1 sh "$INSTALL_SH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"aarch64-apple-darwin"* ]]
}

@test "unsupported architecture fails" {
    run env UNAME_S=Linux UNAME_M=mips INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v1 sh "$INSTALL_SH"
    [ "$status" -eq 1 ]
    [[ "$output" == *"Unsupported architecture"* ]]
}

@test "quiet suppresses info lines" {
    run env UNAME_S=Linux UNAME_M=x86_64 INSTALL_DRY_RUN=1 INSTALL_TEST_VERSION=v1 sh "$INSTALL_SH" --quiet
    [ "$status" -eq 0 ]
    [[ "$output" != *"[INFO]"* ]]
}
