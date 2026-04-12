# CellOracle ↔ Rust parity

Committed JSON under [`golden/`](golden/) matches Python sklearn / scipy reference values. Rust tests in `tests/celloracle_parity.rs` assert agreement.

## Run Rust tests

```bash
cargo test --test celloracle_parity
```

## Layout

- `golden/*.json` — numerical goldens
- `tests/celloracle_parity.rs` — reads goldens via `CARGO_MANIFEST_DIR` + `tests/fixtures/celloracle_parity/golden`
