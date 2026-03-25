# Tabular I/O benchmark: pandas + PyArrow vs Polars (Rust)

Synthetic data: **500** `float64` columns; row counts **small / medium / large** (5k / 50k / 200k). Same RNG seed and distribution in both implementations.

## Caveats (read before comparing)

- **CSV read:** Polars uses a **fixed `Float64` schema** (no inference). pandas/pyarrow infers types, which can add overhead on wide files.
- **Feather vs IPC:** Python uses **Feather** via pyarrow. Rust uses **Arrow IPC** (`IpcWriter` / `IpcReader`) with the same compression names where applicable—sizes and timings are comparable but not identical to `.feather` files.
- **Platform:** Both runs below were on **Darwin arm64**; absolute seconds will differ on other machines, but relative patterns usually hold.

## Row counts

| Label | Rows |
|-------|------|
| small | 5,000 |
| medium | 50,000 |
| large | 200,000 |

## Methodology

### pandas + PyArrow (Python)

- **Libraries:** pandas + pyarrow (Parquet/Feather/CSV read/write).
- **Repeats:** 2 timed runs averaged for 5k and 50k rows; **1** run for 200k.
- **Parallel load:** **8** row shards, Snappy Parquet each; sequential `read_table` + `concat` vs `ThreadPoolExecutor` (workers = min(8, CPU count)).
- **Feather `default`:** pyarrow default Feather compression (typically LZ4).

### Polars (Rust)

- **Engine:** Polars **0.51.0**, eager I/O (`CsvWriter`/`CsvReader`, `ParquetWriter`/`ParquetReader`, `IpcWriter`/`IpcReader`).
- **Repeats:** Same as Python (2 / 2 / 1).
- **Parallel load:** Same sharding; sequential read+concat vs **Rayon** (threads = min(8, CPU parallelism)).
- **CSV:** Explicit `Float64` schema on read.

## Environment (both runs)

| Stack | Versions | Platform |
|-------|----------|----------|
| **pandas + pyarrow** | pandas 3.0.1, pyarrow 23.0.1 | Darwin arm64 |
| **Polars (Rust)** | polars 0.51.0 | macos aarch64 |

---

## Head-to-head — large (200,000 rows)

Times in **seconds**; **size** in MB. **Speedup** = pandas time ÷ Polars time (higher = Polars faster). Sharded rows are **load only** (save not timed).

| Format | Compression | pandas save | Polars save | Save speedup | pandas load | Polars load | Load speedup | Size (pandas) | Size (Polars) |
|--------|-------------|------------|-------------|--------------|------------|-------------|--------------|---------------|---------------|
| csv | none | 59.559 | 1.387 | **43×** | 8.659 | 4.147 | **2.1×** | 1872.2 | 1872.2 |
| csv_gzip | gzip | 188.917 | 27.337 | **6.9×** | 11.538 | 6.573 | **1.8×** | 863.3 | 861.0 |
| parquet | none | 3.397 | 0.457 | **7.4×** | 0.400 | 0.267 | **1.5×** | 892.6 | 763.1 |
| parquet | snappy | 3.408 | 0.615 | **5.5×** | 0.541 | 0.221 | **2.4×** | 892.7 | 763.2 |
| parquet | zstd | 3.768 | 1.367 | **2.8×** | 0.199 | 0.182 | **1.1×** | 855.0 | 732.4 |
| parquet | gzip | 25.224 | 12.317 | **2.0×** | 0.558 | 0.589 | 0.95× | 852.0 | 732.2 |
| parquet | brotli | 10.235 | 2.835 | **3.6×** | 0.885 | 0.933 | 0.95× | 837.4 | 731.7 |
| feather / IPC | default ‡ | 0.256 | — | — | 0.219 | — | — | 763.3 | — |
| feather / IPC | uncompressed | 0.330 | 0.368 | 0.90× | 0.161 | 0.101 | **1.6×** | 763.2 | 763.0 |
| feather / IPC | lz4 | 0.444 | 0.912 | 0.49× | 0.156 | 0.539 | 0.29× | 763.3 | 763.1 |
| feather / IPC | zstd | 0.435 | 0.927 | 0.47× | 0.285 | 0.788 | 0.36× | 731.5 | 731.4 |
| parquet_shards | snappy seq load | — | — | — | 0.270 | 0.893 | 0.30× ‡‡ | 944.4 | 764.2 |
| parquet_shards | snappy parallel load | — | — | — | 0.205 | 0.061 | **3.4×** | 944.4 | 764.2 |

‡ pandas **feather `default`** (pyarrow LZ4-style default) has no separate Polars row; compare to **feather lz4** on the Polars side if desired.

‡‡ Polars **sequential** shard load was slower than pandas in this run; **parallel** shard load was much faster (**0.061s** vs **0.205s** pandas).

---

## Full results — pandas + PyArrow

Times are **seconds** (mean of repeats). **Size** is on-disk MB.

### small (5,000 rows)

| Format | Compression | Save (s) | Load (s) | Size (MB) | Notes |
|--------|-------------|----------|----------|-----------|-------|
| csv | none | 1.418 | 0.215 | 46.8 |  |
| csv_gzip | gzip | 4.568 | 0.305 | 21.6 |  |
| parquet | none | 0.078 | 0.666 | 23.2 |  |
| parquet | snappy | 0.082 | 0.013 | 23.2 |  |
| parquet | zstd | 0.104 | 0.018 | 22.3 |  |
| parquet | gzip | 0.679 | 0.040 | 22.4 |  |
| parquet | brotli | 0.274 | 0.025 | 22.2 |  |
| feather | default | 0.012 | 0.009 | 19.3 |  |
| feather | uncompressed | 0.009 | 0.003 | 19.2 |  |
| feather | lz4 | 0.013 | 0.006 | 19.3 |  |
| feather | zstd | 0.010 | 0.006 | 18.5 |  |
| parquet_shards | snappy | — | 0.075 | 24.2 | sequential load 8 files |
| parquet_shards | snappy_parallel | — | 0.033 | 24.2 | parallel load 8 files threads=8 |

### medium (50,000 rows)

| Format | Compression | Save (s) | Load (s) | Size (MB) | Notes |
|--------|-------------|----------|----------|-----------|-------|
| csv | none | 15.644 | 2.128 | 468.0 |  |
| csv_gzip | gzip | 48.065 | 2.717 | 215.8 |  |
| parquet | none | 0.861 | 0.035 | 237.6 |  |
| parquet | snappy | 0.939 | 0.030 | 237.6 |  |
| parquet | zstd | 1.078 | 0.056 | 227.7 |  |
| parquet | gzip | 6.634 | 0.133 | 226.3 |  |
| parquet | brotli | 2.889 | 0.172 | 218.9 |  |
| feather | default | 0.036 | 0.030 | 190.9 |  |
| feather | uncompressed | 0.076 | 0.030 | 190.9 |  |
| feather | lz4 | 0.089 | 0.043 | 190.9 |  |
| feather | zstd | 0.052 | 0.060 | 183.0 |  |
| parquet_shards | snappy | — | 0.121 | 231.7 | sequential load 8 files |
| parquet_shards | snappy_parallel | — | 0.104 | 231.7 | parallel load 8 files threads=8 |

### large (200,000 rows)

| Format | Compression | Save (s) | Load (s) | Size (MB) | Notes |
|--------|-------------|----------|----------|-----------|-------|
| csv | none | 59.559 | 8.659 | 1872.2 |  |
| csv_gzip | gzip | 188.917 | 11.538 | 863.3 |  |
| parquet | none | 3.397 | 0.400 | 892.6 |  |
| parquet | snappy | 3.408 | 0.541 | 892.7 |  |
| parquet | zstd | 3.768 | 0.199 | 855.0 |  |
| parquet | gzip | 25.224 | 0.558 | 852.0 |  |
| parquet | brotli | 10.235 | 0.885 | 837.4 |  |
| feather | default | 0.256 | 0.219 | 763.3 |  |
| feather | uncompressed | 0.330 | 0.161 | 763.2 |  |
| feather | lz4 | 0.444 | 0.156 | 763.3 |  |
| feather | zstd | 0.435 | 0.285 | 731.5 |  |
| parquet_shards | snappy | — | 0.270 | 944.4 | sequential load 8 files |
| parquet_shards | snappy_parallel | — | 0.205 | 944.4 | parallel load 8 files threads=8 |

---

## Full results — Polars (Rust)

Label **feather_ipc** = Arrow IPC file (Feather v2–compatible). Times are **seconds** (mean of repeats). **Size** is on-disk MB.

### small (5,000 rows)

| Format | Compression | Save (s) | Load (s) | Size (MB) | Notes |
|--------|-------------|----------|----------|-----------|-------|
| csv | none | 0.045 | 0.055 | 46.8 |  |
| csv_gzip | gzip | 0.714 | 0.152 | 21.5 |  |
| parquet | none | 0.108 | 0.003 | 19.2 |  |
| parquet | snappy | 0.101 | 0.002 | 19.2 |  |
| parquet | zstd | 0.198 | 0.004 | 18.4 |  |
| parquet | gzip | 0.345 | 0.010 | 18.5 |  |
| parquet | brotli | 0.174 | 0.035 | 18.5 |  |
| feather_ipc | uncompressed | 0.005 | 0.003 | 19.1 |  |
| feather_ipc | lz4 | 0.016 | 0.013 | 19.2 |  |
| feather_ipc | zstd | 0.025 | 0.019 | 18.4 |  |
| parquet_shards | snappy | — | 0.009 | 20.2 | sequential load 8 files |
| parquet_shards | snappy_parallel | — | 0.004 | 20.2 | parallel load 8 files threads=8 |

### medium (50,000 rows)

| Format | Compression | Save (s) | Load (s) | Size (MB) | Notes |
|--------|-------------|----------|----------|-----------|-------|
| csv | none | 0.374 | 0.536 | 468.0 |  |
| csv_gzip | gzip | 6.929 | 1.582 | 215.3 |  |
| parquet | none | 0.161 | 0.013 | 190.9 |  |
| parquet | snappy | 0.172 | 0.012 | 190.9 |  |
| parquet | zstd | 0.350 | 0.028 | 183.1 |  |
| parquet | gzip | 2.925 | 0.078 | 183.2 |  |
| parquet | brotli | 0.737 | 0.204 | 183.1 |  |
| feather_ipc | uncompressed | 0.060 | 0.028 | 190.8 |  |
| feather_ipc | lz4 | 0.199 | 0.106 | 190.9 |  |
| feather_ipc | zstd | 0.237 | 0.178 | 182.9 |  |
| parquet_shards | snappy | — | 0.022 | 191.9 | sequential load 8 files |
| parquet_shards | snappy_parallel | — | 0.017 | 191.9 | parallel load 8 files threads=8 |

### large (200,000 rows)

| Format | Compression | Save (s) | Load (s) | Size (MB) | Notes |
|--------|-------------|----------|----------|-----------|-------|
| csv | none | 1.387 | 4.147 | 1872.2 |  |
| csv_gzip | gzip | 27.337 | 6.573 | 861.0 |  |
| parquet | none | 0.457 | 0.267 | 763.1 |  |
| parquet | snappy | 0.615 | 0.221 | 763.2 |  |
| parquet | zstd | 1.367 | 0.182 | 732.4 |  |
| parquet | gzip | 12.317 | 0.589 | 732.2 |  |
| parquet | brotli | 2.835 | 0.933 | 731.7 |  |
| feather_ipc | uncompressed | 0.368 | 0.101 | 763.0 |  |
| feather_ipc | lz4 | 0.912 | 0.539 | 763.1 |  |
| feather_ipc | zstd | 0.927 | 0.788 | 731.4 |  |
| parquet_shards | snappy | — | 0.893 | 764.2 | sequential load 8 files |
| parquet_shards | snappy_parallel | — | 0.061 | 764.2 | parallel load 8 files threads=8 |

---

## Takeaways (combined)

**pandas + pyarrow (this machine, 200k rows)**

- Fastest columnar **save:** feather (`default`) ≈ **0.26s**.
- Fastest single-file **load:** feather (`lz4`) ≈ **0.16s**.
- **Smallest** columnar file: feather (`zstd`) ≈ **732 MB**.
- Sharded Parquet: sequential **0.27s** vs parallel **0.20s** load (~**1.3×**).

**Polars Rust (this machine, 200k rows)**

- Fastest columnar **save:** feather_ipc (`uncompressed`) ≈ **0.37s** (pandas was faster on feather-style **save** for compressed variants in this run).
- Fastest single-file **load:** feather_ipc (`uncompressed`) ≈ **0.10s**.
- **Smallest:** feather_ipc (`zstd`) ≈ **731 MB** (matches pandas feather zstd ~).
- Parquet files are **smaller** than pandas-written Parquet for the same logical data (~763–732 MB vs ~893–837 MB) in these runs—likely writer defaults / row-group layout.
- Sharded load: Rayon **parallel** **0.061s** vs sequential **0.893s** (~**15×**); **parallel** sharded load beats pandas parallel **0.205s** (~**3.4×**).

**Practical comparison**

- **CSV:** Polars is dramatically faster on **save** at 200k×500; **load** is faster but not by as large a margin (schema + mmap effects).
- **Parquet:** Polars leads on **save** for uncompressed/snappy/gzip/brotli; **zstd load** is near parity.
- **Feather / IPC:** pyarrow **wins** on **save** for lz4/zstd IPC-style compression in this benchmark; Polars **wins** on **uncompressed IPC load**. Treat lz4/zstd rows as implementation-specific, not a pure format win for either side.

---

_Regenerate pandas results: `scripts/bench_tabular_io.py` (see `benchmarks/tabular_io_runs/results.json`). Regenerate Polars results: `cargo run --release --bin bench_tabular_io` (see `benchmarks/tabular_io_rust/results.json`). After re-running either benchmark, update the tables in this file or extend the scripts to emit a merged report._
