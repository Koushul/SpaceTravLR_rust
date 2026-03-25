use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result};
use flate2::write::GzEncoder;
use flate2::Compression as FlateCompression;
use polars::prelude::*;
use rand::SeedableRng;
use rayon::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::ThreadPoolBuilder;
use serde::Serialize;

const N_COLS: usize = 500;
const ROW_COUNTS: &[usize] = &[5_000, 50_000, 200_000];
const LABELS: &[&str] = &["small", "medium", "large"];
const N_SHARDS: usize = 8;
const REPEATS_SMALL: usize = 2;
const REPEATS_LARGE: usize = 1;

#[derive(Serialize)]
struct BenchRow {
    scenario: String,
    rows: usize,
    format: String,
    compression: String,
    save_s: f64,
    load_s: f64,
    size_mb: f64,
    note: String,
}

#[derive(Serialize)]
struct Meta {
    n_cols: usize,
    dtype: &'static str,
    row_counts: std::collections::HashMap<String, usize>,
    implementation: &'static str,
    polars_version: &'static str,
    machine: String,
}

fn machine_label() -> String {
    format!(
        "{} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

fn float_schema() -> SchemaRef {
    static SCHEMA: OnceLock<SchemaRef> = OnceLock::new();
    SCHEMA
        .get_or_init(|| {
            let mut s = Schema::with_capacity(N_COLS);
            for i in 0..N_COLS {
                s.insert(format!("c{i}").into(), DataType::Float64);
            }
            Arc::new(s)
        })
        .clone()
}

fn make_df(n_rows: usize) -> DataFrame {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let d = StandardNormal;
    let mut cols: Vec<Column> = Vec::with_capacity(N_COLS);
    for i in 0..N_COLS {
        let v: Vec<f64> = (0..n_rows).map(|_| d.sample(&mut rng)).collect();
        cols.push(Series::new(format!("c{i}").into(), v).into());
    }
    DataFrame::new(cols).expect("dataframe")
}

fn file_size_mb(path: &Path) -> f64 {
    path.metadata().map(|m| m.len() as f64 / (1024.0 * 1024.0)).unwrap_or(0.0)
}

fn dir_size_mb(dir: &Path) -> f64 {
    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len() as f64)
        .sum::<f64>()
        / (1024.0 * 1024.0)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn concat_vertical_refs(dfs: &[DataFrame]) -> PolarsResult<DataFrame> {
    let mut it = dfs.iter();
    let mut acc = it.next().expect("non-empty").clone();
    for df in it {
        acc.vstack_mut(df)?;
    }
    Ok(acc)
}

fn concat_vertical_owned(mut dfs: Vec<DataFrame>) -> PolarsResult<DataFrame> {
    let mut acc = dfs.remove(0);
    for d in dfs {
        acc.vstack_mut_owned(d)?;
    }
    Ok(acc)
}

fn bench_csv_plain(df: &mut DataFrame, path: &Path, repeats: usize) -> Result<(f64, f64, f64)> {
    let mut save_t = Vec::with_capacity(repeats);
    let mut load_t = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let f = File::create(path).context("create csv")?;
        let t0 = Instant::now();
        let mut w = CsvWriter::new(f);
        w.finish(df)?;
        save_t.push(t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        let _read = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema(Some(float_schema()))
            .try_into_reader_with_file_path(Some(path.to_path_buf()))?
            .finish()?;
        load_t.push(t0.elapsed().as_secs_f64());
    }
    Ok((mean(&save_t), mean(&load_t), file_size_mb(path)))
}

fn bench_csv_gzip(df: &mut DataFrame, path: &Path, repeats: usize) -> Result<(f64, f64, f64)> {
    let mut save_t = Vec::with_capacity(repeats);
    let mut load_t = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let f = File::create(path).context("create csv.gz")?;
        let mut gz = GzEncoder::new(f, FlateCompression::default());
        let t0 = Instant::now();
        {
            let mut w = CsvWriter::new(&mut gz);
            w.finish(df)?;
        }
        gz.finish().context("finish gzip")?;
        save_t.push(t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        let _read = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema(Some(float_schema()))
            .try_into_reader_with_file_path(Some(path.to_path_buf()))?
            .finish()?;
        load_t.push(t0.elapsed().as_secs_f64());
    }
    Ok((mean(&save_t), mean(&load_t), file_size_mb(path)))
}

fn bench_parquet(
    df: &mut DataFrame,
    path: &Path,
    compression: ParquetCompression,
    repeats: usize,
) -> Result<(f64, f64, f64)> {
    let mut save_t = Vec::with_capacity(repeats);
    let mut load_t = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let f = File::create(path).context("create parquet")?;
        let t0 = Instant::now();
        ParquetWriter::new(f)
            .with_compression(compression)
            .finish(df)?;
        save_t.push(t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        let _read = ParquetReader::new(File::open(path).context("open parquet")?).finish()?;
        load_t.push(t0.elapsed().as_secs_f64());
    }
    Ok((mean(&save_t), mean(&load_t), file_size_mb(path)))
}

fn bench_ipc(
    df: &mut DataFrame,
    path: &Path,
    compression: Option<IpcCompression>,
    repeats: usize,
) -> Result<(f64, f64, f64)> {
    let mut save_t = Vec::with_capacity(repeats);
    let mut load_t = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let f = File::create(path).context("create ipc")?;
        let t0 = Instant::now();
        let mut w = IpcWriter::new(f).with_compression(compression);
        w.finish(df)?;
        save_t.push(t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        let _read = IpcReader::new(File::open(path).context("open ipc")?).finish()?;
        load_t.push(t0.elapsed().as_secs_f64());
    }
    Ok((mean(&save_t), mean(&load_t), file_size_mb(path)))
}

fn shard_row_slices(n: usize, n_shards: usize) -> Vec<(usize, usize)> {
    let k = n_shards.min(n).max(1);
    let base = n / k;
    let rem = n % k;
    let mut out = Vec::with_capacity(k);
    let mut start = 0usize;
    for i in 0..k {
        let len = base + if i < rem { 1 } else { 0 };
        if len > 0 {
            out.push((start, len));
            start += len;
        }
    }
    out
}

fn bench_sharded_parallel(
    df: &DataFrame,
    shard_dir: &Path,
    repeats: usize,
    workers: usize,
) -> Result<(f64, f64, f64, usize)> {
    if shard_dir.exists() {
        fs::remove_dir_all(shard_dir).ok();
    }
    fs::create_dir_all(shard_dir).context("shard dir")?;

    let slices = shard_row_slices(df.height(), N_SHARDS);
    let paths: Vec<PathBuf> = slices
        .iter()
        .enumerate()
        .map(|(i, &(off, len))| {
            let p = shard_dir.join(format!("part_{i:04}.parquet"));
            let mut part = df.slice(off as i64, len);
            let f = File::create(&p).expect("shard write");
            ParquetWriter::new(f)
                .with_compression(ParquetCompression::Snappy)
                .finish(&mut part)
                .expect("parquet shard");
            p
        })
        .collect();

    let total_mb = dir_size_mb(shard_dir);
    let n_parts = paths.len();

    let seq_load = || -> Result<()> {
        let mut dfs: Vec<DataFrame> = Vec::with_capacity(paths.len());
        for p in &paths {
            let f = File::open(p).context("open shard")?;
            let d = ParquetReader::new(f)
                .finish()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            dfs.push(d);
        }
        let _out = concat_vertical_refs(&dfs).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    };

    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("rayon pool")?;

    let par_load = || -> Result<()> {
        let dfs: Vec<DataFrame> = pool.install(|| {
            paths
                .par_iter()
                .map(|p| {
                    ParquetReader::new(File::open(p).expect("open shard"))
                        .finish()
                        .expect("read shard")
                })
                .collect()
        });
        let _out = concat_vertical_owned(dfs).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    };

    let mut st = Vec::with_capacity(repeats);
    let mut pt = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t0 = Instant::now();
        seq_load()?;
        st.push(t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        par_load()?;
        pt.push(t0.elapsed().as_secs_f64());
    }

    fs::remove_dir_all(shard_dir).ok();
    Ok((mean(&st), mean(&pt), total_mb, n_parts))
}

fn parquet_cases() -> &'static [(ParquetCompression, &'static str)] {
    &[
        (ParquetCompression::Uncompressed, "none"),
        (ParquetCompression::Snappy, "snappy"),
        (ParquetCompression::Zstd(None), "zstd"),
        (ParquetCompression::Gzip(None), "gzip"),
        (ParquetCompression::Brotli(None), "brotli"),
    ]
}

fn render_markdown(meta: &Meta, rows: &[BenchRow]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    writeln!(
        s,
        "# Tabular I/O benchmark (Polars / Rust)\n\n\
         Synthetic data: **{}** `float64` columns; row counts **small / medium / large**.\n",
        meta.n_cols
    )
    .unwrap();
    writeln!(s, "## Methodology\n").unwrap();
    writeln!(
        s,
        "- **Engine:** Polars {} (Rust), eager I/O APIs (`CsvWriter`/`CsvReader`, `ParquetWriter`/`ParquetReader`, `IpcWriter`/`IpcReader`).\n\
         - **Arrow IPC** is used as the Feather-equivalent on-disk format (Feather v2-compatible readers can consume IPC files).\n\
         - **Repeats:** 2 averaged for 5k and 50k rows; **1** for 200k.\n\
         - **Parallel load:** 8 row shards as Snappy Parquet; sequential read+concat vs Rayon parallel read+concat (thread pool size = min(8, CPU parallelism)).\n\
         - **CSV:** explicit `Float64` schema on read (skips inference).\n",
        meta.polars_version
    )
    .unwrap();
    writeln!(s, "## Environment\n").unwrap();
    writeln!(
        s,
        "- **polars** {}\n- **Platform:** {}\n",
        meta.polars_version, meta.machine
    )
    .unwrap();
    writeln!(s, "## Row counts\n\n| Label | Rows |\n|-------|------|").unwrap();
    for (l, &n) in LABELS.iter().zip(ROW_COUNTS.iter()) {
        writeln!(s, "| {l} | {n} |").unwrap();
    }
    writeln!(
        s,
        "\n## Results\n\nTimes are **seconds** (mean of repeats). **Size** is on-disk MB.\n"
    )
    .unwrap();

    for (label, &n) in LABELS.iter().zip(ROW_COUNTS.iter()) {
        writeln!(s, "### {label} ({n} rows)\n").unwrap();
        writeln!(
            s,
            "| Format | Compression | Save (s) | Load (s) | Size (MB) | Notes |"
        )
        .unwrap();
        writeln!(
            s,
            "|--------|-------------|----------|----------|-----------|-------|"
        )
        .unwrap();
        for r in rows.iter().filter(|r| r.rows == n) {
            let save_cell = if r.save_s.is_nan() {
                "—".to_string()
            } else {
                format!("{:.3}", r.save_s)
            };
            let note = r.note.replace('_', " ");
            writeln!(
                s,
                "| {} | {} | {} | {:.3} | {:.1} | {} |",
                r.format, r.compression, save_cell, r.load_s, r.size_mb, note
            )
            .unwrap();
        }
        writeln!(s).unwrap();
    }

    writeln!(s, "## Takeaways\n").unwrap();
    let large: Vec<_> = rows
        .iter()
        .filter(|r| r.scenario == "large" && r.format != "parquet_shards")
        .collect();
    if !large.is_empty() {
        let fastest_save = large
            .iter()
            .filter(|r| (r.format == "parquet" || r.format == "feather_ipc") && !r.save_s.is_nan())
            .min_by(|a, b| a.save_s.partial_cmp(&b.save_s).unwrap());
        let fastest_load = large
            .iter()
            .min_by(|a, b| a.load_s.partial_cmp(&b.load_s).unwrap());
        let smallest = large
            .iter()
            .min_by(|a, b| a.size_mb.partial_cmp(&b.size_mb).unwrap());
        if let Some(r) = fastest_save {
            writeln!(
                s,
                "- **Fastest save (200k rows, columnar):** {} (`{}`) ≈ **{:.2}s**.",
                r.format, r.compression, r.save_s
            )
            .unwrap();
        }
        if let Some(r) = fastest_load {
            writeln!(
                s,
                "- **Fastest single-file load (200k rows):** {} (`{}`) ≈ **{:.2}s**.",
                r.format, r.compression, r.load_s
            )
            .unwrap();
        }
        if let Some(r) = smallest {
            writeln!(
                s,
                "- **Smallest file (200k rows):** {} (`{}`) ≈ **{:.0} MB**.",
                r.format, r.compression, r.size_mb
            )
            .unwrap();
        }
    }
    let shards: Vec<_> = rows
        .iter()
        .filter(|r| r.scenario == "large" && r.format == "parquet_shards")
        .collect();
    if shards.len() >= 2 {
        let seq = shards.iter().find(|r| r.note.contains("sequential"));
        let par = shards.iter().find(|r| r.note.contains("parallel"));
        if let (Some(a), Some(b)) = (seq, par) {
            let speedup = a.load_s / b.load_s;
            writeln!(
                s,
                "- **Sharded Parquet (snappy), 200k rows:** sequential load **{:.2}s** vs Rayon parallel **{:.2}s** (~**{:.2}×** faster load).",
                a.load_s, b.load_s, speedup
            )
            .unwrap();
        }
    }
    writeln!(
        s,
        "\nCSV remains slow and large; Parquet **zstd** is often a good balance; IPC/Feather-style files are typically very fast for this wide-float layout on a single machine.\n"
    )
    .unwrap();
    writeln!(
        s,
        "_Raw JSON: `benchmarks/tabular_io_rust/results.json`. Regenerate: `cargo run --release --bin bench_tabular_io`._"
    )
    .unwrap();
    s
}

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/tabular_io_rust");
    fs::create_dir_all(&root).context("out dir")?;

    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let workers = N_SHARDS.min(cpus);

    let mut row_counts_map = std::collections::HashMap::new();
    for (&l, &n) in LABELS.iter().zip(ROW_COUNTS.iter()) {
        row_counts_map.insert((*l).to_string(), n);
    }

    let meta = Meta {
        n_cols: N_COLS,
        dtype: "float64",
        row_counts: row_counts_map,
        implementation: "polars-rust",
        polars_version: polars::VERSION,
        machine: machine_label(),
    };

    let mut out_rows: Vec<BenchRow> = Vec::new();

    for (&label, &n) in LABELS.iter().zip(ROW_COUNTS.iter()) {
        let repeats = if n >= 100_000 {
            REPEATS_LARGE
        } else {
            REPEATS_SMALL
        };
        let mut df = make_df(n);
        let base = root.join(format!("data_{label}"));

        let p = base.with_extension("csv");
        let (s, l, sz) = bench_csv_plain(&mut df, &p, repeats)?;
        out_rows.push(BenchRow {
            scenario: label.to_string(),
            rows: n,
            format: "csv".into(),
            compression: "none".into(),
            save_s: s,
            load_s: l,
            size_mb: sz,
            note: String::new(),
        });

        let p_gz = root.join(format!("data_{label}.csv.gz"));
        let (s, l, sz) = bench_csv_gzip(&mut df, &p_gz, repeats)?;
        out_rows.push(BenchRow {
            scenario: label.to_string(),
            rows: n,
            format: "csv_gzip".into(),
            compression: "gzip".into(),
            save_s: s,
            load_s: l,
            size_mb: sz,
            note: String::new(),
        });

        for &(comp, name) in parquet_cases() {
            let path = base.with_extension(format!("parquet_{name}.parquet"));
            let (s, l, sz) = bench_parquet(&mut df, &path, comp, repeats)?;
            out_rows.push(BenchRow {
                scenario: label.to_string(),
                rows: n,
                format: "parquet".into(),
                compression: name.to_string(),
                save_s: s,
                load_s: l,
                size_mb: sz,
                note: String::new(),
            });
        }

        let ipc_cases: [(Option<IpcCompression>, &str); 3] = [
            (None, "uncompressed"),
            (Some(IpcCompression::LZ4), "lz4"),
            (Some(IpcCompression::ZSTD), "zstd"),
        ];
        for (comp, name) in ipc_cases {
            let path = base.with_extension(format!("ipc_{name}.arrow"));
            let (s, l, sz) = bench_ipc(&mut df, &path, comp, repeats)?;
            out_rows.push(BenchRow {
                scenario: label.to_string(),
                rows: n,
                format: "feather_ipc".into(),
                compression: name.to_string(),
                save_s: s,
                load_s: l,
                size_mb: sz,
                note: String::new(),
            });
        }

        let shard_dir = root.join(format!("data_{label}_shards"));
        let (seq_l, par_l, shard_mb, n_parts) =
            bench_sharded_parallel(&df, &shard_dir, repeats, workers)?;
        out_rows.push(BenchRow {
            scenario: label.to_string(),
            rows: n,
            format: "parquet_shards".into(),
            compression: "snappy".into(),
            save_s: f64::NAN,
            load_s: seq_l,
            size_mb: shard_mb,
            note: format!("sequential_load_{n_parts}_files"),
        });
        out_rows.push(BenchRow {
            scenario: label.to_string(),
            rows: n,
            format: "parquet_shards".into(),
            compression: "snappy_parallel".into(),
            save_s: f64::NAN,
            load_s: par_l,
            size_mb: shard_mb,
            note: format!("parallel_load_{n_parts}_files_threads={workers}"),
        });
    }

    let json_path = root.join("results.json");
    let payload = serde_json::json!({ "meta": meta, "rows": out_rows });
    fs::write(&json_path, serde_json::to_string_pretty(&payload)?).context("write json")?;

    let md = render_markdown(&meta, &out_rows);
    let md_path = root.join("RESULTS.md");
    fs::write(&md_path, md).context("write md")?;

    println!("{}", md_path.display());
    Ok(())
}
