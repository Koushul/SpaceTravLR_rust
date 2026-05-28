//! `spacetravlr --verify` smoke test: download (or local) tonsil `.h5ad`, copy with stripped `normalized_count` /
//! `imputed_count` layers so training runs **Rust full preprocess** (QC → normalize → HVG → … → MAGIC), then
//! tiny full-mode train on **AICDA** and **CD74** with **`--parallel 2`**, confirm two `*_betadata.feather` files.
//! Writes a plain-text log (checklist + hardware). `SPACETRAVLR_VERIFY_MAX_LR` caps DB ligand–receptor pairs
//! (default 256; very low values can orphan a target with no `*_betadata.feather`). Set **`SPACETRAVLR_VERIFY_ALLOW_CPU=1`**
//! to pass when no WebGPU adapter is used (default: require training stderr line `CNN/compute backend = WebGPU`).
//! **`SPACETRAVLR_VERIFY_SKIP_PREP_STRIP=1`** uses the raw `.h5ad` (skips forcing full Rust prep + related log checks).
//!
//! Verify sets **`SPACETRAVLR_FORCE_KEEP_GENES=AICDA,CD74`** on the training subprocess so target genes
//! survive dispersion HVG (top-N by default) and reach Lasso/CNN — both betadata feathers are emitted
//! even when CD74 falls below the HVG cut. The env var is honored by [`crate::rust_preprocess`].
//!
//! **`spaceship_config.toml`** is located via **`SPACETRAVLR_ROOT`**, the compile-time manifest when that file
//! still exists, walking up from **cwd** and from the **executable directory**, then the install layout from
//! [`resolve_spaceship_config_toml_path`] — so release binaries built on CI do not require the builder’s path.

use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, bail};
use chrono::Utc;
use colored::Colorize;
use polars::prelude::*;

use crate::condition_split::CONDITION_RUNS_SUBDIR;
use crate::config::{expand_user_path, resolve_spaceship_config_toml_path};
use crate::read_h5ad_var_names;
use crate::scanpy_preprocess::copy_h5ad_for_verify_forcing_rust_full_prep;

const VERIFY_H5AD_URL: &str = concat!(
    "https://raw.githubusercontent.com/Koushul/SpaceTravLR_rust/",
    "main/data/h5ad/SlideTags_human_tonsil.h5ad",
);
const VERIFY_MIN_H5AD_BYTES: u64 = 500_000;

/// CNN grid edge length (H=W) for verify. Must be **≥ 8**: [`crate::model::CellularNicheNetwork`]
/// stacks three 2×2 max-pools; at 4×4 the map becomes 1×1 before the third pool and Burn’s pool
/// output sizing can underflow `usize` in debug builds.
const VERIFY_SPATIAL_DIM: usize = 8;

/// Default top-N DB ligand–receptor pairs by mean expression. Too low (e.g. 5) often yields **zero**
/// modulators for a target gene → only a `{gene}.orphan` marker, no `{gene}_betadata.feather`.
/// Override with `SPACETRAVLR_VERIFY_MAX_LR` (parse as usize).
const VERIFY_DEFAULT_MAX_LR: usize = 256;

const VERIFY_GENE_TARGETS: &[&str] = &["AICDA", "CD74"];

const VERIFY_TRAIN_STDERR_MAX_BYTES: usize = 16 * 1024 * 1024;

const VERIFY_TRAIN_LOG_TAIL_LINES: usize = 48;

const VERIFY_LOG_RUST_FULL_PREP: &str = "running Rust preprocess (QC → log-norm → HVG";
const VERIFY_LOG_MAGIC_CELLTYPE: &str = ">>> MAGIC per cell_type";
const VERIFY_LOG_WEBGPU: &str = "CNN/compute backend = WebGPU";

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
        .unwrap_or(false)
}

fn read_file_utf8_lossy_capped(path: &Path, max_bytes: usize) -> anyhow::Result<String> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buf = vec![0u8; max_bytes];
    let n = f
        .read(&mut buf)
        .with_context(|| format!("read {}", path.display()))?;
    buf.truncate(n);
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() <= n {
        return lines.join("\n");
    }
    let start = lines.len().saturating_sub(n);
    format!(
        "… ({} earlier lines omitted) …\n{}",
        start,
        lines[start..].join("\n")
    )
}

fn format_exit_status(st: &std::process::ExitStatus) -> String {
    if st.success() {
        return "exit code 0".into();
    }
    if let Some(code) = st.code() {
        return format!("exit code {code} (non-zero; Unix convention: 0 = success)");
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = st.signal() {
            return format!("terminated by signal {sig} (no exit code)");
        }
    }
    "terminated abnormally (no exit code)".into()
}

fn append_log_tail_details(details: &mut Vec<String>, label: &str, path: &Path, max_lines: usize) {
    match read_file_utf8_lossy_capped(path, VERIFY_TRAIN_STDERR_MAX_BYTES) {
        Ok(text) if text.trim().is_empty() => {
            details.push(format!("{label}: (empty file at {})", path.display()));
        }
        Ok(text) => {
            details.push(format!("{label} file: {}", path.display()));
            details.push(format!(
                "{label} tail (last {max_lines} lines):\n{}",
                tail_lines(&text, max_lines)
            ));
        }
        Err(e) => {
            details.push(format!(
                "{label}: could not read {} ({e:#})",
                path.display()
            ));
        }
    }
}

fn stderr_error_hints(stderr: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    for line in stderr.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let lower = t.to_ascii_lowercase();
        if lower.contains("error:")
            || lower.contains(" panic")
            || lower.starts_with("panic")
            || lower.contains("thread '")
            || lower.contains("caused by:")
            || lower.contains("bail!")
        {
            out.push(format!("stderr hint: {t}"));
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

fn training_subprocess_command_line(
    exe: &Path,
    cfg: &str,
    h5: &str,
    out: &str,
    genes: &str,
    max_lr: usize,
) -> String {
    format!(
        "{} --plain --config {cfg} --h5ad {h5} --output-dir {out} \
         --training-mode full --genes {genes} --max-genes {} --epochs 2 \
         --spatial_dim {VERIFY_SPATIAL_DIM} --max-lr {max_lr} --parallel 2",
        exe.display(),
        VERIFY_GENE_TARGETS.len()
    )
}

/// Workspace directory (`current_dir` for the training subprocess) and absolute `--config` path.
///
/// Release binaries embed `CARGO_MANIFEST_DIR` from the **build** machine (often a CI path that does
/// not exist on the user’s machine), so we never rely on that alone: walk up from cwd and from the
/// executable location, then fall back to [`resolve_spaceship_config_toml_path`] (install `data/` layout).
fn verify_workspace_and_config() -> Option<(PathBuf, PathBuf)> {
    fn walk_up_for_repo_spaceship(mut dir: PathBuf) -> Option<(PathBuf, PathBuf)> {
        for _ in 0..24 {
            let cfg = dir.join("spaceship_config.toml");
            if cfg.is_file() {
                return Some((dir, cfg));
            }
            if !dir.pop() {
                break;
            }
        }
        None
    }

    if let Ok(raw) = std::env::var("SPACETRAVLR_ROOT") {
        let root = PathBuf::from(expand_user_path(raw.trim()));
        let cfg = root.join("spaceship_config.toml");
        if cfg.is_file() {
            return Some((root, cfg));
        }
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let man_cfg = manifest.join("spaceship_config.toml");
    if man_cfg.is_file() {
        return Some((manifest, man_cfg));
    }

    if let Ok(cwd) = std::env::current_dir() {
        if let Some(pair) = walk_up_for_repo_spaceship(cwd) {
            return Some(pair);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            if let Some(pair) = walk_up_for_repo_spaceship(parent.to_path_buf()) {
                return Some(pair);
            }
        }
    }

    let mut cfg = resolve_spaceship_config_toml_path()?;
    if cfg.is_relative() {
        cfg = std::env::current_dir().ok()?.join(cfg);
    }
    let cfg = std::fs::canonicalize(&cfg).unwrap_or(cfg);
    if !cfg.is_file() {
        return None;
    }

    let parent = cfg.parent()?;
    if parent.file_name() == Some(OsStr::new("data")) {
        let workspace = parent.parent()?.to_path_buf();
        return Some((workspace, cfg));
    }
    Some((parent.to_path_buf(), cfg))
}

fn wgpu_probe_section() -> String {
    let mut out = String::from("WebGPU / wgpu (adapter probe — same path as training CNN backend)\n");
    let adapter_info = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map(|a| a.get_info())
    });
    match adapter_info {
        Some(info) => {
            use std::fmt::Write;
            let _ = writeln!(out, "  name:          {}", info.name);
            let _ = writeln!(out, "  vendor_id:     0x{:x}", info.vendor);
            let _ = writeln!(out, "  device_id:     0x{:x}", info.device);
            let _ = writeln!(out, "  device_type:   {:?}", info.device_type);
            let _ = writeln!(out, "  backend:       {:?}", info.backend);
            let _ = writeln!(out, "  driver:        {}", info.driver);
            let _ = writeln!(out, "  driver_info:   {}", info.driver_info);
        }
        None => out.push_str("  (no adapter returned — training will use CPU NdArray unless WebGPU becomes available)\n"),
    }
    let force_cpu = std::env::var("SPACETRAVLR_FORCE_CPU").unwrap_or_default();
    let disable_wgpu = std::env::var("SPACETRAVLR_DISABLE_WGPU").unwrap_or_default();
    use std::fmt::Write;
    let _ = writeln!(out, "  SPACETRAVLR_FORCE_CPU:     {force_cpu}");
    let _ = writeln!(out, "  SPACETRAVLR_DISABLE_WGPU:  {disable_wgpu}");
    out
}

#[cfg(feature = "tui")]
fn sysinfo_host_section() -> String {
    fn fmt_gib(bytes: u64) -> String {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
    use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
    let mut sys = System::new_with_specifics(
        RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything()),
    );
    sys.refresh_all();
    let mut s = String::from("Host (sysinfo — CPU / RAM / swap)\n");
    use std::fmt::Write;
    let _ = writeln!(
        s,
        "  hostname:           {:?}",
        System::host_name().unwrap_or_else(|| "<unknown>".into())
    );
    let _ = writeln!(s, "  OS name:            {:?}", System::name());
    let _ = writeln!(s, "  long_os_version:    {:?}", System::long_os_version());
    let _ = writeln!(s, "  kernel_version:     {:?}", System::kernel_version());
    let _ = writeln!(s, "  os_version:         {:?}", System::os_version());
    let _ = writeln!(s, "  distribution_id:    {}", System::distribution_id());
    let _ = writeln!(s, "  cpu_arch (sysinfo): {:?}", System::cpu_arch());
    let _ = writeln!(
        s,
        "  load_average 1/5/15: {:.2} / {:.2} / {:.2}",
        System::load_average().one,
        System::load_average().five,
        System::load_average().fifteen
    );
    let n = sys.cpus().len();
    let _ = writeln!(s, "  logical_cpus:       {n}");
    let _ = writeln!(s, "  physical_cores:     {:?}", sys.physical_core_count());
    if let Some(cpu0) = sys.cpus().first() {
        let _ = writeln!(s, "  cpu_brand (first):  {}", cpu0.brand().trim());
        let _ = writeln!(s, "  cpu_freq_mhz (first): {}", cpu0.frequency());
    }
    let _ = writeln!(
        s,
        "  global_cpu_usage %: {:.1}",
        sys.global_cpu_usage()
    );
    let total = sys.total_memory();
    let used = sys.used_memory();
    let avail = sys.available_memory();
    let free = sys.free_memory();
    let _ = writeln!(s, "  RAM total:          {} ({})", total, fmt_gib(total));
    let _ = writeln!(s, "  RAM used:           {} ({})", used, fmt_gib(used));
    let _ = writeln!(s, "  RAM available:      {} ({})", avail, fmt_gib(avail));
    let _ = writeln!(s, "  RAM free:           {} ({})", free, fmt_gib(free));
    let st = sys.total_swap();
    let su = sys.used_swap();
    let _ = writeln!(s, "  swap total:         {} ({})", st, fmt_gib(st));
    let _ = writeln!(s, "  swap used:          {} ({})", su, fmt_gib(su));
    s
}

#[cfg(not(feature = "tui"))]
fn sysinfo_host_section() -> String {
    "Host (sysinfo)\n  (skipped — rebuild with default `tui` feature for RAM/CPU/swap details)\n"
        .to_string()
}

fn std_host_section() -> String {
    let mut s = String::from("Host (std — always available)\n");
    use std::fmt::Write;
    let _ = writeln!(
        s,
        "  std::env::consts::OS:   {}",
        std::env::consts::OS
    );
    let _ = writeln!(
        s,
        "  std::env::consts::ARCH: {}",
        std::env::consts::ARCH
    );
    let _ = writeln!(
        s,
        "  std::env::consts::FAMILY: {}",
        std::env::consts::FAMILY
    );
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let _ = writeln!(s, "  available_parallelism (hint): {threads}");
    if let Ok(u) = std::env::var("USER").or_else(|_| std::env::var("USERNAME")) {
        let _ = writeln!(s, "  USER: {u}");
    }
    s
}

fn spacetravlr_env_section() -> String {
    let mut s = String::from("Environment — SPACETRAVLR_* (set at verify start)\n");
    let mut keys: Vec<String> = std::env::vars()
        .filter_map(|(k, _)| k.starts_with("SPACETRAVLR_").then_some(k))
        .collect();
    keys.sort();
    if keys.is_empty() {
        s.push_str("  (none)\n");
        return s;
    }
    use std::fmt::Write;
    for k in keys {
        if let Ok(v) = std::env::var(&k) {
            let show = if v.len() > 200 {
                format!("{}… ({} bytes total)", &v[..200], v.len())
            } else {
                v
            };
            let _ = writeln!(s, "  {k}={show}");
        }
    }
    s
}

fn build_hardware_block() -> String {
    let mut b = String::new();
    b.push_str(&std_host_section());
    b.push('\n');
    b.push_str(&sysinfo_host_section());
    b.push('\n');
    b.push_str(&wgpu_probe_section());
    b.push('\n');
    b.push_str(&spacetravlr_env_section());
    b
}

struct VerifyLog {
    path: PathBuf,
    w: BufWriter<File>,
}

impl VerifyLog {
    fn create(path: PathBuf) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create log parent {}", parent.display()))?;
        }
        let f = File::create(&path).with_context(|| format!("create log {}", path.display()))?;
        Ok(Self {
            path: path.clone(),
            w: BufWriter::new(f),
        })
    }

    fn writeln_str(&mut self, s: &str) -> anyhow::Result<()> {
        writeln!(self.w, "{s}").with_context(|| format!("write {}", self.path.display()))?;
        Ok(())
    }

    fn flush(&mut self) -> anyhow::Result<()> {
        self.w
            .flush()
            .with_context(|| format!("flush {}", self.path.display()))?;
        Ok(())
    }
}

fn hr(width: usize) -> String {
    "=".repeat(width)
}

fn rule(width: usize) -> String {
    "-".repeat(width)
}

fn emit_check(
    log: &mut VerifyLog,
    all_ok: &mut bool,
    failed_checks: &mut Vec<(String, Vec<String>)>,
    ok: bool,
    label: &str,
) -> anyhow::Result<()> {
    emit_check_details(log, all_ok, failed_checks, ok, label, &[])
}

fn emit_check_details(
    log: &mut VerifyLog,
    all_ok: &mut bool,
    failed_checks: &mut Vec<(String, Vec<String>)>,
    ok: bool,
    label: &str,
    details: &[String],
) -> anyhow::Result<()> {
    let status = if ok { "PASS" } else { "FAIL" };
    log.writeln_str(&format!("[{status}] {label}"))?;
    for d in details {
        log.writeln_str(&format!("       {d}"))?;
    }
    if !ok {
        failed_checks.push((label.to_string(), details.to_vec()));
    }
    *all_ok &= check_line_stdout(ok, label, details);
    Ok(())
}

fn check_line_stdout(ok: bool, label: &str, details: &[String]) -> bool {
    let icon = if ok { "✓" } else { "✗" };
    let text = format!("  [{icon}] {label}");
    if ok {
        println!("{}", text.green());
    } else {
        println!("{}", text.red());
        for d in details.iter().take(6) {
            for line in d.lines().take(8) {
                println!("{}", format!("      {line}").red().dimmed());
            }
        }
        if details.len() > 6 {
            println!(
                "{}",
                format!(
                    "      … ({} more detail lines in verify log)",
                    details.len() - 6
                )
                .red()
                .dimmed()
            );
        }
    }
    ok
}

fn verify_max_ligands_db() -> usize {
    std::env::var("SPACETRAVLR_VERIFY_MAX_LR")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(VERIFY_DEFAULT_MAX_LR)
}

fn find_gene_betadata_in_run_root(run_root: &Path, gene: &str) -> Option<PathBuf> {
    if let Some(p) = find_betadata_feather(run_root, gene) {
        return Some(p);
    }
    let cond = run_root.join(CONDITION_RUNS_SUBDIR);
    let Ok(rd) = std::fs::read_dir(&cond) else {
        return None;
    };
    for ent in rd.flatten() {
        let p = ent.path();
        if p.is_dir() {
            if let Some(f) = find_betadata_feather(&p, gene) {
                return Some(f);
            }
        }
    }
    None
}

fn find_gene_sidecar(run_root: &Path, gene: &str, suffix: &str) -> Option<PathBuf> {
    let direct = run_root.join(format!("{gene}{suffix}"));
    if direct.is_file() {
        return Some(direct);
    }
    let cond = run_root.join(CONDITION_RUNS_SUBDIR);
    let rd = std::fs::read_dir(&cond).ok()?;
    for ent in rd.flatten() {
        let p = ent.path();
        if p.is_dir() {
            let cand = p.join(format!("{gene}{suffix}"));
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

fn list_marker_files_for_debug(run_root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(files) = list_betadata_like_files(run_root) {
        for n in files {
            out.push(n);
        }
    }
    let cond = run_root.join(CONDITION_RUNS_SUBDIR);
    if let Ok(rd) = std::fs::read_dir(&cond) {
        for ent in rd.flatten() {
            let p = ent.path();
            if !p.is_dir() {
                continue;
            }
            let group = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
            if let Ok(files) = list_betadata_like_files(&p) {
                for n in files {
                    out.push(format!("{CONDITION_RUNS_SUBDIR}/{group}/{n}"));
                }
            }
        }
    }
    out.sort();
    out
}

fn find_betadata_feather(out_dir: &Path, gene: &str) -> Option<PathBuf> {
    let direct = out_dir.join(format!("{gene}_betadata.feather"));
    if direct.is_file() {
        return Some(direct);
    }
    let suffix = "_betadata.feather";
    let entries = std::fs::read_dir(out_dir).ok()?;
    for ent in entries.flatten() {
        let p = ent.path();
        let name = p.file_name()?.to_str()?;
        if name.len() <= suffix.len() {
            continue;
        }
        let (stem, rest) = name.split_at(name.len() - suffix.len());
        if !rest.eq_ignore_ascii_case(suffix) {
            continue;
        }
        if stem.eq_ignore_ascii_case(gene) && p.is_file() {
            return Some(p);
        }
    }
    None
}

fn list_betadata_like_files(dir: &Path) -> std::io::Result<Vec<String>> {
    let mut names = Vec::new();
    for ent in std::fs::read_dir(dir)? {
        let name = ent?.file_name().to_string_lossy().into_owned();
        if name.contains("betadata") || name.ends_with(".orphan") || name.ends_with(".tf_ablated")
        {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

fn feather_max_abs_non_id(path: &Path) -> anyhow::Result<f64> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let df = IpcReader::new(f)
        .finish()
        .with_context(|| format!("read feather {}", path.display()))?;
    let mut max_abs = 0.0_f64;
    for col in df.get_columns() {
        let name = col.name();
        if name.eq_ignore_ascii_case("CellID") || name.eq_ignore_ascii_case("Cluster") {
            continue;
        }
        let s = match col.dtype() {
            DataType::Float32 | DataType::Float64 => col.clone().cast(&DataType::Float64)?,
            _ => continue,
        };
        let ca = s.f64().context("cast to f64")?;
        for i in 0..ca.len() {
            if let Some(v) = ca.get(i) {
                if v.is_finite() {
                    max_abs = max_abs.max(v.abs());
                }
            }
        }
    }
    Ok(max_abs)
}

fn download_h5ad(dest: &Path) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let url = std::env::var("SPACETRAVLR_VERIFY_H5AD_URL").unwrap_or_else(|_| VERIFY_H5AD_URL.into());
    let st = Command::new("curl")
        .args(["-fL", "--retry", "3", "--connect-timeout", "30", "-o"])
        .arg(dest)
        .arg(&url)
        .status()
        .context("spawn curl (install curl or set SPACETRAVLR_VERIFY_H5AD to a local .h5ad path)")?;
    anyhow::ensure!(st.success(), "curl download failed with status {:?}", st.code());
    let meta = std::fs::metadata(dest).with_context(|| format!("stat {}", dest.display()))?;
    anyhow::ensure!(
        meta.len() >= VERIFY_MIN_H5AD_BYTES,
        "downloaded file at {} is too small ({} bytes); URL or mirror may be wrong",
        dest.display(),
        meta.len()
    );
    Ok(())
}

fn resolve_h5ad_for_verify(work: &Path) -> anyhow::Result<PathBuf> {
    if let Ok(p) = std::env::var("SPACETRAVLR_VERIFY_H5AD") {
        let exp = expand_user_path(p.trim());
        let pb = PathBuf::from(exp);
        anyhow::ensure!(pb.is_file(), "SPACETRAVLR_VERIFY_H5AD is not a file: {}", pb.display());
        return Ok(pb);
    }
    let dest = work.join("SlideTags_human_tonsil.h5ad");
    download_h5ad(&dest)?;
    Ok(dest)
}

pub fn run_spacetravlr_verify() -> anyhow::Result<()> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    let log_path = if let Ok(raw) = std::env::var("SPACETRAVLR_VERIFY_LOG") {
        let p = PathBuf::from(expand_user_path(raw.trim()));
        if p.as_os_str().is_empty() {
            bail!("SPACETRAVLR_VERIFY_LOG is empty");
        }
        p
    } else {
        std::env::temp_dir().join(format!("spacetravlr_verify_{stamp}.log"))
    };

    let mut log = VerifyLog::create(log_path.clone())?;

    let w = 78;
    log.writeln_str(&hr(w))?;
    log.writeln_str(" SpaceTravLR — verify log")?;
    log.writeln_str(&hr(w))?;
    log.writeln_str(&format!("Started (UTC):     {}", Utc::now().format("%Y-%m-%d %H:%M:%S UTC")))?;
    log.writeln_str(&format!("spacetravlr:       {} (git {})", env!("CARGO_PKG_VERSION"), env!("SPACETRAVLR_GIT_SHA")))?;
    if let Ok(exe) = std::env::current_exe() {
        log.writeln_str(&format!("This binary:       {}", exe.display()))?;
    }
    log.writeln_str(&format!("Log file:          {}", log_path.display()))?;
    log.writeln_str("")?;

    println!(
        "{}",
        "SpaceTravLR — verify (strip prep layers → Rust full preprocess + MAGIC → parallel-2 train AICDA+CD74 → two betadata feathers)"
            .bold()
    );
    println!(
        "{}",
        format!(
            "Tip: SPACETRAVLR_VERIFY_H5AD=…  ·  SPACETRAVLR_ROOT or run from repo / beside data/spaceship_config.toml  ·  SPACETRAVLR_VERIFY_ALLOW_CPU=1 on CPU-only  ·  SPACETRAVLR_VERIFY_SKIP_PREP_STRIP=1 skips prep-layer strip + prep log checks  ·  log → {}",
            log_path.display()
        )
        .dimmed()
    );

    log.writeln_str(&rule(w))?;
    log.writeln_str(" Hardware & environment (for debugging)")?;
    log.writeln_str(&rule(w))?;
    log.writeln_str("")?;
    for line in build_hardware_block().lines() {
        log.writeln_str(line)?;
    }
    log.writeln_str("")?;
    log.flush()?;

    let mut all_ok = true;
    let mut failed_checks: Vec<(String, Vec<String>)> = Vec::new();

    let resolved = verify_workspace_and_config();
    let (workspace, config_path) = match resolved {
        Some(pair) => pair,
        None => {
            let hint = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("spaceship_config.toml");
            log.writeln_str(&rule(w))?;
            log.writeln_str(" Checklist")?;
            log.writeln_str(&rule(w))?;
            emit_check(
                &mut log,
                &mut all_ok,
                &mut failed_checks,
                false,
                &format!(
                    "spaceship_config.toml (set SPACETRAVLR_ROOT, cd into SpaceTravLR_rust, or install data/spaceship_config.toml next to the binary; compile-time path {} is often wrong on release builds)",
                    hint.display()
                ),
            )?;
            log.writeln_str("")?;
            log.writeln_str(&hr(w))?;
            log.writeln_str(" RESULT: FAIL (missing config)")?;
            log.writeln_str(&hr(w))?;
            log.flush()?;
            check_line_stdout(
                false,
                &format!(
                    "spaceship_config.toml (see log; compile-time fallback was {})",
                    hint.display()
                ),
                &[],
            );
            println!(
                "{}",
                format!("Verify log written to: {}", log_path.display()).green().bold()
            );
            bail!(
                "missing spaceship_config.toml — set SPACETRAVLR_ROOT to your SpaceTravLR_rust repo, cd into that repo, or install data/spaceship_config.toml next to this binary (same layout as install.sh). This binary was built with CARGO_MANIFEST_DIR={} (only valid on the builder machine).",
                env!("CARGO_MANIFEST_DIR")
            );
        }
    };

    let work_path = std::env::temp_dir().join(format!("spacetravlr_verify_{stamp}"));
    std::fs::create_dir_all(&work_path).with_context(|| format!("mkdir {}", work_path.display()))?;

    let skip_prep_strip = env_flag("SPACETRAVLR_VERIFY_SKIP_PREP_STRIP");
    let allow_cpu = env_flag("SPACETRAVLR_VERIFY_ALLOW_CPU");

    log.writeln_str(&rule(w))?;
    log.writeln_str(" Verify run parameters")?;
    log.writeln_str(&rule(w))?;
    log.writeln_str(&format!("  work_dir:          {}", work_path.display()))?;
    log.writeln_str(&format!("  workspace_root:    {}", workspace.display()))?;
    log.writeln_str("  training_mode:     full")?;
    let max_lr = verify_max_ligands_db();
    log.writeln_str(&format!(
        "  target_genes:      {} (from AnnData var, comma-separated on CLI)",
        VERIFY_GENE_TARGETS.join(", ")
    ))?;
    log.writeln_str("  epochs:            2")?;
    log.writeln_str(&format!("  spatial_dim:       {VERIFY_SPATIAL_DIM}"))?;
    log.writeln_str(&format!(
        "  max_ligands (DB):  {max_lr}  (--max-lr; env SPACETRAVLR_VERIFY_MAX_LR overrides; default avoids orphan/no-mod genes)"
    ))?;
    log.writeln_str("  parallel:          2  (--parallel)")?;
    log.writeln_str(&format!(
        "  prep_layer_strip:  {}  (off → SPACETRAVLR_VERIFY_SKIP_PREP_STRIP; forces Rust FullPreprocess when on)",
        if skip_prep_strip { "no" } else { "yes" }
    ))?;
    log.writeln_str(&format!(
        "  require_webgpu:    {}  (relax → SPACETRAVLR_VERIFY_ALLOW_CPU=1)",
        if allow_cpu { "no" } else { "yes" }
    ))?;
    log.writeln_str(&format!(
        "  force_keep_genes:  {}  (env SPACETRAVLR_FORCE_KEEP_GENES on training subprocess; preserves targets through HVG)",
        VERIFY_GENE_TARGETS.join(",")
    ))?;
    log.writeln_str("")?;
    log.flush()?;

    log.writeln_str(&rule(w))?;
    log.writeln_str(" Checklist")?;
    log.writeln_str(&rule(w))?;
    log.flush()?;

    let h5ad_raw = resolve_h5ad_for_verify(work_path.as_path()).context("dataset (download or SPACETRAVLR_VERIFY_H5AD)")?;
    emit_check(
        &mut log,
        &mut all_ok,
        &mut failed_checks,
        h5ad_raw.is_file(),
        "Dataset available (download or local path)",
    )?;

    let h5ad_train: Option<PathBuf> = if skip_prep_strip {
        Some(h5ad_raw.clone())
    } else {
        let dst = work_path.join("verify_forcing_rust_full_prep.h5ad");
        match copy_h5ad_for_verify_forcing_rust_full_prep(&h5ad_raw, &dst) {
            Ok(()) => Some(dst),
            Err(e) => {
                log.writeln_str(&format!("  prep_strip_error: {e:#}"))?;
                emit_check_details(
                    &mut log,
                    &mut all_ok,
                    &mut failed_checks,
                    false,
                    "Copy .h5ad + strip layers normalized_count / imputed_count (forces Rust full auto-prep)",
                    &[format!("{e:#}")],
                )?;
                None
            }
        }
    };

    let vars_path = h5ad_train.as_ref().unwrap_or(&h5ad_raw);
    let vars = read_h5ad_var_names(vars_path).with_context(|| format!("read var_names {}", vars_path.display()))?;

    let mut genes_resolved: Vec<String> = Vec::new();
    let mut missing: Vec<&'static str> = Vec::new();
    for want in VERIFY_GENE_TARGETS {
        if let Some(g) = vars.iter().find(|v| v.eq_ignore_ascii_case(want)) {
            genes_resolved.push(g.clone());
        } else {
            missing.push(want);
        }
    }
    let gene_check_details = if missing.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "missing from AnnData var_names: {}",
            missing.join(", ")
        )]
    };
    emit_check_details(
        &mut log,
        &mut all_ok,
        &mut failed_checks,
        missing.is_empty(),
        &format!(
            "AnnData contains verify targets {} (resolved: {})",
            VERIFY_GENE_TARGETS.join(", "),
            if genes_resolved.is_empty() {
                "(none)".into()
            } else {
                genes_resolved.join(", ")
            }
        ),
        &gene_check_details,
    )?;

    let genes_csv = genes_resolved.join(",");
    let exe = std::env::current_exe().context("current_exe for training subprocess")?;
    let out_dir = work_path.join("verify_train_out");
    let out_str = out_dir.to_string_lossy().into_owned();
    let cfg_str = config_path.to_string_lossy().into_owned();
    let stdout_path = work_path.join("verify_training.stdout.log");
    let stderr_path = work_path.join("verify_training.stderr.log");

    log.writeln_str("")?;
    log.writeln_str("(When [data].condition is set, each group trains under output_dir/conditions/<id>/.)")?;
    log.writeln_str(&format!(
        "  training_stdout:   {}",
        stdout_path.display()
    ))?;
    log.writeln_str(&format!(
        "  training_stderr:   {}",
        stderr_path.display()
    ))?;
    log.writeln_str("")?;
    log.flush()?;

    println!(
        "{}",
        format!(
            "  … training subprocess (--parallel 2, {}, prep + 2 CNN epochs, spatial_dim {VERIFY_SPATIAL_DIM}, --max-lr {max_lr})",
            VERIFY_GENE_TARGETS.join("+")
        )
        .dimmed()
    );
    println!(
        "{}",
        format!(
            "  … logs → {} and {}",
            stdout_path.display(),
            stderr_path.display()
        )
        .dimmed()
    );

    let mut st_train_ok = false;
    let mut train_status: Option<std::process::ExitStatus> = None;
    let mut train_fail_details: Vec<String> = Vec::new();

    if h5ad_train.is_none() {
        train_fail_details.push(
            "subprocess not started: prep-layer strip failed (see prep_strip_error above)".into(),
        );
    } else if !missing.is_empty() {
        train_fail_details.push(format!(
            "subprocess not started: missing verify target genes: {}",
            missing.join(", ")
        ));
    } else if let Some(ref h5_train) = h5ad_train {
        let h5_str = h5_train.to_string_lossy().into_owned();
        let cmd_line = training_subprocess_command_line(
            &exe,
            &cfg_str,
            &h5_str,
            &out_str,
            &genes_csv,
            max_lr,
        );
        log.writeln_str(&format!("  training_command:  {cmd_line}"))?;
        log.writeln_str(&format!(
            "  training_cwd:      {}",
            workspace.display()
        ))?;
        log.flush()?;

        let stdout_file =
            File::create(&stdout_path).with_context(|| format!("create {}", stdout_path.display()))?;
        let stderr_file =
            File::create(&stderr_path).with_context(|| format!("create {}", stderr_path.display()))?;
        let st_train = Command::new(&exe)
            .arg("--plain")
            .arg("--config")
            .arg(&cfg_str)
            .arg("--h5ad")
            .arg(&h5_str)
            .arg("--output-dir")
            .arg(&out_str)
            .arg("--training-mode")
            .arg("full")
            .arg("--genes")
            .arg(&genes_csv)
            .arg("--max-genes")
            .arg(VERIFY_GENE_TARGETS.len().to_string())
            .arg("--epochs")
            .arg("2")
            .arg("--spatial_dim")
            .arg(VERIFY_SPATIAL_DIM.to_string())
            .arg("--max-lr")
            .arg(max_lr.to_string())
            .arg("--parallel")
            .arg("2")
            .env("SPACETRAVLR_FORCE_KEEP_GENES", &genes_csv)
            .stdout(Stdio::from(stdout_file))
            .stderr(Stdio::from(stderr_file))
            .current_dir(&workspace)
            .status()
            .context("spawn spacetravlr training")?;
        train_status = Some(st_train);
        st_train_ok = st_train.success();
        if !st_train_ok {
            train_fail_details.push(format_exit_status(&st_train));
        }
    }

    let train_stdout = if stdout_path.is_file() {
        read_file_utf8_lossy_capped(&stdout_path, VERIFY_TRAIN_STDERR_MAX_BYTES)
            .unwrap_or_else(|_| String::from("(could not read training stdout)"))
    } else {
        String::new()
    };
    let train_stderr = if stderr_path.is_file() {
        read_file_utf8_lossy_capped(&stderr_path, VERIFY_TRAIN_STDERR_MAX_BYTES)
            .unwrap_or_else(|_| String::from("(could not read training stderr)"))
    } else {
        String::new()
    };
    let train_combined = format!("{train_stdout}\n{train_stderr}");

    if !st_train_ok {
        if train_status.is_some() {
            train_fail_details.push(format!("subprocess binary: {}", exe.display()));
        }
        train_fail_details.extend(stderr_error_hints(&train_combined, 12));
        append_log_tail_details(
            &mut train_fail_details,
            "stdout",
            &stdout_path,
            VERIFY_TRAIN_LOG_TAIL_LINES,
        );
        append_log_tail_details(
            &mut train_fail_details,
            "stderr",
            &stderr_path,
            VERIFY_TRAIN_LOG_TAIL_LINES,
        );
        let betas_found: Vec<String> = genes_resolved
            .iter()
            .filter(|g| find_gene_betadata_in_run_root(&out_dir, g).is_some())
            .cloned()
            .collect();
        if !betas_found.is_empty() {
            train_fail_details.push(format!(
                "note: betadata exists for {} despite non-zero exit ({}) — inspect log tails for errors after \"done.\"",
                betas_found.join(", "),
                train_status
                    .as_ref()
                    .map(format_exit_status)
                    .unwrap_or_else(|| "unknown status".into())
            ));
        }
        if train_combined.contains("done.") && !st_train_ok {
            train_fail_details.push(
                "note: stdout contains \"done.\" but exit code was non-zero — likely a late panic/abort during cleanup or HDF5 close".into(),
            );
        }
    }

    emit_check_details(
        &mut log,
        &mut all_ok,
        &mut failed_checks,
        st_train_ok,
        "Training subprocess completed successfully (--parallel 2, AICDA + CD74, full mode, 2 epochs)",
        &train_fail_details,
    )?;

    if skip_prep_strip {
        emit_check(
            &mut log,
            &mut all_ok,
            &mut failed_checks,
            true,
            "Rust full preprocess + MAGIC log markers (skipped — SPACETRAVLR_VERIFY_SKIP_PREP_STRIP)",
        )?;
    } else {
        let has_prep = train_combined.contains(VERIFY_LOG_RUST_FULL_PREP);
        let prep_details = if has_prep {
            Vec::new()
        } else {
            vec![
                format!("expected log substring: {VERIFY_LOG_RUST_FULL_PREP:?}"),
                "searched combined stdout+stderr from training subprocess".into(),
            ]
        };
        emit_check_details(
            &mut log,
            &mut all_ok,
            &mut failed_checks,
            has_prep,
            "Training stderr: Rust full preprocess (normalize / HVG / PCA / UMAP / Leiden / …)",
            &prep_details,
        )?;
        let has_magic = train_combined.contains(VERIFY_LOG_MAGIC_CELLTYPE);
        let magic_details = if has_magic {
            Vec::new()
        } else {
            vec![
                format!("expected log substring: {VERIFY_LOG_MAGIC_CELLTYPE:?}"),
                "searched combined stdout+stderr from training subprocess".into(),
            ]
        };
        emit_check_details(
            &mut log,
            &mut all_ok,
            &mut failed_checks,
            has_magic,
            "Training stderr: clusterwise MAGIC → layers['imputed_count']",
            &magic_details,
        )?;
    }

    let gpu_used = train_combined.contains(VERIFY_LOG_WEBGPU);
    if allow_cpu {
        emit_check(
            &mut log,
            &mut all_ok,
            &mut failed_checks,
            true,
            &format!(
                "CNN compute backend (SPACETRAVLR_VERIFY_ALLOW_CPU: WebGPU={gpu_used}, NdArray allowed)"
            ),
        )?;
    } else {
        let gpu_details = if gpu_used {
            Vec::new()
        } else {
            vec![
                format!("expected log substring: {VERIFY_LOG_WEBGPU:?}"),
                "searched combined stdout+stderr; set SPACETRAVLR_VERIFY_ALLOW_CPU=1 to allow CPU-only".into(),
            ]
        };
        emit_check_details(
            &mut log,
            &mut all_ok,
            &mut failed_checks,
            gpu_used,
            "CNN compute backend = WebGPU (training stderr; set SPACETRAVLR_VERIFY_ALLOW_CPU=1 for CPU-only)",
            &gpu_details,
        )?;
    }

    for gene in &genes_resolved {
        let beta_path = find_gene_betadata_in_run_root(&out_dir, gene)
            .unwrap_or_else(|| out_dir.join(format!("{gene}_betadata.feather")));
        let beta_ok = beta_path.is_file();
        let orphan_path = find_gene_sidecar(&out_dir, gene, ".orphan");
        let tf_ab_path = find_gene_sidecar(&out_dir, gene, ".tf_ablated");
        if beta_ok {
            let shown = beta_path.strip_prefix(&out_dir).unwrap_or(&beta_path);
            emit_check(
                &mut log,
                &mut all_ok,
                &mut failed_checks,
                true,
                &format!(
                    "{} exists ({})",
                    beta_path.file_name().unwrap().to_string_lossy(),
                    shown.display()
                ),
            )?;
            let mx = feather_max_abs_non_id(&beta_path).context("read betadata feather")?;
            let beta_vals = mx > 1e-8;
            let beta_val_details = if beta_vals {
                Vec::new()
            } else {
                vec![format!("max|β| = {mx:.3e} (expected > 1e-8)")]
            };
            emit_check_details(
                &mut log,
                &mut all_ok,
                &mut failed_checks,
                beta_vals,
                &format!("{gene}: betadata feather has finite β values (max|β| ≈ {mx:.3e})"),
                &beta_val_details,
            )?;
        } else {
            let mut detail_lines = vec![format!(
                "{gene}_betadata.feather not found under {} (searched run root and {}/<group>/)",
                out_dir.display(),
                CONDITION_RUNS_SUBDIR
            )];
            if let Some(ref p) = orphan_path {
                detail_lines.push(format!(
                    "found orphan marker {} (no GRN modulators at current --max-lr; try higher SPACETRAVLR_VERIFY_MAX_LR)",
                    p.strip_prefix(&out_dir).unwrap_or(p.as_path()).display()
                ));
            } else if let Some(ref p) = tf_ab_path {
                detail_lines.push(format!(
                    "found tf_ablated marker {} (TF modulator block empty after ablation filters)",
                    p.strip_prefix(&out_dir).unwrap_or(p.as_path()).display()
                ));
            }
            let list = list_marker_files_for_debug(&out_dir);
            if !list.is_empty() {
                let preview: Vec<_> = list.iter().take(24).cloned().collect();
                detail_lines.push(format!("marker files seen: {}", preview.join(", ")));
                if list.len() > 24 {
                    detail_lines.push(format!("… (+{} more marker files)", list.len() - 24));
                }
            }
            emit_check_details(
                &mut log,
                &mut all_ok,
                &mut failed_checks,
                false,
                &format!("{gene}_betadata.feather exists"),
                &detail_lines,
            )?;
        }
    }

    println!();
    log.writeln_str("")?;
    log.writeln_str(&hr(w))?;
    if all_ok {
        log.writeln_str(" RESULT: PASS — all checklist items succeeded")?;
        let _ = std::fs::remove_dir_all(&work_path);
    } else {
        log.writeln_str(" RESULT: FAIL — one or more checklist items failed")?;
        log.writeln_str(&format!(
            " verify work_dir retained: {}",
            work_path.display()
        ))?;
        log.writeln_str("")?;
        log.writeln_str(" Failed checks (detail):")?;
        for (label, details) in &failed_checks {
            log.writeln_str(&format!("  • {label}"))?;
            if details.is_empty() {
                log.writeln_str("      (no extra detail recorded)")?;
            } else {
                for d in details {
                    for line in d.lines() {
                        log.writeln_str(&format!("      {line}"))?;
                    }
                }
            }
        }
    }
    log.writeln_str(&hr(w))?;
    log.writeln_str(&format!("End (UTC): {}", Utc::now().format("%Y-%m-%d %H:%M:%S UTC")))?;
    log.flush()?;

    println!(
        "{}",
        format!("Verify log written to: {}", log_path.display()).green().bold()
    );

    if all_ok {
        Ok(())
    } else {
        let summary: Vec<String> = failed_checks
            .iter()
            .map(|(label, details)| {
                if details.is_empty() {
                    label.clone()
                } else {
                    format!("{label} — {}", details[0])
                }
            })
            .collect();
        bail!(
            "verify finished with {} failed check(s): {}",
            failed_checks.len(),
            summary.join("; ")
        );
    }
}

#[cfg(test)]
mod verify_diag_tests {
    use super::*;

    #[test]
    fn tail_lines_keeps_last_n() {
        let s = (1..=10).map(|i| format!("line{i}")).collect::<Vec<_>>().join("\n");
        let tail = tail_lines(&s, 3);
        assert!(tail.contains("line8"));
        assert!(tail.contains("line10"));
        assert!(!tail.contains("line1\n"));
    }

    #[test]
    fn stderr_error_hints_finds_error_lines() {
        let stderr = "ok\nError: something broke\nCaused by: inner\n";
        let hints = stderr_error_hints(stderr, 8);
        assert!(hints.iter().any(|h| h.contains("something broke")));
        assert!(hints.iter().any(|h| h.contains("Caused by")));
    }
}
