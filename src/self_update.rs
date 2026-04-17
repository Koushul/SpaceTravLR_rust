//! Opt-in self-update: only used when the user runs `spacetravlr --update`.
//! Naming matches `scripts/install.sh` (see `GITHUB_REPO`, `tarball_name`, `prebuilt_tarball_target`).

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use semver::Version;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;

pub const GITHUB_REPO: &str = "Koushul/SpaceTravLR_rust";

pub const DISTRIBUTION_BINARIES: [&str; 3] =
    ["spacetravlr", "spacetravlr-perturb", "spatial_viewer"];

pub const LINUX_GNU_TRIPLE: &str = "x86_64-unknown-linux-gnu";

pub const LINUX_GNU_COMPAT_SUFFIX: &str = "-glibc2.31";

pub fn tarball_name(version_tag: &str, target: &str) -> String {
    format!("spacetravlr-{version_tag}-{target}.tar.gz")
}

pub fn host_target_triple() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Some(LINUX_GNU_TRIPLE),
        ("macos", "aarch64") => Some("aarch64-apple-darwin"),
        _ => None,
    }
}

fn parse_two_part_version_token(s: &str) -> Option<(u32, u32)> {
    let s = s.trim_matches(|c: char| c == '(' || c == ')' || c == ',');
    let mut parts = s.splitn(2, '.');
    let major: u32 = parts.next()?.parse().ok()?;
    let rest = parts.next()?;
    let minor_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let minor: u32 = minor_str.parse().ok()?;
    Some((major, minor))
}

fn glibc_version_from_ldd_first_line(line: &str) -> Option<(u32, u32)> {
    let mut last = None;
    for w in line.split_whitespace() {
        if let Some(v) = parse_two_part_version_token(w) {
            last = Some(v);
        }
    }
    last
}

fn host_glibc_major_minor() -> Result<(u32, u32)> {
    let out = std::process::Command::new("ldd")
        .arg("--version")
        .output()
        .context(
            "run `ldd --version` to detect GNU libc (required for self-update on Linux x86_64)",
        )?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let first = stdout.lines().next().unwrap_or("").trim();
    glibc_version_from_ldd_first_line(first).ok_or_else(|| {
        anyhow::anyhow!("could not parse glibc version from `ldd --version` first line: {first:?}")
    })
}

fn linux_x86_64_prebuilt_tarball_target() -> Result<String> {
    if let Ok(v) = std::env::var("SPACETRAVLR_LINUX_VARIANT") {
        return match v.as_str() {
            "standard" => Ok(LINUX_GNU_TRIPLE.to_string()),
            "compat" => Ok(format!("{LINUX_GNU_TRIPLE}{LINUX_GNU_COMPAT_SUFFIX}")),
            _ => bail!("invalid SPACETRAVLR_LINUX_VARIANT={v:?}; use standard or compat"),
        };
    }
    let (maj, min) = host_glibc_major_minor()?;
    let use_standard = maj > 2 || (maj == 2 && min >= 35);
    if use_standard {
        Ok(LINUX_GNU_TRIPLE.to_string())
    } else {
        Ok(format!("{LINUX_GNU_TRIPLE}{LINUX_GNU_COMPAT_SUFFIX}"))
    }
}

pub fn prebuilt_tarball_target() -> Result<String> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => linux_x86_64_prebuilt_tarball_target(),
        ("linux", "aarch64") => {
            bail!(
                "no prebuilt Linux ARM64 binaries; build from source (e.g. cargo install spacetravlr --locked --features spatial-viewer)"
            )
        }
        ("macos", "aarch64") => Ok("aarch64-apple-darwin".to_string()),
        ("macos", "x86_64") => {
            bail!(
                "no prebuilt Intel Mac binaries; use an Apple Silicon Mac or build from source (cargo install spacetravlr --locked --features spatial-viewer)"
            )
        }
        _ => host_target_triple()
            .map(|s| s.to_string())
            .context("unsupported OS/arch for prebuilt binaries"),
    }
}

#[derive(Debug, Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
}

#[derive(Debug, Deserialize)]
struct GhRelease {
    tag_name: String,
    assets: Vec<GhAsset>,
}

fn gh_api_get_json<T: serde::de::DeserializeOwned>(url: &str) -> Result<T> {
    let agent = ureq::Agent::new();
    let body = agent
        .get(url)
        .set("Accept", "application/vnd.github+json")
        .set("User-Agent", "spacetravlr-self-update")
        .set("X-GitHub-Api-Version", "2022-11-28")
        .call()
        .with_context(|| format!("HTTP GET {url}"))?
        .into_string()
        .with_context(|| format!("read body from {url}"))?;
    serde_json::from_str(&body).with_context(|| format!("parse JSON from {url}"))
}

fn fetch_release(tag: Option<&str>) -> Result<GhRelease> {
    let base = format!("https://api.github.com/repos/{GITHUB_REPO}/releases");
    match tag {
        Some(t) => {
            let url = format!("{base}/tags/{t}");
            gh_api_get_json(&url)
        }
        None => {
            let url = format!("{base}?per_page=1");
            let list: Vec<GhRelease> = gh_api_get_json(&url)?;
            list.into_iter().next().context(
                "no GitHub releases found (publish a release or pass a tag to --update-version)",
            )
        }
    }
}

fn normalize_tag_version(tag: &str) -> Result<Version> {
    let s = tag.strip_prefix('v').unwrap_or(tag);
    Version::parse(s).with_context(|| format!("invalid semver in tag {tag:?}"))
}

fn embedded_version() -> Result<Version> {
    Version::parse(env!("CARGO_PKG_VERSION")).context("CARGO_PKG_VERSION")
}

fn parse_checksums(text: &str, tarball: &str) -> Result<Option<String>> {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let hash = parts.next().context("checksum line empty")?;
        let name = parts.next().context("checksum line missing name")?;
        let name = name.trim_start_matches('*');
        if name == tarball || name.ends_with(&format!("/{tarball}")) {
            return Ok(Some(hash.to_string()));
        }
    }
    Ok(None)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn download_to_path(url: &str, path: &Path) -> Result<()> {
    let agent = ureq::Agent::new();
    let mut reader = agent
        .get(url)
        .set("Accept", "application/octet-stream")
        .set("User-Agent", "spacetravlr-self-update")
        .call()
        .with_context(|| format!("download {url}"))?
        .into_reader();
    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    io::copy(&mut reader, &mut f)?;
    Ok(())
}

fn atomic_replace_file(src: &Path, dst: &Path) -> Result<()> {
    let parent = dst.parent().context("destination has no parent")?;
    fs::create_dir_all(parent)?;
    let name = dst
        .file_name()
        .and_then(|n| n.to_str())
        .context("bad file name")?;
    let tmp = parent.join(format!(".{name}.spacetravlr-new"));
    if tmp.exists() {
        fs::remove_file(&tmp).ok();
    }
    fs::copy(src, &tmp).with_context(|| format!("copy to {}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(src)?.permissions().mode();
        fs::set_permissions(&tmp, fs::Permissions::from_mode(mode | 0o111))?;
    }
    fs::rename(&tmp, dst).with_context(|| format!("rename to {}", dst.display()))?;
    Ok(())
}

fn extract_tar_gz(archive_path: &Path, out_dir: &Path) -> Result<()> {
    let f = File::open(archive_path)?;
    let dec = GzDecoder::new(f);
    let mut arch = tar::Archive::new(dec);
    arch.unpack(out_dir)
        .with_context(|| format!("extract {}", archive_path.display()))?;
    Ok(())
}

pub fn run(update_version: Option<&str>) -> Result<()> {
    let target = prebuilt_tarball_target().with_context(|| {
        format!(
            "unsupported OS/arch for prebuilt binaries: {} {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;
    let exe = std::env::current_exe().context("current_exe")?;
    let install_dir = exe
        .parent()
        .map(Path::to_path_buf)
        .context("current_exe has no parent directory")?;

    eprintln!("Resolving GitHub release ({})…", GITHUB_REPO);
    let release = fetch_release(update_version)?;
    let remote_tag = release.tag_name.trim();
    let remote_ver = normalize_tag_version(remote_tag)?;
    let local_ver = embedded_version()?;

    if local_ver >= remote_ver {
        eprintln!(
            "Already up to date (installed {}, latest release {}).",
            local_ver, remote_ver
        );
        return Ok(());
    }

    let tar_name = tarball_name(remote_tag, &target);
    let tarball_url = release
        .assets
        .iter()
        .find(|a| a.name == tar_name)
        .map(|a| a.browser_download_url.as_str())
        .with_context(|| format!("release {} has no asset {tar_name:?}", release.tag_name))?;

    let sums_asset = release
        .assets
        .iter()
        .find(|a| a.name == "SHA256SUMS")
        .context("release missing SHA256SUMS (required for self-update)")?;

    let sums: String = ureq::get(&sums_asset.browser_download_url)
        .set("User-Agent", "spacetravlr-self-update")
        .call()
        .with_context(|| "download SHA256SUMS")?
        .into_string()
        .context("read SHA256SUMS")?;

    let expected = parse_checksums(&sums, &tar_name)?
        .with_context(|| format!("SHA256SUMS has no entry for {tar_name}"))?;

    let tmp_root = std::env::temp_dir().join(format!(
        "spacetravlr-update-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    ));
    fs::create_dir_all(&tmp_root)?;
    let archive_path = tmp_root.join(&tar_name);
    eprintln!("Downloading {} …", tar_name);
    download_to_path(tarball_url, &archive_path)?;

    let actual = sha256_file(&archive_path)?;
    if actual != expected {
        bail!(
            "SHA256 mismatch for {} (expected {expected}, got {actual})",
            tar_name
        );
    }

    let extract_dir = tmp_root.join("extract");
    fs::create_dir_all(&extract_dir)?;
    eprintln!("Extracting…");
    extract_tar_gz(&archive_path, &extract_dir)?;

    for bin in DISTRIBUTION_BINARIES {
        let src = extract_dir.join(bin);
        if !src.is_file() {
            bail!("archive missing executable {bin}");
        }
        let dst = install_dir.join(bin);
        eprintln!("Installing {} → {}", bin, dst.display());
        /*
         * On Windows replacing a running .exe often fails; this tool targets Unix-style installs.
         */
        atomic_replace_file(&src, &dst)?;
    }

    fs::remove_dir_all(&tmp_root).ok();

    eprintln!(
        "Update complete. Installed {} (was {}). Run `{} --version` to verify.",
        remote_ver, local_ver, DISTRIBUTION_BINARIES[0]
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glibc_from_ldd_ubuntu() {
        assert_eq!(
            glibc_version_from_ldd_first_line("ldd (Ubuntu GLIBC 2.35-0ubuntu3.8) 2.35"),
            Some((2, 35))
        );
    }

    #[test]
    fn glibc_from_ldd_gnu() {
        assert_eq!(
            glibc_version_from_ldd_first_line("ldd (GNU libc) 2.34"),
            Some((2, 34))
        );
    }
}
