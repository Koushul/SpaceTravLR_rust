//! Cross-process barrier for shared run setup (repro TOML, adata prep, condition
//! dirs, ligand-field DB, spatial caches).
//!
//! Gene workers already coordinate with `{gene}.lock`. This file is **not** named
//! `*.lock` so [`crate::spatial_estimator::remove_stale_lock_files_in_dir`] never
//! deletes it mid-setup.
//!
//! # Protocol
//!
//! 1. If `spacetravlr_setup.ready` exists → follower (setup already finished).
//! 2. Else `create_new(spacetravlr_setup.flock)` → this process is the setup leader.
//!    Heartbeat touches the flock while prep/caches run; `mark_ready` writes the
//!    ready file (temp + rename) then removes the flock.
//! 3. Otherwise wait: poll ready, steal the flock if its mtime is older than the
//!    stale threshold (leader SIGKILL), or time out.
//!
//! `--join-output-dir` sets `can_lead = false` and never creates the flock.

use anyhow::Context;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

pub const SETUP_LOCK_FILENAME: &str = "spacetravlr_setup.flock";
pub const SETUP_READY_FILENAME: &str = "spacetravlr_setup.ready";

const HEARTBEAT_SECS: u64 = 15;
const WAIT_LOG_EVERY_SECS: u64 = 10;
/// When `[execution].stale_lock_secs` is 0, still steal a dead setup flock after this.
pub const DEFAULT_SETUP_STALE_SECS: u64 = 7200;
const JOIN_WAIT_TIMEOUT: Duration = Duration::from_secs(24 * 3600);

pub fn setup_lock_path(output_dir: &Path) -> PathBuf {
    output_dir.join(SETUP_LOCK_FILENAME)
}

pub fn setup_ready_path(output_dir: &Path) -> PathBuf {
    output_dir.join(SETUP_READY_FILENAME)
}

pub fn setup_is_ready(output_dir: &Path) -> bool {
    setup_ready_path(output_dir).is_file()
}

fn host_label() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".into())
}

fn unix_secs(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn lock_payload() -> String {
    format!(
        "spacetravlr_setup 1\nhost={}\npid={}\nstarted_unix={}\n",
        host_label(),
        std::process::id(),
        unix_secs(SystemTime::now())
    )
}

fn ready_payload() -> String {
    format!(
        "spacetravlr_setup_ready 1\nhost={}\npid={}\nfinished_unix={}\n",
        host_label(),
        std::process::id(),
        unix_secs(SystemTime::now())
    )
}

fn file_age_secs(path: &Path) -> Option<u64> {
    let meta = fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    SystemTime::now().duration_since(modified).ok().map(|d| d.as_secs())
}

fn resolved_stale_secs(configured: u64) -> u64 {
    if configured > 0 {
        configured
    } else {
        DEFAULT_SETUP_STALE_SECS
    }
}

fn try_create_lock(lock_path: &Path) -> anyhow::Result<Option<fs::File>> {
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(lock_path)
    {
        Ok(mut f) => {
            f.write_all(lock_payload().as_bytes())?;
            let _ = f.flush();
            Ok(Some(f))
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(None),
        Err(e) => Err(e).with_context(|| format!("create {}", lock_path.display())),
    }
}

fn steal_if_stale(lock_path: &Path, stale_secs: u64, log: &dyn Fn(&str)) -> bool {
    let Some(age) = file_age_secs(lock_path) else {
        return false;
    };
    if age < stale_secs {
        return false;
    }
    match fs::remove_file(lock_path) {
        Ok(()) => {
            log(&format!(
                "setup: stole stale {} ({age}s >= {stale_secs}s)",
                SETUP_LOCK_FILENAME
            ));
            true
        }
        Err(_) => false,
    }
}

fn spawn_heartbeat(lock_path: PathBuf, stop: Arc<AtomicBool>) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut acc_ms = 0u64;
        const TICK_MS: u64 = 200;
        while !stop.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(TICK_MS));
            if stop.load(Ordering::Relaxed) {
                break;
            }
            acc_ms += TICK_MS;
            if acc_ms < HEARTBEAT_SECS * 1000 {
                continue;
            }
            acc_ms = 0;
            if let Ok(mut f) = OpenOptions::new().write(true).truncate(true).open(&lock_path) {
                let _ = f.write_all(lock_payload().as_bytes());
                let _ = f.flush();
            }
        }
    })
}

/// Holds the exclusive setup flock until [`SetupLeaderGuard::mark_ready`] or drop.
pub struct SetupLeaderGuard {
    lock_path: PathBuf,
    ready_path: PathBuf,
    stop: Arc<AtomicBool>,
    heartbeat: Option<JoinHandle<()>>,
    marked: AtomicBool,
}

impl SetupLeaderGuard {
    fn new(output_dir: &Path) -> Self {
        let lock_path = setup_lock_path(output_dir);
        let stop = Arc::new(AtomicBool::new(false));
        let heartbeat = spawn_heartbeat(lock_path.clone(), stop.clone());
        Self {
            lock_path,
            ready_path: setup_ready_path(output_dir),
            stop,
            heartbeat: Some(heartbeat),
            marked: AtomicBool::new(false),
        }
    }

    pub fn is_marked_ready(&self) -> bool {
        self.marked.load(Ordering::SeqCst)
    }

    /// Publish `spacetravlr_setup.ready` then drop the flock so waiters proceed.
    /// Idempotent.
    pub fn mark_ready(&self) -> anyhow::Result<()> {
        if self.marked.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        self.stop.store(true, Ordering::Relaxed);
        if let Some(parent) = self.ready_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = self.ready_path.with_extension("ready.tmp");
        fs::write(&tmp, ready_payload())
            .with_context(|| format!("write {}", tmp.display()))?;
        fs::rename(&tmp, &self.ready_path).with_context(|| {
            format!(
                "rename {} -> {}",
                tmp.display(),
                self.ready_path.display()
            )
        })?;
        let _ = fs::remove_file(&self.lock_path);
        Ok(())
    }
}

impl Drop for SetupLeaderGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.heartbeat.take() {
            let _ = h.join();
        }
        if !self.marked.load(Ordering::SeqCst) {
            let _ = fs::remove_file(&self.lock_path);
        }
    }
}

pub enum SetupRole {
    Leader(SetupLeaderGuard),
    Follower,
}

pub struct SetupParticipateOpts<'a> {
    pub output_dir: &'a Path,
    /// False for `--join-output-dir` (never create the flock).
    pub can_lead: bool,
    pub stale_lock_secs: u64,
    pub wait_timeout: Option<Duration>,
    pub log: &'a dyn Fn(&str),
}

/// Elect a setup leader or wait until [`SETUP_READY_FILENAME`] exists.
pub fn participate_run_setup(opts: SetupParticipateOpts<'_>) -> anyhow::Result<SetupRole> {
    let output_dir = opts.output_dir;
    fs::create_dir_all(output_dir)?;
    let lock_path = setup_lock_path(output_dir);
    let ready_path = setup_ready_path(output_dir);
    let stale = resolved_stale_secs(opts.stale_lock_secs);
    let timeout = opts.wait_timeout.unwrap_or(if opts.can_lead {
        Duration::from_secs(u64::MAX / 4)
    } else {
        JOIN_WAIT_TIMEOUT
    });
    let started = SystemTime::now();
    let mut last_log = started;
    let log = opts.log;

    loop {
        if ready_path.is_file() {
            log(&format!("setup: {} present — joining", SETUP_READY_FILENAME));
            return Ok(SetupRole::Follower);
        }

        if opts.can_lead {
            match try_create_lock(&lock_path)? {
                Some(_f) => {
                    log(&format!(
                        "setup: this process is leader ({})",
                        SETUP_LOCK_FILENAME
                    ));
                    return Ok(SetupRole::Leader(SetupLeaderGuard::new(output_dir)));
                }
                None => {}
            }
        }

        if lock_path.is_file() && steal_if_stale(&lock_path, stale, log) && opts.can_lead {
            continue;
        }

        let elapsed = SystemTime::now()
            .duration_since(started)
            .unwrap_or(Duration::ZERO);
        if elapsed >= timeout {
            anyhow::bail!(
                "timed out after {}s waiting for {} in {}. Start a leader `spacetravlr` on this directory (without --join-output-dir), or delete a stuck {}.",
                elapsed.as_secs(),
                SETUP_READY_FILENAME,
                output_dir.display(),
                SETUP_LOCK_FILENAME
            );
        }

        if SystemTime::now()
            .duration_since(last_log)
            .map(|d| d.as_secs() >= WAIT_LOG_EVERY_SECS)
            .unwrap_or(true)
        {
            last_log = SystemTime::now();
            let holder = if lock_path.is_file() {
                fs::read_to_string(&lock_path)
                    .unwrap_or_default()
                    .lines()
                    .take(3)
                    .collect::<Vec<_>>()
                    .join("; ")
            } else {
                "no flock yet".into()
            };
            log(&format!(
                "setup: waiting for leader ({:.0}s) — {holder}",
                elapsed.as_secs_f64()
            ));
        }

        thread::sleep(Duration::from_millis(250));
    }
}

/// Join-only wait used when `--join-output-dir` is set before the repro TOML exists.
pub fn wait_for_setup_ready(
    output_dir: &Path,
    stale_lock_secs: u64,
    log: &dyn Fn(&str),
) -> anyhow::Result<()> {
    match participate_run_setup(SetupParticipateOpts {
        output_dir,
        can_lead: false,
        stale_lock_secs,
        wait_timeout: Some(JOIN_WAIT_TIMEOUT),
        log,
    })? {
        SetupRole::Follower => Ok(()),
        SetupRole::Leader(_) => unreachable!("can_lead=false"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn temp_dir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "st_setup_{}_{}_{}",
            std::process::id(),
            name,
            unix_secs(SystemTime::now())
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn logs() -> (Arc<Mutex<Vec<String>>>, impl Fn(&str)) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let b2 = buf.clone();
        let f = move |s: &str| {
            b2.lock().unwrap().push(s.to_string());
        };
        (buf, f)
    }

    #[test]
    fn ready_file_is_immediate_follower() {
        let dir = temp_dir("ready");
        fs::write(setup_ready_path(&dir), b"ok").unwrap();
        let (_b, log) = logs();
        let role = participate_run_setup(SetupParticipateOpts {
            output_dir: &dir,
            can_lead: true,
            stale_lock_secs: 60,
            wait_timeout: Some(Duration::from_secs(2)),
            log: &log,
        })
        .unwrap();
        assert!(matches!(role, SetupRole::Follower));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn first_process_is_leader_second_waits_then_follows() {
        let dir = temp_dir("elect");
        let d2 = dir.clone();
        let (_b, log) = logs();
        let leader = participate_run_setup(SetupParticipateOpts {
            output_dir: &dir,
            can_lead: true,
            stale_lock_secs: 60,
            wait_timeout: Some(Duration::from_secs(2)),
            log: &log,
        })
        .unwrap();
        let SetupRole::Leader(guard) = leader else {
            panic!("expected leader");
        };
        assert!(setup_lock_path(&dir).is_file());
        let handle = thread::spawn(move || {
            participate_run_setup(SetupParticipateOpts {
                output_dir: &d2,
                can_lead: true,
                stale_lock_secs: 60,
                wait_timeout: Some(Duration::from_secs(8)),
                log: &|s| {
                    let _ = s;
                },
            })
        });
        thread::sleep(Duration::from_millis(400));
        guard.mark_ready().unwrap();
        drop(guard);
        let role = handle.join().unwrap().unwrap();
        assert!(matches!(role, SetupRole::Follower));
        assert!(setup_ready_path(&dir).is_file());
        assert!(!setup_lock_path(&dir).is_file());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn drop_without_ready_releases_lock_so_another_can_lead() {
        let dir = temp_dir("drop");
        let (_b, log) = logs();
        {
            let role = participate_run_setup(SetupParticipateOpts {
                output_dir: &dir,
                can_lead: true,
                stale_lock_secs: 60,
                wait_timeout: Some(Duration::from_secs(2)),
                log: &log,
            })
            .unwrap();
            assert!(matches!(role, SetupRole::Leader(_)));
        }
        assert!(!setup_lock_path(&dir).is_file());
        let role = participate_run_setup(SetupParticipateOpts {
            output_dir: &dir,
            can_lead: true,
            stale_lock_secs: 60,
            wait_timeout: Some(Duration::from_secs(2)),
            log: &log,
        })
        .unwrap();
        assert!(matches!(role, SetupRole::Leader(_)));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn join_cannot_create_lock_times_out() {
        let dir = temp_dir("join");
        let err = match participate_run_setup(SetupParticipateOpts {
            output_dir: &dir,
            can_lead: false,
            stale_lock_secs: 60,
            wait_timeout: Some(Duration::from_millis(400)),
            log: &|_| {},
        }) {
            Err(e) => e,
            Ok(_) => panic!("join without a leader should time out"),
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("timed out"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_flock_is_stolen_by_leader() {
        let dir = temp_dir("stale");
        let lock = setup_lock_path(&dir);
        fs::write(&lock, b"dead").unwrap();
        let old = SystemTime::now() - Duration::from_secs(100);
        let f = OpenOptions::new().write(true).open(&lock).unwrap();
        let _ = f.set_modified(old);
        drop(f);
        let (_b, log) = logs();
        let role = participate_run_setup(SetupParticipateOpts {
            output_dir: &dir,
            can_lead: true,
            stale_lock_secs: 1,
            wait_timeout: Some(Duration::from_secs(5)),
            log: &log,
        })
        .unwrap();
        assert!(matches!(role, SetupRole::Leader(_)));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn gene_lock_suffix_does_not_match_setup_flock() {
        assert!(!SETUP_LOCK_FILENAME.ends_with(".lock"));
        assert!(SETUP_LOCK_FILENAME.ends_with(".flock"));
    }
}
