use ratatui::style::Color;
use ratatui_themes::{ThemeName, ThemePalette};

/// Golden-angle step (degrees) for additional cell types beyond the palette seeds.
const GOLDEN_ANGLE_DEG: f64 = 137.50776405003785;

fn rgb_norm(c: Color) -> Option<(f64, f64, f64)> {
    match c {
        Color::Rgb(r, g, b) => Some((
            f64::from(r) / 255.0,
            f64::from(g) / 255.0,
            f64::from(b) / 255.0,
        )),
        _ => None,
    }
}

fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let v = max;
    let s = if max <= 1e-9 { 0.0 } else { d / max };
    let h = if d <= 1e-9 {
        0.0
    } else if (max - r).abs() <= 1e-9 {
        60.0 * (((g - b) / d).rem_euclid(6.0))
    } else if (max - g).abs() <= 1e-9 {
        60.0 * (((b - r) / d) + 2.0)
    } else {
        60.0 * (((r - g) / d) + 4.0)
    };
    (h.rem_euclid(360.0), s.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let h = h.rem_euclid(360.0);
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
    let m = v - c;
    let (rp, gp, bp) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let clamp_byte = |t: f64| -> u8 { (t * 255.0).round().clamp(0.0, 255.0) as u8 };
    (
        clamp_byte(rp + m),
        clamp_byte(gp + m),
        clamp_byte(bp + m),
    )
}

fn hue_deg(c: Color) -> f64 {
    rgb_norm(c)
        .map(|(r, g, b)| rgb_to_hsv(r, g, b).0)
        .unwrap_or(0.0)
}

fn theme_palette_for_slot(slot: usize) -> ThemePalette {
    let all = ThemeName::all();
    if all.is_empty() {
        return ThemeName::Dracula.palette();
    }
    all[slot % all.len()].palette()
}

/// Distinct foreground color for a cell-type label in the training TUI, derived from the
/// active `ratatui-themes` palette: up to seven types map to evenly spaced semantic colors
/// (sorted by hue); further types use golden-angle hues with saturation/value tuned for
/// light vs dark themes.
pub fn cell_type_color_for_label(theme_slot: usize, label: &str, sorted_unique: &[String]) -> Color {
    let n = sorted_unique.len().max(1);
    let idx = sorted_unique.iter().position(|s| s == label).unwrap_or(0) % n;
    cell_type_color(theme_slot, idx, sorted_unique.len())
}

pub fn cell_type_color(theme_slot: usize, type_index: usize, n_types: usize) -> Color {
    let p = theme_palette_for_slot(theme_slot);
    let n = n_types.max(1);
    let ti = type_index % n;
    let max_palette = n.min(7);
    const SEEDS: usize = 7;
    let seeds = [
        p.error,
        p.warning,
        p.success,
        p.info,
        p.accent,
        p.secondary,
        p.fg,
    ];
    if max_palette > 0 && ti < max_palette {
        let mut pairs: Vec<(f64, usize)> = seeds
            .iter()
            .enumerate()
            .map(|(i, &c)| (hue_deg(c), i))
            .collect();
        pairs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sorted_idx = ((ti * SEEDS) / max_palette).min(SEEDS - 1);
        let seed_i = pairs[sorted_idx].1;
        return seeds[seed_i];
    }
    let synth_i = ti.saturating_sub(max_palette);
    let accent_h = hue_deg(p.accent);
    let h = (accent_h + (f64::from(synth_i as u32) + 1.0) * GOLDEN_ANGLE_DEG).rem_euclid(360.0);
    let is_light = p.is_light();
    let s = if is_light {
        (0.72 + 0.06 * (f64::from((synth_i % 5) as u32) / 4.0)).min(1.0)
    } else {
        (0.55 + 0.10 * (f64::from((synth_i % 4) as u32) / 3.0)).min(1.0)
    };
    let v = if is_light {
        (0.38 + 0.08 * (f64::from((synth_i % 3) as u32) / 2.0)).min(0.55)
    } else {
        (0.82 + 0.12 * (f64::from((synth_i % 3) as u32) / 2.0)).min(1.0)
    };
    let (r, g, b) = hsv_to_rgb(h, s, v);
    Color::Rgb(r, g, b)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TuiColors {
    pub bg: Color,
    pub outer_bord: Color,
    pub tel_bord: Color,
    pub work_bord: Color,
    pub rocket_bord: Color,
    pub gauge_empty: Color,
    pub label: Color,
    pub value: Color,
    pub lilac: Color,
    pub sky: Color,
    pub grape: Color,
    pub muted: Color,
    pub title: Color,
    pub c_wrote: Color,
    pub c_fail: Color,
    pub c_skip: Color,
    pub c_topr2: Color,
    pub c_botr2: Color,
    pub perf_bord: Color,
}

impl TuiColors {
    pub fn from_ratatui_palette(p: ThemePalette) -> Self {
        Self {
            bg: p.bg,
            outer_bord: p.muted,
            tel_bord: p.info,
            work_bord: p.warning,
            rocket_bord: p.secondary,
            gauge_empty: p.muted,
            label: p.warning,
            value: p.success,
            lilac: p.secondary,
            sky: p.info,
            grape: p.accent,
            muted: p.muted,
            title: p.fg,
            c_wrote: p.success,
            c_fail: p.error,
            c_skip: p.muted,
            c_topr2: p.info,
            c_botr2: p.muted,
            perf_bord: p.warning,
        }
    }

    /// Initial **t** theme index: **Gruvbox Dark** ([`ThemeName::GruvboxDark`]).
    pub fn default_theme_slot() -> usize {
        ThemeName::all()
            .iter()
            .position(|&t| t == ThemeName::GruvboxDark)
            .unwrap_or(0)
    }

    pub fn default_palette() -> Self {
        Self::resolve(Self::default_theme_slot())
    }

    pub fn resolve(slot: usize) -> Self {
        let all = ThemeName::all();
        if all.is_empty() {
            return Self::from_ratatui_palette(ThemeName::Dracula.palette());
        }
        let idx = slot % all.len();
        Self::from_ratatui_palette(all[idx].palette())
    }

    pub fn theme_count() -> usize {
        ThemeName::all().len().max(1)
    }

    pub fn advance_slot(slot: usize) -> usize {
        (slot + 1) % Self::theme_count()
    }

    pub fn theme_label(slot: usize) -> String {
        let all = ThemeName::all();
        if all.is_empty() {
            return "Dracula".to_string();
        }
        all[slot % all.len()].display_name().to_string()
    }
}
