use ratatui::style::Color;
use ratatui_themes::{ThemeName, ThemePalette};

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
