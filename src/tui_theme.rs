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
    pub const LEGACY: Self = Self {
        bg: Color::Rgb(40, 40, 40),
        outer_bord: Color::Rgb(60, 56, 54),
        tel_bord: Color::Rgb(69, 133, 136),
        work_bord: Color::Rgb(215, 153, 33),
        rocket_bord: Color::Rgb(131, 165, 152),
        gauge_empty: Color::Rgb(60, 56, 54),
        label: Color::Rgb(215, 153, 33),
        value: Color::Rgb(142, 192, 124),
        lilac: Color::Rgb(184, 187, 38),
        sky: Color::Rgb(69, 133, 136),
        grape: Color::Rgb(211, 134, 155),
        muted: Color::Rgb(146, 131, 116),
        title: Color::Rgb(235, 219, 178),
        c_wrote: Color::Rgb(142, 192, 124),
        c_fail: Color::Rgb(204, 36, 29),
        c_skip: Color::Rgb(146, 131, 116),
        c_topr2: Color::Rgb(69, 133, 136),
        c_botr2: Color::Rgb(146, 131, 116),
        perf_bord: Color::Rgb(215, 153, 33),
    };

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

    pub fn resolve(slot: usize) -> Self {
        if slot == 0 {
            Self::LEGACY
        } else {
            let all = ThemeName::all();
            let idx = slot - 1;
            if idx >= all.len() {
                return Self::LEGACY;
            }
            Self::from_ratatui_palette(all[idx].palette())
        }
    }

    pub fn theme_count() -> usize {
        1 + ThemeName::all().len()
    }

    pub fn advance_slot(slot: usize) -> usize {
        (slot + 1) % Self::theme_count()
    }

    pub fn theme_label(slot: usize) -> String {
        if slot == 0 {
            "default".to_string()
        } else {
            ThemeName::all()[slot - 1].display_name().to_string()
        }
    }
}
