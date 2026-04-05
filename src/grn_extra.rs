//! User-supplied LR pairs (`extra_lr`) and extra modulator genes (fourth Lasso group).
//! See plan: merge after DB LR selection; filter extras against an occupied gene set.

use std::collections::HashSet;
use std::path::Path;

/// Parse `LIG$REC` (first `$` only) or `LIG,REC`.
pub fn parse_extra_lr_token(token: &str) -> Option<(String, String)> {
    let t = token.trim();
    if t.is_empty() || t.starts_with('#') {
        return None;
    }
    if let Some(pos) = t.find('$') {
        let (l, r) = t.split_at(pos);
        let r = &r[1..];
        let l = l.trim();
        let r = r.trim();
        if l.is_empty() || r.is_empty() {
            return None;
        }
        return Some((l.to_string(), r.to_string()));
    }
    if let Some(pos) = t.find(',') {
        let (l, r) = t.split_at(pos);
        let r = &r[1..];
        let l = l.trim();
        let r = r.trim();
        if l.is_empty() || r.is_empty() {
            return None;
        }
        return Some((l.to_string(), r.to_string()));
    }
    None
}

/// One pair per line; `#` comments; supports `LIG$REC` or `LIG,REC`.
pub fn load_extra_lr_file(path: &Path) -> anyhow::Result<Vec<(String, String)>> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read extra_lr file {}: {}", path.display(), e))?;
    let mut out = Vec::new();
    for line in s.lines() {
        if let Some(p) = parse_extra_lr_line(line) {
            out.push(p);
        }
    }
    Ok(out)
}

fn parse_extra_lr_line(line: &str) -> Option<(String, String)> {
    let t = line.trim();
    if t.is_empty() || t.starts_with('#') {
        return None;
    }
    parse_extra_lr_token(t)
}

fn canonical_lr_pair(l: &str, r: &str) -> String {
    format!("{}${}", l.trim(), r.trim())
}

/// Comma-separated gene symbols (same as `--genes`). Empty tokens and `#…` comments are skipped; order preserved, duplicates dropped.
pub fn parse_extra_modulators_cli(s: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for part in s.split(',') {
        let g = part.trim();
        if g.is_empty() || g.starts_with('#') {
            continue;
        }
        if seen.insert(g.to_string()) {
            out.push(g.to_string());
        }
    }
    out
}

/// Parse `--extra-lr` the same way as multiple TOML `extra_lr` entries (merged with config).
///
/// - **`LIG$REC` list:** comma-separated, e.g. `L1$R1,L2$R2`.
/// - **`LIG,REC` list:** semicolon-separated pairs, e.g. `L1,R1;L2,R2`.
/// - **Single `LIG,REC` pair:** one comma, no `$`, e.g. `VEGFA,FLT1`.
pub fn parse_extra_lr_cli(s: &str) -> anyhow::Result<Vec<String>> {
    let s = s.trim();
    if s.is_empty() || s.starts_with('#') {
        return Ok(Vec::new());
    }

    let mut out: Vec<String> = Vec::new();

    if s.contains(';') {
        for part in s.split(';') {
            let t = part.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            let Some((l, r)) = parse_extra_lr_token(t) else {
                anyhow::bail!(
                    "invalid ligand–receptor {:?}: use LIG$REC or LIG,REC (separate pairs with ';')",
                    t
                );
            };
            out.push(canonical_lr_pair(&l, &r));
        }
        return Ok(out);
    }

    if s.contains('$') {
        for part in s.split(',') {
            let t = part.trim();
            if t.is_empty() {
                continue;
            }
            let Some((l, r)) = parse_extra_lr_token(t) else {
                anyhow::bail!(
                    "invalid segment {:?}: with '$' in the value, use comma-separated LIG$REC tokens (e.g. L1$R1,L2$R2)",
                    t
                );
            };
            out.push(canonical_lr_pair(&l, &r));
        }
        return Ok(out);
    }

    let comma_n = s.matches(',').count();
    if comma_n == 0 {
        anyhow::bail!(
            "expected LIG$REC, or LIG,REC, or multiple pairs (L1$R1,L2$R2 or L1,R1;L2,R2)"
        );
    }
    if comma_n > 1 {
        anyhow::bail!(
            "multiple comma-separated pairs: use ';' between LIG,REC pairs (e.g. L1,R1;L2,R2) or use L1$R1,L2$R2"
        );
    }

    let Some((l, r)) = parse_extra_lr_token(s) else {
        anyhow::bail!("invalid ligand–receptor pair {:?}", s);
    };
    out.push(canonical_lr_pair(&l, &r));
    Ok(out)
}

/// Genes from file: one symbol per line (or comma-separated on a line); `#` comments.
pub fn load_extra_modulators_file(path: &Path) -> anyhow::Result<Vec<String>> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read extra_modulators file {}: {}", path.display(), e))?;
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for line in s.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        for part in t.split(',') {
            let g = part.trim();
            if g.is_empty() {
                continue;
            }
            if seen.insert(g.to_string()) {
                out.push(g.to_string());
            }
        }
    }
    Ok(out)
}

/// Append user LR pairs after the database list. Returns number appended.
pub fn merge_extra_lr_into(
    ligands: &mut Vec<String>,
    receptors: &mut Vec<String>,
    lr_pairs: &mut Vec<String>,
    pairs: &[(String, String)],
    target_gene: &str,
    var_set: &HashSet<String>,
) -> usize {
    let mut seen: HashSet<String> = lr_pairs.iter().cloned().collect();
    let mut n = 0usize;
    for (l, r) in pairs {
        if l == target_gene || r == target_gene {
            continue;
        }
        if !var_set.contains(l) || !var_set.contains(r) {
            continue;
        }
        let pair = format!("{}${}", l, r);
        if seen.insert(pair.clone()) {
            ligands.push(l.clone());
            receptors.push(r.clone());
            lr_pairs.push(pair);
            n += 1;
        }
    }
    n
}

/// Filter requested extra modulators: must be in `var_set`, not in `occupied`. Preserves order, dedupes.
pub fn filter_extra_modulators(
    requested: &[String],
    occupied: &HashSet<String>,
    var_set: &HashSet<String>,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for g in requested {
        if !var_set.contains(g) {
            continue;
        }
        if occupied.contains(g) {
            continue;
        }
        if seen.insert(g.clone()) {
            out.push(g.clone());
        }
    }
    out
}

pub fn verify_modulator_invariants(
    target_gene: &str,
    regulators: &[String],
    lr_pairs: &[String],
    extra_modulators: &[String],
) -> anyhow::Result<()> {
    if regulators.iter().any(|g| g == target_gene) {
        anyhow::bail!(
            "GRN invariant violated: target gene {:?} appears as a TF regulator for itself; fix the network or priors",
            target_gene
        );
    }
    for pair in lr_pairs {
        let parts: Vec<&str> = pair.splitn(2, '$').collect();
        if parts.len() != 2 {
            continue;
        }
        if parts[0] == target_gene || parts[1] == target_gene {
            anyhow::bail!(
                "LR invariant violated: target {:?} must not be ligand or receptor in pair {:?}",
                target_gene,
                pair
            );
        }
    }
    if extra_modulators.iter().any(|g| g == target_gene) {
        anyhow::bail!(
            "extra_modulators invariant violated: target {:?} must not be listed as an extra modulator",
            target_gene
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_lr_dollar_and_comma() {
        assert_eq!(parse_extra_lr_token("A$B"), Some(("A".into(), "B".into())));
        assert_eq!(
            parse_extra_lr_token("gene1,gene2"),
            Some(("gene1".into(), "gene2".into()))
        );
        assert!(parse_extra_lr_token("nosep").is_none());
    }

    #[test]
    fn merge_skips_target_and_dup() {
        let mut l = vec!["L0".into()];
        let mut r = vec!["R0".into()];
        let mut p = vec!["L0$R0".into()];
        let var: HashSet<_> = ["L0", "R0", "A", "B", "T"]
            .into_iter()
            .map(String::from)
            .collect();
        let pairs = vec![
            ("A".into(), "B".into()),
            ("T".into(), "A".into()),
            ("A".into(), "B".into()),
        ];
        let n = merge_extra_lr_into(&mut l, &mut r, &mut p, &pairs, "T", &var);
        assert_eq!(n, 1);
        assert_eq!(p, vec!["L0$R0", "A$B"]);
    }

    #[test]
    fn filter_extra_respects_occupied() {
        let req = vec!["A".into(), "B".into(), "C".into()];
        let occupied: HashSet<String> = ["A", "target"].into_iter().map(String::from).collect();
        let var: HashSet<String> = ["A", "B", "C"].into_iter().map(String::from).collect();
        let f = filter_extra_modulators(&req, &occupied, &var);
        assert_eq!(f, vec!["B", "C"]);
    }

    #[test]
    fn invariants_reject_target_in_lr() {
        let e = verify_modulator_invariants("TG", &[], &["X$TG".into()], &[]);
        assert!(e.is_err());
    }

    #[test]
    fn parse_extra_modulators_cli_dedup_and_trim() {
        assert_eq!(
            parse_extra_modulators_cli(" A , B , A ,#c"),
            vec!["A".to_string(), "B".to_string()]
        );
        assert!(parse_extra_modulators_cli("  ,  ").is_empty());
    }

    #[test]
    fn parse_extra_lr_cli_variants() {
        assert_eq!(
            parse_extra_lr_cli("L1$R1,L2$R2").unwrap(),
            vec!["L1$R1".to_string(), "L2$R2".to_string()]
        );
        assert_eq!(
            parse_extra_lr_cli("L1,R1;L2,R2").unwrap(),
            vec!["L1$R1".to_string(), "L2$R2".to_string()]
        );
        assert_eq!(
            parse_extra_lr_cli("VEGFA,FLT1").unwrap(),
            vec!["VEGFA$FLT1".to_string()]
        );
        assert!(parse_extra_lr_cli("").unwrap().is_empty());
        assert!(parse_extra_lr_cli("a,b,c,d").is_err());
    }
}
