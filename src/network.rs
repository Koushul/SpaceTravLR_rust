use anyhow::Result;
use polars::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct Modulators {
    pub regulators: Vec<String>,
    pub ligands: Vec<String>,
    pub receptors: Vec<String>,
    pub tfl_ligands: Vec<String>,
    pub tfl_regulators: Vec<String>,
    pub lr_pairs: Vec<String>,
    pub tfl_pairs: Vec<String>,
}

impl Modulators {
    pub fn apply_modulator_mask(
        mut self,
        use_tf_modulators: bool,
        use_lr_modulators: bool,
        use_tfl_modulators: bool,
    ) -> Self {
        if !use_tf_modulators {
            self.regulators.clear();
        }
        if !use_lr_modulators {
            self.ligands.clear();
            self.receptors.clear();
            self.lr_pairs.clear();
        }
        if !use_tfl_modulators {
            self.tfl_ligands.clear();
            self.tfl_regulators.clear();
            self.tfl_pairs.clear();
        }
        self
    }
}

#[derive(Clone)]
pub struct GeneNetwork {
    pub species: String,
    pub network_path: String,
    pub network_df: DataFrame,
}

pub fn infer_species(var_names: &[String]) -> &'static str {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    let sample_size = std::cmp::min(100, var_names.len());
    let sample: Vec<&String> = var_names.choose_multiple(&mut rng, sample_size).collect();

    let mut mouse_count = 0;
    let mut human_count = 0;

    for gene in sample {
        let chars: Vec<char> = gene.chars().collect();
        if chars.is_empty() {
            continue;
        }

        let mouse_match = chars.len() > 1
            && chars[0].is_uppercase()
            && chars[1..].iter().all(|c| !c.is_uppercase());
        if mouse_match {
            mouse_count += 1;
        }

        let human_match = chars.iter().all(|c| c.is_uppercase() || !c.is_alphabetic())
            && chars.iter().any(|c| c.is_uppercase());
        if human_match {
            human_count += 1;
        }
    }

    if mouse_count > human_count {
        "mouse"
    } else {
        "human"
    }
}

impl GeneNetwork {
    pub fn new(species: &str, var_names: &[String]) -> Result<Self> {
        let network_path = format!("data/{}_network.parquet", species);

        let full_df = LazyFrame::scan_parquet(
            polars_utils::plpath::PlPath::from_string(network_path.clone()),
            ScanArgsParquet::default(),
        )?
        .collect()?;

        let mut source_keep = Vec::new();
        let var_names_set: HashSet<&str> = var_names.iter().map(|s| s.as_str()).collect();

        if let (Ok(s_col), Ok(t_col)) = (
            full_df.column("source")?.cast(&DataType::String)?.str(),
            full_df.column("target")?.cast(&DataType::String)?.str(),
        ) {
            for (s, t) in s_col.into_no_null_iter().zip(t_col.into_no_null_iter()) {
                source_keep.push(var_names_set.contains(s) && var_names_set.contains(t));
            }
        }

        let filter_chunk = BooleanChunked::new("".into(), &source_keep);
        let network_df = full_df.filter(&filter_chunk)?;

        Ok(Self {
            species: species.to_string(),
            network_path,
            network_df,
        })
    }

    pub fn get_modulators(
        &self,
        target_gene: &str,
        tf_ligand_cutoff: f64,
        max_lr_pairs: Option<usize>,
        top_lr_pairs_by_mean_expression: Option<usize>,
        gene_mean_expression: Option<&HashMap<String, f64>>,
    ) -> Result<Modulators> {
        let lf = self.network_df.clone().lazy();

        // --- 1. Regulators (edge_type == "grn") ---
        let grn_df = lf
            .clone()
            .filter(
                col("edge_type")
                    .cast(DataType::String)
                    .eq(lit("grn"))
                    .and(col("target").cast(DataType::String).eq(lit(target_gene))),
            )
            .select([col("source")])
            .collect()?;

        let mut regulators = Vec::new();
        if let Ok(src) = grn_df.column("source")?.cast(&DataType::String)?.str() {
            let mut seen = HashSet::new();
            for v in src.into_no_null_iter() {
                if v != target_gene {
                    if seen.insert(v.to_string()) {
                        regulators.push(v.to_string());
                    }
                }
            }
        }

        // --- 2. LR Pairs (edge_type == "lr") ---
        let lr_df = lf
            .clone()
            .filter(col("edge_type").cast(DataType::String).eq(lit("lr")))
            .select([col("source"), col("target")])
            .collect()?;

        let mut ligands = Vec::new();
        let mut receptors = Vec::new();
        let mut lr_pairs = Vec::new();

        if let (Ok(l_col), Ok(r_col)) = (
            lr_df.column("source")?.cast(&DataType::String)?.str(),
            lr_df.column("target")?.cast(&DataType::String)?.str(),
        ) {
            let mut seen_pairs = HashSet::new();
            for (l, r) in l_col.into_no_null_iter().zip(r_col.into_no_null_iter()) {
                if l == target_gene || r == target_gene {
                    continue;
                }
                let pair = format!("{}${}", l, r);
                if seen_pairs.insert(pair.clone()) {
                    ligands.push(l.to_string());
                    receptors.push(r.to_string());
                    lr_pairs.push(pair);
                }
            }
        }

        Self::select_lr_pairs(
            &mut ligands,
            &mut receptors,
            &mut lr_pairs,
            max_lr_pairs,
            top_lr_pairs_by_mean_expression,
            gene_mean_expression,
        );

        // --- 3. NicheNet Pairs (edge_type == "nichenet") ---
        let regs_len = regulators.len() as u32;
        let mut tfl_ligands = Vec::new();
        let mut tfl_regulators = Vec::new();
        let mut tfl_pairs = Vec::new();

        if regs_len > 0 {
            let nn_df = lf
                .clone()
                .filter(
                    col("edge_type")
                        .cast(DataType::String)
                        .eq(lit("nichenet"))
                        .and(
                            col("weight")
                                .cast(DataType::Float64)
                                .gt(lit(tf_ligand_cutoff)),
                        ),
                )
                .select([col("source"), col("target"), col("weight")])
                .collect()?;

            if let (Ok(l_col), Ok(tf_col), Ok(w_col)) = (
                nn_df.column("source")?.cast(&DataType::String)?.str(),
                nn_df.column("target")?.cast(&DataType::String)?.str(),
                nn_df.column("weight")?.cast(&DataType::Float64)?.f64(),
            ) {
                let ligands_set: HashSet<&String> = ligands.iter().collect();
                let regs_set: HashSet<&String> = regulators.iter().collect();

                let mut tf_candidates: HashMap<String, Vec<(String, f64)>> = HashMap::new();

                for i in 0..nn_df.height() {
                    if let (Some(l), Some(tf), Some(w)) =
                        (l_col.get(i), tf_col.get(i), w_col.get(i))
                    {
                        let l_string = l.to_string();
                        let tf_string = tf.to_string();
                        if ligands_set.contains(&l_string) && regs_set.contains(&tf_string) {
                            tf_candidates
                                .entry(tf_string)
                                .or_default()
                                .push((l_string, w));
                        }
                    }
                }

                // Sort top 5 for each TF
                for reg in regulators.iter() {
                    if let Some(mut candidates) = tf_candidates.remove(reg) {
                        candidates.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        for (l, _w) in candidates.into_iter().take(5) {
                            tfl_ligands.push(l.clone());
                            tfl_regulators.push(reg.clone());
                            tfl_pairs.push(format!("{}#{}", l, reg));
                        }
                    }
                }
            }
        }

        Ok(Modulators {
            regulators,
            ligands,
            receptors,
            tfl_ligands,
            tfl_regulators,
            lr_pairs,
            tfl_pairs,
        })
    }

    fn lr_pair_mean_expr_score(ligand: &str, receptor: &str, means: &HashMap<String, f64>) -> f64 {
        let ml = means.get(ligand).copied().unwrap_or(0.0);
        let mr = means.get(receptor).copied().unwrap_or(0.0);
        0.5 * (ml + mr)
    }

    fn select_lr_pairs(
        ligands: &mut Vec<String>,
        receptors: &mut Vec<String>,
        lr_pairs: &mut Vec<String>,
        max_lr_pairs: Option<usize>,
        top_by_mean_expr: Option<usize>,
        gene_mean_expression: Option<&HashMap<String, f64>>,
    ) {
        let n = lr_pairs.len();
        if n == 0 {
            return;
        }

        if let (Some(k), Some(means)) = (top_by_mean_expr, gene_mean_expression) {
            let k = k.min(n);
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                let sa = Self::lr_pair_mean_expr_score(&ligands[a], &receptors[a], means);
                let sb = Self::lr_pair_mean_expr_score(&ligands[b], &receptors[b], means);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut new_l = Vec::with_capacity(k);
            let mut new_r = Vec::with_capacity(k);
            let mut new_p = Vec::with_capacity(k);
            for &i in order.iter().take(k) {
                new_l.push(ligands[i].clone());
                new_r.push(receptors[i].clone());
                new_p.push(lr_pairs[i].clone());
            }
            *ligands = new_l;
            *receptors = new_r;
            *lr_pairs = new_p;
            return;
        }

        if let Some(k) = max_lr_pairs {
            let k = k.min(n);
            ligands.truncate(k);
            receptors.truncate(k);
            lr_pairs.truncate(k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_species_mouse_genes() {
        let genes: Vec<String> = vec![
            "Gapdh", "Actb", "Sox2", "Pou5f1", "Nanog", "Klf4", "Myc", "Bmp4", "Fgf2", "Wnt3a",
            "Shh", "Notch1", "Dll1", "Jag1", "Hes1",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(infer_species(&genes), "mouse");
    }

    #[test]
    fn infer_species_human_genes() {
        let genes: Vec<String> = vec![
            "GAPDH", "ACTB", "SOX2", "POU5F1", "NANOG", "KLF4", "MYC", "BMP4", "FGF2", "WNT3A",
            "SHH", "NOTCH1", "DLL1", "JAG1", "HES1",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(infer_species(&genes), "human");
    }

    #[test]
    fn infer_species_mixed_defaults_to_majority() {
        // Mostly mouse-style
        let genes: Vec<String> = vec![
            "Gapdh", "Actb", "Sox2", "Pou5f1", "Nanog", "Klf4", "Myc", "BRCA1", "TP53",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(infer_species(&genes), "mouse");
    }

    #[test]
    fn infer_species_empty_defaults_human() {
        let genes: Vec<String> = vec![];
        let result = infer_species(&genes);
        assert_eq!(result, "human");
    }

    #[test]
    fn infer_species_numeric_genes() {
        // Genes with numbers like "123" → no uppercase letters
        let genes: Vec<String> = vec!["123", "456", "789"]
            .into_iter()
            .map(String::from)
            .collect();
        let result = infer_species(&genes);
        assert!(result == "human" || result == "mouse");
    }

    #[test]
    fn modulators_struct_fields() {
        let m = Modulators {
            regulators: vec!["A".into()],
            ligands: vec!["B".into()],
            receptors: vec!["C".into()],
            tfl_ligands: vec!["D".into()],
            tfl_regulators: vec!["E".into()],
            lr_pairs: vec!["B$C".into()],
            tfl_pairs: vec!["D#E".into()],
        };
        assert_eq!(m.regulators.len(), 1);
        assert_eq!(m.lr_pairs[0], "B$C");
        assert_eq!(m.tfl_pairs[0], "D#E");
    }

    #[test]
    fn apply_modulator_mask_lr_only() {
        let m = Modulators {
            regulators: vec!["A".into()],
            ligands: vec!["B".into()],
            receptors: vec!["C".into()],
            tfl_ligands: vec!["D".into()],
            tfl_regulators: vec!["E".into()],
            lr_pairs: vec!["B$C".into()],
            tfl_pairs: vec!["D#E".into()],
        };
        let m = m.apply_modulator_mask(false, true, false);
        assert!(m.regulators.is_empty());
        assert_eq!(m.lr_pairs.len(), 1);
        assert!(m.tfl_pairs.is_empty());
    }
}
