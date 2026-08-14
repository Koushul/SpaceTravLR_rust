use polars::prelude::{DataFrame, NamedFrom, Series};
use spacetravlr::condition_split::{
    CONDITION_LABEL_FILENAME, CONDITION_RUNS_SUBDIR, SAMPLE_LABEL_FILENAME, SAMPLE_RUNS_SUBDIR,
    find_condition_dir_matching_label, prepare_sample_splits_from_obs, resolve_condition_dir_names,
    sanitize_condition_value,
};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn sanitize_condition_value_replaces_unsafe_chars() {
    assert_eq!(
        sanitize_condition_value("batch A / drug+B"),
        "batch_A_drug_B"
    );
    assert_eq!(sanitize_condition_value("  "), "group");
    assert_eq!(sanitize_condition_value("..."), "group");
}

#[test]
fn resolve_condition_dir_names_disambiguates_collisions() {
    let labels = vec![
        "A/B".to_string(),
        "A_B".to_string(),
        "A B".to_string(),
        "A-B".to_string(),
    ];
    let dirs = resolve_condition_dir_names(&labels);
    assert_eq!(dirs, vec!["A_B", "A_B_2", "A_B_3", "A-B"]);
}

#[test]
fn find_condition_dir_matches_label_file_not_sanitized_name() {
    let tmp = std::env::temp_dir().join(format!(
        "stlr_cond_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&tmp);
    let cond = tmp.join("conditions").join("weird_Old_Name");
    fs::create_dir_all(&cond).unwrap();
    fs::write(cond.join("condition_label.txt"), "Patient / 1\n").unwrap();
    let got = find_condition_dir_matching_label(tmp.to_str().unwrap(), "Patient / 1");
    assert_eq!(
        got.as_ref().map(|p| p.file_name().unwrap()),
        Some("weird_Old_Name".as_ref())
    );
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn prepare_sample_splits_standalone_uses_conditions_and_condition_label() {
    let tmp = std::env::temp_dir().join(format!(
        "stlr_sample_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let obs = DataFrame::new(vec![
        Series::new("sample".into(), vec!["s1", "s1", "s2", "s2", "s1", "s2"]).into(),
    ])
    .unwrap();
    let plans = prepare_sample_splits_from_obs(&obs, tmp.to_str().unwrap(), "sample", false, false)
        .unwrap();
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].label, "s1");
    assert_eq!(plans[1].label, "s2");
    assert_eq!(plans[0].n_obs, 3);
    assert_eq!(plans[1].n_obs, 3);
    let d1 = tmp.join(CONDITION_RUNS_SUBDIR).join("s1");
    let d2 = tmp.join(CONDITION_RUNS_SUBDIR).join("s2");
    assert!(d1.is_dir());
    assert!(d2.is_dir());
    assert_eq!(
        fs::read_to_string(d1.join(CONDITION_LABEL_FILENAME))
            .unwrap()
            .trim(),
        "s1"
    );
    assert_eq!(
        fs::read_to_string(d2.join(CONDITION_LABEL_FILENAME))
            .unwrap()
            .trim(),
        "s2"
    );
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn prepare_sample_splits_nested_uses_samples_and_sample_label() {
    let tmp = std::env::temp_dir().join(format!(
        "stlr_sample_nested_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let obs = DataFrame::new(vec![
        Series::new("sample".into(), vec!["a", "b", "a"]).into(),
    ])
    .unwrap();
    let plans =
        prepare_sample_splits_from_obs(&obs, tmp.to_str().unwrap(), "sample", false, true).unwrap();
    assert_eq!(plans.len(), 2);
    let d_a = tmp.join(SAMPLE_RUNS_SUBDIR).join("a");
    let d_b = tmp.join(SAMPLE_RUNS_SUBDIR).join("b");
    assert!(d_a.is_dir());
    assert!(d_b.is_dir());
    assert_eq!(
        fs::read_to_string(d_a.join(SAMPLE_LABEL_FILENAME))
            .unwrap()
            .trim(),
        "a"
    );
    assert_eq!(
        fs::read_to_string(d_b.join(SAMPLE_LABEL_FILENAME))
            .unwrap()
            .trim(),
        "b"
    );
    let _ = fs::remove_dir_all(&tmp);
}
