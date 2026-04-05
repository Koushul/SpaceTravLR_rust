use space_trav_lr_rust::condition_split::{
    find_condition_dir_matching_label, resolve_condition_dir_names, sanitize_condition_value,
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
