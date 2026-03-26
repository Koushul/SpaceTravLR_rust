use space_trav_lr_rust::condition_split::{resolve_condition_dir_names, sanitize_condition_value};

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
