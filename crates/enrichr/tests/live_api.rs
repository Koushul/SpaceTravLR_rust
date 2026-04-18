use enrichr::{EnrichrClient, EnrichrSite};

#[test]
#[ignore = "calls maayanlab.cloud Enrichr"]
fn live_add_list_and_enrich() {
    let client = EnrichrClient::new(EnrichrSite::HumanMouse);
    let added = client
        .add_list(&["TP53", "MDM2", "CDKN1A"], Some("enrichr-rs live test"))
        .expect("addList");
    assert!(!added.short_id.is_empty());
    let table = client
        .enrich(added.user_list_id, "KEGG_2021_Human")
        .expect("enrich");
    assert!(!table.rows.is_empty());
}
