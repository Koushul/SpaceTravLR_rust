//! Blocking client for the [Enrichr](https://maayanlab.cloud/Enrichr) REST API
//! ([API reference](https://maayanlab.cloud/Enrichr/help#api)).
//!
//! Example:
//!
//! ```no_run
//! use enrichr::{EnrichrClient, EnrichrSite};
//!
//! let client = EnrichrClient::new(EnrichrSite::HumanMouse);
//! let added = client.add_list(&["TP53", "MDM2", "CDKN1A"], Some("demo"))?;
//! let table = client.enrich(added.user_list_id, "KEGG_2021_Human")?;
//! for row in table.rows.iter().take(5) {
//!     println!("{} p={:.2e}", row.term, row.p_value);
//! }
//! # Ok::<(), enrichr::EnrichrError>(())
//! ```

mod client;
mod error;
mod multipart;
mod types;

pub use client::{EnrichrClient, EnrichrSite};
pub use error::EnrichrError;
pub use types::{
    AddListResponse, DatasetStatistics, EnrichmentRow, EnrichmentTable, LibraryCategory,
    LibraryStatistic,
};
