use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::EnrichrError;
use crate::multipart::{encode_multipart, random_boundary};
use crate::types::{AddListResponse, DatasetStatistics, EnrichmentTable};

const DEFAULT_USER_AGENT: &str = concat!(
    "enrichr-rs/",
    env!("CARGO_PKG_VERSION"),
    " (SpaceTravLR; https://github.com/Koushul/SpaceTravLR_rust)"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnrichrSite {
    HumanMouse,
    Fly,
    Yeast,
    Worm,
    Fish,
}

impl EnrichrSite {
    pub fn path_segment(self) -> &'static str {
        match self {
            EnrichrSite::HumanMouse => "Enrichr",
            EnrichrSite::Fly => "FlyEnrichr",
            EnrichrSite::Yeast => "YeastEnrichr",
            EnrichrSite::Worm => "WormEnrichr",
            EnrichrSite::Fish => "FishEnrichr",
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnrichrClient {
    base_url: String,
    user_agent: String,
    agent: ureq::Agent,
}

impl EnrichrClient {
    pub fn new(site: EnrichrSite) -> Self {
        Self::with_base_url(format!("https://maayanlab.cloud/{}", site.path_segment()))
    }

    pub fn with_base_url(base_url: String) -> Self {
        let agent = ureq::Agent::new();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            user_agent: DEFAULT_USER_AGENT.to_string(),
            agent,
        }
    }

    pub fn set_user_agent(&mut self, ua: impl Into<String>) {
        self.user_agent = ua.into();
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn dataset_statistics(&self) -> Result<DatasetStatistics, EnrichrError> {
        let url = format!("{}/datasetStatistics", self.base_url);
        self.get_json(&url)
    }

    pub fn library_names(&self) -> Result<Vec<String>, EnrichrError> {
        let stats = self.dataset_statistics()?;
        Ok(stats
            .statistics
            .into_iter()
            .map(|s| s.library_name)
            .collect())
    }

    pub fn add_list(
        &self,
        genes: &[impl AsRef<str>],
        description: Option<&str>,
    ) -> Result<AddListResponse, EnrichrError> {
        let list = genes
            .iter()
            .map(|g| g.as_ref().trim())
            .filter(|g| !g.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        if list.is_empty() {
            return Err(EnrichrError::UnexpectedResponse(
                "gene list is empty after trimming".into(),
            ));
        }
        let desc = description.unwrap_or("Gene list");
        let boundary = random_boundary();
        let body = encode_multipart(&boundary, &[("list", list.as_str()), ("description", desc)]);
        let url = format!("{}/addList", self.base_url);
        let ct = format!("multipart/form-data; boundary={boundary}");
        let resp = self
            .agent
            .post(&url)
            .set("User-Agent", &self.user_agent)
            .set("Content-Type", &ct)
            .send_bytes(&body)?;

        self.json_from_response(resp)
    }

    pub fn enrich(
        &self,
        user_list_id: u64,
        gene_set_library: &str,
    ) -> Result<EnrichmentTable, EnrichrError> {
        let url = format!(
            "{}/enrich?userListId={}&backgroundType={}",
            self.base_url,
            user_list_id,
            urlencoding::encode(gene_set_library)
        );
        let value: Value = self.get_json(&url)?;
        EnrichmentTable::from_enrich_json(value, gene_set_library)
    }

    pub fn enrich_libraries(
        &self,
        user_list_id: u64,
        libraries: &[impl AsRef<str>],
    ) -> Result<Vec<EnrichmentTable>, EnrichrError> {
        libraries
            .iter()
            .map(|lib| self.enrich(user_list_id, lib.as_ref()))
            .collect()
    }

    pub fn view_list(&self, user_list_id: u64) -> Result<Value, EnrichrError> {
        let url = format!("{}/view?userListId={user_list_id}", self.base_url);
        self.get_json(&url)
    }

    pub fn export_tsv(
        &self,
        user_list_id: u64,
        gene_set_library: &str,
    ) -> Result<String, EnrichrError> {
        let url = format!(
            "{}/export?userListId={}&backgroundType={}&filename=export",
            self.base_url,
            user_list_id,
            urlencoding::encode(gene_set_library)
        );
        self.get_text(&url)
    }

    pub fn gene_set_library_text(&self, library_name: &str) -> Result<String, EnrichrError> {
        let url = format!(
            "{}/geneSetLibrary?mode=text&libraryName={}",
            self.base_url,
            urlencoding::encode(library_name)
        );
        self.get_text(&url)
    }

    pub fn find_terms_by_gene(
        &self,
        gene: &str,
        include_json: bool,
        include_setup: bool,
    ) -> Result<Value, EnrichrError> {
        let url = format!(
            "{}/genemap?gene={}&json={}&setup={}",
            self.base_url,
            urlencoding::encode(gene),
            if include_json { "true" } else { "false" },
            if include_setup { "true" } else { "false" },
        );
        self.get_json(&url)
    }

    fn get_json<T: DeserializeOwned>(&self, url: &str) -> Result<T, EnrichrError> {
        let resp = self
            .agent
            .get(url)
            .set("User-Agent", &self.user_agent)
            .call()?;
        self.json_from_response(resp)
    }

    fn get_text(&self, url: &str) -> Result<String, EnrichrError> {
        let resp = self
            .agent
            .get(url)
            .set("User-Agent", &self.user_agent)
            .call()?;
        self.text_from_response(resp)
    }

    fn json_from_response<T: DeserializeOwned>(
        &self,
        resp: ureq::Response,
    ) -> Result<T, EnrichrError> {
        let status = resp.status();
        if status >= 400 {
            let body_preview = resp.into_string().unwrap_or_default();
            let body_preview = truncate_body(&body_preview);
            return Err(EnrichrError::BadStatus {
                status,
                body_preview,
            });
        }
        Ok(resp.into_json()?)
    }

    fn text_from_response(&self, resp: ureq::Response) -> Result<String, EnrichrError> {
        let status = resp.status();
        if status >= 400 {
            let body_preview = resp.into_string().unwrap_or_default();
            let body_preview = truncate_body(&body_preview);
            return Err(EnrichrError::BadStatus {
                status,
                body_preview,
            });
        }
        Ok(resp.into_string()?)
    }
}

fn truncate_body(s: &str) -> String {
    const MAX: usize = 512;
    if s.len() <= MAX {
        s.to_string()
    } else {
        format!("{}…", &s[..MAX])
    }
}
