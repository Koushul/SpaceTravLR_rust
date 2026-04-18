use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LibraryCategory {
    pub category_id: i32,
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LibraryStatistic {
    pub gene_coverage: i32,
    pub genes_per_term: i32,
    pub library_name: String,
    pub link: String,
    pub num_terms: i32,
    pub appyter: String,
    pub category_id: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatasetStatistics {
    pub statistics: Vec<LibraryStatistic>,
    #[serde(default)]
    pub categories: Vec<LibraryCategory>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AddListResponse {
    pub short_id: String,
    pub user_list_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnrichmentRow {
    pub rank: i32,
    pub term: String,
    pub p_value: f64,
    pub odds_ratio: f64,
    pub combined_score: f64,
    pub genes: Vec<String>,
    pub adjusted_p_value: f64,
    pub old_p_value: f64,
    pub old_adjusted_p_value: f64,
}

#[derive(Debug, Clone)]
pub struct EnrichmentTable {
    pub library: String,
    pub rows: Vec<EnrichmentRow>,
}

impl EnrichmentTable {
    pub(crate) fn from_enrich_json(
        value: Value,
        expected_library: &str,
    ) -> Result<Self, super::EnrichrError> {
        let obj = value.as_object().ok_or_else(|| {
            super::EnrichrError::UnexpectedResponse("enrichment JSON was not an object".into())
        })?;

        let (library, rows_value) = if let Some(v) = obj.get(expected_library) {
            (expected_library.to_string(), v)
        } else if obj.len() == 1 {
            let (k, v) = obj.iter().next().expect("len checked");
            (k.clone(), v)
        } else {
            return Err(super::EnrichrError::UnexpectedResponse(format!(
                "enrichment JSON missing library {expected_library:?}; keys: {:?}",
                obj.keys().collect::<Vec<_>>()
            )));
        };

        let rows_json = rows_value.as_array().ok_or_else(|| {
            super::EnrichrError::UnexpectedResponse("enrichment rows were not an array".into())
        })?;

        let mut rows = Vec::with_capacity(rows_json.len());
        for row in rows_json {
            rows.push(parse_enrichment_row(row)?);
        }

        Ok(EnrichmentTable { library, rows })
    }
}

fn parse_enrichment_row(row: &Value) -> Result<EnrichmentRow, super::EnrichrError> {
    let arr = row.as_array().ok_or_else(|| {
        super::EnrichrError::UnexpectedResponse("enrichment row was not an array".into())
    })?;
    if arr.len() != 9 {
        return Err(super::EnrichrError::UnexpectedResponse(format!(
            "expected 9 fields per enrichment row, got {}",
            arr.len()
        )));
    }

    let genes = arr[5]
        .as_array()
        .ok_or_else(|| {
            super::EnrichrError::UnexpectedResponse("Genes field was not a JSON array".into())
        })?
        .iter()
        .map(|g| {
            g.as_str().map(str::to_string).ok_or_else(|| {
                super::EnrichrError::UnexpectedResponse("gene entry was not a string".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(EnrichmentRow {
        rank: json_as_i32(&arr[0])?,
        term: json_as_string(&arr[1])?,
        p_value: json_as_f64(&arr[2])?,
        odds_ratio: json_as_f64(&arr[3])?,
        combined_score: json_as_f64(&arr[4])?,
        genes,
        adjusted_p_value: json_as_f64(&arr[6])?,
        old_p_value: json_as_f64(&arr[7])?,
        old_adjusted_p_value: json_as_f64(&arr[8])?,
    })
}

fn json_as_string(v: &Value) -> Result<String, super::EnrichrError> {
    v.as_str()
        .map(str::to_string)
        .ok_or_else(|| super::EnrichrError::UnexpectedResponse("expected string".into()))
}

fn json_as_f64(v: &Value) -> Result<f64, super::EnrichrError> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_u64().map(|u| u as f64))
        .ok_or_else(|| super::EnrichrError::UnexpectedResponse("expected number".into()))
}

fn json_as_i32(v: &Value) -> Result<i32, super::EnrichrError> {
    v.as_i64()
        .map(|i| i as i32)
        .or_else(|| v.as_u64().map(|u| u as i32))
        .ok_or_else(|| super::EnrichrError::UnexpectedResponse("expected integer".into()))
}
