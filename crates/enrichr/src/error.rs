use thiserror::Error;

#[derive(Debug, Error)]
pub enum EnrichrError {
    #[error("I/O error reading response: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP request failed: {0}")]
    Http(Box<ureq::Error>),

    #[error("Enrichr returned HTTP {status}: {body_preview}")]
    BadStatus { status: u16, body_preview: String },

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("unexpected Enrichr response: {0}")]
    UnexpectedResponse(String),
}

impl From<ureq::Error> for EnrichrError {
    fn from(value: ureq::Error) -> Self {
        Self::Http(Box::new(value))
    }
}
