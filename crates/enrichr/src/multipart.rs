pub(crate) fn encode_multipart(boundary: &str, fields: &[(&str, &str)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, value) in fields {
        out.extend_from_slice(b"--");
        out.extend_from_slice(boundary.as_bytes());
        out.extend_from_slice(b"\r\nContent-Disposition: form-data; name=\"");
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b"\"\r\n\r\n");
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"--");
    out.extend_from_slice(boundary.as_bytes());
    out.extend_from_slice(b"--\r\n");
    out
}

pub(crate) fn random_boundary() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("----enrichrRustBoundary{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_roundtrip_shape() {
        let b = "----x";
        let body = encode_multipart(b, &[("list", "A\nB"), ("description", "d")]);
        let s = String::from_utf8(body).unwrap();
        assert!(s.contains("name=\"list\""));
        assert!(s.contains("A\nB"));
        assert!(s.contains("name=\"description\""));
        assert!(s.ends_with("--\r\n"));
    }
}
