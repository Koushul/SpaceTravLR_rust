//! Rewrite `var` in place (read → set_var) so datasets use current anndata write defaults.
//! After patching anndata to default to gzip, this fixes existing `.h5ad` files that used Blosc.
//!
//! ```text
//! cargo run --example repack_h5ad_var -- /path/to/file.h5ad
//! ```

use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("usage: repack_h5ad_var <file.h5ad>"))?,
    );
    let adata = AnnData::<H5>::open(H5::open_rw(&path)?)?;
    let v = adata.read_var()?;
    adata.set_var(v)?;
    adata.close()?;
    println!("repacked var: {}", path.display());
    Ok(())
}
