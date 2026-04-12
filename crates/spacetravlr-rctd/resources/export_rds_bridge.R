# Bridge for Rust RCTD CLI: .rds → matrix.bin + sidecars (same pipeline as .h5ad).
#
# Supported:
#   Spatial: spacexr::SpatialRNA, or Seurat (default assay counts, e.g. VisiumHD "Spatial")
#   Reference (ref_sc): spacexr::Reference, or Seurat (RNA counts + metadata cell-type column)
#   Reference (ref_profiles): numeric matrix / dgCMatrix, types × genes
#
# Optional subsampling (large objects): set before invoking the Rust CLI:
#   RCTD_RDS_MAX_PIXELS       — max spatial barcodes (random seed 42)
#   RCTD_RDS_MAX_REFERENCE_CELLS — max reference cells for ref_sc (seed 42)
# Seurat reference cell-type column (ref_sc only):
#   RCTD_RDS_SEURAT_CELLTYPE_COL — default "Lineage"
#
# Usage: Rscript export_rds_bridge.R <in.rds> <out_dir> <spatial|ref_sc|ref_profiles>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("usage: Rscript export_rds_bridge.R <in.rds> <out_dir> <spatial|ref_sc|ref_profiles>")
}
inpath <- args[[1]]
out_dir <- args[[2]]
bridge_mode <- args[[3]]

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

obj <- readRDS(inpath)

max_pixels_n <- suppressWarnings(as.integer(Sys.getenv("RCTD_RDS_MAX_PIXELS", unset = "0")))
if (length(max_pixels_n) != 1L || is.na(max_pixels_n)) max_pixels_n <- 0L
max_ref_cells_n <- suppressWarnings(as.integer(Sys.getenv("RCTD_RDS_MAX_REFERENCE_CELLS", unset = "0")))
if (length(max_ref_cells_n) != 1L || is.na(max_ref_cells_n)) max_ref_cells_n <- 0L

subsample_mat_cols <- function(cnt, max_n, seed = 42L) {
  if (max_n <= 0L || ncol(cnt) <= max_n) return(cnt)
  set.seed(seed)
  keep <- sample(ncol(cnt), max_n)
  cnt[, keep, drop = FALSE]
}

write_manifest <- function(kind, nrows, ncols, gene_names, obs_names, cell_types = NULL) {
  meta <- sprintf(
    '{"format":"rctd_rds_bridge_v1","kind":"%s","nrows":%d,"ncols":%d}\n',
    kind, as.integer(nrows), as.integer(ncols)
  )
  writeLines(meta, file.path(out_dir, "meta.json"))
  writeLines(gene_names, file.path(out_dir, "genes.txt"))
  writeLines(obs_names, file.path(out_dir, "obs.txt"))
  if (!is.null(cell_types)) {
    writeLines(cell_types, file.path(out_dir, "cell_types.txt"))
  }
}

write_matrix_le <- function(m) {
  con <- file(file.path(out_dir, "matrix.bin"), "wb")
  on.exit(close(con), add = TRUE)
  writeBin(as.double(t(m)), con, size = 8, endian = "little")
}

if (bridge_mode == "spatial") {
  if (inherits(obj, "SpatialRNA")) {
    cnt <- obj@counts
  } else if (inherits(obj, "Seurat")) {
    if (!requireNamespace("Seurat", quietly = TRUE)) {
      stop("install.packages(\"Seurat\") to load spatial Seurat .rds")
    }
    assay <- Seurat::DefaultAssay(obj)
    cnt <- Seurat::GetAssayData(obj, assay = assay, layer = "counts")
  } else {
    stop("spatial .rds must be spacexr::SpatialRNA or Seurat (Visium/VisiumHD counts)")
  }
  cnt <- subsample_mat_cols(cnt, max_pixels_n)
  cnt <- as.matrix(cnt)
  gene_names <- rownames(cnt)
  obs_names <- colnames(cnt)
  if (is.null(gene_names) || is.null(obs_names)) {
    stop("counts must have rownames (genes) and colnames (spots/cells)")
  }
  m <- t(cnt)
  write_manifest("spatial_rna", nrow(m), ncol(m), gene_names, obs_names, NULL)
  write_matrix_le(m)
} else if (bridge_mode == "ref_sc") {
  if (inherits(obj, "Reference")) {
    cnt <- obj@counts
    cnt <- subsample_mat_cols(cnt, max_ref_cells_n)
    cnt <- as.matrix(cnt)
    gene_names <- rownames(cnt)
    obs_names <- colnames(cnt)
    if (is.null(gene_names) || is.null(obs_names)) {
      stop("Reference @counts must have rownames (genes) and colnames (cells)")
    }
    ct <- obj@cell_types
    if (is.null(names(ct))) {
      stop("Reference @cell_types must be a named factor (cell barcodes)")
    }
    cell_types <- as.character(ct[obs_names])
    if (anyNA(cell_types)) {
      stop("some colnames(counts) missing from names(cell_types)")
    }
    m <- t(cnt)
  } else if (inherits(obj, "Seurat")) {
    if (!requireNamespace("Seurat", quietly = TRUE)) {
      stop("install.packages(\"Seurat\") to load reference Seurat .rds")
    }
    meta_col <- Sys.getenv("RCTD_RDS_SEURAT_CELLTYPE_COL", unset = "Lineage")
    if (!meta_col %in% colnames(obj@meta.data)) {
      stop(
        "metadata column \"", meta_col, "\" not found in Seurat object; set env RCTD_RDS_SEURAT_CELLTYPE_COL (columns: ",
        paste(colnames(obj@meta.data), collapse = ", "), ")"
      )
    }
    assay <- Seurat::DefaultAssay(obj)
    cnt <- Seurat::GetAssayData(obj, assay = assay, layer = "counts")
    cnt <- subsample_mat_cols(cnt, max_ref_cells_n)
    cnt <- as.matrix(cnt)
    gene_names <- rownames(cnt)
    obs_names <- colnames(cnt)
    meta <- obj@meta.data[obs_names, , drop = FALSE]
    cell_types <- as.character(meta[[meta_col]])
    if (anyNA(cell_types)) {
      stop("NAs in cell type column ", meta_col, " for some cells")
    }
    m <- t(cnt)
  } else {
    stop("reference ref_sc .rds must be spacexr::Reference or Seurat")
  }
  write_manifest("reference_sc", nrow(m), ncol(m), gene_names, obs_names, cell_types)
  write_matrix_le(m)
} else if (bridge_mode == "ref_profiles") {
  if (inherits(obj, "SpatialRNA") || inherits(obj, "Reference")) {
    stop("for Reference/SpatialRNA use ref_sc or spatial mode, not ref_profiles")
  }
  if (inherits(obj, "dgCMatrix")) {
    cnt <- as.matrix(obj)
  } else if (inherits(obj, "matrix")) {
    cnt <- obj
  } else {
    cnt <- tryCatch(
      as.matrix(obj),
      error = function(e) stop("ref_profiles .rds must be a numeric matrix (types × genes), dgCMatrix, or coercible matrix")
    )
  }
  gene_names <- colnames(cnt)
  obs_names <- rownames(cnt)
  if (is.null(gene_names) || is.null(obs_names)) {
    stop("reference profile matrix must have rownames (cell types) and colnames (genes)")
  }
  m <- cnt
  write_manifest("reference_profiles", nrow(m), ncol(m), gene_names, obs_names, NULL)
  write_matrix_le(m)
} else {
  stop("bridge_mode must be spatial, ref_sc, or ref_profiles")
}

invisible(TRUE)
