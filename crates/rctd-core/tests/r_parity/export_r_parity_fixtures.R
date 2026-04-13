#!/usr/bin/env Rscript
#
# Generate parity fixtures for Rust vs R (spacexr) RCTD comparison.
#
# Exports intermediate data from spacexr's RCTD pipeline so that Rust can
# run the same algorithm steps on identical inputs and compare outputs.
#
# Requires: spacexr (>= 2.0), Matrix
# Install:  install.packages("remotes"); remotes::install_github("dmcable/spacexr")
#
# Usage: Rscript export_r_parity_fixtures.R [out_dir]
#
# The synthetic data is deterministic (seed 42) with:
#   n_genes=48, n_types=5, n_pixels=28, n_cells_per_type=50.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) {
  out_dir <- file.path(getwd(), "fixtures")
} else {
  out_dir <- args[[1]]
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(spacexr)
  library(Matrix)
})

set.seed(42)
n_genes  <- 48L
n_types  <- 5L
n_pixels <- 28L

gene_names <- paste0("gene", seq_len(n_genes))
type_names <- paste0("t", seq_len(n_types) - 1L)
pixel_names <- paste0("px", seq_len(n_pixels))

profiles <- matrix(0, nrow = n_genes, ncol = n_types,
                   dimnames = list(gene_names, type_names))
for (k in seq_len(n_types)) {
  profiles[, k] <- exp(runif(n_genes)) * 1e-3
}
markers_per_type <- n_genes %/% n_types
for (k in seq_len(n_types)) {
  start_g <- (k - 1L) * markers_per_type + 1L
  end_g   <- min(k * markers_per_type, n_genes)
  profiles[start_g:end_g, k] <- profiles[start_g:end_g, k] * 10
}
for (k in seq_len(n_types)) {
  profiles[, k] <- profiles[, k] / sum(profiles[, k])
}

counts <- matrix(0, nrow = n_genes, ncol = n_pixels,
                 dimnames = list(gene_names, pixel_names))
numi_vec <- numeric(n_pixels)
for (i in seq_len(n_pixels)) {
  numi_vec[i] <- sample(200:2999, 1)
  mix <- runif(n_types)
  mix <- mix / sum(mix)
  for (g in seq_len(n_genes)) {
    counts[g, i] <- floor(sum(profiles[g, ] * mix) * numi_vec[i])
  }
}

coords <- data.frame(
  x = runif(n_pixels),
  y = runif(n_pixels),
  row.names = pixel_names
)

n_cells_per_type <- 50L
n_cells <- n_types * n_cells_per_type
cell_names <- paste0("cell", seq_len(n_cells))
cell_types_vec <- rep(type_names, each = n_cells_per_type)
names(cell_types_vec) <- cell_names
cell_types_factor <- factor(cell_types_vec, levels = type_names)
names(cell_types_factor) <- cell_names

ref_counts <- matrix(0, nrow = n_genes, ncol = n_cells,
                     dimnames = list(gene_names, cell_names))
for (ci in seq_len(n_cells)) {
  type_idx <- ((ci - 1L) %/% n_cells_per_type) + 1L
  total_umi <- sample(500:2000, 1)
  ref_counts[, ci] <- as.integer(rmultinom(1, total_umi, profiles[, type_idx]))
}

spatial_rna <- SpatialRNA(
  coords = coords,
  counts = as(counts, "dgCMatrix"),
  nUMI = colSums(counts)
)

reference <- Reference(
  counts = as(ref_counts, "dgCMatrix"),
  cell_types = cell_types_factor,
  nUMI = colSums(ref_counts)
)

# ---------------------------------------------------------------------------
# Run RCTD in all three modes
# ---------------------------------------------------------------------------
message("=== Running RCTD full mode ===")
rctd_full <- create.RCTD(spatial_rna, reference, max_cores = 1, CELL_MIN_INSTANCE = 5)
rctd_full <- run.RCTD(rctd_full, doublet_mode = "full")

message("\n=== Running RCTD doublet mode ===")
rctd_doublet <- create.RCTD(spatial_rna, reference, max_cores = 1, CELL_MIN_INSTANCE = 5)
rctd_doublet <- run.RCTD(rctd_doublet, doublet_mode = "doublet")

message("\n=== Running RCTD multi mode ===")
rctd_multi <- create.RCTD(spatial_rna, reference, max_cores = 1, CELL_MIN_INSTANCE = 5)
rctd_multi <- run.RCTD(rctd_multi, doublet_mode = "multi")

# ---------------------------------------------------------------------------
# Extract intermediate data from the RCTD object
# ---------------------------------------------------------------------------

# Gene list used for regression (DE genes)
gene_list_reg <- rctd_full@internal_vars$gene_list_reg

# Normalized (renormalized) cell type profiles after fitBulk + platform normalization
# These are G x K matrices, columns = cell types
renorm_profiles <- as.matrix(rctd_full@cell_type_info$renorm[[1]])
renorm_type_names <- rctd_full@cell_type_info$renorm[[2]]

# Subset to the DE gene list (this is what RCTD actually deconvolves on)
renorm_profiles_reg <- renorm_profiles[gene_list_reg, , drop = FALSE]

# Column-normalize (each type sums to 1) — this is what Rust calls "norm_profiles"
norm_profiles <- renorm_profiles_reg
for (k in seq_len(ncol(norm_profiles))) {
  s <- sum(norm_profiles[, k])
  if (s > 0) norm_profiles[, k] <- norm_profiles[, k] / s
}

# Spatial counts subset to DE genes, transposed to pixels x genes
spatial_counts_reg <- as.matrix(rctd_full@spatialRNA@counts[gene_list_reg, ])
spatial_counts_pxg <- t(spatial_counts_reg)  # pixels x genes

# nUMI per pixel
numi_final <- rctd_full@spatialRNA@nUMI

# Q-matrix and X_vals from choose_sigma_c
q_mat <- rctd_full@internal_vars$Q_mat
x_vals <- rctd_full@internal_vars$X_vals
sigma_val <- rctd_full@internal_vars$sigma

message("sigma = ", sigma_val)
message("Q_mat dim: ", nrow(q_mat), " x ", ncol(q_mat))
message("X_vals length: ", length(x_vals))
message("norm_profiles dim: ", nrow(norm_profiles), " x ", ncol(norm_profiles))
message("spatial_counts dim: ", nrow(spatial_counts_pxg), " x ", ncol(spatial_counts_pxg))
message("Gene list: ", paste(gene_list_reg, collapse = ", "))

# ---------------------------------------------------------------------------
# Extract results
# ---------------------------------------------------------------------------

# Full mode: weights matrix (pixels x types)
full_weights <- as.matrix(rctd_full@results$weights)

# Doublet mode results
doublet_weights <- as.matrix(rctd_doublet@results$weights)
doublet_results_df <- rctd_doublet@results$results_df
d_spot_class <- as.character(doublet_results_df$spot_class)
d_first_type <- as.character(doublet_results_df$first_type)
d_second_type <- as.character(doublet_results_df$second_type)

# Multi mode results — stored as a list of per-pixel results by process_beads_multi
multi_results_list <- rctd_multi@results
n_pix_m <- length(multi_results_list)
multi_all_weights <- matrix(0, nrow = n_pix_m, ncol = n_types)
colnames(multi_all_weights) <- renorm_type_names
for (i in seq_len(n_pix_m)) {
  w <- multi_results_list[[i]]$all_weights
  multi_all_weights[i, names(w)] <- w
}

# ---------------------------------------------------------------------------
# Write binary fixtures
# ---------------------------------------------------------------------------

write_matrix_f64 <- function(m, filename) {
  con <- file(file.path(out_dir, filename), "wb")
  on.exit(close(con), add = TRUE)
  writeBin(as.double(t(m)), con, size = 8, endian = "little")
}

write_vector_f64 <- function(v, filename) {
  con <- file(file.path(out_dir, filename), "wb")
  on.exit(close(con), add = TRUE)
  writeBin(as.double(v), con, size = 8, endian = "little")
}

write_lines_file <- function(lines, filename) {
  writeLines(lines, file.path(out_dir, filename))
}

# Inputs
write_matrix_f64(spatial_counts_pxg, "spatial_counts.bin")
write_vector_f64(as.double(numi_final), "numi.bin")
write_matrix_f64(norm_profiles, "norm_profiles.bin")

# Q-matrix
write_matrix_f64(q_mat, "q_mat.bin")
write_vector_f64(x_vals, "x_vals.bin")

# Full mode output
write_matrix_f64(full_weights, "r_full_weights.bin")

# Doublet mode output
write_matrix_f64(doublet_weights, "r_doublet_weights_full.bin")
write_lines_file(d_spot_class, "r_doublet_spot_class.txt")
write_lines_file(d_first_type, "r_doublet_first_type.txt")
write_lines_file(d_second_type, "r_doublet_second_type.txt")

# Multi mode output
write_matrix_f64(multi_all_weights, "r_multi_weights_full.bin")

# Metadata
write_lines_file(gene_list_reg, "gene_names.txt")
write_lines_file(renorm_type_names, "type_names.txt")
write_lines_file(names(numi_final), "pixel_names.txt")

n_pix <- nrow(spatial_counts_pxg)
n_gen <- ncol(spatial_counts_pxg)
n_typ <- ncol(norm_profiles)

meta_json <- sprintf(
  paste0(
    '{"n_pixels":%d,"n_genes":%d,"n_types":%d,',
    '"q_nrows":%d,"q_ncols":%d,"n_xvals":%d,',
    '"sigma":%d}'
  ),
  n_pix, n_gen, n_typ,
  nrow(q_mat), ncol(q_mat), length(x_vals),
  as.integer(sigma_val * 100)
)
writeLines(meta_json, file.path(out_dir, "meta.json"))

# Also write a CSV of full weights for human inspection
write.csv(full_weights, file.path(out_dir, "r_full_weights.csv"))

message("\nWrote R parity fixtures to ", out_dir)
message("  n_pixels=", n_pix, " n_genes=", n_gen, " n_types=", n_typ)
message("  Q_mat: ", nrow(q_mat), " x ", ncol(q_mat))
message("  sigma=", sigma_val, " (stored as ", as.integer(sigma_val * 100), ")")
message("Done.")
