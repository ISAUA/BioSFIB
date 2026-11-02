#!/usr/bin/env Rscript

# Convert the official Hu et al. ccRCC Seurat RDS into H5AD without filtering,
# normalization, feature selection, relabeling, or use of any SpaMV artifact.

args <- commandArgs(trailingOnly = TRUE)

arg_value <- function(flag, default = NULL) {
  index <- match(flag, args)
  if (is.na(index)) {
    return(default)
  }
  if (index == length(args)) {
    stop(sprintf("Missing value after %s", flag))
  }
  args[[index + 1]]
}

input_rds <- arg_value("--input")
output_h5ad <- arg_value("--output")
assay_name <- arg_value("--assay", "RNA")
overwrite <- "--overwrite" %in% args

if (is.null(input_rds) || is.null(output_h5ad)) {
  stop(
    paste(
      "Usage: Rscript convert_ccrcc_snrna_reference.R",
      "--input <snRNA.rds> --output <converted.h5ad>",
      "[--assay RNA] [--overwrite]"
    )
  )
}

required <- c("Seurat", "SeuratDisk")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop(sprintf("Missing R packages: %s", paste(missing, collapse = ", ")))
}
if (!file.exists(input_rds)) {
  stop(sprintf("Input RDS does not exist: %s", input_rds))
}
if (file.exists(output_h5ad) && !overwrite) {
  stop(sprintf("Output exists; pass --overwrite to replace it: %s", output_h5ad))
}

dir.create(dirname(output_h5ad), recursive = TRUE, showWarnings = FALSE)
message("Reading official ccRCC reference: ", normalizePath(input_rds))
reference <- readRDS(input_rds)
if (!inherits(reference, "Seurat")) {
  stop(sprintf("Expected a Seurat object, received: %s", paste(class(reference), collapse = ", ")))
}
if (!(assay_name %in% names(reference@assays))) {
  stop(sprintf("Assay '%s' is absent; available assays: %s", assay_name, paste(names(reference@assays), collapse = ", ")))
}

Seurat::DefaultAssay(reference) <- assay_name
message("Seurat object: ", ncol(reference), " cells x ", nrow(reference), " features")
message("Metadata columns: ", paste(colnames(reference[[]]), collapse = ", "))

h5seurat_path <- sub("\\.h5ad$", ".h5seurat", output_h5ad, ignore.case = TRUE)
if (identical(h5seurat_path, output_h5ad)) {
  h5seurat_path <- paste0(output_h5ad, ".h5seurat")
}

SeuratDisk::SaveH5Seurat(
  reference,
  filename = h5seurat_path,
  overwrite = TRUE
)
SeuratDisk::Convert(
  h5seurat_path,
  dest = "h5ad",
  assay = assay_name,
  overwrite = TRUE
)

if (!file.exists(output_h5ad)) {
  stop(sprintf("Conversion did not create the expected H5AD: %s", output_h5ad))
}
unlink(h5seurat_path)
message("Source MD5: ", unname(tools::md5sum(input_rds)))
message("Converted H5AD: ", normalizePath(output_h5ad))
