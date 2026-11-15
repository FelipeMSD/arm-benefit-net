# Step 9a: one-off export of IMvigor210 (Mariathasan et al., Nature 2018) to CSV.
#
# The data ships inside the R package IMvigor210CoreBiologies as a DESeq
# "CountDataSet" (a Biobase eSet). Biobase is not built for Windows on conda,
# so we read the S4 slots directly with attr(): this works in plain R even
# when the class definition is not installed.
#
# Prerequisites (one-off, separate env so the Python env stays clean):
#   conda create -n rextract -c conda-forge r-base=4.3
#   curl -O http://research-pub.gene.com/IMvigor210CoreBiologies/packageVersions/IMvigor210CoreBiologies_1.0.1.tar.gz
#   tar -xzf IMvigor210CoreBiologies_1.0.1.tar.gz -C data_raw/imvigor210
# Usage (repo root):  Rscript scripts/03_export_imvigor210.R

src <- "data_raw/imvigor210/IMvigor210CoreBiologies/data"
out <- "data_processed/imvigor210"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

load(file.path(src, "cds.RData"))            # object: cds (genes x patients, raw counts)
slot   <- function(obj, name) attr(obj, name, exact = TRUE)
counts <- get("counts", envir = slot(cds, "assayData"))   # assayData is an environment
pheno  <- slot(slot(cds, "phenoData"), "data")            # AnnotatedDataFrame@data
feat   <- slot(slot(cds, "featureData"), "data")
cat("counts:", dim(counts), "| pheno:", dim(pheno), "| features:", dim(feat), "\n")

gz <- function(df, name, rn = TRUE) {
  con <- gzfile(file.path(out, name), "w")
  write.csv(df, con, row.names = rn)
  close(con)
}
gz(counts, "counts.csv.gz")
gz(pheno, "pheno.csv.gz")
gz(feat, "features.csv.gz")

load(file.path(src, "human_gene_signatures.RData"))   # list of named gene-ID vectors
sig <- do.call(rbind, lapply(names(human_gene_signatures), function(n)
  data.frame(signature = n, gene = as.character(human_gene_signatures[[n]]))))
gz(sig, "signatures.csv.gz", rn = FALSE)
cat("written to", out, "\n")
