import argparse
import scanpy as sc
import anndata as ad

def main(integrated_baseline, integrated_helical_p2, outfile):
    print("Concatenating integrated results from baseline models and helical...")
    adata_baseline = sc.read_h5ad(integrated_baseline)
    adata_helical = sc.read_h5ad(integrated_helical_p2)
    adata_concat = ad.concat(
        [adata_baseline, adata_helical], 
        axis=0, 
        join="outer", 
        uns_merge="same", # to preserve .uns
        label="source", 
        keys=["baseline", "helical_p2"]
        )
    
    # Save h5ad object
    adata_concat.write_h5ad(
        filename = outfile,
        compression = "gzip"
    )
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Input and output files for h5ad concatenation"
    )
    parser.add_argument(
        "--baseline",
        type = str,
        help = "Path of directory containing scRNAseq h5ad file integrated using baseline models"
    )
    parser.add_argument(
        "--helical",
        type = str,
        help = "Path of directory containing scRNAseq h5ad file integrated using Helical (foundation models)"
    )
    parser.add_argument(
        "--outfile",
        type = str,
        help = "Filepath for saving concatenated output from scRNA-seq integration"
    )
    args = parser.parse_args()
    main(
        integrated_baseline = args.baseline,
        integrated_helical_p2 = args.helical,
        outfile = args.outfile
    )