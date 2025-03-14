import argparse 
import os 
import sys 
sys.path.append("src/python/")
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

import scanpy as sc
import anndata as ann
import numpy as np

from utils import faiss_kmeans # this should work because we are using the integrate environment

def none_or_str(value):
    if value == 'None':
        return None
    return value

def main(h5ad_file, save_loc):
    # Load h5ad file
    integrated_concat = sc.read_h5ad(h5ad_file, as_sparse = "raw/X")
    
    # Perform kmeans clustering on integrated data 
    # Define method subsets and iterate over them until the same number of k clusters is found
    k = 10
    k_initial = k # Integers are immutable 
    methods = ["uce", "scgpt", "geneformer"]
    method_kmeans_adatas = []
    i = 0
    while i < len(methods):
        # Create a copy of adata to avoid overwriting the original
        adata_copy = integrated_concat.copy()
        
        # Define method subset
        adata_subset = adata_copy[adata_copy.obs["integration_method"] == methods[i]]
        
        # Perform HVG selection on raw (unnormalized, unlogged) data
        adata_subset.X = adata_subset.layers["raw"]
        sc.pp.normalize_total(
            adata_subset,
            target_sum = 1e4
        )
        sc.pp.log1p(adata_subset)
        sc.pp.highly_variable_genes(
            adata_subset,
            n_top_genes = 2500,
            flavor = "seurat"
        )
        
        # Perform faiss kmeans clustering
        adata_subset, k_method = faiss_kmeans(adata_subset, k)
        
        # Test concordance of k values and either append or reset
        if k_method != k:
            k = k_method
            i = 0 
            method_kmeans_adatas.clear()
            continue 
        else:
            i += 1
            method_kmeans_adatas.append(adata_subset)
    
    # Append kmeans cluster info to integrated data
    for method, method_kmeans_adata in zip(methods, method_kmeans_adatas):
        method_kmeans_clusters = method_kmeans_adata.obs["kmeans_faiss"].__array__().astype('str')
        integrated_concat.obs.loc[
            integrated_concat.obs["integration_method"] == method,
            "kmeans_faiss"
        ] = method_kmeans_clusters
    
    # Append information about kmeans faiss clusters to .uns of adata_concat
    integrated_concat.uns["kmeans_stats"] = {
        "kmeans_initial_k": k_initial,
        "kmeans_final_k": k
    }
        
    # Save integrated h5ad object
    integrated_concat.write_h5ad(
        filename = save_loc,
        compression = "gzip"
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Input and output files for kmeans clustering of data integrated by Helical"
    )
    parser.add_argument(
        "--inputfile",
        type = str,
        help = "Input h5ad file from rule integrate_helical"
    )
    parser.add_argument(
        "--outfile",
        type = str,
        help = "Filepath for saving output from kmeans clustering of data integrated by Helical"
    )
    args = parser.parse_args()
    main(
        h5ad_file= args.inputfile,
        save_loc = args.outfile
    )
