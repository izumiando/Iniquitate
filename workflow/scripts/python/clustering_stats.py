import argparse 
import os 
import sys 

import numpy as np
import pandas as pd
import anndata as ann
import scanpy as sc
from sklearn import metrics

def main(h5ad_loc, save_loc, dataset_name, rep):
    # Load h5ad file 
    adata = sc.read_h5ad(h5ad_loc)
    
    # Extract summary statistics from h5ad file
    num_batches_ds = adata.uns["downsampling_stats"]["num_batches"]
    num_celltypes_ds = adata.uns["downsampling_stats"]["num_celltypes_downsampled"]
    prop_ds = adata.uns["downsampling_stats"]["proportion_downsampled"]
    
    # Subset h5ad based on batch-correction method used
    adata_method_sub = []
    methods = ["harmony", "scvi", "bbknn", "scanorama", "seurat", "liger"]
    for method in methods:
        adata_sub = adata[adata.obs["integration_method"] == method]
        adata_method_sub.append(
            adata_sub
        )
        
    # Get ARI, NMI, Homogeneity, Completeness values for each batch-correction method
    # and batch and celltype subsets
    celltype_aris = []
    celltype_amis = []
    celltype_homs = []
    celltype_comps = []
    batch_aris = []
    batch_amis = []
    batch_homs = []
    batch_comps = []
    for adata_sub in adata_method_sub:
        celltype_aris.append(
            metrics.adjusted_rand_score(
                adata_sub.obs["celltype"].__array__(),
                adata_sub.obs["leiden"].__array__()
            )
        )
        celltype_amis.append(
            metrics.adjusted_mutual_info_score(
                adata_sub.obs["celltype"].__array__(),
                adata_sub.obs["leiden"].__array__()
            )
        )
        celltype_homs.append(
            metrics.homogeneity_score(
                adata_sub.obs["celltype"].__array__(),
                adata_sub.obs["leiden"].__array__()
            )
        )
        celltype_comps.append(
            metrics.completeness_score(
                adata_sub.obs["celltype"].__array__(),
                adata_sub.obs["leiden"].__array__()                
            )
        )
        batch_aris.append(
            1 - metrics.adjusted_rand_score(
                adata_sub.obs["batch"].__array__(),
                adata_sub.obs["leiden"].__array__()
            )
        )
        batch_amis.append(
            1 - metrics.adjusted_mutual_info_score(
                adata_sub.obs["batch"].__array__(),
                adata_sub.obs["leiden"].__array__()
            )
        )
        batch_homs.append(
            1 - metrics.homogeneity_score(
                adata_sub.obs["batch"].__array__(),
                adata_sub.obs["leiden"].__array__()
            )
        )
        batch_comps.append(
            1 - metrics.completeness_score(
                adata_sub.obs["batch"].__array__(),
                adata_sub.obs["leiden"].__array__()
            )
        )
    
    # Get number of clusters per method 
    cluster_nums = []
    for adata_sub in adata_method_sub:
        cluster_nums.append(
            len(np.unique(adata_sub.obs["leiden"].__array__()))
        )
    
    # Get number of cells per method 
    cell_nums = []
    for adata_sub in adata_method_sub:
        cell_nums.append(
            adata_sub.n_obs
        )

    # Create summary dataframe for clustering statistics
    cluster_summary_df = pd.DataFrame({
        "Dataset": dataset_name,
        "Batches downsampled": num_batches_ds,
        "Number of celltypes downsampled": num_celltypes_ds,
        "Proportion downsampled": prop_ds,
        "Replicate": rep,
        "Method": methods,
        "Cluster number": cluster_nums,
        "Cell number": cell_nums,
        "Celltype ARI": celltype_aris,
        "Celltype AMI": celltype_amis,
        "Celltype Homogeneity": celltype_homs,
        "Celltype Completeness": celltype_comps,
        "Batch ARI": batch_aris,
        "Batch AMI": batch_amis,
        "Batch Homogeneity": batch_homs,
        "Batch Completeness": batch_comps
    })
    
    # Save clustering summary dataframe to tsv
    cluster_summary_df.to_csv(
        save_loc,
        index=False,
        sep="\t"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Input and output files for clustering results summary"
    )
    parser.add_argument(
        "--infile",
        type = str,
        help = "Path of integrated h5ad file"
    )
    parser.add_argument(
        "--outfile",
        type = str,
        help = "Filepath for saving clustering results of integrated h5ad file"
    )
    parser.add_argument(
        "--dataset",
        type = str,
        help = "Name of dataset"
    )
    parser.add_argument(
        "--rep",
        type = int,
        help = "Repetition number"
    )
    args = parser.parse_args()
    main(
        h5ad_loc = args.infile,
        save_loc = args.outfile,
        dataset_name = args.dataset,
        rep = args.rep
    )