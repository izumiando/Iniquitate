import argparse 
import os 
import sys 
import importlib.util
sys.path.append("src/python/")
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(script_dir, "../../src/python/utils"))

import scanpy as sc
import anndata as ann
import numpy as np

from utils_helical import IntegrationHelical

# Load sample.py and import downsample
sample_path = os.path.join(utils_dir, "sample.py")
spec_sample = importlib.util.spec_from_file_location("sample", sample_path)
sample = importlib.util.module_from_spec(spec_sample)
spec_sample.loader.exec_module(sample)

def none_or_str(value):
    if value == 'None':
        return None
    return value


def main(h5ad_dir, save_loc, ds_celltypes, ds_proportions, num_batches, seed):
    # Load h5ad files 
    files_list = os.listdir(h5ad_dir)
    adata_loaded = []
    for f in files_list:
        adata = sc.read_h5ad(os.path.join(h5ad_dir, f), as_sparse = "raw/X")
        adata.layers["raw"] = adata.X # Store raw counts
        adata.obs = adata.obs[["batch", "celltype"]] # Only store relevant columns
        if "gene" not in adata.var.columns:
            adata.var["gene"] = adata.var_names # Add gene names if not present
        adata.var = adata.var[["gene"]] # Only store relevant columns
        adata_loaded.append(adata)
    
    # Downsample loaded h5ad files based on params 
    if num_batches == 0:
        selected_celltypes_downsampled = "None" # Placeholder - not used
        batches_ds = "None" # Placeholder - not used
    else:
        # Initialize random number generator using the seed
        rng = np.random.default_rng(seed)
        
        # Select indices for downsampling
        selected_indices = rng.choice(
            len(adata_loaded), num_batches, replace = False
        )
        adata_selected = [adata_loaded[i] for i in selected_indices]
        adata_unselected = [adata_loaded[i] for i in range(len(adata_loaded)) if i not in selected_indices]
        
        # Downsample the same selected celltypes across all of the batches - this change will not affect
        # previous runs, as they all downsampled either 0 or only 1 celltype, in either 0 or 1 batches 
        # NOTE - this setup operates on the assumption that the celltypes are the same across all batches
        celltypes_all = np.unique(np.concatenate([adata.obs["celltype"].__array__() for adata in adata_selected]))
        rng.shuffle(celltypes_all)
        celltypes_selected = rng.choice(celltypes_all, ds_celltypes, replace = False)
        selected_celltypes_downsampled = np.array(celltypes_selected)
        adata_downsampled = []
        for adata in adata_selected:
            adata_ds, selected_celltypes_ds = sample.downsample(
                adata = adata, 
                num_celltypes = None,
                celltype_names = celltypes_selected,
                proportion = ds_proportions,
                random_state = seed,
            )
            adata_downsampled.append(adata_ds)
        adata_loaded = adata_unselected + adata_downsampled
        batches_ds = np.unique(np.concatenate([adata.obs["batch"].__array__() for adata in adata_downsampled]))

    # Store batch name separately for each anndata object
    for adata in adata_loaded:
        adata.obs["batch_name"] = adata.obs["batch"]

    # Concatenate files (assume data is raw counts)
    adata_concat = ann.AnnData.concatenate(*adata_loaded)
    adata_concat.obs_names = range(len(adata_concat.obs_names))
    adata_concat.obs_names_make_unique()
    adata_concat.obs["batch"] = adata_concat.obs["batch_name"]
    adata_concat.obs.drop("batch_name", axis = 1, inplace = True)
    
    # Create integration class instance 
    integration = IntegrationHelical(adata = adata_concat)
    
    # Integrate across subsets
    uce_integrated = integration.uce_integrate()
    scgpt_integrated = integration.scgpt_integrate()
    geneformer_integrated = integration.geneformer_integrate()
    
    # Add integration type to each subset and concatenate
    uce_integrated.obs["integration_method"] = "uce" 
    scgpt_integrated.obs["integration_method"] = "scgpt"
    geneformer_integrated.obs["integration_method"] = "geneformer"
    
    geneformer_integrated.var = uce_integrated.var # might cause errors because code below was moved
    scgpt_integrated.var = uce_integrated.var
    
    integrated_concat = ann.concat([
        uce_integrated,
        scgpt_integrated,
        geneformer_integrated],
        axis=0,         # concatenate along cells
        join="inner",   # or "outer" if you want all genes even if some are missing
        merge="same"    # assumes .var is the same across objects
    )
    integrated_concat.obs_names = range(len(integrated_concat.obs_names))
    integrated_concat.obs_names_make_unique()
    
    # Add placeholder in entire obs dataframe for kmeans clustering
    integrated_concat.obs["kmeans_faiss"] = np.zeros(len(integrated_concat.obs_names))

    # If downsampled celltypes and batches are of array length greater than one, combine them 
    if len(batches_ds) > 1:
        batches_ds = np.array(",".join(batches_ds))
    if len(selected_celltypes_downsampled) > 1:
        selected_celltypes_downsampled = np.array(",".join(selected_celltypes_downsampled))

    # Add data about downsampling to .uns of adata_concat
    if num_batches == 0:
        integrated_concat.uns["downsampling_stats"] = {
            "num_batches": 0,
            "num_celltypes_downsampled": ds_celltypes,
            "ds_batch_names": "None",
            "proportion_downsampled": ds_proportions,
            "downsampled_celltypes": "None"
        }
    else:
        integrated_concat.uns["downsampling_stats"] = {
            "num_batches": num_batches,
            "num_celltypes_downsampled": ds_celltypes,
            "ds_batch_names": "Placeholder due to h5py bug",
            "proportion_downsampled": ds_proportions,
            "downsampled_celltypes": selected_celltypes_downsampled
        }
    
    output_dir = os.path.dirname(save_loc)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save integrated h5ad object
    integrated_concat.write_h5ad(
        filename = save_loc,
        compression = "gzip"
    )
    
    print("done with this round of integrate_helical")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Input and output files for scRNA-seq integration"
    )
    parser.add_argument(
        "--filedir",
        type = str,
        help = "Path of directory containing scRNA-seq h5ad files"
    )
    parser.add_argument(
        "--ds_celltypes",
        type = int,
        help = "Number of celltypes to randomly downsample in given batch"
    )
    parser.add_argument(
        "--ds_proportions",
        type = float,
        help = "Proportion of downsampling per celltype in a given batch"
    )
    parser.add_argument(
        "--num_batches",
        type = int,
        help = "Number of batches to perform downsampling on"
    )
    parser.add_argument(
        "--outfile",
        type = str,
        help = "Filepath for saving output from scRNA-seq integration"
    )
    parser.add_argument(
        "--seed",
        type = int,
        help = "Seed for reproducibility"
    )
    args = parser.parse_args()
    main(
        h5ad_dir = args.filedir,
        save_loc = args.outfile,
        ds_celltypes = args.ds_celltypes,
        ds_proportions = args.ds_proportions,
        num_batches = args.num_batches,
        seed = args.seed
    )
