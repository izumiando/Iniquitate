from functools import reduce
import gc
import random 

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ann
# import scvi 
# import bbknn 
import torch

# adding imports for UCE
import argparse
from evaluate import AnndataProcessor
from accelerate import Accelerator

# Undoing scvi's random seed setting
random.seed(None)
np.random.seed(None)
torch.manual_seed(random.randint(1, 10000000000000000000))

# from utils.seurat_integrate import SeuratIntegrate
# from utils.liger_integrate import LigerIntegrate

class Integration:
    """Class for integrating scRNA-seq data and returning processed data."""
    
    def __init__(self, adata, gpu = True):
        """
        Args:
            adata (AnnData): AnnData object to be utilized in integration methods.
                Assumes that the counts being input are unnormalized (raw counts),
                and that raw counts are stored in "counts" layer, and batch covariate
                is available.
            gpu (bool): Whether or not to use GPU for scVI.
        """
        self.adata = adata
        # Check anndata object 
        if not isinstance(adata, ann.AnnData):
            raise Exception("Please input an AnnData object.")
        # Check if gpu is available
        if gpu is True:
            if torch.cuda.is_available():
                self.gpu = True
            else:
                raise Exception("GPU not available. Please set gpu = False.")
        else:
            self.gpu = False

    def uce_integrate(self, n_neighbors = 15, n_pcs = 20):
        print("Performing UCE integration.." + "\n")
        # auce = self.adata.copy()

        # modified code from UCE/eval_single_anndata.py
        args = self.setup_args()
        accelerator = Accelerator(project_dir=args.dir)
        processor = AnndataProcessor(args, accelerator)
        processor.preprocess_anndata()
        processor.generate_idxs()
        auce = processor.run_evaluation(save_file=False)

        # TO DO: write appropriately
        # note: auce.absm["X_uce"] is already the embedding
        # auce.obsm["X_kmeans"] = ...

        # remove later
        # scvi.data.setup_anndata(ascvi, batch_key = "batch")
        # vae = scvi.model.SCVI(ascvi)
        # vae.train(use_gpu = self.gpu)
        # ascvi.obsm["X_scVI"] = vae.get_latent_representation()
        # ascvi.obsm["X_kmeans"] = ascvi.obsm["X_scVI"][:, 0:n_pcs]
        # sc.pp.neighbors(
        #     ascvi,
        #     n_neighbors = n_neighbors,
        #     n_pcs = n_pcs,
        #     use_rep = "X_scVI"
        # )

        sc.tl.leiden(auce)
        sc.tl.umap(auce)
        print("Done!" + "\n")
        return auce
    
    def setup_args():
        parser = argparse.ArgumentParser(
        description='Embed a single anndata using UCE.')

        # Anndata Processing Arguments
        parser.add_argument('--adata_path', type=str,
                            default=None,
                            help='Full path to the anndata you want to embed.')
        parser.add_argument('--dir', type=str,
                            default="./",
                            help='Working folder where all files will be saved.')
        parser.add_argument('--species', type=str, default="human",
                            help='Species of the anndata.')
        parser.add_argument('--filter', type=bool, default=True,
                            help='Additional gene/cell filtering on the anndata.')
        parser.add_argument('--skip', type=bool, default=True,
                            help='Skip datasets that appear to have already been created.')

        # Model Arguments
        parser.add_argument('--model_loc', type=str,
                            default=None,
                            help='Location of the model.')
        parser.add_argument('--batch_size', type=int, default=25,
                            help='Batch size.')
        parser.add_argument('--pad_length', type=int, default=1536,
                            help='Batch size.')
        parser.add_argument("--pad_token_idx", type=int, default=0,
                            help="PAD token index")
        parser.add_argument("--chrom_token_left_idx", type=int, default=1,
                            help="Chrom token left index")
        parser.add_argument("--chrom_token_right_idx", type=int, default=2,
                            help="Chrom token right index")
        parser.add_argument("--cls_token_idx", type=int, default=3,
                            help="CLS token index")
        parser.add_argument("--CHROM_TOKEN_OFFSET", type=int, default=143574,
                            help="Offset index, tokens after this mark are chromosome identifiers")
        parser.add_argument('--sample_size', type=int, default=1024,
                            help='Number of genes sampled for cell sentence')
        parser.add_argument('--CXG', type=bool, default=True,
                            help='Use CXG model.')
        parser.add_argument('--nlayers', type=int, default=4,
                            help='Number of transformer layers.')
        parser.add_argument('--output_dim', type=int, default=1280,
                            help='Output dimension.')
        parser.add_argument('--d_hid', type=int, default=5120,
                            help='Hidden dimension.')
        parser.add_argument('--token_dim', type=int, default=5120,
                            help='Token dimension.')
        parser.add_argument('--multi_gpu', type=bool, default=False,
                            help='Use multiple GPUs')

        # Misc Arguments
        parser.add_argument("--spec_chrom_csv_path",
                            default="./model_files/species_chrom.csv", type=str,
                            help="CSV Path for species genes to chromosomes and start locations.")
        parser.add_argument("--token_file",
                            default="./model_files/all_tokens.torch", type=str,
                            help="Path for token embeddings.")
        parser.add_argument("--protein_embeddings_dir",
                            default="./model_files/protein_embeddings/", type=str,
                            help="Directory where protein embedding .pt files are stored.")
        parser.add_argument("--offset_pkl_path",
                            default="./model_files/species_offsets.pkl", type=str,
                            help="PKL file which contains offsets for each species.")

        args = parser.parse_args([
            '--adata_path', 'None', # add in
            '--dir', './', # add in
            '--species', 'human',
            '--filter', 'True',
            '--skip', 'True',
            '--model_loc', 'None', # add in
            '--batch_size', '25',
            '--pad_length', '1536',
            '--pad_token_idx', '0',
            '--chrom_token_left_idx', '1',
            '--chrom_token_right_idx', '2',
            '--cls_token_idx', '3',
            '--CHROM_TOKEN_OFFSET', '143574',
            '--sample_size', '1024',
            '--CXG', 'True',
            '--nlayers', '33',
            '--output_dim', '1280',
            '--d_hid', '5120',
            '--token_dim', '5120',
            '--multi_gpu', 'False',
            '--spec_chrom_csv_path', './model_files/species_chrom.csv',
            '--token_file', './model_files/all_tokens.torch',
            '--protein_embeddings_dir', './model_files/protein_embeddings/',
            '--offset_pkl_path', './model_files/species_offsets.pkl'
        ])

        return args