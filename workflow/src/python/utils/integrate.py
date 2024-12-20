import random 

import numpy as np
import scanpy as sc
import anndata as ann
import scvi 
import torch

# Undoing scvi's random seed setting
random.seed(None)
np.random.seed(None)
torch.manual_seed(random.randint(1, 10000000000000000000))

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

    def scvi_integrate(self, n_neighbors = 15, n_pcs = 20):
        print("Performing scVI integration.." + "\n")
        ascvi = self.adata.copy()
        scvi.data.setup_anndata(ascvi, batch_key = "batch")
        vae = scvi.model.SCVI(ascvi)
        vae.train(use_gpu = self.gpu)
        ascvi.obsm["X_scVI"] = vae.get_latent_representation()
        ascvi.obsm["X_kmeans"] = ascvi.obsm["X_scVI"][:, 0:n_pcs]
        sc.pp.neighbors(
            ascvi,
            n_neighbors = n_neighbors,
            n_pcs = n_pcs,
            use_rep = "X_scVI"
        )
        sc.tl.leiden(ascvi)
        sc.tl.umap(ascvi)
        print("Done!" + "\n")
        return ascvi
    
    def harmony_integrate(self, n_neighbors = 15, n_pcs = 20, num_hvgs = 2500):
        print("Performing Harmony integration.." + "\n")
        aharmony = self.adata.copy()
        sc.pp.normalize_total(
            aharmony,
            target_sum = 1e4
        )
        sc.pp.log1p(aharmony)
        sc.pp.highly_variable_genes(
            aharmony,
            n_top_genes = num_hvgs,
            flavor = "seurat"
        )
        sc.pp.pca(aharmony, svd_solver="arpack")
        sc.external.pp.harmony_integrate(
            aharmony,
            key = "batch",
            random_state = None
        )
        sc.pp.neighbors(
            aharmony,
            n_neighbors = n_neighbors,
            n_pcs = n_pcs,
            use_rep = "X_pca_harmony"
        )
        aharmony.obsm["X_kmeans"] = aharmony.obsm["X_pca_harmony"][:, 0:n_pcs]
        sc.tl.leiden(aharmony)
        sc.tl.umap(aharmony)
        print("Done!" + "\n")
        return aharmony