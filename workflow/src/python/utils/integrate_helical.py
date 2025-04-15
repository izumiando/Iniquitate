import random 

import numpy as np
import scanpy as sc
import anndata as ann
import torch

# Helical-specific imports
from helical.models.uce.model import UCE, UCEConfig
from helical.models.scgpt.model import scGPT, scGPTConfig
from helical.models.geneformer.model import Geneformer, GeneformerConfig

# for UCE step
import os


# Undoing scvi's random seed setting
random.seed(None)
np.random.seed(None)
torch.manual_seed(random.randint(1, 10000000000000000000))

class IntegrationHelical:
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

    def uce_integrate(self):
        print("Performing UCE integration.." + "\n")
        if os.path.exists("../../../test_counts.npz"):
            os.remove("../../../test_counts.npz")
        auce = self.adata.copy()
        configurer_uce = UCEConfig(model_name="33l_8ep_1024t_1280", device="cuda")
        uce = UCE(configurer=configurer_uce)
        data_loader_uce = uce.process_data(auce)
        embeddings_uce = uce.get_embeddings(data_loader_uce)
        auce.obsm["X_UCE"] = embeddings_uce
        print("UCE embedding dimensions are:" +"\n")
        print(embeddings_uce.shape)
        sc.pp.neighbors(
            auce,
            n_neighbors = 15,
            n_pcs = 20,
            use_rep = "X_UCE"
        )
        sc.tl.leiden(auce)
        sc.tl.umap(auce)
        print("Done!" + "\n")
        return auce
    
    def scgpt_integrate(self):
        print("Performing scGPT integration.." + "\n")
        ascgpt = self.adata.copy()
        configurer_scgpt = scGPTConfig(device="cuda")
        scgpt = scGPT(configurer=configurer_scgpt)
        data_loader_scgpt = scgpt.process_data(ascgpt)
        embeddings_scgpt = scgpt.get_embeddings(data_loader_scgpt)
        ascgpt.obsm["X_scGPT"] = embeddings_scgpt
        print("scGPT embedding dimensions are" + "\n")
        print(embeddings_scgpt.shape)
        sc.pp.neighbors(
            ascgpt,
            n_neighbors = 15,
            n_pcs = 20,
            use_rep = "X_scGPT"
        )
        sc.tl.leiden(ascgpt)
        sc.tl.umap(ascgpt)
        print("Done!" + "\n")
        return ascgpt
        
    def geneformer_integrate(self):
        print("Performing Geneformer integration.." + "\n")
        ageneformer = self.adata.copy()
        configurer_geneformer = GeneformerConfig(model_name="gf-12L-30M-i2048", device="cuda")
        geneformer = Geneformer(configurer=configurer_geneformer)
        data_loader_geneformer = geneformer.process_data(ageneformer)
        embeddings_geneformer = geneformer.get_embeddings(data_loader_geneformer)
        ageneformer.obsm["X_Geneformer"] = embeddings_geneformer
        print("Geneformer embedding dimensions are" + "\n")
        print(embeddings_geneformer.shape)
        sc.pp.neighbors(
            ageneformer,
            n_neighbors = 15,
            n_pcs = 20,
            use_rep = "X_Geneformer"
        )
        sc.tl.leiden(ageneformer)
        sc.tl.umap(ageneformer)
        print("Done!" + "\n")
        return ageneformer
