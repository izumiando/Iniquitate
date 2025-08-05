# import random  # i think this was for the unseeding which we may reintroduce later

import numpy as np
import scanpy as sc
import anndata as ann
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True # Suppress errors for torch dynamo triggered by Transcriptformer

# Helical-specific imports
from helical.models.uce.model import UCE, UCEConfig
from helical.models.scgpt.model import scGPT, scGPTConfig
from helical.models.geneformer.model import Geneformer, GeneformerConfig
from helical.models.transcriptformer.model import TranscriptFormer
from helical.models.transcriptformer.transcriptformer_config import TranscriptFormerConfig

# for UCE step
import os

# for PCA
from sklearn.preprocessing import StandardScaler as Scale
from sklearn.decomposition import PCA

# unseeding
# random.seed(None)
# np.random.seed(None)
# torch.manual_seed(random.randint(1, 10000000000000000000))

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
        
        # adding this here because this needs to be cleared before UCE gets called
        # this assumes that uce_integrate is only called once
        if os.path.exists("../../../test_counts.npz"):
            os.remove("../../../test_counts.npz")

    def uce_integrate(self):
        print("Performing UCE integration.." + "\n")
        auce = self.adata.copy()
        configurer_uce = UCEConfig(model_name="33l_8ep_1024t_1280", batch_size=8, device="cuda")
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
        
        # dimensionality reduction with PCA
        scaler_uce = Scale()
        uce_embeddings_scaled = scaler_uce.fit_transform(auce.obsm["X_UCE"])
        uce_pca = PCA(n_components=20)
        uce_embeddings_reduced = uce_pca.fit_transform(uce_embeddings_scaled)
        auce.obsm["X_emb_reduced"] = uce_embeddings_reduced
        auce.obsm["X_kmeans"] = auce.obsm["X_emb_reduced"][:, 0:20] # 20 is the number of PCs
        
        print("Done!" + "\n")
        return auce
    
    def scgpt_integrate(self):
        print("Performing scGPT integration.." + "\n")
        ascgpt = self.adata.copy()
        configurer_scgpt = scGPTConfig(batch_size=8, device="cuda")
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
        
        # dimensionality reduciton with PCA
        scaler_scgpt = Scale()
        scgpt_embeddings_scaled = scaler_scgpt.fit_transform(ascgpt.obsm["X_scGPT"])
        scgpt_pca = PCA(n_components=20)
        scgpt_embeddings_reduced = scgpt_pca.fit_transform(scgpt_embeddings_scaled)
        ascgpt.obsm["X_emb_reduced"] = scgpt_embeddings_reduced
        ascgpt.obsm["X_kmeans"] = ascgpt.obsm["X_emb_reduced"][:, 0:20] # 20 is the number of PCs
        
        print("Done!" + "\n")
        return ascgpt
        
    def geneformer_integrate(self):
        print("Performing Geneformer integration.." + "\n")
        ageneformer = self.adata.copy()
        configurer_geneformer = GeneformerConfig(model_name="gf-12L-30M-i2048", batch_size= 8, device="cuda")
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
        
        # Making sure there are no duplicate var indicies that prevent concatenation
        # If you get similar issues with obs or for other anndata objects, run this for them as well
        ageneformer.var.index = ageneformer.var.index.astype(str)
        ageneformer.var_names_make_unique()
        
        # dimensionality reduction with PCA
        scaler_geneformer = Scale()
        geneformer_embeddings_scaled = scaler_geneformer.fit_transform(ageneformer.obsm["X_Geneformer"])
        geneformer_pca = PCA(n_components=20)
        geneformer_embeddings_reduced = geneformer_pca.fit_transform(geneformer_embeddings_scaled)
        ageneformer.obsm["X_emb_reduced"] = geneformer_embeddings_reduced
        ageneformer.obsm["X_kmeans"] = ageneformer.obsm["X_emb_reduced"][:, 0:20] # 20 is the number of PCs
        
        print("Done!" + "\n")
        return ageneformer

    def transcriptformer_integrate(self):
        print("Performing Transcriptformer integration.." + "\n")
        atranscriptformer = self.adata.copy()
        configurer_transcriptformer = TranscriptFormerConfig(model_name="tf_sapiens", batch_size=8, emb_mode="cell")
        transcriptformer = TranscriptFormer(configurer=configurer_transcriptformer)
        data_loader_transcriptformer = transcriptformer.process_data([atranscriptformer])
        embeddings_transcriptformer = transcriptformer.get_embeddings(data_loader_transcriptformer)
        atranscriptformer.obsm["X_Transcriptformer"] = embeddings_transcriptformer.numpy()
        print("Transcriptformer embedding dimensions are" + "\n")
        print(atranscriptformer.obsm["X_Transcriptformer"].shape)
        sc.pp.neighbors(
            atranscriptformer,
            n_neighbors = 15,
            n_pcs = 20,
            use_rep = "X_Transcriptformer"
        )
        sc.tl.leiden(atranscriptformer)
        sc.tl.umap(atranscriptformer)
        
        # dimensionality reduction with PCA
        scaler_transcriptformer = Scale()
        transcriptformer_embeddings_scaled = scaler_transcriptformer.fit_transform(atranscriptformer.obsm["X_Transcriptformer"])
        transcriptformer_pca = PCA(n_components=20)
        transcriptformer_embeddings_reduced = transcriptformer_pca.fit_transform(transcriptformer_embeddings_scaled)
        atranscriptformer.obsm["X_emb_reduced"] = transcriptformer_embeddings_reduced
        atranscriptformer.obsm["X_kmeans"] = atranscriptformer.obsm["X_emb_reduced"][:, 0:20] # 20 is the number of PCs
        
        print("Done!" + "\n")
        return atranscriptformer