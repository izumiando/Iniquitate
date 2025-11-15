import random  
import sys
import os
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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

# Finetuning imports
from helical.models.geneformer.model import GeneformerConfig
from helical.models.geneformer import GeneformerFineTuningModel
from helical.models.scgpt import scGPTFineTuningModel, scGPTConfig
from helical.models.fine_tune.data_integration_head import DataIntegrationHead

# for UCE step
import os

# for PCA
from sklearn.preprocessing import StandardScaler as Scale
from sklearn.decomposition import PCA

# unseeding, added back in Aug 5, 2025
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
    
    ############################ FINE TUNING #####################################
    def geneformer_integrate_finetuned(self):
        """
        Geneformer is not designed to be fine tuned for batch integration, rather
        it is designed to be fine tuned for downstream tasks.
        """
        print("Performing fine-tuned Geneformer integration.." + "\n")
        ageneformer = self.adata.copy()
        celltypes = list(ageneformer.obs["celltype"])
        label_set = set(celltypes)
        configurer_geneformer = GeneformerConfig(model_name="gf-12L-30M-i2048", batch_size=8, device="cuda")
        geneformer_fine_tune = GeneformerFineTuningModel(geneformer_config=configurer_geneformer, fine_tuning_head="classification", output_size=len(label_set))
        data = geneformer_fine_tune.process_data(ageneformer)
        data = data.add_column('celltype', celltypes)
        class_id_dict = dict(zip(label_set, range(len(label_set))))
        
        def classes_to_ids(example):
            example["celltype"] = class_id_dict[example["celltype"]]
            return example
        data = data.map(classes_to_ids, num_proc=1)
        
        geneformer_fine_tune.train(train_dataset=data, label="celltype")
        embeddings = geneformer_fine_tune.get_embeddings(data) # embedding
        
        ageneformer.obsm["X_Geneformer_ft"] = embeddings
        print("Geneformer embedding dimensions are" + "\n")
        print(embeddings.shape)
        sc.pp.neighbors(
            ageneformer,
            n_neighbors = 15,
            n_pcs = 20,
            use_rep = "X_Geneformer_ft"
        )
        sc.tl.leiden(ageneformer)
        sc.tl.umap(ageneformer)
        
        # dimensionality reduction with PCA
        scaler_geneformer = Scale()
        geneformer_ft_embeddings_scaled = scaler_geneformer.fit_transform(ageneformer.obsm["X_Geneformer_ft"])
        geneformer_pca = PCA(n_components=20)
        geneformer_embeddings_reduced = geneformer_pca.fit_transform(geneformer_ft_embeddings_scaled)
        ageneformer.obsm["X_emb_reduced"] = geneformer_embeddings_reduced
        ageneformer.obsm["X_kmeans"] = ageneformer.obsm["X_emb_reduced"][:, 0:20] # 20 is the number of PCs
        
        print("\n Done!")
        return ageneformer
    
    def scgpt_integrate_finetuned(self):
        batch_key = "batch"
        print("Performing fine-tuned scGPT integration.." + "\n")
        ascgpt = self.adata.copy()
        ascgpt.obs[f"str_{batch_key}"] = ascgpt.obs[batch_key].astype(str)
        batch_id_labels = ascgpt.obs[f"str_{batch_key}"].astype("category").cat.codes.values
        ascgpt.obs["batch_id"] = batch_id_labels
        num_batches = len(ascgpt.obs[f"str_{batch_key}"].unique())

        # Define training parameters
        batch_size = 8
        epochs = 15
        mask_ratio = 0.4
        dab_weight = 1.0

        configurer_scgpt = scGPTConfig(batch_size=batch_size, device="cuda")
        # Create custom data integration head
        integration_head = DataIntegrationHead(
            num_batches=num_batches,
            ecs_threshold=0.8,
            dab_weight=dab_weight,
            use_dsbn=True,
            dropout=0.2
        )

        # Create fine-tuning model
        scgpt_model = scGPTFineTuningModel(
            scGPT_config=configurer_scgpt,
            fine_tuning_head=integration_head,
            output_size=None  # Not needed when passing head instance
        )

        # Process data for scGPT with batch labels for data integration
        logger.info("Processing data for scGPT...")
        dataset = scgpt_model.process_data(ascgpt, fine_tuning=True, use_batch_labels=True)

        # Train the integration model
        logger.info("Starting integration training...")
        scgpt_model.train_data_integration(
            train_input_data=dataset,
            train_batch_labels=batch_id_labels,
            epochs=epochs,
            mask_ratio=mask_ratio,
            ecs_weight=10.0,
            dab_weight=dab_weight,
            optimizer_params={"lr": 1e-4},
            lr_scheduler_params={
                'name': 'linear',
                'num_warmup_steps': 0,
                'num_training_steps': len(dataset) // batch_size * epochs
            }
        )

        outputs = scgpt_model.get_outputs(dataset)

        # If using DataIntegrationHead, extract embeddings from the output dict
        if isinstance(outputs, dict) and 'embeddings' in outputs:
            embeddings = outputs['embeddings']
        else:
            embeddings = outputs

        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        ascgpt.obsm["X_scGPT_ft"] = embeddings
        print("scGPT embedding dimensions are" + "\n")
        print(embeddings.shape)
        sc.pp.neighbors(
            ascgpt,
            n_neighbors = 15,
            n_pcs = 20,
            use_rep = "X_scGPT_ft"
        )
        sc.tl.leiden(ascgpt)
        sc.tl.umap(ascgpt)

        # dimensionality reduction with PCA
        scaler_scgpt = Scale()
        scgpt_ft_embeddings_scaled = scaler_scgpt.fit_transform(ascgpt.obsm["X_scGPT_ft"])
        scgpt_pca = PCA(n_components=20)
        scgpt_embeddings_reduced = scgpt_pca.fit_transform(scgpt_ft_embeddings_scaled)
        ascgpt.obsm["X_emb_reduced"] = scgpt_embeddings_reduced
        ascgpt.obsm["X_kmeans"] = ascgpt.obsm["X_emb_reduced"][:, 0:20] # 20 is the number of PCs
        print("\n Done!")
        return ascgpt