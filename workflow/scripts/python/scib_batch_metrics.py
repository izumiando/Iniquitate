import argparse
import scanpy as sc
import scib
import pandas as pd
import numpy as np


def main(adata, adata_int, dataset_name, rep, ds_celltypes, ds_proportions,
         num_batches, save_loc):
    # Load h5ad file
    adata = sc.read_h5ad(adata)
    adata_int = sc.read_h5ad(adata_int)

    integration_method = adata_int.obs["integration_method"].values
    scgpt = adata_int[integration_method == "scgpt"]
    geneformer = adata_int[integration_method == "geneformer"]
    transcriptformer = adata_int[integration_method == "transcriptformer"]
    uce = adata_int[integration_method == "uce"]
    scvi = adata_int[integration_method == "scvi"]
    harmony = adata_int[integration_method == "harmony"]

    # Dictionary to store integration methods and their corresponding adata
    integration_methods = {
        "scgpt": scgpt,
        "geneformer": geneformer,
        "transcriptformer": transcriptformer,
        "uce": uce,
        "scvi": scvi,
        "harmony": harmony
    }

    # List to store results for each method
    results_list = []

    # Run scib metrics for each integration method
    for method_name, adata_method in integration_methods.items():
        if len(adata_method) > 0:  # Only process if there are cells
            try:
                scib_results = scib.metrics.metrics(
                    adata,
                    adata_int=adata_method,
                    batch_key="batch",
                    label_key="celltype",
                    embed="X_kmeans",
                    ari_=True,
                    nmi_=True,
                    silhouette_=True,
                    pcr_=True,
                    organism="human",
                    graph_conn_=True,
                    lisi_graph_=True
                )

                # Create a row for this method's results
                result_row = {
                    "Dataset": dataset_name,
                    "Number of batches downsampled": num_batches,
                    "Number of celltypes downsampled": ds_celltypes,
                    "Proportion downsampled": ds_proportions,
                    "Replicate": rep,
                    "Method": method_name,
                    "NMI_cluster/label": scib_results.get(
                        "NMI_cluster/label", np.nan),
                    "ARI_cluster/label": scib_results.get(
                        "ARI_cluster/label", np.nan),
                    "ASW_label": scib_results.get("ASW_label", np.nan),
                    "ASW_label/batch": scib_results.get(
                        "ASW_label/batch", np.nan),
                    "PCR_batch": scib_results.get("PCR_batch", np.nan),
                    "graph_conn": scib_results.get("graph_conn", np.nan),
                    "iLISI": scib_results.get("iLISI", np.nan),
                    "cLISI": scib_results.get("cLISI", np.nan)
                }
                results_list.append(result_row)

            except Exception as e:
                print(f"Error processing {method_name}: {e}")
                # Add a row with NaN values for failed methods
                result_row = {
                    "Dataset": dataset_name,
                    "Number of batches downsampled": num_batches,
                    "Number of celltypes downsampled": ds_celltypes,
                    "Proportion downsampled": ds_proportions,
                    "Replicate": rep,
                    "Method": method_name,
                    "NMI_cluster/label": np.nan,
                    "ARI_cluster/label": np.nan,
                    "ASW_label": np.nan,
                    "ASW_label/batch": np.nan,
                    "PCR_batch": np.nan,
                    "graph_conn": np.nan,
                    "iLISI": np.nan,
                    "cLISI": np.nan
                }
                results_list.append(result_row)

    # Create summary dataframe for scib batch metrics
    scib_batch_metrics_df = pd.DataFrame(results_list)

    scib_batch_metrics_df.to_csv(
        save_loc,
        index=False,
        sep="\t"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Input and output files for scib batch metric results"
    )
    parser.add_argument(
        "--adata",
        type=str,
        help="Path to non-integrated h5ad file"
    )
    parser.add_argument(
        "--adata_int",
        type=str,
        help="Path to integrated h5ad file"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of dataset"
    )
    parser.add_argument(
        "--rep",
        type=str,
        help="Repetition number (wildcard identifier)"
    )
    parser.add_argument(
        "--ds_celltypes",
        type=str,
        help="Number of celltypes downsampled"
    )
    parser.add_argument(
        "--ds_proportions",
        type=str,
        help="Proportion downsampled"
    )
    parser.add_argument(
        "--num_batches",
        type=str,
        help="Number of batches downsampled"
    )
    parser.add_argument(
        "--save_loc",
        type=str,
        help="Filepath for saving scib batch metric results"
    )
    args = parser.parse_args()
    main(
        adata=args.adata,
        adata_int=args.adata_int,
        dataset_name=args.dataset_name,
        rep=args.rep,
        ds_celltypes=args.ds_celltypes,
        ds_proportions=args.ds_proportions,
        num_batches=args.num_batches,
        save_loc=args.save_loc
    )
