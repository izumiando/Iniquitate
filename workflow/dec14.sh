#!/bin/bash
#SBATCH --account=def-kieranc
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=28-00:00:00
#SBATCH --mail-user=izumi.ando@mail.utoronto.ca
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate snakemake

# Run Snakemake
# not specifying partitions for Graham
snakemake --unlock
snakemake -j 10 \
    --use-conda \
    --cluster-config cluster_cedar.json \
    --cluster "sbatch \
		    --account=def-kieranc \
        --mem=148G \
        --gres=gpu:1 \
        --time=28-00:00:00 \
        --mail-user=izumi.ando@mail.utoronto.ca \
        --mail-type=ALL" \
    --restart-times 0 \
    --latency-wait 300 \
    --keep-going \
    --rerun-incomplete
    
echo "We have reached the end of the workflow" 


# IMPORTANT NOTES
## this was run with the following settings to nov29.sh
### 1) increased time allocation to master (48 to 72)
### 2) added cluster_cedar.json
### 3) update on Dec 9th, changed time limit to 28 days for everything
