#!/bin/bash
#SBATCH --account=def-kieranc
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=izumi.ando@mail.utoronto.ca
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate snakemake

# Run Snakemake
# not specifying partitions for Graham
snakemake --unlock
snakemake -j 10 \
		--snakefile Snakefile_uce \
    --use-conda \
    --cluster-config cluster_cedar.json \
    --cluster "sbatch \
		    --account=def-kieranc \
        --mem=75G \
        --cpus-per-task=10 \
        --time=2-00:00:00 \
        --mail-user=izumi.ando@mail.utoronto.ca \
        --mail-type=ALL" \
    --restart-times 0 \
    --latency-wait 300 \
    --keep-going \
    --rerun-incomplete
    
echo "We have reached the end of the workflow" 
