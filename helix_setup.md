### Instructions to install helical on Cedar/CC

module load python/3.11
virtualenv --no-download HELICAL
module load gcc arrow 
source HELICAL/bin/activate
pip install helical 
pip install scanpy 
pip install anndata