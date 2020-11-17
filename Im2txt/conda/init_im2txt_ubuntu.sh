conda create -n im2txt python=3.6 pip --yes
source ~/anaconda3/etc/profile.d/conda.sh
conda activate im2txt

cd /path/to/Im2txt

# Install python libraries
pip install -r requirement.txt

echo "Done"
