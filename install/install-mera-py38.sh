# Install system wide dependencies
sudo apt update && sudo apt install llvm-10 libgomp1 ocl-icd-libopencl1 software-properties-common \
    libgoogle-glog0v5 libboost-graph-dev virtualenv wget build-essential
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install libstdc++6

# Create virtual envinronment with Python 3.8
MERA_VENV=mera-env
virtualenv -p python3.8 $MERA_VENV
source $MERA_VENV/bin/activate

# Auto load the environment
cat <<EOF >> ~/.bashrc
source $MERA_VENV/bin/activate
EOF

# Install MERA host-only version (change to runtime or full for version capable of running on DNA IP machines)
pip install --upgrade pip
pip install tqdm easydict wget notebook pandas matplotlib opencv-python gdown seaborn tensorflow_datasets
pip install mera[host-only]

# Test installation
python -c "mera --version"
