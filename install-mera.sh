# Install system wide dependencies
sudo apt update && sudo apt install python3.6 llvm-10 libgomp1 ocl-icd-libopencl1 software-properties-common \
    libgoogle-glog0v5 libboost-graph-dev virtualenv wget build-essential
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install libstdc++6

# Create virtual envinronment with Python 3.6
MERA_VENV=mera-env
virtualenv -p python3.6 $MERA_VENV
source $MERA_VENV/bin/activate

# Auto load the environment
cat <<EOF >> ~/.bashrc
source $MERA_VENV/bin/activate
EOF

# Install MERA full version (host-only + runtime)
pip install --upgrade pip
pip install mera[full]

# Extra Python dependencies
pip install torch==1.7.1 torchvision==0.8.2
pip install tensorflow==2.6.2 tflite
pip install tqdm easydict wget notebook pandas matplotlib opencv-python gdown seaborn tensorflow_datasets

# Test installation
python -c "import mera;print(mera.get_versions())"
