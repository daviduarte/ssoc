# Oficial REPO of: 

# Installation: 
sudo apt install cmake
sudo apt-get install python3-dev python3-setuptools

# 
wget https://bootstrap.pypa.io/pip/3.6/get-pip.py
python3 get-pip.py


# Create the resut folder copying the folder from server
scp -r denis@200.145.39.86:/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/graph_detector/results/* ./results

# Download de 'files' dir
mkdir ../files
scp -r denis@200.145.39.86:/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/files/gt-camnuvem.npy ../files
scp -r denis@200.145.39.86:/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/files/gt-camnuvem-anomaly-only.npy ../files
scp -r denis@200.145.39.86:/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/files/graph_detector_test_05s.list ../files
scp -r denis@200.145.39.87:/media/denis/dados/CamNuvem/pesquisa/anomalyDetection/files/coco_labels.txt ../files


#mkdir -p ./results/pretext_task
#mkdir -p ./results/downstream_task

# Download and extract the dataset
cd ~
scp denis@200.145.39.86:/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/data.tar.gz .
tar -xf data.tar.gz

# Download and extract the current result folder
mkdir results
scp -r denis@200.145.39.86:/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/graph_detector/results/* ./results
