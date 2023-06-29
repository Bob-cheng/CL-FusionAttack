FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install python=3.8 openmpi=4.0.4 mpi4py=3.0.3 Pillow=8.4.0
RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
RUN pip install tqdm torchpack
RUN pip install -U openmim
RUN mim install mmcv-full==1.4.0
RUN pip install mmdet==2.20.0 mmsegmentation==0.14.1
RUN pip install nuscenes-devkit spconv-cu113
RUN pip install numpy==1.19.5 pandas==1.3.5 networkx==2.2 shapely==1.7.1 numba==0.48.0
# for deepinteraction
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
RUN pip install trimesh==2.35.39 scikit-image==0.19.3 tensorboardx timm setuptools==59.5.0
