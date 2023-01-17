FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic

###################################### user #####################################

ENV SHELL=/bin/bash \
    NB_USER=arg \
    NB_UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} 

RUN echo "root:root" | chpasswd
RUN echo "${NB_USER}:111111" | chpasswd

ENV HOME=/home/${NB_USER}

# RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
#     && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \
#     curl \
#     dpkg \
#     && apt-key del 7fa2af80 \
#     && curl -L -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
#     && dpkg -i cuda-keyring_1.0-1_all.deb 


##################################### basic tools #####################################
RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \
    vim \
    wget \
    sudo \
    cmake \
    gedit \
    gnupg2 \
    gtk+-3.0 \
    locales \
    dirmngr \
    mercurial \
    net-tools \
    lsb-release \
    openssh-client \
    build-essential \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libpcap-dev \
    libusb-1.0 \
    tzdata \
    libglfw3-dev \
    libmetis-dev \
    software-properties-common \
    ruby \
    python3-pip \
    python3-setuptools \
    python3-empy  \
    python3-opencv \
    python3-dbg \
    python3-venv \
    python3-scipy   \
    python3-dev \
    python3-yaml \
    libfltk1.3-dev \ 
    freeglut3-dev \ 
    libpng-dev \ 
    libjpeg-dev \
    libxft-dev \ 
    libxinerama-dev \ 
    libtiff5-dev \ 
    netbase \ 
    fluid \
    xterm \
    qtbase5-dev \
    && add-apt-repository -y ppa:git-core/ppa \
    && apt-get update && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


###################################### GNN #####################################
# RUN apt-get remove python3-numpy -y
# RUN pip3 install \
#     numpy==1.20 \
#     graph_nets \
#     tensorflow_gpu==2.10.0-rc1 \
#     dm-sonnet==2.0.0b0 \
#     tensorflow_probability 

RUN pip3 install \
    pyg-lib \
    torch-scatter \
    torch-sparse \
    torchvision \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-1.13.0+cu116.html \
    torch-geometric 



###################################### PIP3 #####################################
RUN pip3 install \
    matplotlib \
    networkx \
    jupyter \
    gym \
    pygame \
    numpy==1.21.1


###################################### Colab Local #####################################  

RUN pip3 install --upgrade jupyter_http_over_ws>=0.0.7 && \
    jupyter serverextension enable --py jupyter_http_over_ws

##################################### Settings #####################################


# Support of nvidia-docker 2.0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all