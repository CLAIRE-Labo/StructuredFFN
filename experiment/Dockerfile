FROM nvcr.io/nvidia/pytorch:24.02-py3
MAINTAINER Xiuying Wei<xiuying.wei@epfl.ch>


ARG DEBIAN_FRONTEND=noninteractive

# package install
RUN apt-get update &&  apt-get install -y \
    curl vim htop\
    ca-certificates \
    openssh-server \
    cmake \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    zip \
    unzip ssh \
    tmux \
 && rm -rf /var/lib/apt/lists/*


# Install Python 3.8 with Miniconda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O ~/miniconda.sh \
#    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
#    && rm ~/miniconda.sh \
#    && /opt/conda/bin/conda install mkl numpy scipy pandas openmpi ipython jupyter \
#    && /opt/conda/bin/conda clean --all -y


# ENV PATH="~/.local/bin:/opt/conda/bin:/usr/local/cuda/bin:${PATH}" \
#     LD_LIBRARY_PATH="/usr/local/cuda/lib64"
ENV PATH="~/.local/bin:/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64"

# Make $PATH and $LD_LIBRARY PATH available to all users
RUN echo PATH="${PATH}" >> /etc/environment && \
    echo LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" >> /etc/environment

# transformers==4.34.0
# datasets
# evaluate
# accelerate
# RUN pip uninstall transformer-engine --yes
# The following two rows are for butterfly
RUN pip --no-cache-dir install \
    easydict \
    h5py \
    pyyaml \
    tqdm \
    pillow \
    protobuf \
    seaborn \
    scipy \
    scikit-learn \
    wandb \
    hydra-core \
    transformers==4.34.0 \
    datasets \
    evaluate \
    accelerate \
    sentencepiece

# RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
# RUN pip3 install --upgrade flash-attn==2.4.2 --no-build-isolation
# entrypoint
RUN pip install --upgrade protobuf==3.20.0
ENV ENTRYPOINTS_ROOT=/opt/entrypoints
RUN mkdir -p ${ENTRYPOINTS_ROOT}


# The entrypoint is run in an interactive shell so that the conda environment is activated before.
# Don't overwrite the entrypoint, it is installing the project
# and testing that you correctly mounted the project code and data and output directories.
# It also performs some other important setup depending on the deployment platform.
COPY --link entrypoint.sh ${ENTRYPOINTS_ROOT}/entrypoint.sh
ENTRYPOINT ["/bin/bash", "-i", "/opt/entrypoints/entrypoint.sh"]
CMD ["/bin/bash"]


# userconfig
# define your own config here

