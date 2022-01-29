FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel
COPY requirements.txt ./

RUN apt-get update && apt-get install -y --no-install-recommends \
    make build-essential git wget gzip cmake &&\
    rm -rf /var/lib/apt/lists/* 

RUN python -m venv venv &&\ 
    venv/bin/pip install --upgrade pip &&\
    venv/bin/pip install -r requirements.txt && rm ./requirements.txt &&\
    venv/bin/pip install pycocotools

RUN venv/bin/pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/NVIDIA/apex.git &&\
    cd apex &&\
    ../venv/bin/pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . &&\
    rm -rf ../apex

#### detectron 2 installation
RUN venv/bin/pip install 'git+https://github.com/facebookresearch/fvcore'
RUN venv/bin/python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@ffff8ac'
################## v1 ##########################

####### horovod for multi-GPU (distributed) training #######

# update OpenMPI to avoid horovod bug
#RUN rm -r /usr/local/mpi &&\ 
RUN wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz &&\
    gunzip -c openmpi-4.0.0.tar.gz | tar xf - &&\
    cd openmpi-4.0.0 &&\
    ./configure --prefix=/usr/local/mpi --enable-orterun-prefix-by-default &&\
    make -j$(nproc) all && make install &&\
    ldconfig &&\
    cd - && rm -r openmpi-4.0.0 && rm openmpi-4.0.0.tar.gz

ENV OPENMPI_VERSION=4.0.0
ENV PATH=/usr/local/mpi:$PATH
RUN echo $PATH

# horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 \
    venv/bin/pip install --no-cache-dir horovod==0.19.4 &&\
    ldconfig

# ssh
RUN apt-get update &&\
    apt-get install -y --no-install-recommends openssh-client openssh-server &&\
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# fix ssh permissions
RUN bash -c "chmod -R 600 /etc/ssh/ && chmod 600 /var/run/sshd/ && chmod 600 /root"

# use the faster pillow-simd instead of the original pillow
# https://github.com/uploadcare/pillow-simd
RUN venv/bin/pip uninstall -y pillow && \
CC="cc -mavx2" venv/bin/pip install -U --force-reinstall pillow-simd

WORKDIR /clipbert


