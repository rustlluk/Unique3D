# get the development image from nvidia cuda 12.1
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04


ENV DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM

#Install all neccesary thingies
RUN apt update -y && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa -y  \
    && apt update && apt install wget sudo gedit unzip apt-utils curl \
    nano mesa-utils curl htop net-tools sshfs \
    screen git\
    pkg-config python3-venv python3-dev gdb gitk autoconf locales gdebi \
    terminator libopengl0 libgl1 -y


# Set the locale
RUN locale-gen en_US.UTF-8

#Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages

# Remove the default 'ubuntu' user/group that comes with Ubuntu 24.04
RUN touch /var/mail/ubuntu && chown ubuntu /var/mail/ubuntu && \
    userdel -f ubuntu && groupdel ubuntu || true
ARG UID=1000
ARG GID=1000
#Add docker user with correct UID and GID; and add him to sudoers
RUN groupadd -g $GID docker_users && useradd -l -u $UID -G $GID -md /home/docker -s /bin/bash docker && \
    echo 'docker:docker' | chpasswd && usermod -aG sudo docker && \
    sed -i.bkp -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers && \
    echo "docker  ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER docker
WORKDIR /home/docker

RUN python3 -m venv venv
RUN git clone https://github.com/rustlluk/Unique3D
WORKDIR /home/docker/Unique3D
ENV PATH="/home/docker/venv/bin:$PATH"
ENV FORCE_CUDA=1
RUN pip install --upgrade pip
RUN pip install "setuptools<70.0.0" wheel huggingface_hub
ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9+PTX"
RUN echo "setuptools<70.0.0" > constraint.txt &&  \
    PIP_CONSTRAINT=constraint.txt pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.1/index.html --no-build-isolation  \
    && pip install "torch==2.3.1" --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
ENV MAX_JOBS=$(nproc)-6
ENV NVCC_THREADS=4
RUN pip install -r requirements.txt --no-build-isolation
WORKDIR /home/docker/

RUN rm -r Unique3D
ENV HF_HOME=/home/docker/data/HF_data/

RUN cat <<EOF > download_script.py
from huggingface_hub import snapshot_download

def download_checkpoints():
    print("Downloading checkpoints...")
    snapshot_download(
        repo_id="Wuvin/Unique3D",
        repo_type="space",
        allow_patterns="ckpt/*",
        local_dir="/home/docker/data/",
    )
    print("Download complete!")

if __name__ == "__main__":
    download_checkpoints()
EOF