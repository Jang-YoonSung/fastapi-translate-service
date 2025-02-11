FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.9
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

RUN apt-get install -y python3-pip && \
    pip3 install --upgrade pip

RUN python3.9 -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

COPY ./requirements.txt /tmp/requirements.txt
RUN python3.9 -m pip install -r /tmp/requirements.txt