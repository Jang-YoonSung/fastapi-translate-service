FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt update && apt-get install -y software-properties-common curl

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.11 python3.11-dev python3.11-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# pip 업그레이드 및 필수 패키지 설치
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install networkx==3.1 html5lib

RUN python3.11 -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

COPY ./requirements.txt /tmp/requirements.txt
RUN python3.11 -m pip install -r /tmp/requirements.txt