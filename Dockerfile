FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN mkdir "/data"
COPY data /data

RUN apt-get update --fix-missing
RUN apt-get install -y python3 python3-pip psmisc git libturbojpeg ffmpeg libsm6 libxext6 wget 

RUN pip install Cython
RUN pip install --upgrade wandb scikit-learn scikit-image matplotlib scipy==1.9.1 tqdm pyyaml \
opencv-python imgaug pandas diffdist mmcv mmcls faiss-gpu gdown
RUN pip install --upgrade --no-cache-dir --trusted-host download.pytorch.org --extra-index-url https://download.pytorch.org/whl/cu113 torch==1.12.1+cu113 torchvision==0.13.1+cu113
RUN pip install libmr wandb scikit-learn scikit-image matplotlib scipy tqdm pyyaml