FROM tensorflow/tensorflow:2.4.0-gpu
LABEL maintainer="Rui Zhang (rui.zhang@cern.ch)"

ARG PROJECT=CaloShowerGAN
ENV SHAREDIR /myshare

#RUN apt-get update && \
#    apt-get install -y --no-install-recommends vim && \
#    apt-get clean
COPY . /$PROJECT
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /$PROJECT/docker/requirements.txt
