ARG VLLM_VERSION=0.17.0
ARG CUDA_VERSION=12.9
ARG CUDNN_VERSION=9.20
ARG PYTORCH_VERSION=2.10


FROM hieupth/mamba
# Recall build args.
ARG VLLM_VERSION
ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG PYTORCH_VERSION
# Install cuda
RUN conda install -y -c nvidia \
      cuda-toolkit=${CUDA_VERSION} \
      cudnn=${CUDNN_VERSION} && \
    conda clean -ay
# Install pytorch
RUN conda install -y -c pytorch \
      pytorch=${PYTORCH_VERSION} && \
    conda clean -ay
# Install huggingface-hub and vllm
RUN pip install --no-cache-dir \
      huggingface-hub \
      vllm==${VLLM_VERSION}