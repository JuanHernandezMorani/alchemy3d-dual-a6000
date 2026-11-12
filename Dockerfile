# Alchemy3D + TRELLIS.2 + llama.cpp, single container with two dedicated GPUs.
# NVIDIA driver is provided by Empirio's GPU host, NOT by the image.
# CUDA extensions are compiled during the build (no physical GPU required if nvcc
# and TORCH_CUDA_ARCH_LIST=8.6 are configured). GPU runtime must be checked on Empirio.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG INSTALL_FULL_EXTENSIONS=1
ARG BUILD_LLAMA=1
ARG LLAMA_REF=2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c
ARG VCS_REF=unknown
ARG SOURCE_REPO=unknown
LABEL org.opencontainers.image.title="Alchemy3D Dual A6000 Lab" \
      org.opencontainers.image.description="Alchemy3D CUDA 12.4 and llama.cpp for two RTX A6000 GPUs" \
      org.opencontainers.image.source="${SOURCE_REPO}" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAX_JOBS=2 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    CUDA_HOME=/usr/local/cuda \
    LAB_ROOT=/workspace \
    HF_HOME=/workspace/.hf \
    HF_HUB_CACHE=/workspace/.hf/hub \
    PYTHONPATH=/workspace/repos/Alchemy3D:/opt/lab \
    BLENDER_PATH=/workspace/repos/Alchemy3D/blender-4.3.2-linux-x64/blender \
    LAB_GPU=0 QWEN_GPU=1 \
    LAB_PORT=8765 QWEN_PORT=8081

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl ca-certificates pkg-config ninja-build build-essential cmake \
    libjpeg-dev libgl1 libglib2.0-0 libegl1 libxrender1 libsm6 libxext6 \
    ffmpeg sudo libomp-dev libgomp1 libgl1-mesa-dev unzip \
    && rm -rf /var/lib/apt/lists/*
RUN git lfs install --system && python -m pip install --upgrade pip wheel setuptools

WORKDIR /opt/lab
COPY . /opt/lab
RUN python -m pip install -r requirements-runtime.txt

# Immutable upstream source refs recorded in config/revisions.json.
RUN bash /opt/lab/scripts/clone_sources.sh

# Alchemy3D's official CUDA dependencies are built into the IMAGE, not on GPU rental.
# This step is intentionally strict: if CUDA compilation fails, the image does not publish.
RUN if [ "$INSTALL_FULL_EXTENSIONS" = "1" ]; then \
      cd /workspace/repos/Alchemy3D && \
      bash -lc '. ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --flexgemm --blender' && \
      python -m pip install /workspace/repos/TRELLIS.2/o-voxel --no-build-isolation && \
      touch /opt/lab/.cuda-extensions-built; \
    else \
      echo "WARN: Image built without CUDA extensions (diagnostics only)"; \
    fi

# Pin llama.cpp commit to include the current Qwen3.8 architecture and CUDA server.
RUN if [ "$BUILD_LLAMA" = "1" ]; then \
      git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp && \
      git -C /opt/llama.cpp checkout --detach "$LLAMA_REF" && \
      cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release && \
      cmake --build /opt/llama.cpp/build --target llama-server -j 2 && \
      ln -s /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server; \
    fi

RUN mkdir -p /workspace/weights /workspace/results /workspace/logs /workspace/qwen /workspace/.hf \
    && chmod +x /opt/lab/scripts/*.sh

EXPOSE 8765
WORKDIR /workspace
ENTRYPOINT ["/opt/lab/scripts/start_dual_gpu.sh"]
