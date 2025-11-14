# syntax=docker/dockerfile:1.7
ARG CUDA_VER=12.9.1
ARG UBUNTU_VER=22.04

############################
# Base CUDA toolchain
############################
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu${UBUNTU_VER} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# System deps (C++ toolchain, Python, venv, git, neovim, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
      gosu tini \
      build-essential \
      gcc g++ gfortran \
      ninja-build \
      cmake \
      git curl ca-certificates \
      python3 python3-pip python3-venv python3-dev \
      pkg-config \
      libssl-dev \
      ripgrep fd-find unzip neovim \
      clangd \
      nodejs npm \
      && rm -rf /var/lib/apt/lists/*

# Make fd available as `fd` (Ubuntu calls it fdfind)
RUN update-alternatives --install /usr/bin/fd fd /usr/bin/fdfind 50

# Default editor env
ENV EDITOR=nvim VISUAL=nvim

# System-wide git editor default (users can override)
RUN git config --system core.editor "nvim"

# Pre-warm plugins (optional, speeds first start)
# This runs Neovim headless to install plugins via lazy.nvim.
RUN nvim --headless "+Lazy! sync" +qa || true

############################
# Convenient command-line tools
############################

# ---- after base tooling is installed, before switching USER ----
# Create a tools dir and put it on PATH
RUN mkdir -p /opt/tools
ENV PATH="/opt/tools:${PATH}"

# Copy all helper scripts
COPY docker/scripts/ /opt/tools/
# Ensure executables
RUN find /opt/tools -type f -name "*.sh" -exec chmod +x {} \; \
    && for f in /opt/tools/*.sh; do ln -sf "$f" "/usr/local/bin/$(basename "${f%.sh}")"; done
# The symlink makes command-line tools available (no .sh needed)

############################
# Configure non-root user and Python venv
############################

# Python venv in /opt/venv (global, fast, easy)
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip/setuptools/wheel early
RUN pip install --upgrade pip setuptools wheel build

# Optional: pin pip resolver behavior and set default index/extra-index if needed
# COPY docker/pip.conf /etc/pip.conf

# Put a base Neovim config in /etc/skel so future users get it,
# and also install for the existing dev user.
#RUN mkdir -p /etc/skel/.config/nvim
#COPY docker/nvim/ /etc/skel/.config/nvim/
#RUN mkdir -p /home/${USERNAME}/.config/nvim && \
#    cp -r /etc/skel/.config/nvim/* /home/${USERNAME}/.config/nvim/ && \
#    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.config/nvim

############################
# Cache Python wheels separately (optional but recommended)
############################
FROM base AS wheels
WORKDIR /tmp/wheels
# If you use requirements.txt:
COPY docker/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel -r requirements.txt -w /tmp/wheelhouse

# If you use Poetry/pyproject, replace the above with:
# COPY pyproject.toml poetry.lock ./
# RUN pip install "poetry>=1.8.0"
# RUN --mount=type=cache,target=/root/.cache/pip poetry export -f requirements.txt --without-hashes | tee req.txt
# RUN --mount=type=cache,target=/root/.cache/pip pip wheel -r req.txt -w /tmp/wheelhouse

############################
# Dev image
############################
FROM base AS dev
WORKDIR /workspace

# Load for login shells
#RUN printf 'alias vi=nvim\nalias vim=nvim\nexport TERM=xterm-256color\n' >> /home/${USERNAME}/.bashrc

# Bring in prebuilt wheels (fast installs)
COPY --from=wheels /tmp/wheelhouse /tmp/wheelhouse
COPY docker/requirements.txt /workspace/docker/requirements.txt

# Install Python deps from wheels first, then fall back to index
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/tmp/wheelhouse -r /workspace/docker/requirements.txt \
 || pip install -r /workspace/docker/requirements.txt

# (Optional) CUDA env defaults for many toolchains
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX"  \
    CUDA_CACHE_MAXSIZE=2147483647

# Put your entrypoint in place
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Keep the workspace owned by the non-root user
#RUN chown -R ${USERNAME}:${USERNAME} /workspace
#USER ${USERNAME}

#ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
#CMD ["/bin/bash"]

ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
