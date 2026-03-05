# Configure docker environment with GPU support

## Redhat

### Install NVIDIA Container Toolkit

NVIDIA Container Toolkit enables GPU support in Docker containers.

1. Configure the production repository
```
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
```

2. Install the NVIDIA Container Toolkit packages
```
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.2-1
  sudo dnf install -y \
      nvidia-container-toolkit-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1-${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

3. Let docker use NVIDIA's **containerd** runtime
```
sudo nvidia-ctk runtime configure --runtime=containerd
```

4. Restart containerd servier
```
sudo systemctl restart containerd
```

## Ubuntu

### Install NVIDIA Container Toolkit

1. Add NVIDIA’s GPG key
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

2. Add the apt repo (stable)
```
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
```

3. Install
```
sudo apt-get install -y nvidia-container-toolkit
```

4. Configure docker to use NVIDIA's runtime
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
