# paddle
Python Atmospheric Dynamics: Discovering and Learning about Exoplanets. An open-source, user-friendly python version of canoe.

## Install docker and docker-compose plugin
1. install docker with compose
```bash
curl -fsSL https://get.docker.com | sudo sh
```

1. start docker
sudo systemctl start docker

## Test package
1. Create a python virtual environment
```bash
python -m venv pyenv
```

2. Install paddle package
```bash
pip install paddle
```

3. Run test
```bash
cd tests
python test_saturn_adiabat.py
```

## Docker user guide

2. Create a docker container
```bash
make up
```

3. Terminate a docker container
```bash
make down
```

4. Start a docker container
```bash
make start
```

5. Build a new docker image (rarely used)
```bash
make build
```

## For Developer

### Clone repository
```bash
https://github.com/elijah-mullens/paddle
```

### Cache your github credential
```bash
git config credential.helper 'cache --timeout=86400'
```

### Create a python virtual environment
```bash
python -m venv .pyenv
```

### Install paddle package

### Install pre-commit hook
```bash
pip install pre-commit
```

### Install pre-commit hook
```bash
pre-commit install
```

## Troubleshooting
1. If you do not have docker compose
Remove your current docker installation, it could be docker or docker.io
