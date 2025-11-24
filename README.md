# Paddle

Python Atmospheric Dynamics: Discovering and Learning about Exoplanets. An open-source, user-friendly Python version of [canoe](https://github.com/chengcli/canoe).

## Test the package

Testing the package can be done very easily by following the steps below. All we need is to create a python virtual environment, install the package, and run the test script.

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

## Develop with Docker

You may need to install Docker to compose up and install the package inside a container if your device or operating system does not support certain dependencies. Follow the instructions below to install docker and docker-compose plugin.

1. Install docker with compose

   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   ```

2. Start docker using the command below or open docker desktop if applicable.

   ```bash
   sudo systemctl start docker
   ```

After installing Docker, you can use the Makefile commands below to manage your docker containers from the terminal. By default, the container will mount the current directory to `/paddle` inside the container.

> Mounting a local directory allows you to edit files on your local machine while running and testing the code inside the container; or use the container as a development environment and sync files to your local machine.

If you want to change the mounted directory, you can create a `docker-compose.overrides.yml` file based on the provided `docker-compose.overrides.yml.tmp` template file.

- Create a docker container

  ```bash
  make up
  ```

- Start a docker container (only if previously created)

  ```bash
  make start
  ```

- Terminate a docker container

  ```bash
  make down
  ```

- Build a new docker image (rarely used)

  ```bash
  make build
  ```

If you use VSCode, it is recommended to install extensions including [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack), [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) and [Container Tools](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers) for a better development experience within a container.

1. Install the extensions mentioned above.
2. Start the container using `make up` or `make start`.
3. Open VSCode and click on the "Containers" icon on the left sidebar. Note that if you have too many extensions installed, the icon may be hidden under the "..." menu.

   <img src="docs/_static/readme-extension.png" width="200" style="vertical-align: top;">
   <img src="docs/_static/readme-attach.png" width="200" style="vertical-align: top;">

4. Right click on the running container named `paddle` with a green triangle icon (indicating it is running), and select "Attach Visual Studio Code" (see above images). This will open a new VSCode window connected to the container.
5. Open either the default folder `paddle` mounted from your local machine, or any custom workspace folder you have set up inside the `docker-compose.overrides.yml` file. Now you can start developing inside the container as if you were working on your local machine!

   <img src="docs/_static/readme-open-folder.png" width="500" style="vertical-align: top;">

## For Developers

Follow the steps below to set up your development environment.

1. Clone the repository

   ```bash
   https://github.com/elijah-mullens/paddle
   ```

2. Cache your github credential. This will prevent you from being prompted for your username and password every time you push changes to github.

   ```bash
   git config credential.helper 'cache --timeout=86400'
   ```

3. Create a python virtual environment.

   ```bash
   python -m venv .pyenv
   ```

4. Install paddle package

   ```bash
   # Install the package normally
   pip install paddle

   # [Alternatively] if you want to install in editable mode
   pip install -e .
   ```

5. Install pre-commit hook. This will automatically format your code before each commit to ensure consistent code style.

   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Troubleshooting

1. If you have Docker installed but do not have Docker Compose, remove your current Docker installation, which could be docker or docker.io, and re-install it following the guide provided in the [Develop with Docker](#develop-with-docker) section above.
