# Development

### Prerequisites
- Python <3.13,>=3.11
- [pdm](https://pdm-project.org)
- Docker
- A .env file in the root environment with the requisite values needed to start your workload.

### Basic Workflow
This project has optional dependency groups in its `pyproject.toml` for various distributions of the source code, including a cluster-baked image and a jetstreamer that pushes events to one or more source queues. In order to use this codebase in a dockerized dev environment, use [`Dockerfile.dev`](../Dockerfile.dev) and your local virtual environmen using the following steps:

1. After checking out the repo, sync your virutal environment from the default lockfile using the following command:

```sh
pdm sync -G :all --no-self --no-isolation --fail-fast
```

2. If your packages are synced successfully, from the project root you can run `docker compose up`. This will build the Dockerfile.dev file and start it.

3. In a separate terminal session, you can exec into the container with a shell `docker exec -ti grazer sh`

4. Inside the container shell install your environment again by running the same command
```sh
pdm sync -G :all --no-self -no-self --no-isolation --fail-fast
```

5. The project root from your local filesystem is mounted in the container. You can run commands inside the container shell as you would locally (as pdm commands to activate your venv) ie

```sh
pdm start_streamer
```
