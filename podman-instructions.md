# Running HoneInspecror with Podman

These instructions help you build and run the HoneInspecror application using Podman containers. Podman is a drop-in replacement for Docker and can use the same Dockerfile and commands.

## Prerequisites

- [Podman installed](https://podman.io/getting-started/installation)
- (Optional) [Podman Compose](https://github.com/containers/podman-compose) if you wish to use `docker-compose.yml` (not present by default in this repo)
- Clone this repository:
  ```sh
  git clone https://github.com/chandrasmailbox/HoneInspecror.git
  cd HoneInspecror
  ```

## 1. Build the Container Image

The project already contains a `Dockerfile` compatible with Podman.

```sh
podman build -t honeinspecror .
```

- This builds the application as defined in the multi-stage `Dockerfile` in the root of the repo.
- The final stage bundles the frontend (React), backend (Python), and configures Nginx as a reverse proxy.

## 2. Set Environment Variables

The application may require some environment variables (see `entrypoint.sh` for required variables):

- `run_id`
- `code_server_password`
- `preview_endpoint`
- `base_url`
- `monitor_polling_interval`

Export these in your shell or supply them inline to `podman run`:

```sh
export run_id=your_run_id
export code_server_password=your_password
export preview_endpoint=http://localhost:8001
export base_url=http://localhost:8001
export monitor_polling_interval=60
```

## 3. Run the Container

```sh
podman run --rm -it \
  -e run_id=$run_id \
  -e code_server_password=$code_server_password \
  -e preview_endpoint=$preview_endpoint \
  -e base_url=$base_url \
  -e monitor_polling_interval=$monitor_polling_interval \
  -p 80:80 -p 8001:8001 \
  honeinspecror
```

- This will start the application and expose necessary ports for frontend (80) and backend (8001).
- Adjust ports if needed for your environment.

## 4. Access the Application

- Visit: [http://localhost](http://localhost) for the frontend UI.
- Backend APIs are available at [http://localhost:8001](http://localhost:8001)

## Notes

- Podman runs containers rootless by default; if you encounter permission issues with ports below 1024, consider using a higher port or run with `sudo`.
- If you need volume mounts (for persistent data, logs, etc.), add `-v` options as needed.
- For multi-container setups (e.g., adding MongoDB), you can use Podman Compose and a `docker-compose.yml` file (not included by default).

---

**For any customizations, refer to the environment variable checks in `entrypoint.sh` to ensure the container starts up successfully.**
