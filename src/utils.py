import docker
import os
from typing import Literal


def run_docker_container(image_name: Literal["dmrst", "dplp"]) -> str:
    """Run a Docker container with the specified image name.

    Args:
        image_name (str): The name of the Docker image to run. Either "dmrst" or "dplp".

    Returns:
        str: The name of the running container.
    """
    client = docker.from_env()
    # check if the container is already running
    if image_name in [c.name for c in client.containers.list()]:
        return image_name

    if image_name == "dmrst":
        image = f"psandhaas/{image_name}-parser:latest"
        ports = {"8000/tcp": 8000}
        container = client.containers.run(
            image=image,
            name=image_name,
            ports=ports,
            detach=True
        )
        return container.name
    elif image_name == "dplp":
        image = f"mohamadisara20/{image_name}-env:ger"
        ports = {"5000/tcp": 5000}
        container = client.containers.run(
            image=image,
            name=image_name,
            ports={"5000/tcp": 5000},
            volumes={
                r"C:/Users/Philipp/Repos/DPLP-German": {"bind": "/home/DPLP", "mode": "rw"}
            },
            working_dir="/home/DPLP",
            command="python3 ger_rest_api.py",
            detach=True
        )
    else:
        raise ValueError(
            "Invalid image name. Must be one of: 'dmrst', 'dplp'."
        )


if __name__ == "__main__":
    from pprint import pprint
    import requests

    run_docker_container("dplp")
    resp = requests.post(
        "http://localhost:5000/dplp",
        json={"text": "Some text to parse"},
        headers={"Content-Type": "application/json"}
    )
    pprint(resp.json(), sort_dicts=False)
