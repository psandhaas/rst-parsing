from bs4 import BeautifulSoup
import docker
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Literal, Union


def ensure_docker_desktop_running(
    timeout: int = 60,
    docker_desktop_path: str = "C:/Program Files/Docker/Docker/Docker Desktop.exe"  # noqa
) -> None:
    """Checks if Docker Desktop is running on Windows and starts it if not.

    Args:
        timeout (int): Maximum seconds to wait for Docker to become available.
        docker_desktop_path (str): Path to the Docker Desktop executable.
    Raises:
        RuntimeError: If Docker Desktop could not be started or is not
        available.
    """
    import docker
    try:
        # Try to ping Docker daemon
        client = docker.from_env()
        client.ping()
        return
    except Exception:  # Docker not running
        pass

    try:  # Try to start Docker Desktop
        subprocess.Popen([docker_desktop_path], shell=True)
    except FileNotFoundError:
        raise RuntimeError(
            "Docker Desktop executable not found at expected path."
        )

    # Wait for Docker daemon to become available
    for _ in range(timeout):
        try:
            client = docker.from_env()
            client.ping()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Docker Desktop did not start within the timeout period.")


def run_docker_container(image_name: Literal["dmrst", "dplp"]) -> str:
    """Run a Docker container with the specified image name.

    Args:
        image_name (str): The name of the Docker image to run. Either "dmrst" or "dplp".

    Returns:
        str: The name of the running container.
    """
    ensure_docker_desktop_running()
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


def stop_and_rm_container(container_name: str) -> None:
    """Stop and remove a Docker container by its name.

    Args:
        container_name (str): The name of the Docker container to stop and remove.
    """
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        container.stop()
        container.remove()
    except docker.errors.NotFound:
        print(f"Container '{container_name}' not found.")


def write_to_json(
    res: dict,
    file_path: str,
    overwrite: bool = False
) -> None:
    """Save the parse result to a JSON file.

    Args:
        res (dict): The parse result to save.
        file_path (str): The path to the file where the result should be saved.
        overwrite (bool): Whether to overwrite the file if it already exists.
    """
    if not overwrite and os.path.exists(file_path):
        print(f"File '{file_path}' already exists. Skipping overwrite.")
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=4, default=str)
    except Exception as e:
        print(
            f"Encountered an error writing '{file_path}'.\n"
            + f"Error: {e}\n"
            + "Skipping write."
        )
        return


def load_texts() -> dict[str, list[str]]:
    """Returns a dictionary of raw texts as lists of sentences, keyed by their
    files' base-names."""
    texts = {}
    for p in Path.joinpath(
        Path(os.getcwd()).parent, Path("data/texts/")
    ).glob("*.txt"):
        with open(p, "r", encoding="utf-8") as f:
            texts[p.stem] = [
                l.strip() for l in f.readlines() if len(l.strip()) > 0
            ]
    return texts


def load_gold_annotations() -> dict[str, str]:
    """Returns a dictionary of gold annotations as strings (.rs3 XML-format),
    keyed by their files' base-names."""
    annotations = {}
    for p in Path.joinpath(
        Path(os.getcwd()).parent, Path("data/gold/")
    ).glob("*.rs3"):
        with open(p, "r", encoding="utf-8") as f:
            annotations[p.stem] = f.read().strip()
    return annotations


def make_rs3() -> str:
    soup = BeautifulSoup(features="xml")
    soup.append(soup.new_tag("rst"))
    soup.rst.extend([soup.new_tag("header"), soup.new_tag("body")])

    relations = soup.header.append(soup.new_tag("relations"))
    for rel in [
        {'name': 'antithesis', 'type': 'rst'},
        {'name': 'attribution', 'type': 'rst'},
        {'name': 'background', 'type': 'rst'},
        {'name': 'cause', 'type': 'rst'},
        {'name': 'circumstance', 'type': 'rst'},
        {'name': 'concession', 'type': 'rst'},
        {'name': 'condition', 'type': 'rst'},
        {'name': 'conjunction', 'type': 'multinuc'},
        {'name': 'contrast', 'type': 'multinuc'},
        {'name': 'e-elaboration', 'type': 'rst'},
        {'name': 'elaboration', 'type': 'rst'},
        {'name': 'enablement', 'type': 'rst'},
        {'name': 'evaluation-N', 'type': 'rst'},
        {'name': 'evaluation-S', 'type': 'rst'},
        {'name': 'evidence', 'type': 'rst'},
        {'name': 'interpretation', 'type': 'rst'},
        {'name': 'joint', 'type': 'multinuc'},
        {'name': 'justify', 'type': 'rst'},
        {'name': 'list', 'type': 'multinuc'},
        {'name': 'motivation', 'type': 'rst'},
        {'name': 'otherwise', 'type': 'rst'},
        {'name': 'preparation', 'type': 'rst'},
        {'name': 'purpose', 'type': 'rst'},
        {'name': 'reason-N', 'type': 'rst'},
        {'name': 'reason', 'type': 'rst'},
        {'name': 'restatement', 'type': 'rst'},
        {'name': 'result', 'type': 'rst'},
        {'name': 'sameunit', 'type': 'multinuc'},
        {'name': 'sequence', 'type': 'multinuc'},
        {'name': 'solutionhood', 'type': 'rst'},
        {'name': 'summary', 'type': 'rst'}
    ]:
        rel_tag = soup.new_tag("rel")
        for k, v in rel.items():
            rel_tag[k] = v
        relations.append(rel_tag)

    # TODO: populate body
    # <body>
	# 	<segment id="1" >Ministerium bestätigte Omikron-Fall in Österreich</segment>
	# 	<segment id="2" parent="35" relname="span">Die Coronavirus-Variante Omikron ist offiziell in Österreich angekommen:</segment>
    #     ...
    #     <group id="23" type="span" parent="36" relname="span"/>
    #     ...
	# 	<group id="43" type="span" parent="23" relname="elaboration"/>
	# 	<group id="44" type="span" />
	# </body>

    return str(soup.prettify())


if __name__ == "__main__":
    from pprint import pprint
    import requests

    # run_docker_container("dplp")
    # resp = requests.post(
    #     "http://localhost:5000/dplp",
    #     json={"text": "Some text to parse"},
    #     headers={"Content-Type": "application/json"}
    # )
    # pprint(resp.json(), sort_dicts=False)
