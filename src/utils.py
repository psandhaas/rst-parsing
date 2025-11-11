#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Mon, 14.09.25                                                     #
# ========================================================================== #

"""Utilities for RST parsing using dockerized parsers."""

import boto3
from bs4 import BeautifulSoup, SoupStrainer
import docker
from dotenv import load_dotenv
from glob import glob
from IPython import get_ipython
import json
from langchain_aws import ChatBedrockConverse
from langchain_openai import AzureChatOpenAI
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Literal, Optional, Union

from tree import Node


def ensure_docker_desktop_running(
    timeout: int = 60,
    docker_desktop_path: str = "C:/Program Files/Docker/Docker/Docker Desktop.exe",  # noqa
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
        raise RuntimeError("Docker Desktop executable not found at expected path.")

    # Wait for Docker daemon to become available
    for _ in range(timeout):
        try:
            client = docker.from_env()
            client.ping()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Docker Desktop did not start within the timeout period.")


def wait_for_container(container, timeout: int = 60) -> None:
    """Waits for a Docker container to be in 'running' state.

    Args:
        container: The Docker container object.
        timeout (int): Maximum seconds to wait for the container to start.

    Raises:
        RuntimeError: If the container does not start within the timeout period.
    """
    for _ in range(timeout):
        container.reload()
        if container.status == "running":
            return
        time.sleep(1)
    raise RuntimeError("Container did not start within the timeout period.")


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
            image=image, name=image_name, ports=ports, detach=True
        )
        wait_for_container(container)
        return container.name
    elif image_name == "dplp":
        image = f"mohamadisara20/{image_name}-env:ger"
        ports = {"5000/tcp": 5000}
        container = client.containers.run(
            image=image,
            name=image_name,
            ports={"5000/tcp": 5000},
            volumes={
                r"C:/Users/SANDHAP/Repos/DPLP-German": {
                    "bind": "/home/DPLP",
                    "mode": "rw",
                }
            },
            working_dir="/home/DPLP",
            command="python3 ger_rest_api.py",
            detach=True,
        )
        wait_for_container(container)
        return container.name
    else:
        raise ValueError("Invalid image name. Must be one of: 'dmrst', 'dplp'.")


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


def write_to_json(res: dict, file_path: str, overwrite: bool = False) -> None:
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


def load_gold_annotations() -> dict[str, str]:
    """Returns a dictionary of gold annotations as strings (.rs3 XML-format),
    keyed by their files' base-names."""
    annotations = {}
    for p in Path.joinpath(Path(os.getcwd()).parent, Path("data/gold/")).glob("*.rs3"):
        with open(p, "r", encoding="utf-8") as f:
            annotations[p.stem] = f.read().strip()
    return annotations


def get_relations_set(rs3: str, lower: bool = True) -> set[tuple[str, str]]:
    rels = set()
    soup = BeautifulSoup(rs3, "xml", parse_only=SoupStrainer("rel"))
    for rel in soup.find_all(True):
        if lower:
            rels.add(tuple([a.lower() for a in rel.attrs.values()]))
        else:
            rels.add(tuple(rel.attrs.values()))
    return rels


def map_fine2coarse(relation: str, replace_unknown: bool = True) -> Optional[str]:
    """Maps a fine-grained relation label (either from DPLP or DMRST) to
    a coarse-grained one that's compatible with the evaluation script by
    Joty, 2011 (https://github.com/mohamadi-sara20/DPLP-German/tree/master/parsing_eval_metrics).
    """
    if relation is None:
        return None

    # mapping of cross-lingual RST-relations following Braud et al., 2017
    # (https://doi.org/10.48550/arXiv.1701.02946) and Carlson et al., 2001
    general_mapping = {
        # taken from github.com/seq-to-mind/DMRST_Parser/blob/main/Preprocess_RST_Data/1_uniform_treebanks/code/src/relationSet.py#L11
        "ahalbideratzea": "Enablement",
        "alderantzizko-baldintza": "Condition",
        "alternativa": "Condition",
        "analogy": "Comparison",
        "antitesia": "Contrast",
        "antithesis": "Contrast",
        "antítesis": "Contrast",
        "arazo-soluzioa": "Topic-Comment",
        "attribution": "Attribution",
        "attribution-negative": "Attribution",
        "aukera": "Condition",
        "background": "Background",
        "baldintza": "Condition",
        "bateratzea": "Joint",
        "birformulazioa": "Summary",
        "capacitación": "Enablement",
        "causa": "Cause",
        "cause": "Cause",
        "cause-result": "Cause",
        "circumstance": "Background",
        "circunstancia": "Background",
        "comment": "Evaluation",
        "comment-topic": "Topic-Comment",
        "comparison": "Comparison",
        "concesión": "Contrast",
        "concession": "Contrast",
        "conclusion": "Evaluation",
        "condición": "Condition",
        "condición-inversa": "Condition",
        "condition": "Condition",
        "conjunción": "Joint",
        "conjunction": "Joint",
        "consequence": "Cause",
        "contingency": "Condition",
        "contrast": "Contrast",
        "contraste": "Contrast",
        "definition": "Elaboration",
        "definitu-gabeko-erlazioa": "Summary",
        "disjunction": "Joint",
        "disjuntzioa": "Joint",
        "disyunción": "Joint",
        "e-elaboration": "Elaboration",
        "ebaluazioa": "Evaluation",
        "ebidentzia": "Explanation",
        "elaboración": "Elaboration",
        "elaboration": "Elaboration",
        "elaboration-additional": "Elaboration",
        "elaboration-general-specific": "Elaboration",
        "elaboration-object-attribute": "Elaboration",
        "elaboration-part-whole": "Elaboration",
        "elaboration-process-step": "Elaboration",
        "elaboration-set-member": "Elaboration",
        "elaborazioa": "Elaboration",
        "enablement": "Enablement",
        "evaluación": "Evaluation",
        "evaluation": "Evaluation",
        "evidence": "Explanation",
        "evidencia": "Explanation",
        "example": "Elaboration",
        "explanation": "Explanation",
        "explanation-argumentative": "Explanation",
        "ez-baldintzatzailea": "Condition",
        "fondo": "Background",
        "helburua": "Enablement",
        "hypothetical": "Condition",
        "interpretación": "Evaluation",
        "interpretation": "Evaluation",
        "interpretazioa": "Evaluation",
        "inverted-sequence": "Temporal",
        "joint": "Joint",
        "justificación": "Explanation",
        "justifikazioa": "Explanation",
        "justify": "Explanation",
        "kausa": "Cause",
        "konjuntzioa": "Joint",
        "kontrastea": "Contrast",
        "kontzesioa": "Contrast",
        "laburpena": "Summary",
        "list": "Joint",
        "lista": "Joint",
        "manner": "Manner-Means",
        "means": "Manner-Means",
        "medio": "Manner-Means",
        "metodoa": "Manner-Means",
        "motibazioa": "Explanation",
        "motivación": "Explanation",
        "motivation": "Explanation",
        "non-volitional-cause": "Cause",
        "non-volitional-result": "Cause",
        "nonvolitional-cause": "Cause",
        "nonvolitional-result": "Cause",
        "ondorioa": "Cause",
        "otherwise": "Condition",
        "parenthetical": "Elaboration",
        "preference": "Comparison",
        "preparación": "Background",
        "preparation": "Background",
        "prestatzea": "Background",
        "problem-solution": "Topic-Comment",
        "proportion": "Comparison",
        "propósito": "Enablement",
        "purpose": "Enablement",
        "question-answer": "Topic-Comment",
        "reason": "Explanation",
        "reformulación": "Summary",
        "restatement": "Summary",
        "restatement-mn": "Summary",
        "result": "Cause",
        "resultado": "Cause",
        "resumen": "Summary",
        "rhetorical-question": "Topic-Comment",
        "same-unit": "Same-unit",
        "sameunit": "Same-unit",  # added
        "secuencia": "Temporal",
        "sekuentzia": "Temporal",
        "sequence": "Temporal",
        "solución": "Topic-Comment",
        "solutionhood": "Topic-Comment",
        "statement-response": "Topic-Comment",
        "summary": "Summary",
        "temporal-after": "Temporal",
        "temporal-before": "Temporal",
        "temporal-same-time": "Temporal",
        "testuingurua": "Background",
        "textual-organization": "Textual-organization",
        "textualorganization": "Textual-organization",
        "topic-comment": "Topic-Comment",
        "topic-drift": "Topic-Change",
        "topic-shift": "Topic-Change",
        "unconditional": "Condition",
        "unión": "Joint",
        "unless": "Condition",
        "volitional-cause": "Cause",
        "volitional-result": "Cause",
        "zirkunstantzia": "Background",
        "question": "Topic-Comment",
    }
    other2joty = {
        # taken from github.com/mohamadi-sara20/DPLP-German/blob/master/parsing_eval_metrics/RelationClasses.txt
        "attribution": "Attribution",
        "background": "Background",
        "cause": "Cause",
        "causemult": "Cause",
        "comparison": "Comparison",
        "comparisonmult": "Comparison",
        "condition": "Condition",
        "conditionmult": "Condition",
        "contrast": "Contrast",
        "contrastmult": "Contrast",
        "dummy": "Dummy",
        "elaboration": "Elaboration",
        "enablement": "Enablement",
        "evaluation": "Evaluation",
        "evaluationmult": "Evaluation",
        "explanation": "Explanation",
        "explanationmult": "Explanation",
        "jointmult": "Joint",
        "manner-means": "Manner-Means",
        "same": "Same-Unit",
        "span": "span",
        "summary": "Summary",
        "temporal": "Temporal",
        "temporalmult": "Temporal",
        "textualorganization": "TextualOrganization",
        "topichange": "Topic-Change",
        "topichangemult": "Topic-Change",
        "topicomment": "Topic-Comment",
        "topicommentmult": "Topic-Comment",
        "topidriftmult": "Topic-Change",
        "virtual": "Dummy",
        "virtual-root": "Dummy",
    }
    all2joty = {
        "same-unit": "Same-Unit",
        "Same-unit": "Same-Unit",
        "Same-unit": "Same-Unit",
        "textual-organization": "TextualOrganization",
        "Textual-organization": "TextualOrganization",
        "Textual-Organization": "TextualOrganization",
        # Rest are identical:
        # 'Attribution',
        # 'Background',
        # 'Cause',
        # 'Comparison',
        # 'Condition',
        # 'Contrast',
        # 'Elaboration',
        # 'Enablement',
        # 'Evaluation',
        # 'Explanation',
        # 'Joint',
        # 'Manner-Means',
        # 'Summary',
        # 'Temporal',
        # 'Topic-Change',
        # 'Topic-Comment'
    }
    all2joty |= other2joty

    def try_strip_suffix(relation: str) -> str:
        if (
            "-" in relation
            and len(splt := relation.split("-"))  # try to strip nuclearity suffixes
            == 2
            and splt[-1].strip().lower() in ["s", "n", "mn"]
        ):
            return splt[0].strip()
        return relation

    def map_to_general(relation: str) -> str:
        if relation in general_mapping:
            return general_mapping[relation]
        elif relation.lower() in general_mapping:
            return general_mapping[relation.lower()]
        elif (stripped := try_strip_suffix(relation)) in general_mapping:
            return general_mapping[stripped]
        elif (stripped.lower()) in general_mapping:
            return general_mapping[stripped.lower()]
        return relation

    def map_to_joty(relation: str) -> str:
        if relation in all2joty:
            return all2joty[relation]
        elif relation.lower() in all2joty:
            return all2joty[relation.lower()]
        elif (stripped := try_strip_suffix(relation)) in all2joty:
            return all2joty[stripped]
        elif (stripped.lower()) in all2joty:
            return all2joty[stripped.lower()]
        return relation

    rel = map_to_general(relation)
    coarse = map_to_joty(rel)
    if coarse == relation and replace_unknown:
        coarse = "unknown"
    return coarse


def build_relations_map(
    annotations_dir: str, replace_unknown: bool = True
) -> dict[str, dict[str, list[str] | str]]:
    """Builds a mapping of all unique relations found across all .rs3 files in
    the given directory, where fine-grained relations are mapped to
    coarse-grained ones."""
    relations_set = set()
    for p in glob(f"{annotations_dir}/*.rs3"):
        with open(p, "r", encoding="utf-8") as f:
            rs3 = f.read()
        relations_set.update(get_relations_set(rs3))

    fine2coarse = {
        (rel[0], rel[1]): map_fine2coarse(rel[0], replace_unknown=replace_unknown)
        for rel in relations_set
    }

    mapped = {coarse: {"aliases": [], "type": None} for coarse in fine2coarse.values()}
    for (fine, rel_type), coarse in fine2coarse.items():
        if fine not in mapped[coarse]["aliases"]:
            mapped[coarse]["aliases"].append(fine)
        mapped[coarse]["type"] = rel_type

    return mapped


def load_env_vars():
    """Load environment variables from a .env file.

    :raises ValueError: If any of `'AZURE_OPENAI_BASE_URL'`,
                        `'AZURE_OPENAI_API_KEY'`, or
                        `'AZURE_OPENAI_API_VERSION'` is missing.
    """

    def in_notebook() -> bool:
        try:
            shell = get_ipython().__class__.__name__
            return shell == "ZMQInteractiveShell"
        except Exception:
            return False

    if in_notebook():
        if (ipython := get_ipython()) is not None:
            ipython.run_line_magic("load_ext", "dotenv")
            ipython.run_line_magic("dotenv", "")
    else:
        load_dotenv()
    for env_var in [
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
    ]:
        if os.environ.get(env_var) is None:
            raise ValueError(f"{env_var} not found in environment variables.")


def load_texts(texts_dir: str = None) -> Dict[str, List[str]]:
    """Returns a dictionary of raw texts as lists of sentences, keyed by their
    files' base-names."""
    texts = {}
    if texts_dir is None:
        files = Path.joinpath(Path(os.getcwd()).parent, Path("data/texts/")).glob(
            "*.txt"
        )
    else:
        files = glob(f"{texts_dir}/*.txt")
    for path in (Path(p) for p in files):
        with open(path, "r", encoding="utf-8") as f:
            texts[path.stem] = [line for line in f if len(line.strip()) > 0]
    return texts


def load_rs3(
    dir: str,
    read_as: Literal["node", "soup", "string"] = "node",
    exclude_disjunct_segments: bool = False,
) -> Dict[str, Union[Node, BeautifulSoup, str]]:
    if not os.path.isdir(dir):
        raise FileNotFoundError(f"'{dir}' is not a directory path.")
    if len(paths := [Path(p) for p in glob(f"{dir}/*.rs3")]) == 0:
        raise FileNotFoundError(f"No .rs3-files found in directory '{dir}'.")
    res = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                rs3 = f.read()
        except Exception as e:
            print(f"Encountered an error trying to read '{p.absolute}'.")
            raise e
        if read_as == "node":
            try:
                res[p.stem] = Node.from_xml(
                    rs3_str=rs3, exclude_disjunct_segments=exclude_disjunct_segments
                )
            except Exception as e:
                print(f"Encountered an error trying to read '{p.stem}' as Node.")
                raise e
        elif read_as == "soup":
            try:
                res[p.stem] = BeautifulSoup(rs3, features="xml")
            except Exception as e:
                print(f"Encountered an error trying to read '{p.stem}' as XML.")
                raise e
        else:
            res[p.stem] = rs3
    return res


def parse_write_rs3(
    parser, texts_dir: str, out_dir: str, **kwargs
) -> Dict[str, List[Dict]]:
    res = {
        k: parser.parse(text="".join(v), **kwargs)
        for k, v in load_texts(texts_dir).items()
    }
    for k, v in res.items():
        with open(f"{out_dir}/{k}_dmrst.rs3", "w", encoding="utf-8") as f:
            f.write(v["parsed"][0]["rs3"])
    return res


def _get_aws_bedrock_client():
    load_env_vars()
    for env_var in ["AWS_API_KEY", "ANTHROPIC_BASE_URL", "AWS_REGION_NAME"]:
        if os.environ.get(env_var) is None:
            raise ValueError(f"{env_var} not found in environment variables.")
    boto_session = boto3.Session(
        aws_access_key_id=os.environ["AWS_API_KEY"],
        aws_secret_access_key=os.environ["AWS_API_KEY"],
        aws_session_token=os.environ["AWS_API_KEY"],
    )
    bedrock_client = boto_session.client(
        service_name="bedrock-runtime",
        endpoint_url=os.environ["ANTHROPIC_BASE_URL"],
        region_name=os.environ["AWS_REGION_NAME"],
    )

    # API-Key needs to be appended via event hook, as AWS Auth does not
    # directly support API Key authentication.
    def _add_api_key(request, operation_name, **kwargs):
        request.headers["api-key"] = os.environ["AWS_API_KEY"]

    bedrock_client.meta.events.register("request-created.bedrock-runtime", _add_api_key)
    return bedrock_client


def _init_llm(
    model: Literal[
        "gpt-4.1",
        "gpt-4o",
        "o4-mini",
        "claude-sonnet-4",
        "claude-3-7-sonnet",
        "claude-3-5-sonnet",
        "claude-3-sonnet",
    ] = "gpt-4.1",
) -> Union[AzureChatOpenAI, ChatBedrockConverse]:
    load_env_vars()
    if model in ["gpt-4.1", "gpt-4o", "o4-mini"]:
        if model == "o4-mini":
            api_version = "2025-02-01-preview"
        else:
            api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        return AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=api_version,
            base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
            model=model,
        )
    else:
        if model == "claude-sonnet-4":
            deployment_name = os.getenv("CLAUDE_4_DEPLOYMENT_NAME")
        elif model == "claude-3-7-sonnet":
            deployment_name = os.getenv("CLAUDE_3_7_DEPLOYMENT_NAME")
        elif model == "claude-3-5-sonnet":
            deployment_name = os.getenv("CLAUDE_3_5_DEPLOYMENT_NAME")
        elif model == "claude-3-sonnet":
            deployment_name = os.getenv("CLAUDE_3_DEPLOYMENT_NAME")
        else:
            raise ValueError(f"Model '{model}' not recognized for AWS Bedrock.")
        # Initialize langchain class with pre-configured boto3 client to allow
        # for API-Key Authentication
        return ChatBedrockConverse(
            model=deployment_name, client=_get_aws_bedrock_client()
        )
