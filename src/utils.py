from bs4 import BeautifulSoup, SoupStrainer
import docker
import json
from glob import glob
import os
from pathlib import Path
import subprocess
import time
from typing import Literal, Optional


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


def get_relations_set(rs3: str, lower: bool = True) -> set[tuple[str, str]]:
    rels = set()
    soup = BeautifulSoup(rs3, "xml", parse_only=SoupStrainer("rel"))
    for rel in soup.find_all(True):
        if lower:
            rels.add(tuple([a.lower() for a in rel.attrs.values()]))
        else:
            rels.add(tuple(rel.attrs.values()))
    return rels


def map_fine2coarse(relation: str) -> Optional[str]:
    """Maps a fine-grained relation label (either from DPLP or DMRST) to
    a coarse-grained one that's compatible ."""
    # mapping of cross-lingual RST-relations following Braud et al., 2017
    # (https://doi.org/10.48550/arXiv.1701.02946) and Carlson et al., 2001
    general_mapping = {
        # taken from github.com/seq-to-mind/DMRST_Parser/blob/main/Preprocess_RST_Data/1_uniform_treebanks/code/src/relationSet.py#L11
        u'ahalbideratzea':'Enablement',
        u'alderantzizko-baldintza':'Condition',
        u'alternativa':'Condition',
        u'analogy':'Comparison',
        u'antitesia':'Contrast',
        u'antithesis':'Contrast',
        u'antítesis':'Contrast',
        u'arazo-soluzioa':'Topic-Comment',
        u'attribution':'Attribution',
        u'attribution-negative':'Attribution',
        u'aukera':'Condition',
        u'background':'Background',
        u'baldintza':'Condition',
        u'bateratzea':'Joint',
        u'birformulazioa':'Summary',
        u'capacitación':'Enablement',
        u'causa':'Cause',
        u'cause':'Cause',
        u'cause-result':'Cause',
        u'circumstance':'Background',
        u'circunstancia':'Background',
        u'comment':'Evaluation',
        u'comment-topic':'Topic-Comment',
        u'comparison':'Comparison',
        u'concesión':'Contrast',
        u'concession':'Contrast',
        u'conclusion':'Evaluation',
        u'condición':'Condition',
        u'condición-inversa':'Condition',
        u'condition':'Condition',
        u'conjunción':'Joint',
        u'conjunction':'Joint',
        u'consequence':'Cause',
        u'contingency':'Condition',
        u'contrast':'Contrast',
        u'contraste':'Contrast',
        u'definition':'Elaboration',
        u'definitu-gabeko-erlazioa':'Summary',
        u'disjunction':'Joint',
        u'disjuntzioa':'Joint',
        u'disyunción':'Joint',
        u'e-elaboration':'Elaboration',
        u'ebaluazioa':'Evaluation',
        u'ebidentzia':'Explanation',
        u'elaboración':'Elaboration',
        u'elaboration':'Elaboration',
        u'elaboration-additional':'Elaboration',
        u'elaboration-general-specific':'Elaboration',
        u'elaboration-object-attribute':'Elaboration',
        u'elaboration-part-whole':'Elaboration',
        u'elaboration-process-step':'Elaboration',
        u'elaboration-set-member':'Elaboration',
        u'elaborazioa':'Elaboration',
        u'enablement':'Enablement',
        u'evaluación':'Evaluation',
        u'evaluation':'Evaluation',
        u'evidence':'Explanation',
        u'evidencia':'Explanation',
        u'example':'Elaboration',
        u'explanation':'Explanation',
        u'explanation-argumentative':'Explanation',
        u'ez-baldintzatzailea':'Condition',
        u'fondo':'Background',
        u'helburua':'Enablement',
        u'hypothetical':'Condition',
        u'interpretación':'Evaluation',
        u'interpretation':'Evaluation',
        u'interpretazioa':'Evaluation',
        u'inverted-sequence':'Temporal',
        u'joint':'Joint',
        u'justificación':'Explanation',
        u'justifikazioa':'Explanation',
        u'justify':'Explanation',
        u'kausa':'Cause',
        u'konjuntzioa':'Joint',
        u'kontrastea':'Contrast',
        u'kontzesioa':'Contrast',
        u'laburpena':'Summary',
        u'list':'Joint',
        u'lista':'Joint',
        u'manner':'Manner-Means',
        u'means':'Manner-Means',
        u'medio':'Manner-Means',
        u'metodoa':'Manner-Means',
        u'motibazioa':'Explanation',
        u'motivación':'Explanation',
        u'motivation':'Explanation',
        u'non-volitional-cause':'Cause',
        u'non-volitional-result':'Cause',
        u'nonvolitional-cause':'Cause',
        u'nonvolitional-result':'Cause',
        u'ondorioa':'Cause',
        u'otherwise':'Condition',
        u'parenthetical':'Elaboration',
        u'preference':'Comparison',
        u'preparación':'Background',
        u'preparation':'Background',
        u'prestatzea':'Background',
        u'problem-solution':'Topic-Comment',
        u'proportion':'Comparison',
        u'propósito':'Enablement',
        u'purpose':'Enablement',
        u'question-answer':'Topic-Comment',
        u'reason':'Explanation',
        u'reformulación':'Summary',
        u'restatement':'Summary',
        u'restatement-mn':'Summary',
        u'result':'Cause',
        u'resultado':'Cause',
        u'resumen':'Summary',
        u'rhetorical-question':'Topic-Comment',
        u'same-unit':'Same-unit',
        u'sameunit':'Same-unit',  # added
        u'secuencia':'Temporal',
        u'sekuentzia':'Temporal',
        u'sequence':'Temporal',
        u'solución':'Topic-Comment',
        u'solutionhood':'Topic-Comment',
        u'statement-response':'Topic-Comment',
        u'summary':'Summary',
        u'temporal-after':'Temporal',
        u'temporal-before':'Temporal',
        u'temporal-same-time':'Temporal',
        u'testuingurua':'Background',
        u'textual-organization':'Textual-organization',
        u'textualorganization':'Textual-organization',
        u'topic-comment':'Topic-Comment',
        u'topic-drift':'Topic-Change',
        u'topic-shift':'Topic-Change',
        u'unconditional':'Condition',
        u'unión':'Joint',
        u'unless':'Condition',
        u'volitional-cause':'Cause',
        u'volitional-result':'Cause',
        u'zirkunstantzia':'Background',
        u'question': 'Topic-Comment'
    }
    other2joty = {
        # taken from github.com/mohamadi-sara20/DPLP-German/blob/master/parsing_eval_metrics/RelationClasses.txt
        'attribution': 'Attribution',
        'background': 'Background',
        'cause': 'Cause',
        'causemult': 'Cause',
        'comparison': 'Comparison',
        'comparisonmult': 'Comparison',
        'condition': 'Condition',
        'conditionmult': 'Condition',
        'contrast': 'Contrast',
        'contrastmult': 'Contrast',
        'dummy': 'Dummy',
        'elaboration': 'Elaboration',
        'enablement': 'Enablement',
        'evaluation': 'Evaluation',
        'evaluationmult': 'Evaluation',
        'explanation': 'Explanation',
        'explanationmult': 'Explanation',
        'jointmult': 'Joint',
        'manner-means': 'Manner-Means',
        'same': 'Same-Unit',
        'span': 'span',
        'summary': 'Summary',
        'temporal': 'Temporal',
        'temporalmult': 'Temporal',
        'textualorganization': 'TextualOrganization',
        'topichange': 'Topic-Change',
        'topichangemult': 'Topic-Change',
        'topicomment': 'Topic-Comment',
        'topicommentmult': 'Topic-Comment',
        'topidriftmult': 'Topic-Change',
        'virtual': 'Dummy',
        'virtual-root': 'Dummy'
    }
    dmrst2joty = {
        "same-unit": "Same-Unit",
        "Same-unit": "Same-Unit",
        "textual-organization": "TextualOrganization",
        "Textual-organization": "TextualOrganization"
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

    if (rel := relation) is None:
        return None
    
    if rel in general_mapping:
        coarse_dmrst = general_mapping[rel]
    elif (rel := relation.lower()) not in general_mapping:
        # try to strip nuclearity suffixes
        if "-" in rel and len(
            splt := [x.strip() for x in rel.split("-") if len(x.strip()) > 0]
        ) == 2 and splt[-1] in ["s", "n", "mn"]:
            if splt[0] in general_mapping:
                coarse_dmrst = general_mapping[splt[0]]

    if coarse_dmrst in dmrst2joty:
        coarse_label = dmrst2joty[coarse_dmrst]
    elif rel in other2joty:
        coarse_label = other2joty[rel]
    else:
        coarse_label = "unknown"
    return coarse_label


def build_relations_map(
    annotations_dir: str
) -> dict[str, dict[str, list[str] | str]]:
    """Builds a mapping of all unique relations found across all .rs3 files in
    the given directory, where fine-grained relations are mapped to
    coarse-grained ones."""
    relations_set = set()
    for p in glob(f"{annotations_dir}/*.rs3"):
        with open(p, "r", encoding="utf-8") as f:
            rs3 = f.read()
        relations_set.update(get_relations_set(rs3))

    return {rel[0]: {"aliases": [],
                     "type": rel[1]}
            for rel in relations_set}


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
