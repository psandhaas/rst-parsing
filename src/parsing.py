import requests
from typing import Union
from utils import run_docker_container, stop_and_rm_container, write_to_json


class DMRSTParser:
    def __init__(self):
        self.container_name = run_docker_container("dmrst")

    def parse(self, text: Union[str, list[str]]):
        resp = requests.post(
            "http://localhost:8000/parse",
            json={"sentences": text}
        )
        return resp.json()


class DPLPParser:
    def __init__(self):
        self.container_name = run_docker_container("dplp")

    def parse(self, text: Union[str, list[str]]):
        if isinstance(text, str):
            text = [text]
        res = []
        for t in text:
            resp = requests.post(
                "http://localhost:5000/dplp",
                json={"text": t},
                headers={"Content-Type": "application/json"}
            )
            res.append(resp.json())
        return res


class LLMParser:
    pass


if __name__ == "__main__":
    from pprint import pprint
    print(dmrst_container_name := run_docker_container("dmrst"))
