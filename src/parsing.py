import requests
from typing import Union
from utils import (
    load_texts,
    run_docker_container, stop_and_rm_container,
    write_to_json
)


class DMRSTParser:
    def __init__(self):
        self.container_name = run_docker_container("dmrst")

    def parse(
        self,
        sentences: Union[str, list[str]],
        ignore_size_limit: bool = False,
        tokenized_output_sentences: bool = True
    ) -> list[dict]:
        if isinstance(sentences, str):
            sentences = [sentences]
        resp = requests.post(
            "http://localhost:8000/parse",
            json={"sentences": sentences,
                  "batch_size": len(sentences),
                  "ignore_size_limit": ignore_size_limit,
                  "tokenized_output_sentences": tokenized_output_sentences}
        )
        return resp.json()

    def stop(self):
        stop_and_rm_container(self.container_name)


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

    def stop(self):
        stop_and_rm_container(self.container_name)


class LLMParser:
    pass


if __name__ == "__main__":
    from pprint import pprint

    texts = load_texts()
    # dmrst = DMRSTParser()
    # dplp = DPLPParser()
    # dplp_rst_out_dir = "C:/Users/Philipp/Repos/DPLP-German"
    res_dir = "C:/Users/Philipp/Repos/rst-parsing/data/parsed"
    # for doc_name, sents in texts.items():
    #     pprint((doc_res := dplp.parse("\n".join(sents))[0]), sort_dicts=False)
    #     with open(f"{res_dir}/{doc_name}_dplp.rs3", "w", encoding="utf-8") as rs3_f:
    #         rs3_f.write(
    #             "<?xml version='1.0' encoding='UTF-8'?>"
    #         )
    #         rs3_f.write(doc_res["rs3"])
    #     with open(f"{res_dir}/{doc_name}_dplp.dis", "w", encoding="utf-8") as dis_f:
    #         dis_f.write(doc_res["dis"])
