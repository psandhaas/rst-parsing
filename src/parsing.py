from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion, ParsedChatCompletion
import os
from pydantic import BaseModel
import requests
from typing import Dict, List, Literal, Optional, Union

from tree import binarize, dmrst_nodes
from utils import (
    run_docker_container, stop_and_rm_container,
    in_notebook
)
from output_formats import (
    Mononuclear, Multinuclear,
    EDU, RSTNode, RSTRelation, RSTTree
)


class DMRSTParser:
    def __init__(self):
        self.container_name = run_docker_container("dmrst")

    def parse(
        self,
        text: Union[str, list[str]],
        ignore_size_limit: bool = False,
        multinuc_relations: Optional[List[str]] = None,
        return_raw_parse: bool = False
    ) -> list[dict]:
        if isinstance(text, str):
            text = [text]
        resp = requests.post(
            "http://localhost:8000/parse",
            json={"texts": text,
                  "batch_size": len(text),
                  "ignore_size_limit": ignore_size_limit,
                  "multinuc_relations": multinuc_relations,
                  "return_raw_parse": return_raw_parse}
        )

        return resp.json()
    
    @staticmethod
    def dmrst2rs3(
        dmrst_tree: Union[str, List[str]],
        edu_spans: Dict[int, str],
        save_path: Optional[str] = None
    ) -> str:
        nodes = dmrst_nodes(dmrst_tree, edu_spans)
        tree = binarize(nodes)
        rs3 = tree.to_rs3()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(rs3)
        return rs3

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
    def __init__(self):
        if in_notebook():
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                ipython.run_line_magic("load_ext", "dotenv")
                ipython.run_line_magic("dotenv", "")
        else:
            load_dotenv()
        for env_var in [
            "AZURE_OPENAI_BASE_URL",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_API_VERSION"
        ]:
            if os.environ.get(env_var) is None:
                raise ValueError(
                    f"{env_var} not found in environment variables."
                )
        self.client = AzureOpenAI(
            base_url=os.environ["AZURE_OPENAI_BASE_URL"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"]
        )

    def _call_llm(
        self,
        model: Literal["gpt-4o", "gpt-4.1", "o4-mini"],
        messages: list[Dict[str, str]],
        output_format: Optional[BaseModel] = None,
        return_full_response: bool = False,
        **kwargs
    ) -> Union[str, BaseModel, ChatCompletion, ParsedChatCompletion]:
        try:
            if output_format is not None:
                completion = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=output_format,
                    **kwargs
                )
                if return_full_response:
                    return completion
                return completion.choices[0].message.parsed  # BaseModel
            else:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                if return_full_response:
                    return completion
                return completion.choices[0].message.content  # str
        except Exception as e:
            print(f"Error calling LLM!")
            raise e


if __name__ == "__main__":
    from pprint import pprint
    from utils import (
        build_relations_map, map_fine2coarse, load_texts, parse_write_rs3
    )

    # gold_dir = "C:/Users/SANDHAP/Repos/rst-parsing/data/gold_annotations"
    texts_dir = "C:/Users/SANDHAP/Repos/rst-parsing/data/texts"
    res_dir = "C:/Users/SANDHAP/Repos/rst-parsing/data/parsed"


#     texts = load_texts()
#     llm = LLMParser()
#     prompt = f"""
# Analysiere die RST-Struktur des folgenden Textes:
# {list(texts.values())[3]}
# """
#     res = llm._call_llm(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#         output_format=RSTTree
#     )
#     pprint(res.model_dump(), sort_dicts=False)
