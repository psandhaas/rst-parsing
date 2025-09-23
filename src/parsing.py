#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Fri, 01.08.25                                                     #
# ========================================================================== #

"""Interfaces for comparing different RST parsers."""

from langchain_core.language_models.chat_models import BaseChatModel
import os
import requests
from typing import Dict, List, Literal, Optional, Union

from llm_graph import parse_rst
from tree import binarize, dmrst_nodes, Node
from utils import (
    run_docker_container, stop_and_rm_container,
    _init_llm
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
    ) -> List[Dict]:
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
    def __init__(
        self,
        model: Literal[
            "gpt-4.1", "gpt-4o", "o4-mini",
            "claude-sonnet-4", "claude-3-7-sonnet", "claude-3-5-sonnet",
            "claude-3-sonnet"
        ] = "gpt-4.1"
    ):
        # self.llm: BaseChatModel = _init_llm(model=model)
        self.model = model
        # self.prompt = str(
        #     "Analysiere die RST-Struktur des folgenden Texts:\n{text}\n\n" +
        #     "Hier sind die EDU-Segmente des Texts:\n{segmentation}"
        # )

    def parse(
        self,
        text: Union[str, list[str]],
        return_structured_output: bool = False
    ) -> Union[List[Node], List[Dict]]:
        """Segment the provided texts into EDUs & parse their RST trees using
        an LLM and structured prompting.
        
        :param text: The input text(s) to be parsed.
        :type text: `str | List[str]`
        :param return_structured_output: Whether to return the raw structured
            output. If `False`, the structured output will be converted to a binary
            tree of `Node` objects. Defaults to `False`.
        :type return_structured_output: `bool`, optional

        :return: The root `Node` of the constructed tree or the raw structured
            output as a dict with keys `'text'`, `'edus'`, `'spans'`, `'queue'`,
            `'current_parent'`, and `'current_children'`. If a list of texts is
            provided, a list of `Node` or dicts is returned.
        :rtype: `Node` | `Dict[str, Union[
            str,
            List[EDU],
            Dict[int, EDU],
            List[Span],
            Optional[Span],
            List[Optional[Span]
        ]`
        """
        return parse_rst(
            text=text,
            model=self.model,
            return_structured_output=return_structured_output
        )

    # @property
    # def prompt(self) -> str:
    #     return self._prompt
    
    # @prompt.setter
    # def prompt(self, prompt: str = "{text}\n\n{segmentation}"):
    #     if len(prompt := prompt.strip()) == 0:
    #         raise ValueError(
    #             "Prompt cannot be an empty string. It must contain at least " +
    #             "the placeholders {text} and {segmentation} to insert the " +
    #             "input text and its EDU segmentation."
    #         )
    #     self._prompt = prompt

    # def _call_llm(
    #     self,
    #     model: Literal["gpt-4o", "gpt-4.1"],
    #     messages: list[Dict[str, str]],
    #     output_format: Optional[BaseModel] = None,
    #     return_full_response: bool = False,
    #     **kwargs
    # ) -> Union[str, BaseModel, ChatCompletion, ParsedChatCompletion]:
    #     try:
    #         if output_format is not None:
    #             completion = self.client.beta.chat.completions.parse(
    #                 model=model,
    #                 messages=messages,
    #                 response_format=output_format,
    #                 **kwargs
    #             )
    #             if return_full_response:
    #                 return completion
    #             return completion.choices[0].message.parsed  # BaseModel
    #         else:
    #             completion = self.client.chat.completions.create(
    #                 model=model,
    #                 messages=messages,
    #                 **kwargs
    #             )
    #             if return_full_response:
    #                 return completion
    #             return completion.choices[0].message.content  # str
    #     except Exception as e:
    #         print(f"Error calling LLM!")
    #         raise e

    # def _segment(
    #     self,
    #     text: str,
    #     model: Literal["gpt-4o", "gpt-4.1"],
    #     **kwargs
    # ) -> Segmentation:
    #     """Prompts the LLM to segment the input text into EDUs using the
    #     `Segmentation` output format.
        
    #     :returns: Original document, partitioned into EDUs.
    #     :rtype: `Segmentation`
    #     """
    #     kwargs.pop("return_full_response", None)
    #     msg = [{
    #         "role": "user",
    #         "content": f"Segmentiere den folgenden Text in EDUs:\n{text}"
    #     }]
    #     resp = self._call_llm(
    #         model=model,
    #         messages=msg,
    #         output_format=Segmentation,
    #         return_full_response=False,
    #         **kwargs
    #     )
    #     return resp

    # def parse(
    #     self,
    #     text: Union[str, list[str]],
    #     model: Literal["gpt-4o", "gpt-4.1"] = "gpt-4.1",
    #     **kwargs
    # ) -> List[Union[str, BaseModel, ChatCompletion, ParsedChatCompletion]]:
    #     if isinstance(text, str):
    #         text = [text]
    #     output_format = kwargs.pop("output_format", RSTTree)
    #     res = []
    #     for t in text:
    #         seg: Segmentation = self._segment(t, model, **kwargs)
    #         msg = [{
    #             "role": "user",
    #             "content": self.prompt.format(
    #                 text=t, segmentation=seg.model_dump_json()
    #             )
    #         }]
    #         res.append(self._call_llm(
    #             model=model,
    #             messages=msg,
    #             output_format=output_format,
    #             **kwargs
    #         ))
    #     return res


if __name__ == "__main__":
    from glob import glob
    from pathlib import Path
    from pprint import pprint
    from utils import load_texts

    gold_dir = "C:/Users/SANDHAP/Repos/rst-parsing/data/gold_annotations"
    texts_dir = "C:/Users/SANDHAP/Repos/rst-parsing/data/texts"
    segmented_texts_dir = "C:/Users/SANDHAP/Repos/rst-parsing/data/segmented_texts/gold_excluding_disjunct_segments"
    res_dir = "C:/Users/SANDHAP/Repos/rst-parsing/data/parsed"

    texts = load_texts(segmented_texts_dir)
    text = "".join(list(texts.values())[3])
    llm_rst = LLMParser("gpt-4.1")
    root = llm_rst.parse(text)[0]
