#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Fri, 01.08.25                                                     #
# ========================================================================== #

"""Interfaces for comparing different RST parsers."""

from langchain_core.language_models.chat_models import BaseChatModel
import requests
from typing import Dict, List, Literal, Optional, Union

from llm_graph import parse_rst
from tree import Node
from utils import run_docker_container, stop_and_rm_container, _init_llm


class DMRSTParser:
    def __init__(self):
        self.container_name = run_docker_container("dmrst")

    def parse(
        self,
        text: Union[str, list[str]],
        ignore_size_limit: bool = False,
        multinuc_relations: Optional[List[str]] = None,
        return_raw_parse: bool = False,
    ) -> List[Dict]:
        if isinstance(text, str):
            text = [text]
        resp = requests.post(
            "http://localhost:8000/parse",
            json={
                "texts": text,
                "batch_size": len(text),
                "ignore_size_limit": ignore_size_limit,
                "multinuc_relations": multinuc_relations,
                "return_raw_parse": return_raw_parse,
            },
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
                headers={"Content-Type": "application/json"},
            )
            res.append(resp.json())
        return res

    def stop(self):
        stop_and_rm_container(self.container_name)


class LLMParser:
    def __init__(
        self,
        model: Literal[
            "gpt-4.1",
            "gpt-4o",
            "o4-mini",
            "claude-sonnet-4",
            "claude-3-7-sonnet",
            "claude-3-5-sonnet",
            "claude-3-sonnet",
        ] = "gpt-4.1",
    ):
        llm = _init_llm(model=model)
        # ensure structured output is supported
        if type(llm).bind_tools is BaseChatModel.bind_tools:
            raise NotImplementedError(
                "with_structured_output is not implemented for this model."
            )
        self.model = llm

    def parse(
        self,
        text: Optional[Union[str, List[str]]] = None,
        edus: Optional[Union[List[str], List[List[str]]]] = None,
        return_structured_output: bool = False,
    ) -> Union[List[Node], List[Dict]]:
        """Parse RST trees using an LLM and structured prompting.

        :param text: The input text(s) to be segmented and parsed. If ´None´,
            `edus` must be provided.
        :type text: `str | List[str] | None`, optional
        :param edus: Pre-segmented EDUs for the input text(s). If `None`,
            `text` must be provided.
        :type edus: `List[str] | List[List[str]] | None`, optional
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

        :raises ValueError: If neither `text` nor `edus` or both is provided.
        """
        if (text is None and edus is None) or (text is not None and edus is not None):
            raise ValueError("Exactly one of `text` or `edus` must be provided.")
        if edus is not None:
            return parse_rst(
                model=self.model,
                edus=edus,
                return_structured_output=return_structured_output,
            )
        return parse_rst(
            model=self.model,
            text=text,
            return_structured_output=return_structured_output,
        )


if __name__ == "__main__":
    from utils import load_texts

    texts = load_texts()

    llm_rst = LLMParser("gpt-4.1")
    unsegmented_w_linebreaks_parsed = llm_rst.parse(
        text=list("".join(t) for t in texts.values())
    )
    print(unsegmented_w_linebreaks_parsed[0].to_rs3())
