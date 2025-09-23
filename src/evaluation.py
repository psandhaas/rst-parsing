#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Mon, 14.09.25                                                     #
# ========================================================================== #

"""PARSEVAL evaluation metrics for RST trees."""

from glob import glob
import pandas as pd
from pandas.io.formats.style import Styler
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Tuple, Optional, Union

from tree import Node
from utils import load_rs3, map_fine2coarse


# GOLD_SEGMENTATION: Dict[str, List[str]] = load_texts(
#     "C:/Users/SANDHAP/Repos/rst-parsing/data/segmented_texts/gold_excluding_disjunct_segments"  # noqa
# )


# class EDUSpans(object):
#     def __init__(
#         self,
#         segmentation: Union[Node, Dict[Any, List[str]]],
#         gold: Union[str, List[str]]  # file-name or list of EDUs
#     ):
#         if isinstance(gold, str):
#             if gold not in GOLD_SEGMENTATION:
#                 raise ValueError(
#                     f"'{gold}' is not a valid identifier of a known document."
#                 )
#             else:
#                 segments = GOLD_SEGMENTATION[gold]
#         else:
#             segments = gold
#         self.gold_segments = segments
#         self.edus = [
#             seg.text.strip() for seg in segmentation.rs3_segments
#         ] if isinstance(segmentation, Node) else [
#             seg.strip() for seg in segmentation.values()
#         ]
#         self.recall = segments
#         self.precision = segments
#         self.f1 = segments

#     @property
#     def edus(self) -> List[str]:
#         """Get the list of EDUs in the document."""
#         return self._edus

#     @edus.setter
#     def edus(self, edus: List[str]):
#         if not hasattr(self, "_edus"):
#             if len(edus) == 0:
#                 raise ValueError("EDU list cannot be empty.")
#             if not all(isinstance(edu, str) and len(edu) > 0 for edu in edus):
#                 raise ValueError("All EDUs must be non-empty strings.")
#             if not EDUSpans.same_text(edus, self.gold_segments):
#                 raise ValueError(
#                     "Text of EDU Spans doesn't correspond to the provided " +
#                     "gold text."
#                 )
#             self._edus = edus

#     @property
#     def recall(self) -> float:
#         if not hasattr(self, "_recall"):
#             self.recall = self.gold_segments
#         return self._recall
    
#     @recall.setter
#     def recall(self, gold_segments: List[str]):
#         expected: List[str] = [
#             ln.strip() for ln in gold_segments if len(ln.strip()) > 0
#         ]
#         correct = sum(
#             True for edu in self.edus
#             if edu in expected
#         )
#         self._recall = correct / len(gold_segments)

#     @property
#     def precision(self) -> float:
#         if not hasattr(self, "_precision"):
#             self.precision = self.gold_segments
#         return self._precision
    
#     @precision.setter
#     def precision(self, gold_segments: List[str]):
#         expected: List[int] = [
#             ln.strip() for ln in gold_segments if len(ln.strip()) > 0
#         ]
#         correct = sum(
#             True for edu in self.edus
#             if edu in expected
#         )
#         self._precision = correct / len(self.edus)

#     @property
#     def f1(self) -> float:
#         if not hasattr(self, "_f1"):
#             self.f1 = self.gold_segments
#         return self._f1
    
#     @f1.setter
#     def f1(self, gold_segments: List[str]):
#         expected: List[int] = [
#             ln.strip() for ln in gold_segments if len(ln.strip()) > 0
#         ]
#         correct = sum(
#             True for edu in self.edus
#             if edu in expected
#         )
#         self._f1 = 2 * correct / (
#             len(expected) + len(self.edus)
#         )

#     @staticmethod
#     def tokenize(text: str) -> List[str]:
#         return [t.strip() for t in text.split(" ") if len(t.strip()) > 0]
    
#     @staticmethod
#     def join(tokens: List[str]) -> str:
#         return " ".join([t.strip() for t in tokens if len(t.strip()) > 0])

#     @staticmethod
#     def boundaries(segments: List[str]) -> List[int]:
#         """Given a sorted list of segments, return the token-indices that
#         correspond to the split points between segments."""
#         toks: List[str] = EDUSpans.tokenize(EDUSpans.join(segments))
#         edus: List[List[str]] = [
#             EDUSpans.tokenize(edu) for edu in segments
#         ]
#         assert toks == [t for edu in edus for t in edu], (
#             "Tokenization of EDUs & text doesn't match!"
#         )
#         boundaries = []
#         bound = 0
#         for edu in edus:
#             n = len(edu)
#             span_toks, toks = toks[:n], toks[n:]
#             if span_toks != edu:
#                 raise ValueError("Tokenization of EDUs & text doesn't match!")
#             boundaries.append(bound + n)
#             bound += n
#         return boundaries

#     @staticmethod
#     def same_text(segments1: List[str], segments2: List[str]) -> bool:
#         """Ensure two segmented lists correspond to the same original text."""
#         return " ".join([
#             seg.strip() for seg in segments1 if len(seg.strip()) > 0
#         ]) == " ".join([
#             seg.strip() for seg in segments2 if len(seg.strip()) > 0
#         ])


def tokens_2_edus(tree: Node) -> Dict[int, int]:
    """Map token indices to EDU indices for a given tree."""
    doc_tokens: List[str] = tree.tokens
    edus: List[Node] = sorted(tree.rs3_segments, key=lambda x: x.span[0])
    res = {}
    tok_idx = 0
    while edus:
        edu = edus.pop(0)
        for i, t in enumerate(edu_toks := Node.tokenize(edu)):
            if t == doc_tokens[tok_idx]:
                res[tok_idx] = edu.id
                tok_idx += 1
            else:
                expected_ctxt = " ".join(doc_tokens[
                    max(0, tok_idx-5):min(tok_idx+5, len(doc_tokens))
                ])
                actual_ctxt = " ".join(
                    edu_toks[max(0, i-5):min(i+5, len(edu_toks))]
                )
                raise ValueError(
                    "The tokenization of EDU segments does not match the" +
                    " tokenization of the full text. Encountered " +
                    f"unexpected token in EDU {edu.id}:\n" +
                    f"  Expected: '... {expected_ctxt} ...'\n" +
                    f"  Got:   '... {actual_ctxt} ...'"
                )
    return res


def edus2tokens(tree: Node) -> Dict[int, List[int]]:
    toks2edus = tokens_2_edus(tree)
    edus2tokens = {
        edu: [] for edu in sorted(list(set(toks2edus.values())))
    }
    for tok_idx, edu_idx in toks2edus.items():
        edus2tokens[edu_idx].append(tok_idx)
    return {
        edu: sorted(tok_indices)
        for edu, tok_indices in edus2tokens.items()
    }


def tree_spans_2_token_spans(tree: Node) -> Dict[
    Tuple[int, int], Tuple[int, int]
]:
    """Map EDU-spans to token-spans for all constituents in the tree."""
    root = tree.root
    edu_toks = edus2tokens(root)
    last_edu_id = max(seg.id for seg in root.rs3_segments)
    res = {}
    for n in root:
        if (span := n.span) is None:
            desc = sorted([
                d for d in n._descendants("all")
                if d.id <= last_edu_id 
            ], key=lambda x: x.id)
            span = (desc[0].span[0], desc[-1].span[-1])
        if span not in res:
            start = edu_toks[span[0]][0]
            end = edu_toks[span[1]][-1] + 1
            res[span] = (start, end)

    return {
        edu_span: tok_span for edu_span, tok_span in sorted(
            res.items(),
            key=lambda x: (x[0][1] - x[0][0], x[0][0])
        )
    }


def align_constituents(
    gold: Node, parsed: Node,
    doc_name: str,
    map_relations: bool = True,
) -> pd.DataFrame:
    """Align the constituents of two RST trees. A constituent of a tree is
    any node within it and that node's span is defined as the token indices
    of the EDUs that a given node dominates.
    
    If a constituent exists in both trees and the spans of the constituents,
    as well as the spans of the EDUs of those constituents are identical, the
    constituents are considered aligned.

    :param gold: The gold RST tree.
    :type gold: `Node`
    :param parsed: The parsed RST tree.
    :type parsed: `Node`
    :param map_relations: Whether to map fine-grained relation labels to
        coarse-grained ones. Defaults to `True`.
    :type map_relations: `bool`, optional

    :return: A DataFrame with a multi-index of token spans and columns for
        gold & parsed spans, nuclearity, and relation labels.
    :rtype: `pd.DataFrame`
    """
    def get_node_id(edu_span: Tuple[int, int], tree: Node) -> int:
        """Get the ID of the node that corresponds to the provided EDU span."""
        def walk_up(node: Node) -> List[Node]:
            res = []
            while (parent := node.parent) is not None:
                res.append(parent)
                node = parent
            res.append(node)
            return res

        try:
            return tree[edu_span].id
        except KeyError:  # no span set -> determine ID
            leftmost_child = tree[(edu_span[0], edu_span[0])]
            rightmost_child = tree[(edu_span[1], edu_span[1])]
            left_parents = walk_up(leftmost_child)
            right_parents = walk_up(rightmost_child)
            while left_parents and right_parents and (
            left_parents[-1] == right_parents[-1]
            ):
                lca = left_parents.pop()
                right_parents.pop()
        return lca.id

    gold_constituents = tree_spans_2_token_spans(gold)
    gold_root_span = max(
        gold_constituents.values(), key=lambda x: (x[1]-x[0], x[0])
    )
    parsed_constituents = tree_spans_2_token_spans(parsed)
    parsed_root_span = max(
        parsed_constituents.values(), key=lambda x: (x[1]-x[0], x[0])
    )
    if gold_root_span != parsed_root_span:
        raise ValueError(
            "The root spans of the gold and parsed trees do not match. " +
            f"Gold: {gold_root_span}, Parsed: {parsed_root_span}"
        )

    token_spans = sorted(list(set(
        list(gold_constituents.values()) +
        list(parsed_constituents.values())
    )), key=lambda x: (x[0], x[1] - x[0]))
    token_spans.remove(gold_root_span)  # remove root span
    df = pd.DataFrame(
        {
            ("Spans", "gold"): [
                True if span in gold_constituents.values() else False
                for span in token_spans
            ],
            ("Spans", "parsed"): [
                True if span in parsed_constituents.values() else False
                for span in token_spans
            ],
        ("Nuclearity", "gold"): [None] * len(token_spans),
        ("Nuclearity", "parsed"): [None] * len(token_spans),
        ("Relations", "gold"): [None] * len(token_spans),
        ("Relations", "parsed"): [None] * len(token_spans),
        },
        index=token_spans
    )
    df.index.name = "token_span"

    # add nuclearity & (coarse) relation labels
    try:
        for tok_span in token_spans:
            if tok_span in gold_constituents.values():
                gold_edu_span = [
                    edu_span for edu_span, t_span in gold_constituents.items()
                    if t_span == tok_span
                ][0]
                node_id = get_node_id(gold_edu_span, gold)
                gold_nuc = gold[node_id].nuc
                gold_rel = gold[node_id].relname
            else:
                gold_nuc = None
                gold_rel = None
            if tok_span in parsed_constituents.values():
                parsed_edu_span = [
                    edu_span for edu_span, t_span in parsed_constituents.items()
                    if t_span == tok_span
                ][0]
                node_id = get_node_id(parsed_edu_span, parsed)
                parsed_nuc = parsed[node_id].nuc
                parsed_rel = parsed[node_id].relname
            else:
                parsed_nuc = None
                parsed_rel = None
            df.at[tok_span, ("Nuclearity", "gold")] = gold_nuc
            df.at[tok_span, ("Nuclearity", "parsed")] = parsed_nuc
            if map_relations:
                if gold_rel is not None:
                    if (mapped_rel := map_fine2coarse(gold_rel)) == "unknown":
                        gold_rel = gold_rel  # keep original if no mapping exists
                    else:
                        gold_rel = mapped_rel
                if parsed_rel is not None:
                    if (mapped_rel := map_fine2coarse(parsed_rel)) == "unknown":
                        parsed_rel = parsed_rel  # keep original if no mapping exists
                    else:
                        parsed_rel = mapped_rel
            df.at[tok_span, ("Relations", "gold")] = gold_rel
            df.at[tok_span, ("Relations", "parsed")] = parsed_rel
    except Exception as e:
        print(f"Encountered an error while aligning '{doc_name}': {e}")
        raise e

    for col in [
        ("Nuclearity", "gold"), ("Nuclearity", "parsed"),
        ("Relations", "gold"), ("Relations", "parsed")
    ]:
        df[col] = df[col].astype(object)

    return df


# class Metrics:
#     category: Literal["segmentation", "spans", "nuclearity", "relations"]
#     precision: float
#     recall: float
#     f1: float

#     def __init__(
#         self,
#         category: Literal["segmentation", "spans", "nuclearity", "relations"],
#         gold: Node,
#         parsed: Node
#     ):
#         self.category = category
#         self.gold = gold
#         self.parsed = parsed
#         if category == "segmentation":
#             self._compute_segmentation()
#         elif category == "spans":
#             self._compute_spans()
#         elif category == "nuclearity":
#             self._compute_nuclearity()
#         elif category == "relations":
#             self._compute_relations()
#         else:
#             raise ValueError(
#                 f"'{category}' is not a valid metric category. Must be one " +
#                 "of 'segmentation', 'spans', 'nuclearity', or 'relations'."
#             )

#     def _compute_segmentation(self):
#         edus = EDUSpans(
#             segmentation=self.parsed, gold=self.gold.document_lines
#         )
#         self.precision = edus.precision
#         self.recall = edus.recall
#         self.f1 = edus.f1

#     # TODO
#     def _compute_spans(self):
#         df = align_constituents(self.gold, self.parsed)


#     # TODO
#     def _compute_nuclearity(self):
#         pass

#     # TODO
#     def _compute_relations(self):
#         pass


class RSTEval:
    def __init__(
        self,
        parsed_dir: str,
        gold_dir: str,
        exclude_disjunct_segments: bool = True,
        map_relations: bool = True
    ):
        """Initialize with paths to the parsed & gold directories, containing
        `.rs3` files.
        
        :param parsed_dir: Path to the directory with parsed `.rs3` files.
        :type parsed_dir: `str`
        :param gold_dir: Path to the directory with gold `.rs3` files.
        :type gold_dir: `str`
        :param exclude_disjunct_segments: Whether to exclude disjunct segments
            such as headings without an annotated parent when loading the
            `.rs3` files. Defaults to `True`.
        :type exclude_disjunct_segments: `bool`, optional
        :param map_relations: Whether to map fine-grained relation labels to
            coarse-grained ones. Defaults to `True`.
        :type map_relations: `bool`, optional
        
        :raises ValueError: If the parsed & gold directories don't contain the
            same files.
        """
        self.gold, self.parsed = self._load_data(
            gold_dir, parsed_dir, exclude_disjunct_segments
        )
        self.name: Optional[str] = None
        self.aligned: Dict[str, pd.DataFrame] = self._align_all(map_relations)
        self.metrics: pd.DataFrame = self._compute_metrics()

    def _load_data(
        self,
        gold_dir: str, parsed_dir: str,
        exclude_disjunct_segments: bool
    ) -> Tuple[Dict[str, Node]]:
        parsed_names = [Path(p).stem for p in glob(f"{parsed_dir}/*.rs3")]
        parsed2base_names = {
            Path(p).stem: re.search(
                r"(.+\d(_blog)?)(_[^\d]+)", p, re.I
            ).group(1) for p in parsed_names
        }
        if set(Path(p).stem for p in glob(f"{gold_dir}/*.rs3")) != set(
            list(parsed2base_names.values())
        ):
            raise ValueError(
                "The parsed & gold directories don't contain the same files."
            )
        gold: Dict[str, Node] = load_rs3(
            gold_dir, read_as="node",
            exclude_disjunct_segments=exclude_disjunct_segments
        )
        parsed: Dict[str, Node] = {
            parsed2base_names[name]: node for name, node in load_rs3(
                parsed_dir, read_as="node",
                exclude_disjunct_segments=exclude_disjunct_segments
            ).items()
        }
        
        if (m := re.search(
            r"(.+\d(_blog)?)(_[^\d]+)", parsed_names[0], re.I
        )) is not None:
            if len(splt := (n := m.group(3)[1:]).split("_", 1)) == 2:
                self.name = " ".join([
                    splt[0].strip().upper(),
                    f"({splt[1].strip().capitalize().replace('_', '-')})"
                ])
            else:
                self.name = n.strip().upper()

        return gold, parsed

    def _precision(
        self, aligned: pd.DataFrame,
        category: Literal["Spans", "Nuclearity", "Relations"]
    ) -> float:
            if category == "Spans":
                tp = (  # True if both gold and parsed are True
                    (aligned[(category, "gold")] == True)
                    & (aligned[(category, "parsed")] == True)
                ).sum()
                fp = (  # False positive: parsed True, gold False
                    (aligned[(category, "gold")] != True)
                    & (aligned[(category, "parsed")] == True)
                ).sum()
            else:
                tp = (  # True positive: gold == parsed and notna
                    (aligned[(category, "gold")].notna())
                    & (aligned[(category, "parsed")].notna())
                    & (aligned[(category, "gold")] == aligned[(category, "parsed")])  # noqa: E501
                ).sum()
                fp = (  # False positive: parsed notna, gold isna or not equal
                    (aligned[(category, "parsed")].notna())
                    & (
                        (aligned[(category, "gold")].isna())
                        | (aligned[(category, "gold")] != aligned[(category, "parsed")])
                    )
                ).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            return precision

    def _recall(
        self, aligned: pd.DataFrame,
        category: Literal["Spans", "Nuclearity", "Relations"]
    ) -> float:
            if category == "Spans":
                tp = (  # True if both gold and parsed are True
                    (aligned[(category, "gold")] == True)
                    & (aligned[(category, "parsed")] == True)
                ).sum()
                fn = (  # False negative: gold True, parsed False
                    (aligned[(category, "gold")] == True)
                    & (aligned[(category, "parsed")] != True)
                ).sum()
            else:
                tp = (  # True positive: gold == parsed and notna
                    (aligned[(category, "gold")].notna())
                    & (aligned[(category, "parsed")].notna())
                    & (aligned[(category, "gold")] == aligned[(category, "parsed")])  # noqa: E501
                ).sum()
                fn = (  # False negative: gold notna, parsed isna or not equal
                    (aligned[(category, "gold")].notna())
                    & (
                        (aligned[(category, "parsed")].isna())
                        | (aligned[(category, "gold")] != aligned[(category, "parsed")])  # noqa: E501
                    )
                ).sum()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return recall

    def _f1(
        self, aligned: pd.DataFrame,
        category: Literal["Spans", "Nuclearity", "Relations"]
    ) -> float:
            r = self._recall(aligned, category)
            p = self._precision(aligned, category)
            f1 = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
            return f1

    def _align_all(self, map_relations: bool = True) -> Dict[str, pd.DataFrame]:
        res = {}
        for name in self.gold.keys():
            res[name] = align_constituents(
                self.gold[name], self.parsed[name],
                doc_name=name, map_relations=map_relations
            )
        return res

    def _compute_metrics(self) -> pd.DataFrame:
        # compute metrics for all documents
        metrics = {}
        for doc, df in self.aligned.items():
            res = {}
            for cat in ["Spans", "Nuclearity", "Relations"]:
                res[cat] = {"Recall": self._recall(df, cat),
                            "Precision": self._precision(df, cat),
                            "F1": self._f1(df, cat)}
            metrics[doc] = res

        # create DataFrame
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
        metrics_df = pd.concat(
            {cat: metrics_df[cat].apply(pd.Series)
             for cat in metrics_df.columns},
            axis=1
        )
        metrics_df.index.name = "Document"
        metrics_df.loc["Average"] = metrics_df.mean(numeric_only=True)

        return metrics_df

    def styled(self, caption: str = None) -> Styler:
        """Get a styled version of the metrics table for better visualization
        in Jupyter Notebooks.

        :param caption: Optional caption for the table. If not provided,
            the name is inferred from the different file-name-suffixes of gold
            and parsed files, if possible. Defaults to `None`.
        :type caption: `str`, optional

        """
        def highlight_max_f1(s):
            # Highlight the largest F1-values of each category
            is_max = s == s.max()
            return ['font-weight: bold' if v else '' for v in is_max]

        def italicize_average(val):
            return "font-style: italic" if val == "Average" else ""

        # Apply highlighting and center columns using Styler
        styled = (
            self.metrics.style
            .format("{:.3f}")  # Format all values to 3 decimals, no trailing zeros
            .set_table_styles([
                # Divider above the Average row
                {'selector': 'tbody tr:last-child', 'props': [
                    ('border-top', '1px solid black')
                ]}
            ])
            .apply(highlight_max_f1, subset=[('Spans', 'F1')], axis=0)
            .apply(highlight_max_f1, subset=[('Nuclearity', 'F1')], axis=0)
            .apply(highlight_max_f1, subset=[('Relations', 'F1')], axis=0)
            .applymap_index(italicize_average, level=0)
        )
        if self.name is not None or caption is not None:
            styled.set_caption(self.name if caption is None else caption)
        return styled

    # FIXME: use df and kwargs of to_latex()
    def to_latex(self, caption: str = None) -> str:
        """Export the metrics table to LaTeX format.
        
        :param caption: Optional caption for the table. If not provided,
            the name is inferred from the different file-name-suffixes of gold
            and parsed files, if possible. Defaults to `None`.
        :type caption: `str`, optional
        """
        return self.styled(caption=caption).to_latex()


if __name__ == "__main__":
    from pprint import pprint

    parsed_dir = "C:/Users/SANDHAP/Repos/rst-parsing/data/parsed/llm_zeroshot"
    gold_dir = "C:/Users/SANDHAP/Repos/rst-parsing/data/gold_annotations"
    rst_eval = RSTEval(parsed_dir, gold_dir)
    styled = rst_eval.styled()
