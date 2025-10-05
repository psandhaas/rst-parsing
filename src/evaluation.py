#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Mon, 14.09.25                                                     #
# ========================================================================== #

"""PARSEVAL evaluation metrics for RST trees."""

from glob import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas.io.formats.style import Styler
from pathlib import Path
import re
import seaborn as sns
from typing import Dict, List, Literal, Tuple, Optional

from tree import Node
from utils import load_rs3, map_fine2coarse


def edus2tokens(tree: Node) -> Dict[int, List[int]]:
    """Map EDU indices to a list of token indices, spanning all tokens in the
    EDU."""
    res = {}
    tok_indices = [0]
    for i, edu in enumerate(tree.edus):
        prev_tok_idx = tok_indices.pop(-1)
        edu_tok_indices = list(range(
            prev_tok_idx, prev_tok_idx + len(Node.tokenize(edu, "text"))
        ))
        tok_indices += edu_tok_indices
        res[i + 1] = edu_tok_indices
    return res


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
            end = edu_toks[span[1]][-1]
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

    if (  # ensure we're aligning results based on the same document text
        Node._replace_quotation_marks("".join(gold.tokens))
        != Node._replace_quotation_marks("".join(parsed.tokens))
    ):
        gold_text = "\n".join([
            f"\tEDU {i+1}: {edu.strip()}"
            for i, edu in enumerate(gold.edus)
        ])
        parsed_text = "\n".join([
            f"\tEDU {i+1}: {edu.strip()}"
            for i, edu in enumerate(parsed.edus)
        ])
        raise ValueError(
            "The texts of the gold and parsed trees do not match in " +
            f"'{doc_name}'.\n" +
            f"Gold:\n\t{gold_text}\n---\nParsed:\n\t{parsed_text}"
        )

    gold_constituents = tree_spans_2_token_spans(gold)
    gold_edus = {k: v for k, v in gold_constituents.items() if k[0] == k[1]}
    parsed_constituents = tree_spans_2_token_spans(parsed)
    parsed_edus = {k: v for k, v in parsed_constituents.items() if k[0] == k[1]}
    token_spans = sorted(list(set(
        list(gold_constituents.values()) +
        list(parsed_constituents.values())
    )), key=lambda x: (x[0], x[1] - x[0]))
    # remove root spans
    token_spans.remove(max(  # gold root span
        gold_constituents.values(), key=lambda x: (x[1]-x[0], x[0])
    ))
    if (parsed_root_span := max(
        parsed_constituents.values(), key=lambda x: (x[1]-x[0], x[0])
    )) in token_spans:
        token_spans.remove(parsed_root_span)
    
    # align constituent spans
    df = pd.DataFrame({
        ("EDUs", "gold"): [
            True if span in gold_edus.values() else False
            for span in token_spans
        ],
        ("EDUs", "parsed"): [
            True if span in parsed_edus.values() else False
            for span in token_spans
        ],
        ("Spans", "gold"): [
            True if (
                span in gold_constituents.values()
                and span not in gold_edus.values()
            ) else False
            for span in token_spans
        ],
        ("Spans", "parsed"): [
            True if (
                span in parsed_constituents.values()
                and span not in parsed_edus.values()
            ) else False
            for span in token_spans
        ],
        ("Nuclearity", "gold"): [None] * len(token_spans),
        ("Nuclearity", "parsed"): [None] * len(token_spans),
        ("Relations", "gold"): [None] * len(token_spans),
        ("Relations", "parsed"): [None] * len(token_spans),
    }, index=token_spans)
    df.index.name = "token_span"

    # add nuclearity & (coarse or fine) relation labels
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


def eval_doc(
    doc: str, parser: str,
    to_latex: bool = True
) -> str | pd.DataFrame:
    """Evaluate a single document by aligning the constituents of the gold
    and parsed RST trees.
    
    :param doc: The base name of the document (without parser suffix).
    :type doc: `str`
    :param parser: The name of the parser (used as directory & file suffix).
    :type parser: `str`
    :param to_latex: Whether to return the results as a LaTeX table. Defaults to
        `True`.
    :type to_latex: `bool`, optional

    :return: A LaTeX table or DataFrame with a multi-index of token spans and
        columns for gold & parsed spans, nuclearity, and relation labels.
    :rtype: `str` or `pd.DataFrame`
    """
    def latex_formatter(val):
        if val is True:
            return r"\checkmark"
        elif val is False or val is None or pd.isna(val) or str(val).lower() == "none":
            return ""
        elif isinstance(val, str):
            if val == "Nucleus":
                return "N"
            elif val == "Satellite":
                return "S"
            return val.capitalize()
        return val
    def df_to_latex(df: pd.DataFrame) -> str:
        def indent_latex_table(latex_str):
            lines = latex_str.splitlines()
            indented_lines = []
            for line in lines:
                if line.strip().startswith(r"\begin{table}") or line.strip().startswith(r"\end{table}"):
                    indented_lines.append('\t' + line)
                else:
                    indented_lines.append('\t\t' + line)
            return '\n'.join(indented_lines)
        # Format index for LaTeX
        df.index = [f"{a},\\text{{ {b-1} }}" for a, b in df.index]
        # Format all cells
        df_fmt = df.map(latex_formatter)
        # Set column format
        latex_str = df_fmt.to_latex(
            escape=False,
            column_format="D{,}{,}{-1}|cc|cc|cc|cc|cc"
        )
        # Replace header lines programmatically
        lines = latex_str.splitlines()
        # Replace the two header lines after \toprule
        lines[2] = (
            r"\multicolumn{1}{c}{\textbf{Token-span}} & "
            r"\multicolumn{2}{c}{\textbf{EDUs}} & "
            r"\multicolumn{2}{c}{\textbf{Spans}} & "
            r"\multicolumn{2}{c}{\textbf{Nuclearity}} & "
            r"\multicolumn{2}{c}{\textbf{Relations} \small(Fine)} & "
            r"\multicolumn{2}{c}{\textbf{Relations} \small(Coarse)} \\"
        )
        lines[3] = " & gold & parsed & gold & parsed & gold & parsed & gold & parsed & gold & parsed \\\\"
        latex_str = "\n".join(lines)
        # Remove the token_span line if present
        latex_str = re.sub(r"^token_span\s*&.*?\\\\\n", "", latex_str, flags=re.MULTILINE)
        # Add table environment, caption, and label
        experiment = (
            "LLM (Experiment 1)" if parser == "llm_without_linebreaks"
            else "LLM (Experiment 2)" if parser == "llm_with_linebreaks"
            else "LLM (Experiment 3)" if parser == "llm_presegmented"
            else parser.upper()
        )
        caption = f"\\texttt{{{doc.replace('_', r'\_')}}} after alignment; Experiment: {experiment}"
        label = f"tab:aligned_{doc}_{experiment.lower().replace(' (', '-').replace(' ', '_').replace('()', '')}"
        latex_str = (
            "\\begin{table}\n"
            "\\resizebox{\\textwidth}{!}{\n"
            f"{latex_str}"
            "}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\end{table}"
        )
        return indent_latex_table(latex_str)

    gold_rs3 = f"C:/Users/SANDHAP/Repos/rst-parsing/data/gold_annotations/{doc}.rs3"
    parsed_rs3 = f"C:/Users/SANDHAP/Repos/rst-parsing/data/parsed/{parser}/{doc}_{parser}.rs3"
    gold = Node.from_rs3(
        gold_rs3,
        exclude_disjunct_segments=True
    )
    parsed = Node.from_rs3(
        parsed_rs3,
        exclude_disjunct_segments=True
    )
    # Get fine (unmapped) relations
    df_fine = align_constituents(
        gold, parsed, doc_name=doc,
        map_relations=False
    )
    # Get coarse (mapped) relations
    df_coarse = align_constituents(
        gold, parsed, doc_name=doc,
        map_relations=True
    )
    # Build new MultiIndex columns for relations
    # Select all columns except "Relations"
    base_cols = [col for col in df_fine.columns if col[0] != "Relations"]
    # Build MultiIndex for relations
    fine_rel = df_fine[("Relations", "gold")], df_fine[("Relations", "parsed")]
    coarse_rel = df_coarse[("Relations", "gold")], df_coarse[("Relations", "parsed")]
    # Concatenate all columns with correct MultiIndex
    arrays = [
        [col[0] for col in base_cols] + ["Relations (fine)", "Relations (fine)", "Relations (coarse)", "Relations (coarse)"],
        [col[1] for col in base_cols] + ["gold", "parsed", "gold", "parsed"]
    ]
    new_columns = pd.MultiIndex.from_arrays(arrays)
    df = pd.concat(
        [df_fine[base_cols], fine_rel[0].rename(("Relations (fine)", "gold")), fine_rel[1].rename(("Relations (fine)", "parsed")),
         coarse_rel[0].rename(("Relations (coarse)", "gold")), coarse_rel[1].rename(("Relations (coarse)", "parsed"))],
        axis=1
    )
    df.columns = new_columns
    if to_latex:
        return df_to_latex(df)
    return df


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
        self._caption: Optional[str] = None
        self.aligned: Dict[str, pd.DataFrame] = self._align_all(map_relations)
        self.metrics: pd.DataFrame = self._compute_metrics()
        self.conf_matrix: plt.Figure = self._plot_confusion_matrix()

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
        
        self.name = Path(parsed_dir).name
        caption = ""
        for i, splt in enumerate(Path(parsed_dir).name.split("_")):
            if i == 0:
                caption += splt.upper().strip()
            else:
                caption += f" ({splt.strip().capitalize()})"
        self._caption = caption

        return gold, parsed

    def _precision(
        self, aligned: pd.DataFrame,
        category: Literal["EDUs", "Spans", "Nuclearity", "Relations"]
    ) -> float:
            if category in ["EDUs", "Spans"]:
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
        category: Literal["EDUs", "Spans", "Nuclearity", "Relations"]
    ) -> float:
            if category in ["EDUs", "Spans"]:
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
        category: Literal["EDUs", "Spans", "Nuclearity", "Relations"]
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
            for cat in ["EDUs", "Spans", "Nuclearity", "Relations"]:
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

    def _plot_confusion_matrix(
        self, figsize: Tuple[int, int] = (10, 8), cmap: str = "Blues",
        include_empty_columns: bool = True
    ) -> plt.Figure:
        """Plot a confusion matrix for relation labels across all documents.
        
        :param figsize: Size of the figure. Defaults to `(10, 8)`.
        :type figsize: `Tuple[int, int]`, optional
        :param cmap: matplotlib clormap for the heatmap. Defaults to `"Blues"`.
        :type cmap: `str`, optional
        :param include_empty_columns: Whether to include empty columns (i.e.
            those labels that weren't used by the parser) in the confusion
            matrix. Defaults to `False`.
        :type include_empty_columns: `bool`, optional
        
        :return: The confusion matrix as a matplotlib figure.
        :rtype: `plt.Figure`
        """
        all_gold = []
        all_pred = []
        all_gold_labels = set()

        for df in self.aligned.values():
            gold = df[("Relations", "gold")].map(
                lambda x: map_fine2coarse(x, replace_unknown=False)
            )
            pred = df[("Relations", "parsed")].map(
                lambda x: map_fine2coarse(x, replace_unknown=False)
            )
            mask = gold.notna() & pred.notna()
            all_gold.extend(gold[mask])
            all_pred.extend(pred[mask])
            all_gold_labels.update(gold.dropna().unique())
        all_gold_labels = sorted(
            lbl for lbl in all_gold_labels
            if pd.notna(lbl)
        )

        conf_mat = pd.crosstab(
            pd.Series(all_gold, name="Gold"),
            pd.Series(all_pred, name="Parsed")
        )
        if include_empty_columns:
            conf_mat = conf_mat.reindex(
                index=all_gold_labels,
                columns=all_gold_labels,
                fill_value=0
            )

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False,
            square=True
        )
        ax.tick_params(axis='both', which='both', length=0)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='left')
        plt.ylabel("Gold", fontweight="bold")
        plt.xlabel("Parsed", fontweight="bold")
        plt.tight_layout()
        fig = ax.get_figure()
        plt.close(fig)  # prevent double display in notebooks
        return fig

    def plot_relation_distribution(
        self, which: Literal["gold", "parsed"] = "gold",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        dists = {}
        for doc, df in self.aligned.items():
            if which == "gold":
                counts = df.Relations.gold.value_counts()
            else:
                counts = df.Relations.parsed.value_counts()
            dists[doc] = counts
        df = pd.DataFrame.from_dict(dists).fillna(0).astype(int).T
        ax = df.plot(kind='bar', stacked=True, figsize=figsize)
        ax.legend(
            title="Relation label", loc='upper right',
            bbox_to_anchor=(0.98, 0.98)
        )
        ax.set_xticklabels(
            [label.get_text().replace('blogposts_', '').replace('pcc_', '')
             for label in ax.get_xticklabels()],
            rotation=0, fontfamily='monospace'
        )
        ax.tick_params(axis='x', which='both', length=0)
        plt.title(
            f"Distribution of (coarse) relation labels in {which} annotations",
            fontweight="bold"
        )
        plt.tight_layout()
        fig = ax.get_figure()
        plt.close(fig)  # prevent double display in notebooks
        return fig

    def styled(self) -> Styler:
        """Get a styled version of the metrics table for better visualization
        in Jupyter Notebooks.

        :param caption: Optional caption for the table. If not provided,
            the name is inferred from the different file-name-suffixes of gold
            and parsed files, if possible. Defaults to `None`.
        :type caption: `str`, optional

        """
        return self.metrics.style.set_table_styles([
            # Divider above the Average row
            {'selector': 'tbody tr:last-child', 'props': [
                ('border-top', '1px solid black')
            ]}
        ]).highlight_max(
            subset=[
                ("EDUs", "F1"), ("Spans", "F1"),
                ("Nuclearity", "F1"), ("Relations", "F1")
            ],
            color="black",
            axis=0,
            props="textbf:--rwrap;"
        ).map_index(
            lambda x: "textit:--rwrap;" if x == "Average" else "",
            axis=0, level=0
        ).format_index(
            escape="latex"
        ).format(
            precision=3
        )

    def to_latex(self, caption: str = None, label: str = None) -> str:
        """Export the metrics table to LaTeX format.
        
        :param caption: Optional caption for the table. If not provided,
            the name is inferred from the different file-name-suffixes of gold
            and parsed files, if possible. Defaults to `None`.
        :type caption: `str`, optional
        """
        def fix_latex_table(latex_str, caption, label):
            # Remove HTML style line
            latex_str = re.sub(r'\\tbody.*?black\n', '', latex_str)

            # Wrap multicol titles in \textbf{}
            latex_str = re.sub(
                r'(\\multicolumn\{\d+\}\{\w\}\{)([^\}]+)',
                lambda m: "".join([
                    m.group(1),
                    "\\textbf{",
                    m.group(2),
                    "}"
                ]),
                latex_str
            )

            # Combine header lines
            latex_str = re.sub(
                r' & Recall & Precision & F1 & Recall & Precision & F1 & Recall & Precision & F1 & Recall & Precision & F1 \\\\\nDocument &  &  &  &  &  &  &  &  &  &  &  &  \\\\\n',
                r'Document & Recall & Precision & F1 & Recall & Precision & F1 & Recall & Precision & F1 & Recall & Precision & F1 \\\\\n',
                latex_str
            )

            # Add \midrule above Average row
            latex_str = re.sub(
                r'(\\textit{Average}.*?\\\\)',
                r'\\midrule\n\1',
                latex_str
            )

            # Wrap tabular in resizebox
            if (
                tabular_match := re.search(
                    r'(\\begin{tabular}.*?\\end{tabular})',
                    latex_str, re.DOTALL
                )
            ):
                tabular_block = tabular_match.group(1)
                resized_tabular = f'\t\\resizebox{{\\textwidth}}{{!}}{{\n{tabular_block}}}'
                latex_str = latex_str.replace(tabular_block, resized_tabular)

            # Insert caption and label below tabular-environment
            if caption is None:
                caption = ""
            if label is None:
                label = ""
            latex_str = re.sub(
                r"(\\end\{table\*\})",
                lambda m: "".join([
                    "\t\\caption{",
                    caption,
                    "}\n\t\\label{",
                    label,
                    "}\n",
                    m.group(1)
                ]),
                latex_str, re.DOTALL
            )

            return latex_str

        if caption is None:
            caption = self._caption or ""
        if label is None:
            label = f"tab:{self.name}" if self.name is not None else ""
        return fix_latex_table(
            self.styled().to_latex(
                environment="table*",
                column_format="l|rrr|rrr|rrr|rrr",
                multicol_align="c",
                hrules=True
            ),
            caption,
            label
        )


def main(
    parsed_parent_dir: str = "parsed",
    exclude_disjunct_segments: bool = True
) -> Dict[str, RSTEval]:
    """Evaluate all parsers in the provided parent directory by comparing
    their parsed `.rs3` files to the gold `.rs3` files.

    :param parsed_parent_dir: Name of the parent directory in the data
        directory containing subdirectories for each parser, each with
        their respective parsed `.rs3` files.
    :type parsed_parent_dir: `str`
    :param exclude_disjunct_segments: Whether to exclude disjunct segments
        such as headings without an annotated parent when loading the
        parsed `.rs3` files.
    :type exclude_disjunct_segments: `bool`, optional

    :return: A dictionary mapping parser names to their evaluation results.
    :rtype: `Dict[str, RSTEval]`
    """
    data_dir = Path(f"{Path(os.getcwd()).parent}/data").as_posix()
    gold_dir = Path(f"{data_dir}/gold_annotations").as_posix()
    return {
        p_dir: RSTEval(
            f"{data_dir}/{parsed_parent_dir}/{p_dir}",
            gold_dir,
            exclude_disjunct_segments=exclude_disjunct_segments
        ) for p_dir in os.listdir(f"{data_dir}/{parsed_parent_dir}")
        if os.path.isdir(f"{data_dir}/{parsed_parent_dir}/{p_dir}")
    }


if __name__ == "__main__":
    all_results = main()
