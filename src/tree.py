#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Mon, 14.09.25                                                     #
# ========================================================================== #

"""Helpers for converting DMRST output to RS3 format."""

from __future__ import annotations
from bs4 import BeautifulSoup, Tag
from bs4.formatter import XMLFormatter
from nltk import word_tokenize
import os
import re
from typing import Dict, Generator, Iterator, List, Literal, Optional, Tuple, Union


class GroupNode(dict):
    def __init__(
        self,
        span=None,
        parent_span=None,
        children_spans=None,
        depth=None,
        height=None,
        id=None,
        parent=None,
        nuc=None,
        relname=None,
        type=None,
        edus={}
    ):
        if span is not None and span[0] == span[1]:
            id_ = span[0]
            text = edus.get(span[0] - 1, None)
        else:
            id_ = id
            text = None
        super().__init__({
            "span": span,
            "parent_span": parent_span,
            "children_spans": children_spans,
            "depth": depth,
            "height": height,
            "id": id_,
            "parent": parent,
            "nuc": nuc,
            "relname": relname,
            "type": type,
            "text": text
        })
        if self.get("text") is None:
            self.pop("text", None)


class RS3Formatter(XMLFormatter):
    def __init__(self, *args, **kwargs):
        super(XMLFormatter, self).__init__(
            language=self.XML,
            *args, **kwargs
        )

    def attributes(self, tag):
        if tag.name == "segment":
            return [
                (k, tag.attrs.get(k))
                for k in ["id", "parent", "relname"]
                if tag.attrs.get(k) is not None
            ]
        elif tag.name == "group":
            return [
                (k, tag.attrs.get(k))
                for k in ["id", "type", "parent", "relname"]
                if tag.attrs.get(k) is not None
            ]
        else:
            return [
                (k, v) for k, v in tag.attrs.items()
                if v is not None
            ]


def dmrst_nodes(
    dmrst_tree: Union[str, List[str]],
    edu_spans: Dict[int, str],
    include_lvl_spans: bool = False
) -> Dict[Tuple[int, int], Dict[str, Union[str, int, Tuple[int, int]]]]:
    def get_root(nodes: Dict[Tuple[int, int], Dict]) -> Tuple[int, int]:
        root_ = (spans := sorted(
            list(nodes.keys()),
            key=lambda x: (x[0] - x[1], x[0])
        ))[0]
        for span in spans:
            if span[0] < root_[0] or span[1] > root_[1]:
                raise ValueError("Nodes contain multiple roots.")
        return root_

    def get_span(
        node: Union[int, Dict, Tuple[int, int]],
        nodes: Dict[Tuple[int, int], Dict]
    ) -> Optional[Tuple[int, int]]:
        if isinstance(node, tuple):
            return node
        if isinstance(node, int):
            if len(nodes_ := [v for v in nodes.values() if v.get("id") == node]) == 0:
                raise KeyError(f"No node exists with ID={node}.")
            node = nodes_[0]
        if (span := node.get("span")) is None:
            if (id_ := node.get("id")) is None:
                return None
            return list(k for k, v in nodes.items() if v["id"] == id_)[0]
        return span

    def get_node(
        node: Union[int, Dict, Tuple[int, int]],
        nodes: Dict[Tuple[int, int], Dict]
    ) -> Optional[Dict]:
        if (span := get_span(node, nodes)) is None:
            raise ValueError(f"Node-span {span} is not a valid span.")
        if all(isinstance(k, int) for k in nodes.keys()):
            if len(candidates := [
                v for v in nodes.values() if v.get("span") == span
            ]) == 0:
                raise KeyError(f"No node exists with span={span}.")
            return candidates[0]
        return nodes.get(span)

    def get_parent(
        node: Union[int, Dict, Tuple[int, int]],
        nodes: Union[
            Dict[int, Dict],
            Dict[Tuple[int, int], Dict]
        ]
    ) -> Optional[Tuple[int, int]]:
        if all(isinstance(k, int) for k in nodes.keys()):
            nodes = {v["span"]: v for v in nodes.values()}
        if not isinstance(node, tuple):
            node = get_span(node, nodes)
        if node == (root := get_root(nodes)):
            return None
        if len(candidates := sorted(list(
            span for span in nodes.keys() if (
                span[0] == node[0]
                and span[1] > node[1]
            ) or (
                span[0] < node[0]
                and span[1] == node[1]
            )
        ), key=lambda x: x[1] - x[0])) == 0:
            raise ValueError(f"Span {node} doesn't have an immediate parent.")
        return candidates[0]

    def get_children(
        node: Union[int, Dict, Tuple[int, int]],
        nodes: Union[
            Dict[int, Dict],
            Dict[Tuple[int, int], Dict]
        ],
        reverse: bool = False
    ) -> List[Optional[Tuple[int, int]]]:
        span = get_span(node, nodes)
        if all(isinstance(k, int) for k in nodes.keys()):
            nodes = {v["span"]: v for v in nodes.values()}
        candidates = list(k for k in nodes.keys() if (
                            k[0] == span[0] and k[1] < span[1]
                        ) or (
                            k[0] > span[0] and k[1] == span[1]
                        ))
        if len(candidates) == 0:
            return []
        children = []
        if len(l_cands := sorted([
            c for c in candidates if c[0] == span[0]
        ], key=lambda x: x[0] - x[1])) > 0:
            children.append(l_cands[0])
        if len(r_cands := sorted([
            c for c in candidates if c[1] == span[1]
        ], key=lambda x: x[0] - x[1])) > 0:
            children.append(r_cands[0])
        if reverse:
            return children[::-1]
        return children

    def _get_height(
        node: Union[int, Dict, Tuple[int, int]],
        nodes: Dict[Tuple[int, int], Dict]
    ) -> int:
        node = get_node(node, nodes)
        if len(children := node["children_spans"]) == 0:
            return 0
        return 1 + max(
            _get_height(ch, nodes) for ch in children
        )

    def _get_depth(
        node: Union[int, Dict, Tuple[int, int]],
        nodes: Dict[Tuple[int, int], Dict]
    ) -> int:
        node = get_node(node, nodes)
        if (parent := node.get("parent_span")) is None:
            return 0
        return 1 + _get_depth(parent, nodes)

    node_pat = r'\(?(\d+):(.+?)=(.+?):(\d+)\)?'
    span_pat = rf"\(?{node_pat},{node_pat}\)?"
    if isinstance(dmrst_tree, list):
        dmrst_tree = " ".join(dmrst_tree)

    nodes = {}
    for i, m in enumerate(m for m in re.finditer(
        span_pat, dmrst_tree
    ) if m is not None):
        lvl_span = (int(m.group(1)), int(m.group(8)))
        l_span = (lvl_span[0], int(m.group(4)))
        r_span = (int(m.group(5)), lvl_span[1])
        l_rel, l_nuc = m.group(3), m.group(2)
        r_rel, r_nuc = m.group(7), m.group(6)
        
        _nodes = [
            {
                "span": l_span,
                "nuc": l_nuc,
                "relname": l_rel
            },
            {
                "span": r_span,
                "nuc": r_nuc,
                "relname": r_rel
            }
        ]
        if i == 0:  # add root
            _nodes.insert(0, {
                "span": lvl_span
            })
        elif include_lvl_spans:
            _nodes.insert(0, {
                "span": lvl_span
            })
        
        # set IDs & parents
        for node in _nodes:
            if (span := node["span"])[0] == span[1]:
                node["id"] = span[0]
            elif len(nodes) == 0:
                node["id"] = len(edu_spans) + len(nodes) + 1
            else:
                node["id"] = max(list(
                    v.get("id") for v in nodes.values()
                )) + 1
            
            if len(nodes) == 0:
                node["parent"] = None
            else:
                node["parent_span"] = get_parent(span, nodes)

            # add node
            nodes[span] = GroupNode(**node, edus=edu_spans)
    
    # set children
    for span in nodes:
        nodes[span]["children_spans"] = get_children(span, nodes)

    # calculate height & depth
    for span in nodes:
        nodes[span]["height"] = _get_height(span, nodes)
        nodes[span]["depth"] = _get_depth(span, nodes)

    return nodes


def sorted_spans(
    nodes: Union[List[Tuple[int, int]], Dict[Tuple[int, int], Dict]],
    ascending: bool = True
) -> List[Tuple[int, int]]:
    if isinstance(nodes, dict):
        spans = list(nodes.keys())
    else:
        spans = nodes
    if ascending:
        sort_by = lambda x: (x[1] - x[0], x[0])
    else:
        sort_by = lambda x: (x[0] - x[1], x[0])
    return list(sorted(
        spans, key=sort_by
    ))


def get_child_spans(
    root_span: Tuple[int, int],
    nodes: Dict[Tuple[int, int], Dict],
    branch: Literal["left", "right", "all"]
) -> List[Tuple[int, int]]:
    if root_span not in nodes:
        raise KeyError(f"{root_span} is not a valid root span.")
    if root_span[0] == root_span[1]:
        return []
    descendant_spans_left = [
        k for k in sorted_spans(nodes)
        if (
            k[0] == root_span[0]
            and k[1] < root_span[1]
            and k != root_span
        )
    ]
    descendant_spans_right = [
        k for k in sorted_spans(nodes)
        if (
            k[1] == root_span[1]
            and k[0] > root_span[0]
            and k != root_span
        )
    ]
    if branch == "left":
        if len(descendant_spans_left) == 0:
            return []
        return [descendant_spans_left[-1]]
    elif branch == "right":
        if len(descendant_spans_right) == 0:
            return []
        return [descendant_spans_right[-1]]
    else:
        if len(descendant_spans_left + descendant_spans_right) == 0:
            return []
        return [descendant_spans_left[-1], descendant_spans_right[-1]]


def get_parent_span(
    nodes: Union[List[Tuple[int, int]], Dict[Tuple[int, int], Dict]],
    child_span: Optional[Tuple[int, int]] = None
) -> Optional[Tuple[int, int]]:
    spans = sorted_spans(nodes)
    if child_span == (root_span := spans[-1]):
        return None
    if child_span is None:
        return root_span
    elif child_span not in nodes:
        raise KeyError(f"{child_span} is not a valid child span.")
    return [
        span for span in spans
        if (
            span[0] == child_span[0]
            and span[1] > child_span[1]
        ) or (
            span[0] < child_span[0]
            and span[1] == child_span[1]
        )
    ][0]


def get_sibling_spans(
    span: Tuple[int, int],
    nodes: Union[List[Tuple[int, int]], Dict[Tuple[int, int], Dict]]
) -> List[Optional[Tuple[int, int]]]:
    spans = sorted_spans(nodes.keys())
    if (parent_span := get_parent_span(spans, span)) is None:
        return []
    if len(sibling_spans := [
        sib for sib in get_child_spans(parent_span, nodes, "all")
        if sib != span
    ]) == 0:
        return []
    return sibling_spans


def get_depth(
    span: Tuple[int, int],
    spans: List[Tuple[int, int]]
) -> int:
    if (parent_span := get_parent_span(spans, span)) is None:
        return 0
    return 1 + get_depth(parent_span, spans)


def get_height(
    span: Tuple[int, int],
    spans: List[Tuple[int, int]]
) -> int:
    if len(children := get_child_spans(span, spans, "all")) == 0:
        return 0
    return 1 + max(
        get_height(child_span, spans) for child_span in children
    )


def get_spans(
    nodes: Union[List[Tuple[int, int]], Dict[Tuple[int, int], Dict]],
    depth: int = -1  # deepest
) -> List[Tuple[int, int]]:
    if len(spans := sorted_spans(nodes)) == 0:
        return []
    max_depth = max(get_depth(span, spans) for span in spans)
    if depth == -1:
        depth = max_depth
    elif depth > max_depth:
        raise ValueError(
            f"No spans exist at depth {depth}. " +
            f"Maximum depth is {max_depth}."
        )
    return sorted([
        span for span in spans
        if get_depth(span, spans) == depth
    ], key=lambda x: (x[0], x[1] - x[0]))


def get_adjoining_spans(
    nodes: Union[List[Tuple[int, int]], Dict[Tuple[int, int], Dict]],
    depth: int = -1  # deepest
) -> List[Tuple[Tuple[int, int]]]:
    if len(spans := sorted_spans(nodes)) == 0:
        return []
    spans = get_spans(spans, depth=depth)
    adjoining = []
    while len(spans) > 0:
        pair = [spans.pop(0)]
        if len(spans) > 0 and spans[0][0] == pair[-1][-1] + 1:
            pair.append(spans.pop(0))
        adjoining.append(tuple(pair))
    return adjoining


class Node:
    _multinuc_relations = ["Joint", "Same-Unit", "Temporal"]

    def __init__(
        self,
        attrs: Dict,
        children: List[Optional[Union[Node, Dict]]] = [],
        parent: Optional[Union[Node, Dict]] = None
    ):
        self.id = attrs.get("id")
        self.nuc = attrs.get("nuc")
        self.text = attrs.get("text")
        self.span = attrs.get("span")
        self.children = children
        self.parent = parent
        self.siblings = attrs.get("siblings", [])
        self.relname = attrs.get("relname")
        self.type = self.children

    def __repr__(self) -> str:
        return "\n".join([
            f"Node< id={self.id}",
            "\n".join([
                f"      {k}={str(v)}"
                for k, v in list(self.to_dict().items())[1:-2]
            ]),
            f"      nuc={self.nuc} >"
        ])

    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> Node:
        def search(node):
            if isinstance(key, int):
                if node.id == key:
                    return node
            elif isinstance(key, tuple):
                if node.span is not None and node.span == key:
                    return node
            else:
                raise TypeError(
                    f"Node-key must be of {type(int)} or {type(tuple)}. Got {type(key)}."
                )
            for child in node.children:
                if child is not None:
                    try:
                        return search(child)
                    except KeyError:
                        continue
            raise KeyError(f"No node with key={key} found in the tree.")
        return search(self.root)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._traverse("level-order"))

    def _traverse(
        self,
        order: Literal["pre-order", "post-order", "level-order"]
    ) -> Generator[Node, None, None]:
        if order == "pre-order":
            yield self
            for child in self.children:
                if child is not None:
                    yield from child._traverse(order)
        elif order == "post-order":
            for child in self.children:
                if child is not None:
                    yield from child._traverse(order)
            yield self
        elif order == "level-order":
            queue = [self]
            while queue:
                node = queue.pop(0)
                yield node
                queue.extend([c for c in node.children if c is not None])
        else:
            raise ValueError(f"Unknown traversal order: {order}")

    def _ancestors(self) -> Generator[Node, None, None]:
        """Get all ancestors of this node, starting from the parent up to and
        including the root."""
        node = self
        while (parent := node.parent) is not None:
            yield parent
            node = parent
        yield node

    def _descendants(
        self,
        branch: Literal["left", "right", "all"]
    ) -> Generator[Node, None, None]:
        if branch == "left":
            if len(self.children) > 0 and self.children[0] is not None:
                yield self.children[0]
                yield from self.children[0]._descendants(branch)
        elif branch == "right":
            if len(self.children) > 1 and self.children[1] is not None:
                yield self.children[1]
                yield from self.children[1]._descendants(branch)
        else:
            for child in self.children:
                if child is not None:
                    yield child
                    yield from child._descendants(branch)

    def lca(self, other: Node) -> Optional[Node]:
        """Get the lowest common ancestor of this node and another node.
        
        If no common ancestor exists, `None` is returned.
        """
        other_ancestors = list(other._ancestors())
        for ancestor in self._ancestors():
            if ancestor in other_ancestors:
                return ancestor
        return None

    def edu_span2id(self, span: Tuple[int, int]) -> int:
        """Convert a span of EDU-indices to the `Node.id` of the corresponding
        node.
        
        :raises KeyError: If any EDU in the span is not found in the tree, or
            if no node with the exact span exists in the tree.
        :raises ValueError: If any ID in the span doesn't belong to an EDU
            (i.e. a leaf node).
        """
        for edu_id in span:
            try:
                edu = self[edu_id]
                if edu.text is None:
                    raise ValueError(f"EDU with id={edu_id} is not a leaf node.")
            except KeyError:
                raise KeyError(f"EDU with id={edu_id} not found in tree.")
        try:  # look up the span directly
            return self[span].id
        except KeyError:  # span might not be set
            left, right = self[span[0]], self[span[1]]
            if (lca := left.lca(right)) is not None:
                if lca.span is not None:
                    if (left.id, right.id) == lca.span:
                        return lca.id
                    else:
                        raise KeyError(
                            f"No node with span=({left.id}, {right.id}) found."
                        )
                else:  # set span of LCA for future reference
                    lca.span = (left.id, right.id)
                    return lca.id

    def get_edu_span(self) -> Tuple[int, int]:
        """Get the (non-unique) span of EDU IDs covered by this node."""
        edu_descendants = sorted(list(
            n for n in self._descendants("all") if n.text is not None
        ), key=lambda x: x.id)
        if len(edu_descendants) == 1:
            return (edu_descendants[0].id, edu_descendants[0].id)
        return (edu_descendants[0].id, edu_descendants[-1].id)

    @property
    def root(self) -> Node:
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    @property
    def children(self) -> List[Optional[Node]]:
        if not hasattr(self, "_children"):
            self._children = []
        return self._children
    
    @children.setter
    def children(self, children: List[Optional[Union["Node", Dict]]]):
        node_children = []
        for child in children:
            if isinstance(child, dict):
                node_child = Node(child, parent=self)
            else:
                node_child = child
                node_child.parent = self
            node_children.append(node_child)
        self._children = node_children

        if len(self._children) == 2:
            if not all(child.nuc == "Nucleus" for child in self._children):
                nuc_child = next(
                    (child for child in self._children
                     if child.nuc == "Nucleus"),
                    None
                )
                sat_child = next(
                    (child for child in self._children
                     if child.nuc == "Satellite"),
                    None
                )
                if nuc_child is not None and sat_child is not None:
                    nuc_child.parent = self
                    nuc_child.siblings = []
                    sat_child.parent = nuc_child
                    sat_child.siblings = []
                else:
                    raise ValueError(
                        "At least one child must be a Nucleus if there are two children."
                    )

    @property
    def parent(self) -> Optional[Node]:
        if not hasattr(self, "_parent"):
            self._parent = None
        return self._parent
    
    @parent.setter
    def parent(self, parent: Optional[Union[Node, Dict]]):
        if isinstance(parent, dict):
            parent = Node(parent, children=[self])
        elif parent is not None:
            self._parent = parent

    @property
    def siblings(self) -> List[Optional[Node]]:
        if not hasattr(self, "_siblings"):
            self.siblings = []
        return self._siblings

    @siblings.setter
    def siblings(
        self, siblings: List[Optional[Union[
            Dict, Node
        ]]] = []
    ):
        _siblings = (
            [s for s in self._siblings]
            if hasattr(self, "_siblings") else []
        )
        if len(siblings) == 0:
            self._siblings = _siblings
            return
        elif all(
            isinstance(s, dict) for s in siblings
        ):
            siblings = [
                Node(s) for s in siblings
                if s["span"] not in _siblings
            ]
        elif all(
            isinstance(s, Node) for s in siblings
        ):
            spans = []
            for s in siblings:
                try:
                    self[s.id]
                except KeyError:  # not yet in tree -> add
                    spans.append(s)
            siblings = spans
        elif all(
            isinstance(s, tuple) for s in siblings
        ):
            spans = []
            for s in siblings:
                try:  # append if already in tree
                    spans.append(self[s[0]])
                except KeyError:  # not yet in tree -> can't create from span
                    spans.append(s)
            siblings = spans
        else:
            raise TypeError(
                f"Expected siblings to be a list of Dicts or Nodes but got {type(siblings[0])}."
            )

        siblings_ = _siblings + siblings
        if len(siblings_) == 1:  # mononuc
            try:
                sibling = self[siblings[0]]
                if self.nuc == "Nucleus":
                    if sibling.nuc != "Nucleus":
                        sibling.parent = self
                        sibling.siblings = []
                else:  # change parent pointer to N
                    self.parent = sibling
                    siblings_ = []
            except KeyError:  # not in tree yet
                pass
            
        self._siblings = siblings_

    @property
    def type(self) -> Optional[Literal["span", "multinuc"]]:
        self.type = self._children
        return self._type

    @type.setter
    def type(self, children: List[Optional[Node]]):
        if len(children) == 2:
            if all(
                (child.nuc == "Nucleus" if isinstance(child, Node)
                 else child["nuc"] == "Nucleus")
                for child in children
            ):
                self._type = "multinuc"
            else:
                self._type = "span"
        else:
            self._type = None

    @property
    def span(self) -> Optional[Tuple[int, int]]:
        if not hasattr(self, "_span"):
            self._span = None
        return self._span
    
    @span.setter
    def span(self, span: Optional[Tuple[int, int]]):
        if span is None:
            if self.text is not None:  # leaf node
                self._span = (self.id, self.id)
            else:  # get left- & right-most descendants
                if len(descendants := sorted(list(
                    d for d in self._descendants("all")
                    if d.id < self.root.id  # only leafs
                ), key=lambda n: n.id)) > 1:
                    self._span = (descendants[0].id, descendants[-1].id)
                else:
                    self._span = None
        elif span is not None and (
                not isinstance(span, tuple) or len(span) != 2
                or not all(isinstance(i, int) and i >= 0 for i in span)
            ):
            raise TypeError(
                "Span must be a tuple of two non-negative integers. " +
                f"Got {type(span)}."
            )
        else:
            self._span = span

    @property
    def rs3_segments(self) -> List[Node]:
        return sorted([
            node for node in self._traverse("level-order")
            if node.text is not None
        ], key=lambda n: n.id)
    
    @property
    def rs3_groups(self) -> List[Node]:
        return sorted([
            node for node in self._traverse("level-order")
            if node.text is None
        ], key=lambda n: n.id)

    @property
    def rs3_relations(self) -> List[Tuple[str, str]]:
        multinuc_relations = set()
        mononuc_relations = set()
        for node in self.root:
            if node.relname != "span" and node.relname is not None:
                if node.relname in Node._multinuc_relations:
                    multinuc_relations.add(node.relname)
                else:
                    mononuc_relations.add(node.relname)
        relations = {
            (rel, "multinuc") for rel in multinuc_relations
        } | {
            (rel, "rst") for rel in mononuc_relations
        }
        return sorted(list(relations), key=lambda x: x[0])

    @property
    def edus(self) -> List[str]:
        """Sorted list of (unaltered) EDU-texts."""
        if not hasattr(self, "_edus"):
            self._edus = [
                edu.text for edu in sorted(
                    self.rs3_segments, key=lambda x: x.span[0]
                ) if edu.text is not None
            ]
        return self._edus

    @property
    def tokens(self) -> List[str]:
        """Canonicalized tokenization of the document's text."""
        if not hasattr(self, "_tokens"):
            self._tokens = Node.tokenize(self, "doc")
        return self._tokens

    @staticmethod
    def _add_ws_to_intra_sent_punct(text: str) -> str:
        return re.sub(
            r"(.)([,:;\-–—])(\s)?", r"\g<1> \g<2>\g<3>", text
        )
    
    @staticmethod
    def _replace_quotation_marks(text: str) -> str:
        text = re.sub(r'``|´´', r'"', text)
        text = re.sub(r'‘|’', r"'", text)
        text = re.sub(r"''", r'"', text)
        text = re.sub(r'“|”', r'"', text)
        return text

    @staticmethod
    def _rejoin_hyphenated_words(toks: List[str]) -> List[str]:
        tokens = []
        i = 0
        while i < len(toks):
            tok = toks[i]
            # If previous token ends with hyphen, join with current
            if (
                tokens
                and tokens[-1].endswith('-')
                and re.match(r'^[a-zA-Zäöüß\d]', tok)
            ):
                tokens[-1] += tok
            # If current token starts with hyphen, join with previous if exists
            elif (
                tokens
                and tok.startswith('-')
                and re.match(r'[a-zA-Zäöüß\d]$', tokens[-1])
            ):
                tokens[-1] += tok
            # If current token ends with hyphen and next token starts with letter/digit, join them
            elif (
                tok.endswith('-')
                and i + 1 < len(toks)
                and re.match(r'^[a-zA-Zäöüß\d]', toks[i + 1])
            ):
                tokens.append(tok + toks[i + 1])
                i += 1  # Skip next token
            else:
                tokens.append(tok)
            i += 1
            if (
                len(tokens) > 1
                and tokens[-1].startswith("-")
            ):
                tokens.append(
                    "".join([tokens.pop(-2), tokens.pop(-1)])
                )
        return [t.strip() for t in tokens if len(t.strip()) > 0]

    @staticmethod
    def tokenize(
        target: Union[Node, str], text: Literal["doc", "edus", "text"]
    ) -> Union[List[str], List[List[str]]]:
        """Apply canonicalized tokenization such that:
        - punctuation characters are separated iff they are at word boundaries
        - hyphenated words are re-joined and treated as single tokens
        - empty and whitespace-only tokens are excluded

        :param target: The target to tokenize. Either a Node object or a
            string of text.
        :type target: Union[Node, str]
        :param text: What to tokenize. If `target` is a Node, this can be
            `"doc"` (the full document text), `"edus"` (the EDUs as a list of
            strings), or `"text"` (the `Node.text` attribute). If `target` is
            a string, this parameter is ignored.
        :type text: Literal["doc", "edus", "text"]

        :return: A list of tokens or a list of lists of tokens (if
            `text="edus"`).
        :rtype: List[str] | List[List[str]]
        """
        if isinstance(target, str):
            return Node._rejoin_hyphenated_words(
                word_tokenize(
                    Node._replace_quotation_marks(
                        Node._add_ws_to_intra_sent_punct(target)
                    )
                )
            )
        else:  # Node
            if text == "text":  # tokenize only this Node's text (if any)
                if target.text is None:
                    return []
                return Node._rejoin_hyphenated_words(
                    word_tokenize(
                        Node._replace_quotation_marks(
                            Node._add_ws_to_intra_sent_punct(target.text.strip())
                        )
                    )
                )
            edus = [
                Node._replace_quotation_marks(
                    Node._add_ws_to_intra_sent_punct(edu.text.strip())
                )
                for edu in sorted(target.rs3_segments, key=lambda x: x.span[0])
            ]
            if text == "edus":
                return [
                    Node._rejoin_hyphenated_words(word_tokenize(edu))
                    for edu in edus
                ]
            return Node._rejoin_hyphenated_words(word_tokenize(" ".join(edus)))

    def add_child(self, child: Union["Node", Dict]):
        if isinstance(child, dict):
            child = Node(child, parent=self)
        else:
            child.parent = self
        if child.span in [c.span for c in self.children]:
            return
        self.children = sorted([child] + self.children, key=lambda c: c.span[0])

    def to_dict(self) -> Dict:
        d = {
            "id": self.id,
            "span": self.span,
            "parent": (
                self.parent.id if self.parent is not None else None
            ),
            "relname": self.relname,
        }
        if self.type is not None:
            d["type"] = self.type
        if self.text is not None:
            d["text"] = self.text
        d["children"] = [c.id for c in self.children]
        d["nuc"] = self.nuc
        return d

    def to_tag(self) -> Tag:
        if self.text is not None:
            tag = Tag(name="segment", is_xml=True)
            tag["id"] = self.id
            if self.parent is None:
                raise ValueError("Node is missing parent-ID.")
            tag["parent"] = self.parent.id
            tag["relname"] = self.relname
            tag.append(self.text)
        else:
            tag = Tag(name="group", is_xml=True)
            tag["id"] = self.id
            tag["type"] = self.type
            if self.parent is not None:
                tag["parent"] = self.parent.id
                tag["relname"] = self.relname
        return tag

    def to_rs3(self) -> str:
        soup = BeautifulSoup(features="xml")
        soup.append(soup.new_tag("rst"))
        soup.rst.extend([
            soup.new_tag("header"), soup.new_tag("body")
        ])
        soup.header.append(soup.new_tag("relations"))
        soup.body.append(
            soup.new_tag("segments")
        )
        soup.body.append(
            soup.new_tag("groups")
        )

        for relname, reltype in self.rs3_relations:
            reltag = soup.new_tag("rel")
            reltag["name"] = relname
            reltag["type"] = reltype
            soup.header.relations.append(reltag)
        soup.body.segments.extend([
            node.to_tag() for node in self.rs3_segments
        ])
        soup.body.groups.extend([
            node.to_tag() for node in self.rs3_groups
        ])

        soup.body.segments.unwrap()
        soup.body.groups.unwrap()
        soup.smooth()
        xml = "\n".join(str(soup.prettify(  # remove <?xml ...?>
            formatter=RS3Formatter(indent=4)
        )).splitlines()[1:])
        rs3_string = re.sub(  # put segment tags on one line
            r"(<segment [^>]+?>)\n\s*(.+?)\n\s*?(</segment)",
            r"\g<1>\g<2>\g<3>",
            xml,
            flags=re.M
        )
        return rs3_string

    @classmethod
    def from_xml(
        cls,
        rs3_str: str,
        exclude_disjunct_segments: bool = True
    ) -> Node:
        """Convert XML nodes to a Node representation.

        :param rs3_str: `.rs3`-XML content as a string.
        :type rs3_str: str
        :param exclude_disjunct_segments: Whether to exclude disjunct segments
        (i.e. segments without a parent, such as headings). Defaults to `True`.
        :type exclude_disjunct_segments: bool, optional
        :return: A Node object representing the root of the XML structure.
        :rtype: Node
        """
        def mononuc_rels(soup) -> List[str]:
            return [
                rel["name"]
                for rel in soup.find_all("rel")
                if rel.get("type") == "rst"
            ]

        def multinuc_rels(soup) -> List[str]:
            return [
                rel["name"]
                for rel in soup.find_all("rel")
                if rel.get("type") == "multinuc"
            ]

        def nuclearity(
            node: Dict
        ) -> Optional[Literal["Nucleus", "Satellite"]]:
            nonlocal mononuc, multinuc
            if (relname := node.get("relname")) is None:
                return None
            if relname == "span" or relname in multinuc:
                return "Nucleus"
            return "Satellite"

        def segments(
            soup, exclude_disjunct: bool = exclude_disjunct_segments
        ) -> List[Dict]:
            res = []
            for node in soup.find_all("segment"):
                if exclude_disjunct and node.get("parent") is None:
                    continue
                else:
                    res.append({
                        "id": int(node["id"]),
                        "parent": int(node.get("parent")),
                        "text": node.get_text(strip=False),
                        "type": node.get("type", None),
                        "relname": node.get("relname"),
                        "nuc": nuclearity(node),
                        "children": []
                    })
            return sorted(res, key=lambda x: x["id"])

        def groups(soup) -> List[Dict]:
            res = []
            for node in soup.find_all("group"):
                res.append({
                    "id": int(node["id"]),
                    "parent": int(node.get("parent")) if node.get("parent") else None,
                    "text": None,
                    "type": node.get("type"),
                    "relname": node.get("relname"),
                    "nuc": nuclearity(node),
                    "children": []
                })
            return sorted(res, key=lambda x: x["id"])

        def all_nodes(
            soup, exclude_disjunct: bool = exclude_disjunct_segments
        ) -> List[Dict]:
            return segments(soup, exclude_disjunct) + groups(soup)

        def assign_sequential_node_indices(
            nodes: List[Dict]
        ) -> Dict[int, Dict]:
            # Separate segments and groups
            segments = sorted(
                [n for n in nodes if n.get("text") is not None],
                key=lambda x: x["id"]
            )
            groups = sorted(
                [n for n in nodes if n.get("text") is None],
                key=lambda x: x["id"]
            )
            assert len(groups) + len(segments) == len(nodes)

            # Find the root group (parent is None)
            root_group = next(
                g for g in groups if g.get("parent") is None
            )
            groups = sorted(
                [g for g in groups if g is not root_group],
                key=lambda x: x["id"]
            )
            assert len(groups) == len(nodes) - len(segments) - 1

            # Assign new IDs
            old2new = {}

            # Segments: 1..N
            for edu_id, seg in enumerate(segments, start=1):
                old2new[seg["id"]] = edu_id
                seg["id"] = edu_id

            # Root group: N+1
            root_id = len(segments) + 1
            old2new[root_group["id"]] = root_id
            root_group["id"] = root_id

            # Other groups: N+2..N+1+M
            for group_id, group in enumerate(groups, start=root_id + 1):
                old2new[group["id"]] = group_id
                group["id"] = group_id

            # Update parent pointers for all nodes
            for node in segments + [root_group] + groups:
                if (par_id := node.get("parent")) is not None:
                    node["parent"] = old2new.get(par_id, None)

            # Build reindexed dict
            reindexed = {}
            for node in segments + [root_group] + groups:
                reindexed[node["id"]] = node

            return reindexed

        def collect_children(nodes: Dict[int, Dict]) -> Dict[int, Dict]:
            par2children = {k: [] for k in nodes.keys()}
            for child_id, child in nodes.items():
                if (par_id := child.get("parent")) is not None:
                    par2children[par_id].append(child_id)
            res = {}
            for node_id, node in nodes.items():
                node["children"] = sorted(list(set(par2children[node_id])))
                res[node_id] = node
            return res

        def build_node(node: Dict, nodes: Dict[int, Dict]) -> Node:
            return Node(node, children=[
                build_node(nodes[child_id], nodes)
                for child_id in node.get("children", [])
            ])

        soup = BeautifulSoup(rs3_str, features="xml")
        mononuc, multinuc = mononuc_rels(soup), multinuc_rels(soup)
        reindexed = assign_sequential_node_indices(
            all_nodes(soup, exclude_disjunct=exclude_disjunct_segments)
        )
        nodes = collect_children(reindexed)
        root_node = next(
            n for n in nodes.values()
            if n.get("parent") is None and n.get("text") is None
        )
        return build_node(root_node, nodes)

    @classmethod
    def from_rs3(
        cls,
        rs3_path: str,
        exclude_disjunct_segments: bool = True
    ) -> Node:
        """Construct an RST dependency-tree from a `.rs3`-XML file.

        :param rs3_path: Path to the `.rs3`-XML file.
        :type rs3_path: str
        :param exclude_disjunct_segments: Whether to exclude disjunct segments
        (i.e. segments without a parent). Defaults to `True`.
        :type exclude_disjunct_segments: bool, optional

        :return: The root node of the constructed RST tree.
        :rtype: Node
        """
        if not os.path.isfile(rs3_path):
            raise FileNotFoundError(f"File not found: {rs3_path}")
        try:
            with open(rs3_path, "r", encoding="utf-8") as f:
                rs3 = f.read()
        except Exception as e:
            raise IOError(f"Error reading file {rs3_path}: {e}")

        root = cls.from_xml(rs3, exclude_disjunct_segments)
        for node in root:  # compute spans once all nodes were added
            node.span = None  # trigger span.setter
        
        return root


def binarize(nodes) -> Node:
    for span in nodes.keys():
        nodes[span]["siblings"] = get_sibling_spans(span, nodes)

    def build_node(span, nodes_dict) -> Node:
        node_dict = nodes_dict[span]
        children_spans = get_child_spans(span, nodes_dict, "all")
        children = [
            build_node(child_span, nodes_dict) for child_span in children_spans
        ]
        return Node(node_dict, children=children)

    root_span = get_parent_span(nodes)
    root = build_node(root_span, nodes)
    return root


def dmrst2rs3(
    dmrst_tree: Union[str, List[str]],
    edu_spans: Dict[int, str]
) -> str:
    def node2tag(node: Dict, include_all_attrs: bool = False) -> Tag:
        if (text := node.get("text")) is not None:
            tag = soup.new_tag("segment")
            tag["id"] = node.get("id", "?")
            tag["parent"] = node.get("parent", "?")
            tag["relname"] = node.get("relname", "?")
            tag.append(text)
        else:
            tag = soup.new_tag("group")
            tag["id"] = node.get("id", "")
            tag["type"] = node.get("type", "")
            relname = node.get("relname", "")
            if (parent := node.get("parent", 0)) == 0:
                if include_all_attrs:
                    tag["parent"] = parent
                    tag["relname"] = relname
            else:
                tag["parent"] = parent
                tag["relname"] = relname
        return tag

    soup = BeautifulSoup(features="xml")
    soup.append(soup.new_tag("rst"))
    soup.rst.extend([
        soup.new_tag("header"), soup.new_tag("body")
    ])
    soup.header.append(soup.new_tag("relations"))
    soup.body.append(
        soup.new_tag("segments")
    )
    soup.body.append(
        soup.new_tag("groups")
    )

    all_relations = set()
    segments, groups = [], []
    nodes = list(build_tree(dmrst_nodes(dmrst_tree, edu_spans)).values())
    for node in nodes:
        if (span := node.get("span"))[0] == span[1]:
            segments.append(node)
        else:
            groups.append(node)
        # FIXME: extract relation types
        if (type_ := node.get("type")) != "multinuc":
            type_ = "rst"
        relname = node.get("relname", "span")
        if relname != "span" and type_ == "rst":
            all_relations.add((relname, type_))
        elif relname != "span" and type_ == "multinuc":
            # get relname of multinuc from its children
            if node.get("text") is None:
                    if len(child_rels := set([
                        n["relname"] for n in nodes
                        if n.get("parent") == node.get("id")
                    ])) == 1:  # all children have same relname
                        relname = list(child_rels)[0]
                        all_relations.add((relname, type_))

    for relname, type_ in sorted(set(all_relations)):
        rel_tag = soup.new_tag("rel")
        rel_tag["name"] = relname
        rel_tag["type"] = type_
        soup.header.relations.append(rel_tag)
    for leaf in sorted(segments, key=lambda x: x.get("id")):
        soup.body.segments.append(node2tag(leaf))
    for intermediate_node in sorted(groups, key=lambda x: x.get("id")):
        soup.body.groups.append(node2tag(intermediate_node, include_all_attrs=True))

    soup.body.segments.unwrap()
    soup.body.groups.unwrap()
    soup.smooth()
    xml = "\n".join(str(soup.prettify(  # remove <?xml ...?>
        formatter=RS3Formatter(indent=4)
    )).splitlines()[1:])
    rs3_string = re.sub(  # put segment tags on one line
        r"(<segment [^>]+?>)\n\s*(.+?)\n\s*?(</segment)",
        r"\g<1>\g<2>\g<3>",
        xml,
        flags=re.M
    )

    return rs3_string


if __name__ == "__main__":
    # Example usage    
    dmrst_tree = [
        '(1:Satellite=Background:1,2:Nucleus=span:19)',
        '(2:Nucleus=Joint:17,18:Nucleus=Joint:19)',
        '(2:Nucleus=Joint:14,15:Nucleus=Joint:17)',
        '(2:Nucleus=Joint:8,9:Nucleus=Joint:14)',
        '(2:Nucleus=span:6,7:Satellite=Elaboration:8)',
        '(2:Nucleus=span:5,6:Satellite=Elaboration:6)',
        '(2:Nucleus=span:4,5:Satellite=Elaboration:5)',
        '(2:Nucleus=span:2,3:Satellite=Elaboration:4)',
        '(3:Nucleus=span:3,4:Satellite=Attribution:4)',
        '(7:Nucleus=Joint:7,8:Nucleus=Joint:8)',
        '(9:Nucleus=Joint:10,11:Nucleus=Joint:14)',
        '(9:Nucleus=Temporal:9,10:Nucleus=Temporal:10)',
        '(11:Nucleus=Joint:11,12:Nucleus=Joint:14)',
        '(12:Nucleus=span:13,14:Satellite=Attribution:14)',
        '(12:Nucleus=Joint:12,13:Nucleus=Joint:13)',
        '(15:Satellite=Background:15,16:Nucleus=span:17)',
        '(16:Nucleus=span:16,17:Satellite=Enablement:17)',
        '(18:Nucleus=Joint:18,19:Nucleus=Joint:19)'
    ]
    edu_spans = {
        0: 'Ministerium bestätigte Omikron-Fall in Österreich',
        1: 'Die Coronavirus-Variante Omikron ist offiziell in Österreich angekommen:',
        2: '“Im Gesundheitsministerium liegen jetzt sämtliche Ergebnisse vor, die es für eine Bestätigung braucht”,',
        3: 'hieß es aus dem Ressort.',
        4: 'Bei einem Fall aus Tirol handelt es sich “mit Sicherheit” um die Variante mit der wissenschaftlichen Bezeichnung B.1.1.529.',
        5: 'Die Tiroler Behörden hatten den Verdachtsfall am Samstagabend bekanntgegeben.',
        6: 'Von der Infektion betroffen sei eine Person, die nach einer Südafrika-Reise positiv auf Covid-19 getestet wurde und derzeit keine Symptome aufweise.',
        7: 'Laut Elmar Rizzoli, Leiter des Tiroler Corona-Einsatzstabes, wurden alle Kontaktpersonen umgehend abgesondert.',
        8: 'Das Land Tirol forderte alle Personen, die in den vergangenen 14 Tagen aus den Ländern Südafrika, Lesotho, Botswana, Simbabwe, Mosambik, Namibia und Eswatini zurückgekehrt sind, auf, einen PCR-Test zu machen.',
        9: 'Dies sollte am fünften und zehnten Tag nach der Einreise wiederholt werden.',
        10: 'Bisher meldeten sich 31 Menschen.',
        11: '“Neben dem genannten einen positiven Ergebnis liegt für 20 Personen bereits ein negatives Testergebnis vor,',
        12: 'bei elf Personen sind die Testungen aktuell im Gange”,',
        13: 'hieß es.',
        14: 'Die Omikron-Variante von SARS-CoV-2 sorgt weltweit seit Tagen für Schlagzeilen.',
        15: 'Die WHO arbeitet nach eigenen Angaben mit technischen Partnern zusammen,',
        16: 'um die Auswirkungen der Variante auf die bestehenden Gegenmaßnahmen wie Impfstoffe zu bewerten.',
        17: 'Es sei noch unklar, ob B.1.1.259 leichter übertragbar verglichen mit anderen Covid-19-Varianten sei',
        18: 'oder einen schwereren Krankheitsverlauf nach sich ziehe.'
    }

    nodes = dmrst_nodes(dmrst_tree, edu_spans)
    root = binarize(nodes)
    print([n for n in root])
    print(root.to_rs3())
