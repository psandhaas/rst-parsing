#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Mon, 14.09.25                                                     #
# ========================================================================== #

"""PARSEVAL evaluation metrics for RST trees."""

from platform import node
from bs4 import BeautifulSoup, Tag
from collections import defaultdict
import os
from typing import Dict, List, Literal, Tuple, Optional, Union

from tree import Node


if __name__ == "__main__":
    from pprint import pprint

    tst = "C:/Users/SANDHAP/Repos/rst-parsing/data/parsed/dmrst/blogposts_CRE210_Blog_dmrst.rs3"
    root = Node.from_rs3(tst)
    pprint([n for n in root])
