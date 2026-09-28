"""Every notebook listed on the documentation's examples page has exactly one top-level title.

A notebook without one shows each of its sections as a separate entry on that page.
"""
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def listed_notebooks():
    """Names of the notebooks in the toctree of ``docs/examples.md``."""
    toctree = (REPO / "docs" / "examples.md").read_text()
    return re.findall(r"^_generated/examples/(\S+)$", toctree, re.MULTILINE)


def markdown_headings(notebook_path):
    """The ``#`` heading lines of a notebook's markdown cells, outside fenced code blocks."""
    headings = []
    for cell in json.loads(notebook_path.read_text())["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        in_fence = False
        for line in "".join(cell["source"]).splitlines():
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            elif not in_fence and line.startswith("#"):
                headings.append(line)
    return headings


def test_the_examples_page_lists_notebooks():
    assert len(listed_notebooks()) >= 9


@pytest.mark.parametrize("name", listed_notebooks())
def test_the_notebook_has_one_title(name):
    headings = markdown_headings(REPO / "examples" / f"{name}.ipynb")
    titles = [heading for heading in headings if heading.startswith("# ")]
    assert len(titles) == 1, titles
