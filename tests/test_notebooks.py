"""The notebooks are part of the submission: they must be valid, executed top to bottom, and
committed with their outputs (so they read on GitHub without running anything)."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "notebooks").glob("0*.ipynb"))


def test_all_five_notebooks_exist():
    assert [p.name[:2] for p in NOTEBOOKS] == ["01", "02", "03", "04", "05"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_valid_and_executed_in_order(path):
    nb = nbformat.read(str(path), as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code, "notebook has no code cells"
    # Restart & Run All: execution counts are exactly 1..n in cell order.
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
    for c in code:
        assert c.outputs, f"code cell {c.execution_count} has no output (commit the notebook executed)"
        assert not any(o.output_type == "error" for o in c.outputs), f"error output in cell {c.execution_count}"
    assert nb.cells[0].cell_type == "markdown" and nb.cells[0].source.startswith("# ")
