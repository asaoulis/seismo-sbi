"""Importing a core module does not drag in a mapping or plotting stack it only needs to draw."""
import subprocess
import sys

import pytest


def modules_loaded_by(module: str) -> set:
    """The ``sys.modules`` keys after importing ``module`` in a fresh interpreter."""
    code = f"import sys, {module}; print('\\n'.join(sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=True)
    return set(result.stdout.split())


@pytest.mark.parametrize("module, heavy", [
    ("seismo_sbi.simulators.receivers", "cartopy"),
    ("seismo_sbi.sbi.configuration", "cartopy"),
    ("seismo_sbi.sbi.pipeline", "seismo_sbi.plotting.results_plotting"),
    ("seismo_sbi.plotting.lune", "mpl_toolkits.basemap"),
])
def test_importing_a_module_does_not_load_what_only_its_plots_need(module, heavy):
    assert heavy not in modules_loaded_by(module)
