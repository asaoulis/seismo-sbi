"""instaseis' inverse mapping, whose arguments are jitted functions, stays out of numba's disk cache."""
import pytest

from seismo_sbi.simulators.instaseis.querier import keep_inverse_mapping_out_of_the_numba_disk_cache


def test_the_inverse_mapping_is_compiled_per_process():
    finite_elem_mapping = pytest.importorskip("instaseis.finite_elem_mapping")
    caching = pytest.importorskip("numba.core.caching")
    keep_inverse_mapping_out_of_the_numba_disk_cache()
    assert isinstance(finite_elem_mapping._inv_mapping_iterative._cache, caching.NullCache)
