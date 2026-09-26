"""Process-environment settings a compute job must apply before the science imports.

Thread caps, the numba JIT cache directory, the Instaseis querier cache size and arviz's
daily-warning stamp are all read by their libraries at import time or inherited by spawned
workers, so they are set here before the science imports. Call them at the top of a launcher.
"""

import datetime
import getpass
import os
import tempfile
from pathlib import Path

#: Resident size of one open Instaseis database handle, used to report the cache budget.
QUERIER_HANDLE_MB = 55


def cap_blas_threads(num_threads=1):
    """Pin every BLAS backend to ``num_threads``, so multiprocessing workers stay single-threaded."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = str(num_threads)


def configure_numba_cache():
    """Point numba's JIT cache at a writable per-user temp dir; HOME may be read-only on compute nodes.

    Must run before numba is first imported, which happens inside instaseis.
    """
    cache_dir = Path(tempfile.gettempdir()) / f"numba_cache_{getpass.getuser()}"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))


def cap_querier_cache(maxsize):
    """Cap the per-worker LRU of open Instaseis database handles at ``maxsize`` (``None`` = uncapped).

    Exported into the environment because joblib workers are spawned and would not inherit a
    module global; each handle costs about 55 MB resident per worker process.
    """
    if maxsize is None:
        return
    os.environ["SEISMO_QUERIER_CACHE_MAXSIZE"] = str(int(maxsize))
    budget_gb = int(maxsize) * QUERIER_HANDLE_MB / 1024
    print(f"Instaseis querier cache capped at {int(maxsize)} open handles/worker "
          f"(~{budget_gb:.1f} GB per worker process).")


def stamp_arviz_daily_warning():
    """Write today's date into arviz's once-a-day warning stamp, through a per-process temp file.

    arviz rewrites the stamp at import through one shared temp name, so ranks importing it in the
    same second race and the loser dies; with today's stamp in place arviz skips the write.
    """
    try:
        from platformdirs import user_cache_dir
    except ImportError:
        return
    stamp_dir = Path(user_cache_dir("arviz", "arviz"))
    stamp_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp_dir / "daily_warning"
    today = datetime.date.today().isoformat()
    if stamp.is_file() and stamp.read_text().strip() == today:
        return
    temporary = stamp_dir / f"daily_warning.{os.getpid()}.tmp"
    temporary.write_text(today)
    temporary.replace(stamp)
