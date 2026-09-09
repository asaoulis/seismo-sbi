# Writing code in this repository

This file is the house style for everyone who edits code here, human or agent. It applies to
`seismo-sbi` (the library: simulators, noise and theory-error models, compression, training,
inference, evaluation) and to the study repositories built on it (`mt-sbi`: one package per
region, producing and analysing moment-tensor catalogues). The same file lives in both.

## Who reads this code

Seismologists and astrophysicists, not software engineers. A colleague should be able to open
any file cold and read it top to bottom like a method section: what goes in, the steps, what
comes out. Optimise for that reader. Cleverness, indirection and defensive generality cost more
than they save.

## The shape of a module

- A 3–8 line docstring at the top says what the module does and how it is used. The count is
  what `ast.get_docstring` returns: the title line and the blank after it count, the closing
  quotes do not, so the body is at most six lines. It does not argue for the module's existence
  or recount how it came to be.
- Functions are short (aim for under 40 lines, rarely over ~60) and read as the steps of a method
  in the order they are applied. One function, one step. A long function is a list of steps
  that has not been split yet. The exception is a function that lays out one figure: split by
  concern, never by line count.
- Names follow the physics, with units in the name where a number has them:
  `moment_tensor`, `source_depth_km`, `station_azimuth_deg`, `sampling_rate_hz`,
  `green_functions`, `posterior_samples`, `noise_covariance`. Not `arr2`, `cfg_blk`, `tmp`,
  `helper`, `do_stuff`. Loop variables are `station`, `trace`, `event`, not `i`, `x`.
- A class exists to hold state that several steps share (a simulator, a trainer, a
  configuration). Free functions are the default; do not wrap a function in a class.
- No file is longer than it needs to be. When a module passes ~500 lines, something in it is a
  separate concern. New functions never go into a module already past that size; open a new one.

## Comments

Comments are rare and short: at most two lines, and only where the *why* is not obvious from
the code and the names. They never:

- restate what the code does (`# increment the counter`);
- tell a story (`this took down two 500k jobs before...`);
- point outside the repository: no task names, plan files, ledgers, run ids, dates, other
  projects, or private directories. The code is self-referential. If a fact matters, it lives
  in a docstring, a test, or the README;
- serve as a changelog. Git is the changelog.

A `#:` line above an attribute is documentation for that attribute, held to the docstring rules.
A section banner is one line. A region name appears in library prose only as a physical datum a
number needs (a latitude, a station count), never as the label of where a threshold came from.

Docstrings state what a function returns, the units and shapes of its arguments, and any
convention the caller must know. Nothing else.

A design note or plan is a sketch of the code, not the code. Where it conflicts with what the
code actually does, the code wins; say so in the change description rather than following the sketch.

**Before**, the opening of a training script:

```python
# Give numba an explicitly writable on-disk cache directory.
#
# instaseis JITs with @njit(cache=True) (finite_elem_mapping), so on the FIRST instaseis.open_db
# numba must resolve a cache "locator". Its fallback chain is: NUMBA_CACHE_DIR -> the source
# directory (site-packages/instaseis/) -> the user-wide cache (~/.cache). On a cluster the conda
# env AND ~/.cache both live in $HOME, so the moment HOME is full, read-only, or over quota EVERY
# locator fails and numba raises, killing the job at simulator-construction time:
#     RuntimeError: cannot cache function 'compute_theta_r': no locator available for file ...
# That took down two 500k dataset-generation jobs before they ran a single simulation.
# ... (nine more lines)
if not os.environ.get("NUMBA_CACHE_DIR"):
    import tempfile as _tempfile
    _numba_cache = os.path.join(_tempfile.gettempdir(), f"numba_cache_{os.environ.get('USER', 'seismo')}")
    os.makedirs(_numba_cache, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = _numba_cache
```

**After**, one call in the script and a small function in the library:

```python
def configure_numba_cache():
    """Point numba's JIT cache at a writable per-user temp dir; HOME may be read-only on compute nodes."""
    cache_dir = Path(tempfile.gettempdir()) / f"numba_cache_{getpass.getuser()}"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
```

**Before**, a 40-line module docstring that opens with "Why this module exists", names the
study and the task the module was written for, and describes a pre-registered statistic.
**After**:

```python
"""Station-subset disagreement metrics for a posterior.

Whitened divergence between full-station and subset-station posteriors, leave-one-out
influence tables, jackknife variances under station resampling, and covariate-matched null
percentiles. Pure numpy; sampling and file I/O belong to the caller.
"""
```

## Units and conventions

- SI unless the name says otherwise: `_km`, `_deg`, `_s`, `_hz`, `_nm` (newton-metres).
- Moment tensor components in the order `m_rr, m_tt, m_pp, m_rt, m_rp, m_tp` (r, θ, φ = up,
  south, east). Lune angles `gamma_deg`, `delta_deg`. Depth is positive downwards, in km.
- Time is seconds relative to the origin time unless the name says `utc`.
- Arrays are documented by shape in the docstring: `(n_stations, n_components, n_samples)`.

## Configuration and scripts

- Every knob lives in a typed configuration object parsed once from YAML (the library's
  `SBI_Configuration`, and the `TrainingConfiguration` that replaces the training script's raw YAML
  reads). Code never re-opens the YAML to read one key.
- Command-line flags exist only for run identity and for what a scheduler must set (for example
  `--config`, `--run-name`, `--stage`, `--epochs`, `--devices`, a batch size). Anything else is a
  config key.
- A script is a launcher: parse the flags, build the configuration, call one library entry
  point. Under ~80 lines including its docstring. If a script grows a second concern, that
  concern is a library function with a test.
- A configuration dataclass mirrors its YAML block: field names are the YAML keys, one class per
  block, a block with a single value is a field on the parent rather than its own class. A value
  derived from the fields carries units in its name.
- Defaults are the library's defaults, not a particular study's. Study-specific values
  (thresholds, station selections, reference catalogues) live in the study repository's config.

## Library versus study

- Library code takes the region as *data*: a station list, a catalogue path, a database path,
  a config block. It must work unchanged for the next region.
- Study code is anything that names a region, an event, a reference catalogue, a station
  selection, or a figure layout. It lives in the study repository, never in the library.
- The dependency points one way: study imports library. The library never imports, reads, or
  mentions a study.
- Each study region has the same modules in the same order (event selection, data preparation,
  inference, catalogue, probabilistic summary, headline figures). A stage module is one file
  read in run order; split at stage boundaries, not by length. A hypothesis that was tested
  and retired is one paragraph in the region README, not a module.

## Tests

- One test per behaviour, named for the behaviour: `test_lune_angles_round_trip`, not `test_1`.
- Tests use small synthetic inputs or committed fixtures. No test reads private data paths.
- When code is deleted, its tests are deleted with it. A test for something no longer useful is
  bloat, not safety.
- When code is ported or moved, its own earlier outputs on disk are the arbiter: the port is
  compared against them before it replaces anything, and the comparison is reported.

## Removing code

Removing is a first-class edit. When something is superseded, unused, or exploratory and no
longer informative, delete it together with its tests, comments, and config keys. Do not keep
it behind a flag, rename it `_old`, or move it to an attic. Git keeps the history. Removal means
`git rm`: files the repository does not track are left where they are, never deleted from disk.

## Checklist before finishing a change

1. Grep the changed files for pointers outside the repository (agent-tooling directories, task or
   plan names, ledgers, run ids, dates in comments). The grep returns nothing.
2. Every new or edited module has a top docstring of 3–8 lines; no comment block exceeds two lines.
3. No function over ~60 lines; no script over ~80 lines; no new key read from raw YAML.
4. Names carry units; no region name inside library code.
5. The fast test tier is green, and any deleted code took its tests with it.
