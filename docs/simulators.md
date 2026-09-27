# Forward models

Everything that turns source parameters into seismograms lives in `seismo_sbi.simulators`.
The generic parts sit at the root of the package and each backend has its own subdirectory;
`simulators/__init__.py` holds no imports, so using one backend never pulls in another's
dependencies.

| module | what it holds |
|---|---|
| `base.py` | `Simulator`, the interface every forward model implements |
| `sources.py` | `GenericPointSource`, `SourceLocation`, the moment tensors, `build_stf_sliprate` |
| `receivers.py` | `Receiver`, `Receivers` |
| `kernel.py` | seismograms from precomputed moment-tensor sensitivity kernels |
| `gf_ensemble.py` | `GFEnsembleSimulator`: draw one Earth model per simulation |
| `multi_region.py` | `MultiModelSimulator`: a different Earth model per receiver region |
| `simulation_io.py` | the HDF5 layout one simulation is written to and read back from |
| `theory_covariance.py` | per-trace theory-error covariance estimated from an ensemble |
| `registry.py` | `simulation_type` → builder |
| `instaseis/` | Instaseis backend: querier, point source, ensemble, multi-model |
| `cps/` | Computer Programs in Seismology backend |
| `axisem/` | the perturbed 1-D Earth models a database ensemble is built from |

## Nuisance effects

What a real recording does to a synthetic seismogram lives in its own package,
`seismo_sbi.nuisance_effects`, which imports nothing from the simulators:

| module | what it holds |
|---|---|
| `post_processing.py` | `PostProcessingChain`, `EFFECT_REGISTRY` and the chain builders |
| `seismogram_effect.py` | `SeismogramEffect`, the base class of every nuisance effect |
| `amplitude_effect.py`, `dropout_effects.py`, `time_shift_effect.py`, `scattering_coda_effect.py`, `anisotropy_effects.py`, `dispersion_effect.py` | the nuisance effects, one family per module |
| `lanczos_shift.py` | sub-sample time shifts by Lanczos interpolation |

Every `Simulator` runs its output through a `PostProcessingChain`; the same effects run in the
dataloader as training-time augmentation.

## Plug in your own forward model

Subclass `simulators.base.Simulator` and implement one method:

```python
from seismo_sbi.simulators.base import Simulator

class Specfem3DSimulator(Simulator):
    def generic_point_source_simulation(self, source, **kwargs):
        ...  # returns {station_name: {component: np.ndarray}}
```

`source` is a `simulators.sources.GenericPointSource`: a `SourceLocation`
(latitude, longitude, depth in km below the catalogue datum, time shift in s) and a moment
tensor whose `.components` are `m_rr, m_tt, m_pp, m_rt, m_rp, m_tp` in N.m. The receivers to
simulate are the `Receivers` handed to `__init__`, iterated with `self.receivers.iterate()`;
each trace has `seismogram_duration_in_s * sampling_rate` samples. Unknown keyword arguments
must be swallowed, because the base class forwards `velocity_model`, `stf_duration` and
`use_fiducial` whether or not a backend uses them.

The base class does the rest: it builds the source from the parameter dictionary, applies the
per-station time shifts, runs the post-processing chain, and writes the simulation to HDF5.

Then make the model selectable from a configuration file:

```python
from seismo_sbi.simulators.registry import register_simulator

def build_specfem3d(simulation_parameters, simulator_config, pp_effects, data_flattening):
    return Specfem3DSimulator(
        components=simulation_parameters.components,
        receivers=simulation_parameters.receivers,
        seismogram_duration_in_s=simulation_parameters.seismogram_duration,
        synthetics_processing=simulation_parameters.processing,
        post_processing_effects=pp_effects,
    )

register_simulator("specfem3d", build_specfem3d)
```

With that call made before the pipeline is built, `simulation_type: specfem3d` in the
`seismic_context` block selects it. `simulator_config` is `(simulation_type, payload)`, where
the payload carries whatever the type needs beyond the parameters — the sensitivity kernels for
`kernel`, the ensemble simulator for `cps_covariance` — and is `None` otherwise.

## AxiSEM ensembles

`simulators/axisem/` produces the perturbed 1-D models a theory-error ensemble is simulated on.
`perturb.py` draws one perturbed model, `perturbed_models.py` writes a whole ensemble, and
`build_ensemble.py` also renders the solver input files and the member manifest:

```bash
python scripts/build_axisem_ensemble.py --config examples/configs/axisem_ensemble.yaml --dry-run
```

Meshing, solving and repacking into Instaseis databases happen on a cluster; the resulting
directory of databases is what `instaseis/ensemble.py` reads back.
