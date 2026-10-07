# Forward models

`seismo_sbi.simulators` holds the forward models: every class that turns source parameters into
seismograms. The generic parts sit at the root of the package, and each backend has its own
subdirectory.

| module | what it holds |
|---|---|
| `base.py` | `Simulator`, the interface every forward model implements |
| `sources.py` | `GenericPointSource`, `SourceLocation`, the moment tensors, `build_stf_sliprate` |
| `receivers.py` | `Receiver`, `Receivers` |
| `kernel.py` | seismograms from precomputed moment-tensor sensitivity kernels |
| `gf_ensemble.py` | `GFEnsembleSimulator`: draw one Earth model per simulation |
| `multi_region.py` | `MultiModelSimulator`: a different Earth model per receiver region |
| `spectral_filter.py` | a Butterworth filter designed at the observed data's rate, applied to synthetics at their own rate in the frequency domain |
| `simulation_io.py` | the HDF5 layout one simulation is written to and read back from |
| `theory_covariance.py` | per-trace theory-error covariance estimated from an ensemble |
| `registry.py` | `simulation_type` → builder |
| `instaseis/` | Instaseis backend: querier, point source, ensemble, multi-model |
| `cps/` | Computer Programs in Seismology backend |
| `axisem/` | the perturbed 1-D Earth models a database ensemble is built from |

## Receivers

`Receivers` is the ordered set of stations that every seismogram array follows. It is built in
one of four ways:

- `Receivers.from_station_file(stations, components_path, time_shifts_path)` reads
  `name network latitude longitude` lines, with two optional JSON maps.
- `Receivers.from_arrays` takes equal-length sequences of names, networks, latitudes and
  longitudes.
- `Receivers.from_inventory(inventory, channels="?H?")` reads an ObsPy `Inventory`, for example
  from StationXML.
- `Receivers(receivers=[...])` takes `Receiver` records directly.

The `components.json` map is the per-station channel list that an `Inventory` already holds.
`from_inventory` fills it from each station's channels (`1` and `2` read as `E` and `N`).
`receivers.to_inventory()` goes back to ObsPy for plotting or FDSN queries, and
`receivers.network_station_codes()` gives the `(network, station)` pairs to request. A
receiver's `time_shift` is a static correction in samples. An `Inventory` does not carry it.

## Nuisance effects

`seismo_sbi.nuisance_effects` holds the nuisance effects: the changes a real recording makes to a
synthetic seismogram.

| module | what it holds |
|---|---|
| `post_processing.py` | `PostProcessingChain`, `EFFECT_REGISTRY` and the chain builders |
| `seismogram_effect.py` | `SeismogramEffect`, the base class of every nuisance effect |
| `amplitude_effect.py`, `dropout_effects.py`, `time_shift_effect.py`, `scattering_coda_effect.py`, `anisotropy_effects.py`, `dispersion_effect.py` | the nuisance effects, one family per module |
| `lanczos_shift.py` | sub-sample time shifts by Lanczos interpolation |

Every `Simulator` runs its output through a `PostProcessingChain`. The same effects run in the
dataloader as training-time augmentation.

## Plug in your own forward model

Subclass `simulators.base.Simulator` and implement one method:

```python
from seismo_sbi.simulators.base import Simulator

class Specfem3DSimulator(Simulator):
    pre_event_pad_s = 0.0  # seconds each trace starts before the origin time

    def generic_point_source_simulation(self, source, **kwargs):
        ...  # returns {station_name: {component: np.ndarray}}
```

`source` is a `simulators.sources.GenericPointSource`: a `SourceLocation` (latitude, longitude,
depth in km below the catalogue datum, time shift in s) and a moment tensor whose `.components`
are `m_rr, m_tt, m_pp, m_rt, m_rp, m_tp` in N m. The receivers to simulate are the `Receivers`
handed to `__init__`, iterated with `self.receivers.iterate()`. Each trace has
`seismogram_duration_in_s * sampling_rate` samples. The method must accept the keyword arguments
`velocity_model`, `stf_duration` and `use_fiducial`, and can ignore them.

`pre_event_pad_s` states where the origin falls in each trace (60 s for the Instaseis backends, 0
for CPS). `simulators.synthetic_stream.seismogram_map_to_stream` needs it to time-stamp the
traces. The source time is the centroid of the moment-rate function (`stf_alignment` is
`"peak"`). A model whose source time function starts at the source time overrides the
`stf_alignment` property to return `"onset"`, as CPS does.

The station time shifts and the nuisance effects are applied to the output for you.

Then make the model selectable from a configuration file:

```python
from seismo_sbi.simulators.registry import register_simulator

def build_specfem3d(simulation_parameters, *, post_processing_effects):
    return Specfem3DSimulator(
        components=simulation_parameters.components,
        receivers=simulation_parameters.receivers,
        seismogram_duration_in_s=simulation_parameters.seismogram_duration,
        synthetics_processing=simulation_parameters.processing,
        post_processing_effects=post_processing_effects,
    )

register_simulator("specfem3d", build_specfem3d)
```

If that call is made before the pipeline is built, `simulation_type: specfem3d` in the
`seismic_context` block selects the model. A builder receives the `SimulationParameters` and, by
keyword, `post_processing_effects` (the nuisance effects to apply to every simulation), and
returns the `Simulator`. The built-in `kernel` and `theory_covariance` types also receive their
own inputs by keyword: `score_compression_data` (the sensitivity kernels), and
`ensemble_simulator` with `data_flattening`. The
[`custom_forward_model`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/custom_forward_model.ipynb)
notebook registers a toy forward model and inverts with it.

## AxiSEM ensembles

`simulators/axisem/` produces the perturbed 1-D models a theory-error ensemble is simulated on.
`perturb.py` draws one perturbed model. `perturbed_models.py` writes a whole ensemble.
`build_ensemble.py` also renders the solver input files and the member manifest:

```bash
python scripts/build_axisem_ensemble.py --config examples/configs/axisem_ensemble.yaml --dry-run
```

Meshing, solving and repacking into Instaseis databases happen on a cluster. The resulting
directory of databases is what `instaseis/ensemble.py` reads back.
