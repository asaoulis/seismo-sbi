# Working with ObsPy

A seismologist holding ObsPy objects (`Stream`, `Inventory`, `Origin`, `Catalog`) can reach a
posterior and get ObsPy objects back without writing a station file, a YAML or an HDF5 file by
hand. Downloading is ObsPy's job: query an FDSN client directly and pass the result in.

| direction | ObsPy object | function | module |
|---|---|---|---|
| in | `Inventory` | `Receivers.from_inventory` | `simulators.receivers` |
| out | `Inventory` | `Receivers.to_inventory`, `Receivers.network_station_codes` | `simulators.receivers` |
| in | `Stream` + `Inventory` | `deconvolve_and_filter`, `rotate_horizontals_to_north_east` (`Stream` to `Stream`) | `data_handling.preprocessing.processing` |
| in | processed `Stream` | `observation_from_stream`: the event's data vector and station presence mask | `data_handling.preprocessing.sbi_export` |
| in | processed `Stream` | `pre_event_autocorrelations`: the pre-event noise autocorrelations | `data_handling.preprocessing.sbi_export` |
| in | processed `Stream` | `export_to_sbi_h5`: the same event written as the HDF5 file the pipeline reads | `data_handling.preprocessing.sbi_export` |
| in | continuous `Stream` | `noise_windows_from_stream`: event-free noise windows and their presence mask | `data_handling.preprocessing.noise_windows` |
| in | `Catalog` (any ObsPy-readable file) | `load_catalogue`: locations, depths, magnitudes and times for a catalogue prior | `priors.catalogue` |
| in | `Origin` | `source_location_from_origin`: a `SourceLocation` (depth in km, time in s after a stated origin time) | `moment_tensor.quakeml` |
| in | `Tensor` | `moment_tensor_from_tensor`: `m6` in N.m | `moment_tensor.quakeml` |
| out | `Tensor` | `tensor_from_moment_tensor` | `moment_tensor.quakeml` |
| out | `Stream` of synthetics | `seismogram_map_to_stream`; `seismogram_map_from_traces` for the output of `simulate_at(..., return_traces=True)` | `simulators.synthetic_stream` |
| out | `Event` (QuakeML) | `posterior_event`: moment tensor, focal mechanism and Mw of a posterior | `moment_tensor.quakeml` |

All modules are under `seismo_sbi`.

## Conventions

- QuakeML's `Tensor` is `m_rr, m_tt, m_pp, m_rt, m_rp, m_tp` in N.m in r, θ, φ = up, south, east,
  the library's `m6` order, so its components carry over unchanged. An `Origin` gives depth in m;
  a `SourceLocation` gives it in km, positive downwards.
- An observed window holds `compute_data_vector_length(duration, sampling_rate_hz) + 1` samples.
  The stream must already be filtered and resampled to the rate the synthetics use.
- A station without all three components in a window is absent from it: its samples are zeros and
  its entry in the presence mask is False.
- Instaseis synthetics start 60 s before the origin time and CPS synthetics at it; every simulator
  states this as `pre_event_pad_s`, which `seismogram_map_to_stream` reads. A simulator whose
  `pre_event_pad_s` is None (the moment-tensor kernel simulator, whose kernels may come from either
  backend) cannot be turned into a `Stream` until it is set. Observed event windows must start the
  same time before the origin as the synthetics they are compared with.
- `noise_windows_from_stream` returns rows `(n_windows, n_traces * n_samples)` and a mask
  `(n_windows, n_stations)`; `RealNoiseSampler.from_windows(noise_windows, receivers, components,
  present=present)` draws from them and
  `EmpiricalCovarianceEstimator(None, receivers, components).estimate_from_windows(noise_windows,
  present)` estimates the per-trace noise autocovariances from them.

## What a posterior event carries

`posterior_event(posterior_samples, origin)` reports a point estimate (the posterior mean, or the
`point_estimate` given) and the marginal posterior standard deviations: the tensor and its
component errors, the scalar moment, the isotropic, double-couple and CLVD fractions, the nodal
planes and principal axes, and Mw with its spread. Correlations between components, the posterior
mass on the lune and any second mode are not carried; keep the samples for those.
