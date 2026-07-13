"""Data quality utilities: compare a reference synthetic against observed waveforms
and decide, per station, whether to keep / time-shift / drop it.

The package is deliberately small and generic: the metric, policy, alignment and
serialization layers take plain numpy arrays + lightweight descriptors (never a
pipeline or an h5 file), so they work for any simulator/source and are trivially
unit-testable. Scripts (e.g. the Santorini ``qa_forward_check``) do the I/O and forward
modelling and hand arrays in.
"""
from .metrics import (
    SNRMetrics,
    TraceDescriptor,
    TraceMetrics,
    align_best_lag,
    aligned_variance_reduction,
    compute_trace_metrics,
    correlation_misfit,
    envelope_misfit,
    peak_amplitude_ratio,
    signal_window,
    snr_metrics,
    station_reduced_chi2,
    traces_from_receivers,
    variance_reduction,
)
from .policy import (
    KEPT_VERDICTS,
    ComponentVerdict,
    QAThresholds,
    StationSummary,
    StationVerdict,
    VERDICT_COLORS,
    VERDICT_LABELS,
    component_verdicts,
    decide_component,
    decide_station,
    event_contamination,
    sigma_outlier_verdicts,
    snr_station_drop,
    summarise_event,
    summarise_station,
)
from .alignment import (
    ShiftResult,
    nonzero_shifts,
    optimise_event_shifts,
    optimise_station_shift,
)
from .serialization import (
    QAArtifacts,
    components_from_verdicts,
    load_qa_artifacts,
    verdict_to_json,
    write_components_json,
    write_time_shifts_json,
    write_verdicts_json,
)
from .aggregate import StationReliability, station_reliability
from .guards import (
    compose_component_qa,
    data_qa_thresholds,
    neighbour_window_flag,
    obs_dead_components,
    read_noise_sigma,
)

__all__ = [
    # metrics
    "TraceDescriptor", "TraceMetrics", "SNRMetrics", "align_best_lag",
    "aligned_variance_reduction", "compute_trace_metrics", "correlation_misfit",
    "envelope_misfit", "peak_amplitude_ratio", "signal_window", "snr_metrics",
    "station_reduced_chi2", "traces_from_receivers", "variance_reduction",
    # policy
    "KEPT_VERDICTS", "ComponentVerdict", "QAThresholds", "StationSummary",
    "StationVerdict", "VERDICT_COLORS", "VERDICT_LABELS", "component_verdicts",
    "decide_component", "decide_station", "event_contamination",
    "sigma_outlier_verdicts", "snr_station_drop", "summarise_event",
    "summarise_station",
    # alignment
    "ShiftResult", "nonzero_shifts", "optimise_event_shifts", "optimise_station_shift",
    # serialization
    "QAArtifacts", "components_from_verdicts", "load_qa_artifacts", "verdict_to_json",
    "write_components_json", "write_time_shifts_json", "write_verdicts_json",
    # aggregate
    "StationReliability", "station_reliability",
    # guards (calibrated presets + model-free guards + gate composition)
    "compose_component_qa", "data_qa_thresholds", "neighbour_window_flag",
    "obs_dead_components", "read_noise_sigma",
]
