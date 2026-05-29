"""Tests for SBI_Configuration YAML parsing."""

import json
import pytest

from seismo_sbi.sbi.configuration import SBI_Configuration, InvalidConfiguration


# ---------------------------------------------------------------------------
# Compression option parsing (pure dict, no file I/O)
# ---------------------------------------------------------------------------

def test_parse_optimal_score_filtered_block():
    cfg = SBI_Configuration()
    cfg.parse_compression_options([{"optimal_score": {"filtered_block": "/some/path"}}])
    assert len(cfg.compression_methods) == 1
    key, opts = cfg.compression_methods[0]
    assert key == "optimal_score_filtered_block"
    assert opts["type"] == "optimal_score"
    assert opts["covariance"] == "filtered_block"
    assert opts["path"] == "/some/path"


def test_parse_optimal_score_empirical_block():
    cfg = SBI_Configuration()
    cfg.parse_compression_options([{"optimal_score": {"empirical_block": "/x"}}])
    key, opts = cfg.compression_methods[0]
    assert key == "optimal_score_empirical_block"


def test_parse_multiple_compression_methods():
    cfg = SBI_Configuration()
    cfg.parse_compression_options([
        {"optimal_score": {"filtered_block": "/a"}},
        {"optimal_score": {"empirical_block": "/b"}},
    ])
    assert len(cfg.compression_methods) == 2
    keys = [k for k, _ in cfg.compression_methods]
    assert "optimal_score_filtered_block" in keys
    assert "optimal_score_empirical_block" in keys


def test_parse_invalid_compression_type_raises():
    cfg = SBI_Configuration()
    with pytest.raises(InvalidConfiguration, match="Invalid compression type"):
        cfg.parse_compression_options([{"bogus_type": {"x": "y"}}])


def test_parse_optimal_score_missing_covariance_raises():
    """optimal_score entry must have exactly one covariance sub-key."""
    cfg = SBI_Configuration()
    with pytest.raises(InvalidConfiguration):
        cfg.parse_compression_options([{"optimal_score": {}}])


# ---------------------------------------------------------------------------
# SBI config parsing
# ---------------------------------------------------------------------------

def _sbi_config_dict(method="posterior", pipeline="single_event"):
    return {
        "sbi": {
            "method": method,
            "pipeline": pipeline,
            "noise_model": {"type": "real_noise", "noise_catalogue_path": "/x"},
        },
        "likelihood": {"run": False},
    }


def test_parse_sbi_method_posterior():
    cfg = SBI_Configuration()
    cfg.parse_sbi_config(_sbi_config_dict("posterior"))
    assert cfg.sbi_method == "posterior"


def test_parse_sbi_method_likelihood():
    cfg = SBI_Configuration()
    cfg.parse_sbi_config(_sbi_config_dict("likelihood"))
    assert cfg.sbi_method == "likelihood"


def test_parse_sbi_pipeline_type():
    cfg = SBI_Configuration()
    cfg.parse_sbi_config(_sbi_config_dict(pipeline="multi_event"))
    assert cfg.pipeline_type == "multi_event"


def test_parse_sbi_default_pipeline():
    """Missing pipeline key should default to single_event."""
    cfg = SBI_Configuration()
    raw = {
        "sbi": {
            "method": "posterior",
            "noise_model": {"type": "real_noise", "noise_catalogue_path": "/x"},
        },
        "likelihood": {"run": False},
    }
    cfg.parse_sbi_config(raw)
    assert cfg.pipeline_type == "single_event"


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------

def test_parse_invalid_parameter_type_raises():
    cfg = SBI_Configuration()
    with pytest.raises(InvalidConfiguration, match="Invalid parameter type"):
        cfg.parse_parameters({
            "inference": {
                "not_a_real_param": {
                    "fiducial": [1.0],
                    "stencil_deltas": [0.1],
                    "bounds": [[-1.0], [1.0]],
                }
            },
            "nuisance": {},
        })


def test_parse_invalid_nuisance_type_raises():
    """Unknown nuisance parameter types must raise InvalidConfiguration."""
    cfg = SBI_Configuration()
    with pytest.raises(InvalidConfiguration, match="Invalid parameter type"):
        cfg.parse_parameters({
            "inference": {
                "moment_tensor": {
                    "fiducial": [1e13] * 6,
                    "stencil_deltas": [1e10] * 6,
                    "bounds": [[-5e13] * 6, [5e13] * 6],
                }
            },
            "nuisance": {
                "not_a_real_nuisance": {
                    "fiducial": [1.0],
                    "bounds": [0.5, 2.0],
                }
            },
        })


def test_parse_source_location_nuisance():
    """source_location is the canonical constant nuisance — must parse cleanly."""
    cfg = SBI_Configuration()
    cfg.parse_parameters({
        "inference": {
            "moment_tensor": {
                "fiducial": [1e13] * 6,
                "stencil_deltas": [1e10] * 6,
                "bounds": [[-5e13] * 6, [5e13] * 6],
            }
        },
        "nuisance": {
            "source_location": {
                "fiducial": [37.6, -118.9, 5.0, 0.0],
                "bounds": [37.6, -118.9, 5.0, 0.0],
            }
        },
    })
    assert "source_location" in cfg.model_parameters.nuisance
    assert cfg.model_parameters.nuisance["source_location"] == [37.6, -118.9, 5.0, 0.0]


# ---------------------------------------------------------------------------
# Nuisance parameters for new effects (will PASS once the whitelist in
# SBI_Configuration.parameter_types is extended during the refactor).
# ---------------------------------------------------------------------------

def test_parse_stf_duration_nuisance():
    """stf_duration must be parseable as a nuisance parameter after the refactor."""
    cfg = SBI_Configuration()
    cfg.parse_parameters({
        "inference": {
            "moment_tensor": {
                "fiducial": [1e13] * 6,
                "stencil_deltas": [1e10] * 6,
                "bounds": [[-5e13] * 6, [5e13] * 6],
            }
        },
        "nuisance": {
            "source_location": {
                "fiducial": [37.6, -118.9, 5.0, 0.0],
                "bounds": [37.6, -118.9, 5.0, 0.0],
            },
            "stf_duration": {
                "fiducial": [0.0],
                "bounds": [0.5, 5.0],
            },
        },
    })
    assert "stf_duration" in cfg.model_parameters.nuisance


def test_parse_amplitude_error_nuisance():
    """amplitude_error must be parseable as a nuisance parameter after the refactor."""
    cfg = SBI_Configuration()
    cfg.parse_parameters({
        "inference": {
            "moment_tensor": {
                "fiducial": [1e13] * 6,
                "stencil_deltas": [1e10] * 6,
                "bounds": [[-5e13] * 6, [5e13] * 6],
            }
        },
        "nuisance": {
            "source_location": {
                "fiducial": [37.6, -118.9, 5.0, 0.0],
                "bounds": [37.6, -118.9, 5.0, 0.0],
            },
            "amplitude_error": {
                "fiducial": [1.0],
                "bounds": [0.8, 1.2],
            },
        },
    })
    assert "amplitude_error" in cfg.model_parameters.nuisance


def test_parse_instrument_dropout_nuisance():
    """instrument_dropout must be parseable as a nuisance parameter after the refactor."""
    cfg = SBI_Configuration()
    cfg.parse_parameters({
        "inference": {
            "moment_tensor": {
                "fiducial": [1e13] * 6,
                "stencil_deltas": [1e10] * 6,
                "bounds": [[-5e13] * 6, [5e13] * 6],
            }
        },
        "nuisance": {
            "source_location": {
                "fiducial": [37.6, -118.9, 5.0, 0.0],
                "bounds": [37.6, -118.9, 5.0, 0.0],
            },
            "instrument_dropout": {
                "fiducial": [0.0],
                "bounds": [0.0, 0.3],
            },
        },
    })
    assert "instrument_dropout" in cfg.model_parameters.nuisance


def test_parse_time_shift_error_nuisance():
    """time_shift_error must be parseable as a nuisance parameter."""
    cfg = SBI_Configuration()
    cfg.parse_parameters({
        "inference": {
            "moment_tensor": {
                "fiducial": [1e13] * 6,
                "stencil_deltas": [1e10] * 6,
                "bounds": [[-5e13] * 6, [5e13] * 6],
            }
        },
        "nuisance": {
            "source_location": {
                "fiducial": [37.6, -118.9, 5.0, 0.0],
                "bounds": [37.6, -118.9, 5.0, 0.0],
            },
            "time_shift_error": {
                "fiducial": [0.0],
                "bounds": [0.0, 1.0],
                "gaussian_sigma": 2.0,
            },
        },
    })
    assert "time_shift_error" in cfg.model_parameters.nuisance


def test_parse_nuisance_effect_config_scale_range():
    """Extra YAML keys (scale_range) must be stored in nuisance_effect_config."""
    cfg = SBI_Configuration()
    cfg.parse_parameters({
        "inference": {
            "moment_tensor": {
                "fiducial": [1e13] * 6,
                "stencil_deltas": [1e10] * 6,
                "bounds": [[-5e13] * 6, [5e13] * 6],
            }
        },
        "nuisance": {
            "amplitude_error": {
                "fiducial": [0.0],
                "bounds": [0.0, 1.0],
                "scale_range": [0.3, 1.7],
            },
        },
    })
    assert "amplitude_error" in cfg.model_parameters.nuisance_effect_config
    assert cfg.model_parameters.nuisance_effect_config["amplitude_error"]["scale_range"] == [0.3, 1.7]


def test_parse_nuisance_effect_config_gaussian_sigma():
    """gaussian_sigma for time_shift_error must be stored in nuisance_effect_config."""
    cfg = SBI_Configuration()
    cfg.parse_parameters({
        "inference": {
            "moment_tensor": {
                "fiducial": [1e13] * 6,
                "stencil_deltas": [1e10] * 6,
                "bounds": [[-5e13] * 6, [5e13] * 6],
            }
        },
        "nuisance": {
            "time_shift_error": {
                "fiducial": [0.0],
                "bounds": [0.0, 1.0],
                "gaussian_sigma": 3.5,
            },
        },
    })
    assert "time_shift_error" in cfg.model_parameters.nuisance_effect_config
    ec = cfg.model_parameters.nuisance_effect_config["time_shift_error"]
    assert ec["gaussian_sigma"] == 3.5
    # 'fiducial' and 'bounds' must NOT appear in effect_config
    assert "fiducial" not in ec
    assert "bounds" not in ec


def test_parse_scattering_coda_nuisance():
    """scattering_coda must parse cleanly and store alpha in nuisance_effect_config."""
    cfg = SBI_Configuration()
    cfg.parse_parameters({
        "inference": {
            "moment_tensor": {
                "fiducial": [1e13] * 6,
                "stencil_deltas": [1e10] * 6,
                "bounds": [[-5e13] * 6, [5e13] * 6],
            }
        },
        "nuisance": {
            "source_location": {
                "fiducial": [37.6, -118.9, 5.0, 0.0],
                "bounds": [37.6, -118.9, 5.0, 0.0],
            },
            "scattering_coda": {
                "fiducial": [0.0],
                "bounds": [0.0, 1.0],
                "alpha": 0.4,
            },
        },
    })
    assert "scattering_coda" in cfg.model_parameters.nuisance
    assert "scattering_coda" in cfg.model_parameters.nuisance_effect_config
    ec = cfg.model_parameters.nuisance_effect_config["scattering_coda"]
    assert ec["alpha"] == 0.4
    assert "fiducial" not in ec
    assert "bounds" not in ec


def test_parse_nuisance_no_effect_config_when_only_standard_keys():
    """No effect_config entry should be created when only fiducial+bounds are given."""
    cfg = SBI_Configuration()
    cfg.parse_parameters({
        "inference": {
            "moment_tensor": {
                "fiducial": [1e13] * 6,
                "stencil_deltas": [1e10] * 6,
                "bounds": [[-5e13] * 6, [5e13] * 6],
            }
        },
        "nuisance": {
            "amplitude_error": {
                "fiducial": [0.0],
                "bounds": [0.0, 1.0],
            },
        },
    })
    assert "amplitude_error" not in cfg.model_parameters.nuisance_effect_config


def test_parse_moment_tensor_parameters():
    cfg = SBI_Configuration()
    fiducial = [1e13] * 6
    cfg.parse_parameters({
        "inference": {
            "moment_tensor": {
                "fiducial": fiducial,
                "stencil_deltas": [1e10] * 6,
                "bounds": [[-5e13] * 6, [5e13] * 6],
            }
        },
        "nuisance": {},
    })
    assert "moment_tensor" in cfg.model_parameters.theta_fiducial
    assert cfg.model_parameters.theta_fiducial["moment_tensor"] == fiducial


# ---------------------------------------------------------------------------
# Full config round-trip (requires temp files)
# ---------------------------------------------------------------------------

def _minimal_config(tmp_path):
    """Build a complete config dict pointing at temp fixture files."""
    stations_file = tmp_path / "stations.txt"
    # Two rows needed: np.genfromtxt returns 1D for a single row, breaking iteration
    stations_file.write_text("STA1 XX 0.0 0.0\nSTA2 XX 1.0 1.0\n")

    components_file = tmp_path / "components.json"
    components_file.write_text(json.dumps({"STA1": ["Z"], "STA2": ["Z"]}))

    return {
        "run_name": "test_run",
        "output_directory": str(tmp_path),
        "job_name": "test_job",
        "generate_dataset": False,
        "num_jobs": 1,
        "seismic_context": {
            "components": "Z",
            "stations_path": str(stations_file),
            "station_components_path": str(components_file),
            "seismogram_duration": 30,
            "sampling_rate": 1,
            "syngine_address": "syngine://prem_i_2s",
            "processing": {
                "filter": {
                    "type": "bandpass",
                    "freqmin": 0.01,
                    "freqmax": 0.1,
                    "corners": 4,
                    "zerophase": False,
                },
                "sampling_rate": 1,
            },
        },
        "parameters": {
            "inference": {
                "moment_tensor": {
                    "fiducial": [1e13] * 6,
                    "stencil_deltas": [1e10] * 6,
                    "bounds": [[-5e13] * 6, [5e13] * 6],
                }
            },
            "nuisance": {
                "source_location": {
                    "fiducial": [0.0, 0.0, 10.0, 0.0],
                    "bounds": [0.0, 0.0, 10.0, 0.0],
                }
            },
        },
        "simulations": {
            "num_simulations": 100,
            "sampling_method": {"moment_tensor": "uniform"},
            "iterative_least_squares": {"max_iterations": 5, "damping_factor": 0.01},
        },
        "compression": [{"optimal_score": {"filtered_block": "/fake/noise"}}],
        "inference": {
            "sbi": {
                "method": "posterior",
                "pipeline": "single_event",
                "noise_model": {"type": "real_noise", "noise_catalogue_path": "/x"},
            },
            "likelihood": {"run": False},
        },
        "jobs": {
            "real_events": {},
            "simulations": {"random_events": 1, "fixed_events": [], "custom_events": []},
            "noise_models": {"gaussian_noises": [1e-6]},
            "plots": {
                "async_plotting": False,
                "test_posteriors": {"chain_consumer": []},
                "disable_plotting": True,
            },
        },
    }


def test_full_config_parsing_seismic_context(tmp_path):
    cfg = SBI_Configuration()
    cfg.process_configuration_data(_minimal_config(tmp_path))
    assert cfg.sim_parameters is not None
    receivers = list(cfg.sim_parameters.receivers.iterate())
    assert len(receivers) == 2
    station_names = {r.station_name for r in receivers}
    assert "STA1" in station_names


def test_full_config_parsing_model_parameters(tmp_path):
    cfg = SBI_Configuration()
    cfg.process_configuration_data(_minimal_config(tmp_path))
    assert "moment_tensor" in cfg.model_parameters.theta_fiducial
    assert len(cfg.model_parameters.theta_fiducial["moment_tensor"]) == 6


def test_full_config_parsing_compression_method(tmp_path):
    cfg = SBI_Configuration()
    cfg.process_configuration_data(_minimal_config(tmp_path))
    assert len(cfg.compression_methods) == 1
    key, _ = cfg.compression_methods[0]
    assert key == "optimal_score_filtered_block"


def test_full_config_parsing_sbi_method(tmp_path):
    cfg = SBI_Configuration()
    cfg.process_configuration_data(_minimal_config(tmp_path))
    assert cfg.sbi_method == "posterior"
    assert cfg.pipeline_type == "single_event"


def test_full_config_parsing_pipeline_parameters(tmp_path):
    cfg = SBI_Configuration()
    cfg.process_configuration_data(_minimal_config(tmp_path))
    assert cfg.pipeline_parameters.run_name == "test_run"
    assert cfg.pipeline_parameters.num_jobs == 1
