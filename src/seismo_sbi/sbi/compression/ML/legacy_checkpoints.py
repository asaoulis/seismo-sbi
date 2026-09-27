"""Load checkpoints written with the earlier station-CNN parameter names.

Those checkpoints name the station CNN ``CNN_feature_extractor.seismic_trace_CNN`` and carry
the weights of a feed-forward head the forward pass never used. :func:`remap_legacy_state_dict`
renames each old key by the table below and drops the unused head; a current checkpoint passes
through untouched.
"""

#: Old key fragment -> new fragment, or ``None`` for weights the current model does not have.
LEGACY_KEY_RENAMES = {
    ".CNN_feature_extractor.seismic_trace_CNN.": ".station_encoder._cnn.",
    ".CNN_feature_extractor.feedforward_net.": None,
}


def is_legacy_state_dict(state_dict) -> bool:
    """Whether any key uses one of the earlier station-CNN parameter names."""
    return any(old in key for key in state_dict for old in LEGACY_KEY_RENAMES)


def remap_legacy_state_dict(state_dict) -> dict:
    """The state dict with old key names renamed and unused old weights dropped, order kept."""
    if not is_legacy_state_dict(state_dict):
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        old = next((fragment for fragment in LEGACY_KEY_RENAMES if fragment in key), None)
        if old is None:
            remapped[key] = value
        elif LEGACY_KEY_RENAMES[old] is not None:
            remapped[key.replace(old, LEGACY_KEY_RENAMES[old])] = value
    return remapped
