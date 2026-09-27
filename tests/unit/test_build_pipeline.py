"""build_pipeline builds the class a configuration names, or the one it is given."""
from types import SimpleNamespace

from seismo_sbi.sbi.training_data import build_pipeline


class _RecordingPipeline:
    def __init__(self, pipeline_parameters, config_path):
        self.built_with = (pipeline_parameters, config_path)

    def load_seismo_parameters(self, *parameters):
        self.loaded = parameters


def test_build_pipeline_uses_the_class_it_is_given():
    config = SimpleNamespace(pipeline_type="multi_event", pipeline_parameters="pipeline",
                             compression_methods="compression", sim_parameters="simulation",
                             model_parameters="model", dataset_parameters="dataset")

    pipeline = build_pipeline(config, "config.yaml", pipeline_class=_RecordingPipeline)

    assert pipeline.built_with == ("pipeline", "config.yaml")
    assert pipeline.compression_methods == "compression"
    assert pipeline.loaded == ("simulation", "model", "dataset")
