"""Prepare one recorded event for inference from its ``preprocessing:`` configuration block.

Reads the raw MiniSEED and StationXML the block points at, removes the response, filters and
resamples, and writes the event file the pipeline reads under ``<output_dir>/events``.

    python prepare_event.py --config ../examples/configs/LV2_preprocessing.yaml
"""
import argparse
from dataclasses import replace

from seismo_sbi.data_handling.preprocessing.prepare_event import PreprocessingConfiguration, prepare_event
from seismo_sbi.utils.environment import log_progress_to_stdout


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare one recorded event for inference.")
    parser.add_argument("--config", "-c", required=True, help="YAML file with a preprocessing block.")
    parser.add_argument("--event-name", dest="event_name", help="Overrides preprocessing.event_name.")
    return parser.parse_args()


def main():
    log_progress_to_stdout()
    args = parse_arguments()
    config = PreprocessingConfiguration.from_yaml(args.config)
    if args.event_name:
        config = replace(config, event_name=args.event_name)
    prepare_event(config)


if __name__ == "__main__":
    main()
