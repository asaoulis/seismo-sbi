"""Train an NPE neural compressor and flow from an SBI configuration file.

Every knob lives in the configuration file; the flags below are only what a scheduler must set.
``--stage generate`` stops once the simulations are on disk (the CPU half of a two-stage cluster
workflow), ``--stage meta`` rebuilds a run's ``model_meta.json`` sidecar without training, and
``--stage train`` runs the whole thing.
"""

import argparse

from seismo_sbi.utils.environment import cap_blas_threads, configure_numba_cache, cap_querier_cache

# Both read by their libraries at import time, so they run before the science imports.
cap_blas_threads()
configure_numba_cache()

from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.pipeline import (build_pipeline, generate_training_dataset,
                                     prepare_training_data, preload_noise_cache)
from seismo_sbi.sbi.compression.ML.train import (CompressionTrainer, apply_warm_start,
                                                 attach_loggers, enable_mmd_loss)
from seismo_sbi.sbi.scalers import scaler_provenance


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an NPE model from a configuration file.")
    parser.add_argument('--config', '-c', required=True, help="SBI configuration file.")
    parser.add_argument('--run-name', '-n', dest='run_name', default='default_run',
                        help="Names this run's subfolder under the output directory.")
    parser.add_argument('--stage', choices=['generate', 'meta', 'train'], default='train',
                        help="How far to go: simulate only, write the sidecar only, or train.")
    parser.add_argument('--epochs', '-e', type=int, help="Overrides the configured epoch count.")
    parser.add_argument('--devices', type=int,
                        help="Number of GPUs; above one trains with one rank per GPU.")
    parser.add_argument('--architecture', '-a', help="Overrides ml_architecture.")
    parser.add_argument('--train-batch-size', dest='train_batch_size', type=int,
                        help="Per-GPU training batch size; overrides ml_batch.train.")
    parser.add_argument('--num-simulations', dest='num_simulations', type=int,
                        help="Overrides simulations.num_simulations.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = SBI_Configuration.from_file(args.config)
    training = config.training.apply_overrides(station_encoder=args.architecture,
                                               epochs=args.epochs, devices=args.devices,
                                               train_batch_size=args.train_batch_size)
    cap_querier_cache(training.querier_cache_maxsize)

    pipeline = build_pipeline(config, args.config, num_simulations=args.num_simulations)
    simulation_paths = generate_training_dataset(pipeline, config,
                                                 training.skip_compression_stencil)
    if args.stage == 'generate':
        return

    data = prepare_training_data(pipeline, config, simulation_paths, training)
    trainer = CompressionTrainer.from_configuration(
        training, data.components, data.station_locations, data.trace_length,
        scaler_provenance(data.data_scaler))
    enable_mmd_loss(trainer, training, pipeline, data)

    if args.stage == 'meta':
        print(f"Wrote {trainer.write_model_meta(pipeline.models_output_path / args.run_name)}")
        return

    apply_warm_start(trainer, training, pipeline.models_output_path)
    preload_noise_cache(pipeline, training.cache)
    trainer.train(args.run_name, epochs=training.epochs,
                  output_path=pipeline.models_output_path,
                  dataloader_args=training.dataloader_args(pipeline, data),
                  logger=attach_loggers(training.logging,
                                        pipeline.models_output_path / args.run_name),
                  devices=training.devices)


if __name__ == '__main__':
    main()
