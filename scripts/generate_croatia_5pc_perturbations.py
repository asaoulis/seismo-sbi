"""Generate an isolated set of kappa=5 (5%) CPS perturbations for the FULL Croatia
10-station, 4-model geometry (multi_model.json).

Mirrors examples/theory_errors_LV2.ipynb / multi_CPS_perturbations.py, but runs
generate_CPS_perturbations once per 1-D Earth model, filtering the receiver set in
Python (so single-station models like model3/CACV work without a 1-row stations file).

Writes:  <output_base>/<model>/kappa_<k>/            (perturbation Green's functions)
         <output_base>/<model>/kappa_<k>_fiducial/   (fiducial Green's functions)

Example:
  python generate_croatia_5pc_perturbations.py \
      --config configs/croatia/perturbations_5pc/croatia_pert_5pc_base.yaml \
      --output_base /data/alex/cps/cps_croatia/multi_models_5pc \
      --kappa 5 --cps_path /home/alex/work/cps/PROGRAMS.330/bin
"""
import os
import argparse

# prevent BLAS/OpenMP oversubscription against joblib multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from pathlib import Path

from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.cps_simulator.compatibility import load_velocity_model

from generate_CPS_perturbations import generate_CPS_perturbations

# model file -> receiver subset, matching scripts/configs/croatia/multi_model.json
MODEL_TO_RECEIVERS = {
    "model1.txt": ["KALN", "MOSL", "BLY"],
    "model2.txt": ["KRJB", "RABC", "PLIT"],
    "model3.txt": ["CACV"],
    "model4_simple.txt": ["CEY", "MOZS", "PERS"],
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', '-c', required=True,
                        help='Base SBI config (full 10-station geometry).')
    parser.add_argument('--output_base', '-o',
                        default='/data/alex/cps/cps_croatia/multi_models_5pc',
                        help='Base directory for the new perturbation set.')
    parser.add_argument('--kappa', '-k', type=int, default=5,
                        help='Kappa (percent) perturbation level. Default 5. '
                             'Used as the integer dir tag kappa_<k> to match multi_model.json.')
    parser.add_argument('--cps_path', default=None,
                        help='Path to CPS bin dir (hprep96/hspec96/hpulse96/f96tosac).')
    parser.add_argument('--random_events', type=int, default=None,
                        help='Override perturbations per model (default: config value). '
                             'Use a small value for a smoke test.')
    parser.add_argument('--models', default=None,
                        help='Comma-separated subset of model files to run '
                             '(default: all four). e.g. "model3.txt".')
    return parser.parse_args()


def main():
    args = parse_arguments()
    models = ([m.strip() for m in args.models.split(',')]
              if args.models else list(MODEL_TO_RECEIVERS.keys()))

    base_dir = Path(args.output_base)
    velocity_model_dir = Path(
        '/home/alex/work/seismo-sbi/scripts/configs/croatia/multi_models_txt')

    for model in models:
        receiver_subset = MODEL_TO_RECEIVERS[model]

        config = SBI_Configuration()
        config.parse_config_file(args.config)

        kappa_tag = "kappa_" + str(args.kappa)
        cps_GFs_path = base_dir / model[:-4] / kappa_tag
        cps_GFs_fiducial_path = base_dir / model[:-4] / (kappa_tag + "_fiducial")
        cps_GFs_path.mkdir(parents=True, exist_ok=True)
        cps_GFs_fiducial_path.mkdir(parents=True, exist_ok=True)

        replace_kwargs = dict(cps_GFs_path=str(cps_GFs_path),
                              cps_GFs_fiducial_path=str(cps_GFs_fiducial_path))
        if args.cps_path is not None:
            replace_kwargs["cps_path"] = str(args.cps_path)
        config.sim_parameters = config.sim_parameters._replace(**replace_kwargs)

        # swap in this model's 1-D Earth model + kappa level
        velocity_mod_parameters = config.model_parameters.bounds['velocity_model']
        velocity_mod_parameters[0] = str(velocity_model_dir / model)
        velocity_mod_parameters[1] = args.kappa
        config.model_parameters.bounds['velocity_model'] = velocity_mod_parameters
        config.model_parameters.nuisance['velocity_model'] = load_velocity_model(
            velocity_mod_parameters[0])

        # optional perturbation-count override (smoke tests)
        if args.random_events is not None:
            config.test_job_simulations = config.test_job_simulations._replace(
                random_events=args.random_events)

        # restrict receivers to this model's subset (filter the full set in Python)
        receivers = config.sim_parameters.receivers
        receivers.receivers = [rec for rec in receivers.iterate()
                               if rec.station_name in receiver_subset]
        config.sim_parameters = config.sim_parameters._replace(receivers=receivers)

        print(f"\n=== Generating kappa={args.kappa} GFs for {model} "
              f"receivers={[r.station_name for r in receivers.iterate()]} ===", flush=True)
        generate_CPS_perturbations(config)


if __name__ == '__main__':
    main()
