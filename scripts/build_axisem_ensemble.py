#!/usr/bin/env python
"""Stage an AxiSEM ensemble from one configuration file.

    python scripts/build_axisem_ensemble.py --config examples/configs/axisem_ensemble.yaml
    python scripts/build_axisem_ensemble.py --config … --dry-run   # reference + 1 member
"""

import argparse

from seismo_sbi.simulators.axisem.build_ensemble import build_ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="ensemble configuration YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="stage the reference plus one member only")
    parser.add_argument("--from-bm-dir", default=None,
                        help="ingest background models already written under this directory "
                             "(fiducial/ + member_NNN/) instead of perturbing the reference")
    parser.add_argument("--name", default=None, help="override the ensemble name")
    args = parser.parse_args()

    members_path = build_ensemble(args.config, dry_run=args.dry_run,
                                  from_bm_dir=args.from_bm_dir, name=args.name)
    print(f"members manifest: {members_path}")


if __name__ == "__main__":
    main()
