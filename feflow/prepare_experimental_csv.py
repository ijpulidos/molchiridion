"""
Convert experimental binding affinity data from a protein-ligand-benchmark YAML file
(https://github.com/openforcefield/protein-ligand-benchmark) into a CSV suitable for
analyze_switching_results.py.

Converts Ki/Kd measurements to absolute binding free energies via:
    ΔG = RT ln(Ki)  [kcal/mol]
with error propagation:
    δΔG = RT * δKi / Ki

Usage
-----
    python prepare_experimental_csv.py ligands.yml -o exp_dg.csv
    python prepare_experimental_csv.py ligands.yml -o exp_dg.csv --temperature 298.15
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import yaml

R_KCAL = 1.987204258e-3  # kcal / (mol · K)

UNIT_TO_MOLAR = {
    "m": 1.0,
    "molar": 1.0,
    "mm": 1e-3,
    "millimolar": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "micromolar": 1e-6,
    "nm": 1e-9,
    "nanomolar": 1e-9,
    "pm": 1e-12,
    "picomolar": 1e-12,
}


def ki_to_dg(ki_molar: float, temperature: float) -> float:
    return R_KCAL * temperature * math.log(ki_molar)


def ki_error_to_dg_error(ki_molar: float, ki_error_molar: float, temperature: float) -> float:
    return R_KCAL * temperature * abs(ki_error_molar / ki_molar)


def parse_ligands(yaml_path: Path, temperature: float) -> list[dict]:
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)

    # Support both dict-of-dicts (keyed by ligand name) and list-of-dicts formats.
    if isinstance(data, dict):
        items = data.values()
    else:
        items = data

    rows = []
    for entry in items:
        name = entry["name"]
        meas = entry["measurement"]

        unit_key = meas["unit"].lower()
        if unit_key not in UNIT_TO_MOLAR:
            print(
                f"  WARNING: unknown unit '{meas['unit']}' for ligand '{name}' — skipping",
                file=sys.stderr,
            )
            continue

        scale = UNIT_TO_MOLAR[unit_key]
        ki_molar = float(meas["value"]) * scale
        ki_error_molar = float(meas["error"]) * scale

        dg = ki_to_dg(ki_molar, temperature)
        dg_error = ki_error_to_dg_error(ki_molar, ki_error_molar, temperature)

        rows.append({"ligand": name, "DG": dg, "uncertainty": dg_error})

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("yaml_file", metavar="YAML", help="Path to ligands.yml")
    parser.add_argument(
        "-o", "--output", default="exp_dg.csv", metavar="CSV", help="Output CSV path (default: exp_dg.csv)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        metavar="K",
        help="Temperature in Kelvin for ΔG = RT ln(Ki) conversion (default: 298.15)",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml_file)
    if not yaml_path.exists():
        sys.exit(f"Error: file not found: {yaml_path}")

    rows = parse_ligands(yaml_path, args.temperature)

    output_path = Path(args.output)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ligand", "DG", "uncertainty"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} ligand(s) to {output_path}")
    for r in rows:
        print(f"  {r['ligand']:20s}  DG = {r['DG']:8.3f}  ±{r['uncertainty']:.3f}  kcal/mol")


if __name__ == "__main__":
    main()
