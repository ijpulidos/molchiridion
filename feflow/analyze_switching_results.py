"""
Analyze NonEquilibriumSwitching results and produce cinnabar plots.

Each run directory must contain:
  protocol_result.json  — written by run_nonequilibrium_switching.py
  run_info.json         — written by run_nonequilibrium_switching.py (ligand_a, ligand_b)

An optional experimental CSV provides absolute binding free energies for comparison:
  columns: ligand,DG,uncertainty  (values in kcal/mol)

Usage
-----
    # Relative FE network from computed results only
    python analyze_switching_results.py results/run_0/ results/run_1/ ...

    # With experimental absolute FEs for comparison
    python analyze_switching_results.py results/run_*/ --experimental exp_dg.csv \\
        --method-name "NEQ Switching" --target-name "TYK2" --output-dir plots/
"""

import argparse
import json
from pathlib import Path


def _load_single_result(run_dir):
    """Load a run directory, returning (info, result_or_phase_pair).

    For two-phase runs (complex + solvent subdirs present) returns a tuple of
    (result_complex, result_solvent).  For legacy single-phase runs returns the
    single ProtocolResult directly.
    """
    run_dir = Path(run_dir)
    info_path = run_dir / "run_info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Missing run_info.json in {run_dir}")

    with open(info_path) as fh:
        info = json.load(fh)

    from feflow.protocols.nonequilibrium_switching import NonEquilibriumSwitchingProtocolResult

    complex_path = run_dir / "complex" / "protocol_result.json"
    solvent_path = run_dir / "solvent" / "protocol_result.json"
    legacy_path = run_dir / "protocol_result.json"

    if complex_path.exists() and solvent_path.exists():
        result_complex = NonEquilibriumSwitchingProtocolResult.from_json(complex_path)
        result_solvent = NonEquilibriumSwitchingProtocolResult.from_json(solvent_path)
        return info, (result_complex, result_solvent)

    if legacy_path.exists():
        return info, NonEquilibriumSwitchingProtocolResult.from_json(legacy_path)

    raise FileNotFoundError(
        f"No protocol_result.json found in {run_dir} (checked complex/, solvent/, and root)"
    )


def _build_femap(run_dirs, n_bootstraps):
    import numpy as np
    from cinnabar import FEMap, Measurement

    # First pass: collect all (ddg, unc) values per (ligand_a, ligand_b) pair
    raw = {}
    units = None

    for run_dir in run_dirs:
        print(f"  loading {run_dir} ...")
        info, result = _load_single_result(run_dir)

        if isinstance(result, tuple):
            # Two-phase: ΔΔG_binding = ΔG_complex - ΔG_solvent
            result_complex, result_solvent = result
            ddg_complex = result_complex.get_estimate().to("kcal/mol")
            ddg_solvent = result_solvent.get_estimate().to("kcal/mol")
            ddg = ddg_complex - ddg_solvent

            unc_complex = result_complex.get_uncertainty(n_bootstraps=n_bootstraps).to("kcal/mol")
            unc_solvent = result_solvent.get_uncertainty(n_bootstraps=n_bootstraps).to("kcal/mol")
            unc = (
                np.sqrt(unc_complex.magnitude**2 + unc_solvent.magnitude**2)
                * unc_complex.units
            )
        else:
            ddg = result.get_estimate().to("kcal/mol")
            unc = result.get_uncertainty(n_bootstraps=n_bootstraps).to("kcal/mol")

        if units is None:
            units = ddg.units

        key = (info["ligand_a"], info["ligand_b"])
        raw.setdefault(key, []).append((ddg.magnitude, unc.magnitude))

    # Second pass: inverse-variance weighted average for duplicate pairs
    femap = FEMap()
    for (label_a, label_b), entries in raw.items():
        ddg_vals = np.array([e[0] for e in entries])
        unc_vals = np.array([e[1] for e in entries])

        if len(entries) == 1:
            ddg_avg, unc_avg = ddg_vals[0], unc_vals[0]
        else:
            weights = 1.0 / unc_vals**2
            ddg_avg = np.sum(weights * ddg_vals) / np.sum(weights)
            unc_avg = 1.0 / np.sqrt(np.sum(weights))
            print(
                f"  {label_a} -> {label_b}: averaged {len(entries)} replicas "
                f"=> DDG = {ddg_avg:.3f} +/- {unc_avg:.3f} kcal/mol"
            )

        femap.add_measurement(Measurement(
            labelA=label_a,
            labelB=label_b,
            DG=ddg_avg * units,
            uncertainty=unc_avg * units,
            computational=True,
        ))

    return femap


def _load_experimental(exp_csv):
    import csv
    from cinnabar import Measurement, ReferenceState
    from openff.units import unit

    measurements = []
    with open(exp_csv) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            m = Measurement(
                labelA=ReferenceState(),
                labelB=row["ligand"],
                DG=float(row["DG"]) * unit.kilocalorie_per_mole,
                uncertainty=float(row["uncertainty"]) * unit.kilocalorie_per_mole,
                computational=False,
                source="experiment",
            )
            measurements.append(m)
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        metavar="RUN_DIR",
        help="Run output directories (each must contain protocol_result.json and run_info.json)",
    )
    parser.add_argument(
        "--experimental",
        metavar="CSV",
        help="CSV with experimental absolute binding FEs (columns: ligand,DG,uncertainty in kcal/mol)",
    )
    parser.add_argument("--output-dir", default=".", metavar="DIR", help="Directory for output plots (default: .)")
    parser.add_argument("--method-name", default="NEQ Switching", help="Method name for plot labels")
    parser.add_argument("--target-name", default="", help="Target name for plot labels")
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=1000,
        help="Bootstrap samples for uncertainty estimation (default: 1000)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(args.run_dirs)} result(s) ...")
    femap = _build_femap(args.run_dirs, n_bootstraps=args.n_bootstraps)

    has_experimental = bool(args.experimental)
    if has_experimental:
        print(f"Loading experimental data from {args.experimental} ...")
        for m in _load_experimental(args.experimental):
            femap.add_measurement(m)

    femap.generate_absolute_values()

    # Always: draw the perturbation network topology
    network_path = str(output_dir / "network.png")
    femap.draw_graph(title=f"{args.target_name} ({args.method_name})", filename=network_path)
    print(f"Saved: {network_path}")

    if has_experimental:
        from cinnabar.plotting import plot_DDGs, plot_DGs, plot_all_DDGs

        # All pairwise computed vs experimental DDGs scatter
        all_ddg_path = str(output_dir / "all_ddg.png")
        plot_all_DDGs(
            femap,
            "experiment",
            method_name=args.method_name,
            target_name=args.target_name,
            filename=all_ddg_path,
        )
        print(f"Saved: {all_ddg_path}")

        # Computed vs experimental DDGs
        ddg_path = str(output_dir / "ddg.png")
        plot_DDGs(
            femap,
            "experiment",
            method_name=args.method_name,
            target_name=args.target_name,
            filename=ddg_path,
        )
        print(f"Saved: {ddg_path}")

        # Absolute FEs: computed (MLE from network) vs experimental
        dg_path = str(output_dir / "dg.png")
        plot_DGs(
            femap,
            "experiment",
            method_name=args.method_name,
            target_name=args.target_name,
            filename=dg_path,
        )
        print(f"Saved: {dg_path}")
    else:
        print("No --experimental data provided; skipping DDG/DG scatter plots.")
        print("Supply a CSV (columns: ligand,DG,uncertainty in kcal/mol) to enable comparison plots.")


if __name__ == "__main__":
    main()
