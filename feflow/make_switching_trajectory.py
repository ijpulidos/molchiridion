"""
Build PyMOL-ready topology + trajectory files from the output of
run_nonequilibrium_switching.py.

For each forward/reverse NEQ switch replicate, this reconstructs the actual
*hybrid* trajectory that was propagated by the alchemical integrator: the
ForwardSwitchingUnit/ReverseSwitchingUnit save position snapshots split into
an "initial" array (hybrid positions reordered to match the old/stateA
topology) and a "final" array (reordered to match the new/stateB topology).
This script recombines those two arrays back onto the full hybrid topology
using the atom maps from the pickled HybridTopologyFactory produced by the
run's SetupUnit, so you see both the disappearing and appearing atoms at
once -- the same hybrid structure used internally by the simulation.

Requires the run to have been executed with keep_shared=True (the default
in run_nonequilibrium_switching.py) so the shared directory's intermediate
files are still on disk.

Usage
-----
    python make_switching_trajectory.py neq_switching_run
    python make_switching_trajectory.py neq_switching_run/shared --out-dir viz/
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np

_INITIAL_RE = re.compile(r"^(forward|reverse)_initial_(.+)\.npy$")


def _find_htf(shared_dir: Path):
    candidates = list(shared_dir.rglob("hybrid_topology_factory.pickle"))
    if not candidates:
        raise FileNotFoundError(
            f"No hybrid_topology_factory.pickle found under {shared_dir}. "
            "Was the run executed with keep_shared=True?"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Found multiple hybrid_topology_factory.pickle files under {shared_dir}: "
            f"{candidates}. Point this script at a single run's shared directory."
        )
    with open(candidates[0], "rb") as f:
        return pickle.load(f)


def _find_switch_pairs(shared_dir: Path):
    """Yield (direction, name, initial_path, final_path) for every switch replicate found."""
    for initial_path in sorted(shared_dir.rglob("*_initial_*.npy")):
        match = _INITIAL_RE.match(initial_path.name)
        if not match:
            continue
        direction, name = match.groups()
        final_path = initial_path.with_name(f"{direction}_final_{name}.npy")
        if not final_path.exists():
            raise FileNotFoundError(f"Expected matching {final_path} for {initial_path}")
        yield direction, name, initial_path, final_path


def _rebuild_hybrid_trajectory(htf, initial_traj: np.ndarray, final_traj: np.ndarray) -> np.ndarray:
    """
    Recombine old/new-ordered position arrays (nm) back onto the hybrid
    topology, mirroring HybridTopologyFactory._compute_hybrid_positions
    (old atoms placed first, then new atoms).
    """
    n_frames = initial_traj.shape[0]
    n_hybrid_atoms = htf.hybrid_topology.n_atoms
    hybrid_xyz_nm = np.zeros((n_frames, n_hybrid_atoms, 3), dtype=initial_traj.dtype)

    for old_idx, hybrid_idx in htf.old_to_hybrid_atom_map.items():
        hybrid_xyz_nm[:, hybrid_idx, :] = initial_traj[:, old_idx, :]
    for new_idx, hybrid_idx in htf.new_to_hybrid_atom_map.items():
        hybrid_xyz_nm[:, hybrid_idx, :] = final_traj[:, new_idx, :]

    return hybrid_xyz_nm


def _box_dimensions_angstrom(htf, n_frames: int):
    """[lx, ly, lz, alpha, beta, gamma] per frame (Angstrom/degrees), or None if non-periodic."""
    import openmm
    from MDAnalysis.lib.mdamath import triclinic_box

    system = htf.hybrid_system
    is_periodic = any(
        isinstance(system.getForce(i), openmm.NonbondedForce)
        and system.getForce(i).getNonbondedMethod()
        in (openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald)
        for i in range(system.getNumForces())
    )
    if not is_periodic:
        return None

    box = system.getDefaultPeriodicBoxVectors()
    box_angstrom = np.array([v.value_in_unit(openmm.unit.angstrom) for v in box])
    dims = triclinic_box(*box_angstrom)
    return np.tile(dims, (n_frames, 1))


def write_switch_trajectory(htf, direction, name, initial_path, final_path, out_dir: Path):
    import openmm
    from openmm.app import PDBFile
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    initial_traj = np.load(initial_path)
    final_traj = np.load(final_path)

    hybrid_xyz_nm = _rebuild_hybrid_trajectory(htf, initial_traj, final_traj)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{direction}_{name}"
    topology_path = out_dir / f"{stem}.pdb"
    trajectory_path = out_dir / f"{stem}.dcd"

    # Write a single-frame PDB (first frame) as the topology, via OpenMM --
    # MDAnalysis can then read it back in to build the Universe.
    with open(topology_path, "w") as f:
        PDBFile.writeFile(
            htf.omm_hybrid_topology,
            hybrid_xyz_nm[0] * openmm.unit.nanometer,
            f,
        )

    u = mda.Universe(str(topology_path))
    u.load_new(
        hybrid_xyz_nm * 10.0,  # nm -> Angstrom
        format=MemoryReader,
        dimensions=_box_dimensions_angstrom(htf, n_frames=hybrid_xyz_nm.shape[0]),
    )
    with mda.Writer(str(trajectory_path), n_atoms=u.atoms.n_atoms) as writer:
        for _ in u.trajectory:
            writer.write(u.atoms)

    return topology_path, trajectory_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Either the --output-dir passed to run_nonequilibrium_switching.py, "
        "or its 'shared' subdirectory directly.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write topology/trajectory files (default: <run_dir>/visualization)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    shared_dir = run_dir
    if not list(shared_dir.rglob("hybrid_topology_factory.pickle")) and (run_dir / "shared").is_dir():
        shared_dir = run_dir / "shared"

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visualization"

    htf = _find_htf(shared_dir)

    pairs = list(_find_switch_pairs(shared_dir))
    if not pairs:
        raise SystemExit(f"No forward/reverse switch trajectories found under {shared_dir}")

    print(f"Found {len(pairs)} switch replicate(s); writing topology+trajectory to {out_dir}")
    for direction, name, initial_path, final_path in pairs:
        topology_path, trajectory_path = write_switch_trajectory(
            htf, direction, name, initial_path, final_path, out_dir
        )
        print(f"  {direction}: {topology_path.name} + {trajectory_path.name}")

    print(
        "\nIn PyMOL, for each switch:\n"
        "  load <topology>.pdb, switch\n"
        "  load_traj <trajectory>.dcd, switch"
    )


if __name__ == "__main__":
    main()
