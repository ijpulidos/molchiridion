"""
Standalone script to run feflow's NonEquilibriumSwitchingProtocol.

By default this runs a tiny benzene -> toluene transformation in vacuum
(using the small-molecule data bundled with `gufe`'s test suite), so it
works out of the box with no external input files. Pass --ligand-a/-b
(and optionally --protein) to run your own transformation instead.

Usage
-----
    python run_nonequilibrium_switching.py
    python run_nonequilibrium_switching.py --num-switches 5 --eq-steps 1000 --neq-steps 1000
    python run_nonequilibrium_switching.py \
        --ligand-a ligA.sdf --ligand-b ligB.sdf --protein protein.pdb --solvate \
        --platform CUDA --output-dir results/
"""

import argparse
from pathlib import Path

import gufe
from openff.units import unit


def build_default_benzene_toluene_systems():
    """Benzene -> toluene in vacuum, using gufe's bundled test molecules."""
    from importlib.resources import files, as_file
    from rdkit import Chem

    source = files("gufe.tests.data").joinpath("benzene_modifications.sdf")
    with as_file(source) as f:
        supp = Chem.SDMolSupplier(str(f), removeHs=False)
        mols = {m.GetProp("_Name"): m for m in supp}

    benzene = gufe.SmallMoleculeComponent(mols["benzene"])
    toluene = gufe.SmallMoleculeComponent(mols["toluene"])

    state_a = gufe.ChemicalSystem({"ligand": benzene})
    state_b = gufe.ChemicalSystem({"ligand": toluene})
    return state_a, state_b, benzene, toluene


def build_mapping(component_a, component_b):
    """Atom mapping between two SmallMoleculeComponents via Kartograf."""
    from kartograf import KartografAtomMapper

    atom_mapper = KartografAtomMapper()
    return next(atom_mapper.suggest_mappings(component_a, component_b))


def build_custom_systems(ligand_a_path, ligand_b_path, protein_path, solvate):
    ligand_a = gufe.SmallMoleculeComponent.from_sdf_file(ligand_a_path)
    ligand_b = gufe.SmallMoleculeComponent.from_sdf_file(ligand_b_path)

    components_a = {"ligand": ligand_a}
    components_b = {"ligand": ligand_b}

    if protein_path:
        protein = gufe.ProteinComponent.from_pdb_file(protein_path)
        components_a["protein"] = protein
        components_b["protein"] = protein

    if solvate or protein_path:
        solvent = gufe.SolventComponent(positive_ion="Na", negative_ion="Cl")
        components_a["solvent"] = solvent
        components_b["solvent"] = solvent

    state_a = gufe.ChemicalSystem(components_a)
    state_b = gufe.ChemicalSystem(components_b)
    return state_a, state_b, ligand_a, ligand_b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-a", type=str, default=None, help="SDF file for ligand A (state A)")
    parser.add_argument("--ligand-b", type=str, default=None, help="SDF file for ligand B (state B)")
    parser.add_argument("--protein", type=str, default=None, help="PDB file for a shared protein component")
    parser.add_argument("--solvate", action="store_true", help="Add a NaCl solvent component (forced on if --protein is given)")

    parser.add_argument("--num-switches", type=int, default=1, help="Number of forward/reverse NEQ switch replicates")
    parser.add_argument("--eq-steps", type=int, default=250, help="Internal equilibration steps per endpoint")
    parser.add_argument("--neq-steps", type=int, default=250, help="Nonequilibrium switching steps per replicate")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in kelvin")
    parser.add_argument("--platform", type=str, default="CUDA", choices=["CPU", "CUDA", "OpenCL", "HIP"], help="OpenMM compute platform")
    parser.add_argument("--output-dir", type=str, default="neq_switching_run", help="Directory for shared/scratch protocol outputs")

    args = parser.parse_args()

    # --- Build chemical systems + mapping -------------------------------
    if args.ligand_a or args.ligand_b:
        if not (args.ligand_a and args.ligand_b):
            parser.error("--ligand-a and --ligand-b must be given together")
        state_a, state_b, comp_a, comp_b = build_custom_systems(
            args.ligand_a, args.ligand_b, args.protein, args.solvate
        )
    else:
        state_a, state_b, comp_a, comp_b = build_default_benzene_toluene_systems()

    mapping = build_mapping(comp_a, comp_b)

    # --- Build protocol settings -----------------------------------------
    from feflow.protocols import NonEquilibriumSwitchingProtocol

    settings = NonEquilibriumSwitchingProtocol.default_settings()
    settings.engine_settings.compute_platform = args.platform
    settings.thermo_settings.temperature = args.temperature * unit.kelvin
    settings.integrator_settings.equilibrium_steps = args.eq_steps
    settings.integrator_settings.nonequilibrium_steps = args.neq_steps
    settings.num_switches = args.num_switches
    # work/traj save frequencies auto-derive from neq_steps if left as None

    protocol = NonEquilibriumSwitchingProtocol(settings=settings)

    # --- Create and execute the DAG --------------------------------------
    dag = protocol.create(
        stateA=state_a,
        stateB=state_b,
        mapping=mapping,
        name="NEQ switching run",
    )

    from gufe.protocols.protocoldag import execute_DAG

    output_dir = Path(args.output_dir)
    shared = output_dir / "shared"
    scratch = output_dir / "scratch"
    shared.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.num_switches} forward + {args.num_switches} reverse switch(es) "
          f"on platform={args.platform} ...")
    dag_result = execute_DAG(
        dag,
        shared_basedir=shared,
        scratch_basedir=scratch,
        keep_shared=True,
        keep_scratch=True,
    )

    if not dag_result.ok():
        print("Protocol DAG execution FAILED:")
        for failure in dag_result.protocol_unit_failures:
            print(f"  - {failure.name}: {failure.exception}")
        raise SystemExit(1)

    # --- Gather and report -------------------------------------------------
    protocol_result = protocol.gather([dag_result])
    estimate = protocol_result.get_estimate()
    uncertainty = protocol_result.get_uncertainty()

    print(f"\nOutputs written to: {output_dir.resolve()}")
    print(f"ddG estimate:    {estimate:.4f}")
    print(f"ddG uncertainty: {uncertainty:.4f}")


if __name__ == "__main__":
    main()
