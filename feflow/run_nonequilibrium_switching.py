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
    python run_nonequilibrium_switching.py \
        --molecules ligands.sdf --protein protein.pdb \
        --transformation-index 0 --platform CUDA --output-dir results/
    python run_nonequilibrium_switching.py \
        --molecules ligands.sdf --protein protein.pdb \
        --network-json network.json --plan-only
    python run_nonequilibrium_switching.py \
        --network-json network.json --transformation-index 1 --platform CUDA
"""

import argparse
from functools import partial
import json
from pathlib import Path

from rdkit import Chem
from gufe import tokenization
import gufe
import openfe
import kartograf
from kartograf.filters import (
    filter_ringbreak_changes,
    filter_ringsize_changes,
    filter_whole_rings_only,
)
from openff.toolkit import RDKitToolkitWrapper, AmberToolsToolkitWrapper
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
from openff.toolkit.utils.toolkit_registry import toolkit_registry_manager, ToolkitRegistry
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
    atom_mapper = kartograf.KartografAtomMapper()
    return next(atom_mapper.suggest_mappings(component_a, component_b))


def gen_charges(smc, method="am1bcc"):
    """Assign partial charges using am1bcc (RDKit + AmberTools) or nagl (NAGLToolkitWrapper)."""
    offmol = smc.to_openff()
    if method == "nagl":
        import openff.nagl_models
        # NAGLToolkitWrapper expects a model path or filename, not the string "nagl"
        model_path = openff.nagl_models.list_available_nagl_models()[-1]
        registry = ToolkitRegistry([NAGLToolkitWrapper()])
        with toolkit_registry_manager(registry):
            offmol.assign_partial_charges(str(model_path), use_conformers=offmol.conformers)
    else:
        registry = ToolkitRegistry([RDKitToolkitWrapper(), AmberToolsToolkitWrapper()])
        with toolkit_registry_manager(registry):
            offmol.assign_partial_charges(method, use_conformers=offmol.conformers)
    return openfe.SmallMoleculeComponent.from_openff(offmol)


def gen_ligand_network(smcs):
    """Generate a LOMAP ligand network with Kartograf atom mapping."""
    mapping_filters = [
        filter_ringbreak_changes,
        filter_ringsize_changes,
        filter_whole_rings_only,
    ]
    mapper = kartograf.KartografAtomMapper(
        atom_map_hydrogens=True,
        additional_mapping_filter_functions=mapping_filters,
    )
    scorer = partial(openfe.lomap_scorers.default_lomap_score, charge_changes_score=0.1)
    ligand_network = openfe.ligand_network_planning.generate_lomap_network(
        molecules=smcs, mappers=mapper, scorer=scorer
    )
    if not ligand_network.is_connected():
        raise ValueError("Generated ligand network is not connected.")
    return ligand_network


def plan_alchemical_network(molecules_path, protein_path, protocol, charge_method="am1bcc"):
    """Load a multi-ligand SDF, build a LOMAP network, return the full AlchemicalNetwork."""
    rdmols = [m for m in Chem.SDMolSupplier(str(molecules_path), removeHs=False)]
    smcs = [openfe.SmallMoleculeComponent.from_rdkit(m) for m in rdmols]

    method_label = "NAGL" if charge_method == "nagl" else "AM1BCC (AmberTools)"
    print(f"Generating {method_label} partial charges for {len(smcs)} ligand(s)...")
    smcs = [gen_charges(smc, method=charge_method) for smc in smcs]

    ligand_network = gen_ligand_network(smcs)
    edges = list(ligand_network.edges)
    print(f"Ligand network has {len(edges)} edge(s).")

    solvent = gufe.SolventComponent(positive_ion="Na", negative_ion="Cl")
    protein = gufe.ProteinComponent.from_pdb_file(protein_path) if protein_path else None

    transformations = []
    for mapping in edges:
        comp_a = mapping.componentA
        comp_b = mapping.componentB
        components_a = {"ligand": comp_a, "solvent": solvent}
        components_b = {"ligand": comp_b, "solvent": solvent}
        if protein is not None:
            components_a["protein"] = protein
            components_b["protein"] = protein
        transformations.append(
            openfe.Transformation(
                stateA=openfe.ChemicalSystem(components_a),
                stateB=openfe.ChemicalSystem(components_b),
                mapping=mapping,
                protocol=protocol,
                name=f"{comp_a.name}_{comp_b.name}",
            )
        )

    return openfe.AlchemicalNetwork(transformations)


def save_alchemical_network(network, path):
    with open(path, "w") as f:
        json.dump(network.to_dict(), f, cls=tokenization.JSON_HANDLER.encoder)


def load_alchemical_network(path):
    with open(path) as f:
        data = json.load(f, cls=tokenization.JSON_HANDLER.decoder)
    return openfe.AlchemicalNetwork.from_dict(data)


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
    parser.add_argument("--molecules", type=str, default=None, help="SDF file with multiple ligands; triggers network planning mode")
    parser.add_argument("--network-json", type=str, default=None, help="Path to save/load the alchemical network JSON; if the file exists it is loaded instead of replanning")
    parser.add_argument("--plan-only", action="store_true", help="Plan the alchemical network and save it (requires --molecules), then exit without running")
    parser.add_argument("--ligand-a", type=str, default=None, help="SDF file for ligand A (single-pair mode)")
    parser.add_argument("--ligand-b", type=str, default=None, help="SDF file for ligand B (single-pair mode)")
    parser.add_argument("--protein", type=str, default=None, help="PDB file for a shared protein component")
    parser.add_argument("--solvate", action="store_true", help="Add a NaCl solvent component (forced on if --protein is given)")
    parser.add_argument("--transformation-index", type=int, default=0, help="Index of the transformation to run from the network (network mode only)")
    parser.add_argument("--charge-method", type=str, default="am1bcc", choices=["am1bcc", "nagl"], help="Partial charge method used when planning the network (default: am1bcc)")

    parser.add_argument("--num-switches", type=int, default=1, help="Number of forward/reverse NEQ switch replicates")
    parser.add_argument("--eq-steps", type=int, default=250, help="Internal equilibration steps per endpoint")
    parser.add_argument("--neq-steps", type=int, default=250, help="Nonequilibrium switching steps per replicate")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in kelvin")
    parser.add_argument("--platform", type=str, default="CUDA", choices=["CPU", "CUDA", "OpenCL", "HIP"], help="OpenMM compute platform")
    parser.add_argument("--output-dir", type=str, default="neq_switching_run", help="Directory for shared/scratch protocol outputs")

    args = parser.parse_args()

    # --- Validate arg combinations ---------------------------------------
    if (args.molecules or args.network_json) and (args.ligand_a or args.ligand_b):
        parser.error("--molecules / --network-json cannot be combined with --ligand-a / --ligand-b")
    if args.plan_only and not args.molecules:
        parser.error("--plan-only requires --molecules")

    # --- Build protocol from CLI args ------------------------------------
    from feflow.protocols import NonEquilibriumSwitchingProtocol

    settings = NonEquilibriumSwitchingProtocol.default_settings()
    settings.engine_settings.compute_platform = args.platform
    settings.thermo_settings.temperature = args.temperature * unit.kelvin
    settings.integrator_settings.equilibrium_steps = args.eq_steps
    settings.integrator_settings.nonequilibrium_steps = args.neq_steps
    settings.num_switches = args.num_switches
    # work/traj save frequencies auto-derive from neq_steps if left as None

    protocol = NonEquilibriumSwitchingProtocol(settings=settings)

    # --- Build / load chemical systems + mapping -------------------------
    if args.molecules or args.network_json:
        network_json = Path(args.network_json) if args.network_json else None

        if args.molecules:
            if network_json and network_json.exists():
                print(f"Loading alchemical network from {network_json} ...")
                alchemical_network = load_alchemical_network(network_json)
            else:
                alchemical_network = plan_alchemical_network(args.molecules, args.protein, protocol, charge_method=args.charge_method)
                if network_json:
                    save_alchemical_network(alchemical_network, network_json)
                    print(f"Alchemical network saved to {network_json}")
        else:
            print(f"Loading alchemical network from {network_json} ...")
            alchemical_network = load_alchemical_network(network_json)

        if args.plan_only:
            edges = list(alchemical_network.edges)
            print(f"Network has {len(edges)} transformation(s). Done (--plan-only).")
            return

        edges = list(alchemical_network.edges)
        if args.transformation_index >= len(edges):
            raise SystemExit(
                f"--transformation-index {args.transformation_index} is out of range "
                f"(network has {len(edges)} edge(s))."
            )
        selected = edges[args.transformation_index]
        state_a = selected.stateA
        state_b = selected.stateB
        mapping = selected.mapping

    elif args.ligand_a or args.ligand_b:
        if not (args.ligand_a and args.ligand_b):
            parser.error("--ligand-a and --ligand-b must be given together")
        state_a, state_b, comp_a, comp_b = build_custom_systems(
            args.ligand_a, args.ligand_b, args.protein, args.solvate
        )
        mapping = build_mapping(comp_a, comp_b)
    else:
        state_a, state_b, comp_a, comp_b = build_default_benzene_toluene_systems()
        mapping = build_mapping(comp_a, comp_b)

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
