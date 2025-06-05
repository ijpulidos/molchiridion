"""
Minimal example for creating a solvated protein-ligand complex topology.

It assumes there's a PDB file with the protein information and an SDF 
with the ligands information.
"""

import copy
from pathlib import Path
import logging
import numpy as np
import mdtraj
from openmm import MonteCarloBarostat, LangevinMiddleIntegrator, XmlSerializer
from openmm.app import Modeller, PME, HBonds, Simulation, CheckpointReporter, StateDataReporter, DCDReporter, PDBFile
from openmm import unit
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator

# Configure logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# Simulation parameters
timestep = 4 * unit.femtoseconds
pressure = 1.0 * unit.atmosphere
temperature = 300.0 * unit.kelvin
barostat_period = 50
water_model = "tip3p"
solvent_padding = 9.0 * unit.angstroms
ionic_concentration = 0.15 * unit.molar
# Forcefield parameters
hmass = 3.0 * unit.amu
pme_tolerance = 2.5e-04
constraints = HBonds
# Execution parameters
checkpoint_frequency = 250
logging_frequency = 250
traj_frequency = 250
nsteps = 10000

def create_complex(receptor_path: Path | str, ligands_path: Path | str):
    """
    Read input files for receptor/protein and ligands/small-molecules.

    Note: It always takes the first available conformer for the ligands.
    """
    # Make sure we are dealing with Path objects
    receptor_path = Path(receptor_path)
    ligands_path = Path(ligands_path)
    receptor_pdb = PDBFile(str(receptor_path))
    receptor_top = receptor_pdb.topology
    ligands_off = Molecule.from_file(ligands_path)
    if not isinstance(ligands_off, list):
        # Make it a list either way. Treat it as a list from now on.
        ligands_off = [ligands_off]

    # Create mdtraj topology with receptor and add the ligands topologies iteratively
    receptor_md_top = mdtraj.Topology.from_openmm(receptor_top)
    complex_md_topology = copy.copy(receptor_md_top)  # initialize complex topology with only receptor
    # Initialize positions with receptor positions -- this way avoids having Vec3 on each element
    n_atoms_receptor = receptor_top.getNumAtoms()
    complex_positions = unit.Quantity(np.zeros([n_atoms_receptor, 3]), unit=unit.nanometers)
    complex_positions[:n_atoms_receptor, :] = receptor_pdb.positions
    for molecule in ligands_off:
        ligand_top = molecule.to_topology().to_openmm()
        ligand_positions = molecule.conformers[0].to_openmm().in_units_of(unit.nanometers)
        # Append positions to complex positions
        complex_positions = np.append(complex_positions, ligand_positions) * unit.nanometers
        ligand_md_top = mdtraj.Topology.from_openmm(ligand_top)
        complex_md_topology = complex_md_topology.join(ligand_md_top)

    # Make sure number of atoms match
    n_atoms_total = receptor_top.getNumAtoms()
    for molecule in ligands_off:
        n_atoms_total += molecule.n_atoms
    assert n_atoms_total == complex_positions.shape[0], \
            "Number of atoms after merging the protein and ligand topology does not match"




# Read protein
protein_pdb = PDBFile("protein.pdb")
protein_top = protein_pdb.topology
# Read ligands
ligand_off = Molecule.from_file("ligand.sdf")
ligand_top = ligand_off.to_topology().to_openmm()
ligand_positions = ligand_off.conformers[0].to_openmm().in_units_of(unit.nanometers)
small_mols_list = [ligand_off]

# Merging topologies using mdtraj to join topologies
protein_md_top = mdtraj.Topology.from_openmm(protein_top)
ligand_md_top = mdtraj.Topology.from_openmm(ligand_top)
complex_md_top = protein_md_top.join(ligand_md_top)
complex_top = complex_md_top.to_openmm()
# Double check number of atoms 
n_atoms_total = complex_top.getNumAtoms()
n_atoms_protein = protein_top.getNumAtoms() 
n_atoms_ligand = ligand_top.getNumAtoms()
_logger.info(f"Complex topology generated. Total atoms: {n_atoms_total} (protein: {n_atoms_protein}, ligand: {n_atoms_ligand}).")
assert n_atoms_total == n_atoms_protein + n_atoms_ligand, "Number of atoms after merging the protein and ligand topology does not match"
complex_positions = unit.Quantity(np.zeros([n_atoms_total, 3]), unit=unit.nanometers)
complex_positions[:n_atoms_protein, :] = protein_pdb.positions
complex_positions[n_atoms_protein:n_atoms_protein+n_atoms_ligand, :] = ligand_positions


forcefield_kwargs = {'removeCMMotion': True, 'ewaldErrorTolerance': pme_tolerance,
                     'constraints' : constraints, 'rigidWater': True, 'hydrogenMass' : hmass}
periodic_forcefield_kwargs = {"nonbondedMethod": PME}
barostat = MonteCarloBarostat(pressure, temperature, barostat_period)
forcefields_list = ['amber/ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml']
small_molecule_forcefield = "openff-2.1.0"

# Generate OpenMM system
system_generator = SystemGenerator(forcefields=forcefields_list, forcefield_kwargs=forcefield_kwargs, 
                                   periodic_forcefield_kwargs=periodic_forcefield_kwargs,
                                   barostat=barostat, small_molecule_forcefield=small_molecule_forcefield, 
                                   molecules=small_mols_list, cache="cache.db")

# Solvate system
_logger.info("Solvating system")
modeller = Modeller(complex_top, complex_positions)
modeller.addSolvent(system_generator.forcefield, model=water_model, padding=solvent_padding, ionicStrength=ionic_concentration)

# Get solvated topology and positions and create omm solvated system
solvated_top = modeller.getTopology()
solvated_positions = modeller.getPositions()
solvated_system = system_generator.create_system(solvated_top)

# Create simulation
integrator = LangevinMiddleIntegrator(temperature, 1/unit.picosecond, timestep)
simulation = Simulation(solvated_top, solvated_system, integrator)
simulation.context.setPositions(solvated_positions)
simulation.minimizeEnergy(maxIterations=250)
minimized_state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
# Serialize components -- minimized
with open("system.xml", "w") as _file:
    _file.write(XmlSerializer.serialize(solvated_system))
with open("state.xml", "w") as _file:
    _file.write(XmlSerializer.serialize(minimized_state))
with open("integrator.xml", "w") as _file:
    _file.write(XmlSerializer.serialize(integrator))

# Run simulation and monitor the results
simulation.reporters.append(DCDReporter('traj.dcd', traj_frequency))
simulation.reporters.append(CheckpointReporter('checkpoint.chk', checkpoint_frequency))
simulation.reporters.append(StateDataReporter('reporter.log', logging_frequency, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))
simulation.step(nsteps)
