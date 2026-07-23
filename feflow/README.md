# Nonequilibrium Switching Free Energy Calculations

This directory contains scripts for running and analyzing nonequilibrium (NEQ) switching free energy calculations using [feflow](https://github.com/OpenFreeEnergy/feflow)'s `NonEquilibriumSwitchingProtocol`.

The workflow has three stages:

```
run_nonequilibrium_switching.py  →  make_switching_trajectory.py  →  analyze_switching_results.py
        (simulate)                        (visualize)                        (analyze)
```

## Dependencies

- `feflow`, `openfe`, `gufe`, `kartograf`
- `openff-toolkit`, `openff-units`
- `openmm`, `MDAnalysis` (for trajectory writing)
- `cinnabar >= 0.5` (for analysis plots)
- `rdkit`

---

## 1. Running simulations — `run_nonequilibrium_switching.py`

### Modes

The script supports three input modes. In all cases, all simulation settings default to the protocol's built-in defaults unless explicitly overridden.

#### Quick test (no input files needed)

Runs benzene → toluene in vacuum using molecules bundled with `gufe`:

```bash
python run_nonequilibrium_switching.py
```

#### Single ligand pair

```bash
python run_nonequilibrium_switching.py \
    --ligand-a ligA.sdf \
    --ligand-b ligB.sdf \
    --output-dir results/ligA_ligB/
```

Add `--protein receptor.pdb` to include a protein; this also enables solvation automatically. Use `--solvate` without a protein for explicit solvent in the absence of a receptor.

#### Multi-ligand network (recommended for production)

**Step 1 — plan the network** (generates atom mappings, partial charges, serializes the full network to JSON):

```bash
python run_nonequilibrium_switching.py \
    --molecules ligands.sdf \
    --protein receptor.pdb \
    --network-json network.json \
    --plan-only \
    --num-switches 50 \
    --neq-steps 5000
```

The JSON embeds all simulation settings so each run step inherits them automatically.

**Step 2 — run one transformation** (repeat for each edge in the network, typically in parallel):

```bash
python run_nonequilibrium_switching.py \
    --network-json network.json \
    --transformation-index 0 \
    --platform CUDA \
    --output-dir results/edge_0/

python run_nonequilibrium_switching.py \
    --network-json network.json \
    --transformation-index 1 \
    --platform CUDA \
    --output-dir results/edge_1/
```

Settings are inherited from the network JSON. Any flag given on the command line overrides the embedded value.

### Simulation settings

| Flag | Default | Description |
|---|---|---|
| `--num-switches N` | 100 | Number of forward + reverse switch replicates |
| `--neq-steps N` | 2500 | Steps per nonequilibrium switch (10 ps at 4 fs timestep) |
| `--eq-steps N` | 1000 | Equilibration steps per endpoint before each switch |
| `--temperature T` | 298.15 | Temperature in Kelvin |
| `--platform P` | CPU | OpenMM platform: `CPU`, `CUDA`, `OpenCL`, `HIP` |
| `--charge-method M` | am1bcc | Partial charge method during planning: `am1bcc` or `nagl` |

When loading from `--network-json`, all of the above are inherited from the embedded protocol settings unless you explicitly pass the flag to override them.

### Output files

Each run produces the following in `--output-dir` (default: `neq_switching_run/`):

```
neq_switching_run/
├── run_info.json          # ligand names (used by analyze_switching_results.py)
├── protocol_result.json   # serialized ProtocolResult with all work values
├── shared/                # intermediate files kept from the protocol DAG
│   └── ...SetupUnit.../
│       ├── hybrid_topology_factory.pickle   # HybridTopologyFactory (needed for trajectories)
│       ├── forward_initial_<uuid>.npy       # positions at start of each forward switch
│       ├── forward_final_<uuid>.npy         # positions at end of each forward switch
│       ├── reverse_initial_<uuid>.npy       # positions at start of each reverse switch
│       └── reverse_final_<uuid>.npy         # ...
└── scratch/               # temporary per-unit scratch files
```

---

## 2. Visualizing trajectories — `make_switching_trajectory.py`

Reconstructs the full alchemical hybrid trajectory from the saved position snapshots and writes PyMOL-ready PDB + DCD files.

```bash
python make_switching_trajectory.py results/edge_0/
```

Or point it at the `shared/` subdirectory directly:

```bash
python make_switching_trajectory.py results/edge_0/shared --out-dir viz/edge_0/
```

### How it works

During each NEQ switch, positions are saved in two arrays: `*_initial_*.npy` (old/stateA topology ordering) and `*_final_*.npy` (new/stateB topology ordering). This script maps both back onto the full hybrid topology using the `HybridTopologyFactory` pickle, so the trajectory shows all atoms — both the disappearing and appearing regions — simultaneously.

### Output

For each switch replicate, two files are written to `--out-dir` (default: `<run_dir>/visualization/`):

- `forward_<uuid>.pdb` — single-frame topology (first frame of the switch)
- `forward_<uuid>.dcd` — DCD trajectory of the full switch
- `reverse_<uuid>.pdb` / `reverse_<uuid>.dcd` — same for reverse switches

### Viewing in PyMOL

```
load forward_<uuid>.pdb, switch
load_traj forward_<uuid>.dcd, switch
```

The script prints these exact commands for each replicate when it finishes.

> **Note:** The run must have been executed with `keep_shared=True` (the default). If you manually set `keep_shared=False`, the `.npy` files and the `HybridTopologyFactory` pickle will not be on disk.

---

## 3. Analyzing results — `analyze_switching_results.py`

Reads multiple run output directories, builds a [cinnabar](https://github.com/OpenFreeEnergy/cinnabar) `FEMap`, and produces free energy plots.

```bash
python analyze_switching_results.py results/edge_0/ results/edge_1/ results/edge_2/ \
    --method-name "NEQ Switching" \
    --target-name "TYK2" \
    --output-dir plots/
```

Each directory must contain `protocol_result.json` and `run_info.json` (both written by `run_nonequilibrium_switching.py`).

### Options

| Flag | Default | Description |
|---|---|---|
| `--experimental CSV` | — | CSV with experimental absolute binding FEs for comparison |
| `--output-dir DIR` | `.` | Directory for output plots |
| `--method-name STR` | `NEQ Switching` | Method label on plots |
| `--target-name STR` | `` | Target label on plots |
| `--n-bootstraps N` | 1000 | Bootstrap samples for uncertainty estimation (reduce for speed) |

### Output plots

| File | Contents | Requires experimental data? |
|---|---|---|
| `all_ddg.png` | Computed pairwise DDGs across the full network | No |
| `ddg.png` | Computed vs experimental relative FEs (DDG scatter) | Yes |
| `dg.png` | Computed (MLE) vs experimental absolute FEs | Yes |

### Experimental data format

A CSV with one row per ligand and three columns (no units, values in kcal/mol):

```csv
ligand,DG,uncertainty
ligand_A,-10.2,0.3
ligand_B,-9.7,0.4
ligand_C,-11.1,0.2
```

```bash
python analyze_switching_results.py results/edge_*/ \
    --experimental exp_binding_affinities.csv \
    --output-dir plots/
```

---

## Full example workflow

```bash
# 1. Plan the alchemical network (sets all simulation parameters)
python run_nonequilibrium_switching.py \
    --molecules ligands.sdf \
    --protein receptor.pdb \
    --network-json network.json \
    --plan-only \
    --num-switches 50 \
    --neq-steps 5000 \
    --platform CUDA

# 2. Run each edge (parallelise across GPUs / job array)
for i in 0 1 2 3; do
    python run_nonequilibrium_switching.py \
        --network-json network.json \
        --transformation-index $i \
        --platform CUDA \
        --output-dir results/edge_$i/
done

# 3. (Optional) Build trajectories for a run
python make_switching_trajectory.py results/edge_0/ --out-dir viz/edge_0/

# 4. Analyze all results
python analyze_switching_results.py results/edge_*/ \
    --experimental exp_dg.csv \
    --method-name "NEQ Switching" \
    --target-name "TYK2" \
    --output-dir plots/
```
