# Tyk2 benchmark setup and execution
The aim of this quick tutorial is to run a tyk2 benchmark test from the JACS dataset as a sanity
check when we have new releases of the new protocols and tools, for example for a new `feflow`
release.

## Downloading the input data
The data will be obtained from Open Free Energy's industry benchmark 2024 repository, located at
https://github.com/OpenFreeEnergy/IndustryBenchmarks2024.
Please run the following in a terminal to get the input files

Clone the repository contents in current working directory
```bash
git clone https://github.com/OpenFreeEnergy/IndustryBenchmarks2024.git
```
This will create a subdirectory named `IndustryBenchmarks2024` in the current directory.

Inside the created subdirectory, checkout the latest release tag. To date it is the `v1.0.0` release.
```bash
cd IndustryBenchmarks2024
git checkout v1.0.0
```
This will recreate the snapshot for all the files for the specified release. Useful for reproducibility.

Copy the contents of the `industry_benchmarks/input_structures/prepared_structures/jacs_set/tyk2` directory 
into the current directory, for example, with
```bash
cp -r industry_benchmarks/input_structures/prepared_structures/jacs_set/tyk2/ ../input_files
```
This would copy the `tyk2` directory as the `input_files` in the current working directory.

## Preparing the ligand transformation network

We will use the included `plan-rbfe-network.py` script in this directory for creating the atom maps
and transformation networks between ligands by running the following command in the root working
directory

```bash 
python plan-rbfe-network.py --protein input_files/protein.pdb --molecules input_files/ligands.sdf --output-dir network_setup
```

## Simulation execution
The most basic and direct way to run the simulation is by using the OpenFE CLI as
```bash
openfe quickrun network_setup/transformations/complex_ejm_50_ejm_42.json -o results_dir/results_complex.json -d results_dir
```
This will run the specified transformation. In order to run the whole network, we would need to run
all the transformations separately using a similar command. 

### Running all transformations with SLURM

The script `main_exec.sh` provided in this directory is a shell script that will run all the transformations
using the openfe CLI and SLURM in an HPC environment. To run the jobs you would just execute the
following command from the login node in the HPC system.
```bash
bash main_exec.sh
```
this will spawn many jobs, one for each transformation, in your SLURM queue.

