#!/bin/env bash

# This is the main execution script to run all the simulations, one per transformation.
# The idea is that this script is executed by the user in a login/head node and it will
# spawn SLURM jobs in the compute nodes, one for each transformation.

# Adapted from https://industrybenchmarks2024.readthedocs.io/en/latest/public/overview.html

for file in network_setup/transformations/*.json; do
  relpath=$(basename "${file}")  # strip off "network_setup/"
  dirpath=${relpath%.*}  # strip off final ".json"
  jobpath="$(dirname "${file}")/${dirpath}.job"
  echo "Genereating and submitting job for ${file}."
  if [ -f "${jobpath}" ]; then
    echo "${jobpath} already exists"
    exit 1
  fi
  cmd="openfe quickrun ${file} -o results/${relpath} -d results/${dirpath}"
  cat jobscript_template.sh > "${jobpath}"  # Create job file from template
  echo -e "\n${cmd}" >> "${jobpath}"  # APPEND command to created job file!
  sbatch "${jobpath}"  # Submit job
done
