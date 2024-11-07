#!/bin/bash
#SBATCH --job-name=brain_simulation
#SBATCH --output=brain_simulation.out
#SBATCH --error=brain_simulation.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=500G
#SBATCH --mail-type=END
#SBATCH --mail-user=yz886@duke.edu


# Run the Python script
python run.py
