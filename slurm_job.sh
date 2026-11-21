#!/bin/bash
# Job GPU JURECA, d'après le README officiel de l'organisation (section 5).
# À lancer depuis le dossier d'une tâche, qui doit contenir son .venv :
#   cd unlearning && mkdir -p output && sbatch ../slurm_job.sh task_template.py [args...]
# Le compte, la réservation, la partition et le GPU doivent correspondre à ce que les organisateurs ont attribué.
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

source .venv/bin/activate || exit 1
srun python "$@"
echo "done!"
