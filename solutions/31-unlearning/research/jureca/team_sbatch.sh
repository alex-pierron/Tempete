#!/bin/bash
# `sbatch` with the team rule: AT MOST ONE NODE FOR THE WHOLE TEAM. Every job reserves a whole node
# (4 x A100). If a teammate's job is already pending or running, the new job is queued with
# `--dependency=afterany:<their ids>`: it waits for them to end instead of taking a second node.
# Never call `sbatch` directly: use this script, or `make hk-run WHERE=jureca` (the lab applies the
# same rule, hackathon.lab.remote.one_node_guard).
#
# Usage, from the folder the job must run in (its relative output/ path resolves there):
#   bash <team>/Tempete/jureca/team_sbatch.sh [sbatch options] job.sbatch [job args]
#
# The team is read from the ACL of the team folder (the lead's setup script gave each member a
# named entry). TEAM_DIR defaults to two levels above this script (<team>/Tempete/jureca).
# Fails closed (exit 7, nothing queued) when the team cannot be read or squeue fails.
set -u
TEAM_DIR="${TEAM_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
team=$(getfacl -cp "$TEAM_DIR" 2>/dev/null | sed -n 's/^user:\([^:][^:]*\):.*/\1/p' | sort -u | paste -sd, -)
if [ -z "$team" ]; then
    echo "one-node rule: cannot read the team members (getfacl $TEAM_DIR), nothing queued; set TEAM_DIR" >&2
    exit 7
fi
jobs=$(squeue -h -u "$team" -o %i) || { echo "one-node rule: squeue failed, nothing queued" >&2; exit 7; }
deps=$(echo $jobs | tr ' ' ':')
if [ -n "$deps" ]; then
    echo "one-node rule: this job waits for the team jobs $deps" >&2
    exec sbatch --dependency=afterany:"$deps" "$@"
fi
exec sbatch "$@"
