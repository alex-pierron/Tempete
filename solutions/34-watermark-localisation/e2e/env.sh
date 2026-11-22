# X-LOC-B shared environment on JURECA (sourced by the task and the CPU steps).
T="${JURECA_TEAM_ROOT:?set JURECA_TEAM_ROOT}"
PY=$T/Hackathon_Watermark_Localisation_Task/.venv/bin/python
W=$T/agents/x-loc-b
export WMLOC_ROOT=$T/agents/r-wml-b2/data/wmloc
export WMLOC_KIT=$T/Hackathon_Watermark_Localisation_Task
export HF_HOME=$T/agents/r-wml-b2/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $W/data $W/output
