# Environment for mu2e-hydra: mu2e-vision's environment (cvmfs ana env with torch,
# Mu2e/DQM, and the histcompare package, whose synthetic generator defines the benchmark
# set and the detector) plus this directory.
#
#   source mu2e-hydra/setup_env.sh

_HY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VD="${MU2E_VISION_DIR:-${VISIONDQM_DIR:-$_HY/../mu2e-vision}}"   # VISIONDQM_DIR: old name
if [ -f "$_VD/setup_env.sh" ]; then
    source "$_VD/setup_env.sh"
else
    echo "WARNING: mu2e-vision not found at $_VD; set MU2E_VISION_DIR"
fi
export PYTHONPATH="$_HY${PYTHONPATH:+:$PYTHONPATH}"
unset _VD
