#!/usr/bin/env bash
set -euo pipefail

# Fetches external inspiration repos to a local, non-versioned folder.
# This keeps the main experiment repo clean while making references reproducible.

BASE_DIR="${1:-$HOME/waymax_risk_uq_external_refs}"
mkdir -p "${BASE_DIR}"

clone_or_update() {
  local name="$1"
  local url="$2"
  if [ -d "${BASE_DIR}/${name}/.git" ]; then
    git -C "${BASE_DIR}/${name}" fetch --depth 1 origin
    git -C "${BASE_DIR}/${name}" reset --hard FETCH_HEAD
  else
    git clone --depth 1 "${url}" "${BASE_DIR}/${name}"
  fi
  local sha
  sha="$(git -C "${BASE_DIR}/${name}" rev-parse HEAD)"
  echo "${name} ${sha}"
}

clone_or_update waymax https://github.com/waymo-research/waymax.git
clone_or_update temperature_scaling https://github.com/gpleiss/temperature_scaling.git
clone_or_update uncertainty_baselines https://github.com/google/uncertainty-baselines.git
clone_or_update plancp https://github.com/Jiankai-Sun/PlanCP.git
clone_or_update icp https://github.com/tedhuang96/icp.git
clone_or_update rap https://github.com/TRI-ML/RAP.git
clone_or_update racp https://github.com/KhMustafa/Risk-aware-contingency-planning-with-multi-modal-predictions.git
clone_or_update radius https://github.com/roahmlab/RADIUS.git
clone_or_update cuqds https://github.com/HuiqunHuang/CUQDS.git

echo "Fetched inspiration repos under: ${BASE_DIR}"
