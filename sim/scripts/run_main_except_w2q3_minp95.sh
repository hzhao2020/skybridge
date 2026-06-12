#!/usr/bin/env zsh
set -u

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sky

cd /Users/heng/Documents/skybridge/sim

LOG_DIR="results/experiment_logs"
RESULTS_ROOT="results/main_Q1000_S50_dbfixed_minp95_full"
FAILURES_CSV="${LOG_DIR}/main_Q1000_S50_dbfixed_minp95_full_failures.csv"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}"
cp -R "results/main_Q1000_S50_dbfixed_minp95_w2q3/workflow2_Q3" "${RESULTS_ROOT}/" 2>/dev/null || true

methods=(decomposition single_cloud greedy dpgm mtgp)
combos=(workflow1:Q1 workflow1:Q2 workflow1:Q3 workflow2:Q1 workflow2:Q2)

for combo in "${combos[@]}"; do
  workflow="${combo%%:*}"
  quality="${combo##*:}"
  for method in "${methods[@]}"; do
    out="${RESULTS_ROOT}/${workflow}_${quality}/${method}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') START ${workflow}/${quality}/${method}"
    python scripts/run_simulation.py \
      --workflow "${workflow}" \
      --quality "${quality}" \
      --method "${method}" \
      --results-dir "${out}" \
      --heldout-eval
    code=$?
    echo "$(date '+%Y-%m-%d %H:%M:%S') END ${workflow}/${quality}/${method} exit=${code}"
    if [ "${code}" -ne 0 ]; then
      echo "${workflow},${quality},${method},${code}" >> "${FAILURES_CSV}"
    fi
  done
done

echo "$(date '+%Y-%m-%d %H:%M:%S') DONE main experiment except W2-Q3"
