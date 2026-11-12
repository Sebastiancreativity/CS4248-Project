# Runs every auditing-framework experiment reported in the paper (from official/; scores are cached in cache/scores.sqlite).
set -e

python -m framework.run --config framework/configs/qwen1.5b.json
python -m framework.run --config framework/configs/qwen3b.json
python -m framework.run --config framework/configs/mixed_pool.json
python -m framework.probe --config framework/configs/qwen1.5b.json
python -m framework.probe --config framework/configs/qwen3b.json

python -m framework.run --config framework/configs/ensemble_dev.json
python -m framework.run --config framework/configs/ensemble_dev_short.json

python -m framework.submit --config framework/configs/submit_ensemble_framework.json --gpu
python -m framework.submit --config framework/configs/submit_ensemble_consensus.json --gpu
python -m framework.submit --config framework/configs/submit_qwen3b_selfpool.json --gpu
