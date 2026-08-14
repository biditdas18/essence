#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
source .venv_recbole/bin/activate

run() {
    echo "=================================================================="
    echo "RUNNING: python experiments/mind_comirec/train.py $*"
    echo "=================================================================="
    python experiments/mind_comirec/train.py "$@"
}

# Canonical (pretrained-init, default hyperparams)
run --dataset lastfm --model mind
run --dataset amazon --model mind

# Random-init ablation
run --dataset lastfm --model mind --embed-init random --tag randominit
run --dataset amazon --model mind --embed-init random --tag randominit

# LR sweep (Last.fm only)
run --dataset lastfm --model mind --lr 0.0002 --tag lr2e-4
run --dataset lastfm --model mind --lr 0.0001 --tag lr1e-4

# Seed variance
run --dataset lastfm --model mind --seed 1 --tag seed1
run --dataset lastfm --model mind --seed 2 --tag seed2
run --dataset amazon --model mind --seed 1 --tag seed1
run --dataset amazon --model mind --seed 2 --tag seed2

# K sweep (Last.fm only)
run --dataset lastfm --model mind --K 2 --tag K2
run --dataset lastfm --model mind --K 3 --tag K3
run --dataset lastfm --model mind --K 5 --tag K5
run --dataset lastfm --model mind --K 6 --tag K6

echo "=================================================================="
echo "ALL 14 MIND RERUNS COMPLETE"
echo "=================================================================="
