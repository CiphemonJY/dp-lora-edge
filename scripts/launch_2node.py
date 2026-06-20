#!/usr/bin/env python3
"""2-node distributed training using env-based init (NOT torchrun).

torchrun's c10d rendezvous hangs silently on the DGX Spark cluster.
The env-based init (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE) works
perfectly. This script uses env-based init.

Usage (2-node):
  # On primary (rank 0):
  source ~/LISA_FTM/.venv/bin/activate && source ~/LISA_FTM/nccl_cluster_env.sh
  RANK=0 WORLD_SIZE=2 MASTER_ADDR=192.168.100.12 MASTER_PORT=29500 \
    python3 scripts/launch_2node.py --experiment all --seeds 3

  # On secondary (rank 1), simultaneously:
  source ~/LISA_FTM/.venv/bin/activate && source ~/LISA_FTM/nccl_cluster_env.sh
  RANK=1 WORLD_SIZE=2 MASTER_ADDR=192.168.100.12 MASTER_PORT=29500 \
    python3 scripts/launch_2node.py --experiment all --seeds 3

Usage (single-node):
  python3 scripts/multiseed_ablation.py --experiment all --seeds 3
"""
import os
import sys

# Set defaults for env-based init
os.environ.setdefault("MASTER_ADDR", "192.168.100.12")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

# Now import and run the ablation script
from scripts.multiseed_ablation import main

if __name__ == "__main__":
    main()