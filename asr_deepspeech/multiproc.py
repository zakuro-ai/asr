"""Multi-GPU launcher — thin wrapper around torchrun.

Usage:
    python -m asr_deepspeech.multiproc [--nproc-per-node N] [trainer args...]

This replaces the old subprocess-per-rank approach. torchrun handles rank
assignment, rendezvous, and fault tolerance automatically.
"""
import subprocess
import sys


def main():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    known, rest = parser.parse_known_args()

    import torch
    nproc = known.nproc_per_node or torch.cuda.device_count() or 1

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "-m", "asr_deepspeech.trainers",
    ] + rest

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
