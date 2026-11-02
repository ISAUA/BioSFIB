"""Run the SSR pipeline from the standalone project root."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="BioSFIB SSR pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="YAML configuration path")
    source.add_argument("--dataset", help="Config stem, e.g. e11_0_s1 or simulation6")
    parser.add_argument("--steps", default="preprocess,train,evaluate")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if args.config:
        config = Path(args.config).resolve()
    else:
        matches = list((ROOT / "configs").glob(f"*/{args.dataset}.yaml"))
        if len(matches) != 1:
            parser.error(f"Expected one dataset config, found {len(matches)}")
        config = matches[0]
    if not config.is_file():
        parser.error(f"Config not found: {config}")
    steps = [value.strip() for value in args.steps.split(",")]
    if not steps or any(value not in ("preprocess", "train", "evaluate") for value in steps):
        parser.error("Steps must contain only preprocess, train, evaluate")
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(key, "1")
    for step in steps:
        command = [sys.executable, str(ROOT / f"run_{step}.py"), "--config", str(config)]
        if step == "evaluate" and args.checkpoint:
            command.extend(["--checkpoint", args.checkpoint])
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
