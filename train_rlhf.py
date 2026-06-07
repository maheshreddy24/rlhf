import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent
SRC = ROOT / "src"


def run_stage(script_name: str):
    script_path = SRC / script_name

    print("\n" + "=" * 80)
    print(f"Running: {script_name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
    )

    print(f"Finished: {script_name}")
    return result


def main():
    try:
        # Stage 1: SFT
        run_stage("sft_training.py")

        # Stage 2: Reward Model
        run_stage("reward_model_training.py")

        # Stage 3: RLHF / PPO
        run_stage("rlhf_trainer.py")

        print("\nRLHF pipeline completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed at stage: {e.cmd}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()