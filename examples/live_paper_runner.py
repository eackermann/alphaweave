"""
Example: Load a YAML config and execute a paper-trading run.
"""

from alphaweave.live.config import load_live_config
from alphaweave.live.runner import LiveRunner


def main():
    cfg = load_live_config("examples/live_config_template.yaml")
    runner = LiveRunner.from_config(cfg)
    run = runner.run()
    print(f"LiveRunner completed with {len(run.trades)} trades.")


if __name__ == "__main__":
    main()


