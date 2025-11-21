"""
Example: Run a mock live configuration using LiveRunner.
"""

from alphaweave.data.loaders import load_directory
from alphaweave.live.config import parse_live_config
from alphaweave.live.runner import LiveRunner


def main():
    frames = load_directory("alphaweave/data", symbols=["SPY"])
    cfg = parse_live_config(
        {
            "broker": {"name": "mock"},
            "strategy": {
                "class": "examples.strategy_sma_crossover.SMACrossoverStrategy",
            },
            "datafeed": {"type": "replay", "frames": frames, "capital": 100_000.0},
            "monitor": {"dashboard_html": "output/mock_dashboard.html"},
            "persistence": {"state_path": "state/mock_state.pkl"},
        }
    )
    runner = LiveRunner.from_config(cfg)
    run = runner.run()
    print(f"Recorded {len(run.bars)} bars via LiveRunner (mock broker).")


if __name__ == "__main__":
    main()


