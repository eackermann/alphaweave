# Sprint 18 — Broker Connectors & Live Stack

**Status:** 🚧 In progress

## Goal

Bridge Sprint 17’s monitoring layer with a broker/datafeed orchestration stack so that strategies can move from replay → paper → real brokers without changing their API.

## Core Deliverables

1. **Broker protocol & adapters**
   - `alphaweave.live.broker`: `Broker`, `BrokerFill`, `BrokerAccountState`
   - `alphaweave.live.adapters.base`: `BrokerConfig`, `BrokerAdapter`
   - Concrete adapters:
     - `MockBrokerAdapter` – Portfolio-backed fills for integration tests
     - `PaperBrokerAdapter` – Slightly friendlier config layer
     - Skeletons for IBKR, Alpaca, Binance (NotImplemented placeholders for future work)
2. **Config-driven live runs**
   - `alphaweave.live.config.LiveConfig` + `load_live_config/parse_live_config`
   - Supports YAML or JSON; mirrors the sample file `examples/live_config_template.yaml`
3. **State persistence hooks**
   - `alphaweave.live.state.LiveState`, `save_live_state`, `load_live_state`
   - `Strategy.get_state()/set_state()` default hooks for user-defined state
4. **Runner orchestration**
   - `alphaweave.live.runner.LiveRunner`
     - Instantiates broker adapter, replay datafeed, strategy, `LiveEngine`, `InMemoryMonitor`
     - Runs the strategy (currently via replay → VectorBacktester), emits dashboards, and persists broker state snapshots
5. **Datafeed scaffolding**
   - `alphaweave.live.datafeed.ReplayDataFeed` for deterministic tests/replay
6. **Monitoring integration**
   - Runner auto-injects `InMemoryMonitor`, writes HTML dashboards when configured, and persists state snapshots for resuming

## Example Workflows

- `examples/live_mock_runner.py` – programmatic config targeting the mock broker
- `examples/live_paper_runner.py` – load YAML config and run PaperBrokerAdapter
- `examples/live_config_template.yaml` – documented template showing all top-level keys

## Testing

- `tests/test_live_stack.py` covers mock broker fills and an end-to-end `LiveRunner` replay with persistence + dashboard output.

## Next Steps

- Flesh out real broker adapters (IBKR, Alpaca, Binance) using their APIs
- Implement incremental datafeeds (WebSocket, REST polling) beyond replay
- Feed broker fills back into strategy state in real time
- Expand risk hooks (live kill-switch, notional guards per symbol)
- Persist/restore strategy-specific state by exposing richer serialization helpers


