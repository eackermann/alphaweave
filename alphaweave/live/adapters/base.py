"""Base types for broker adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, TypeVar

from alphaweave.live.broker import Broker


@dataclass
class BrokerConfig:
    """Configuration required to initialize a broker adapter."""

    name: str
    account_id: str | None = None
    extra: Mapping[str, Any] | None = None


T_BrokerAdapter = TypeVar("T_BrokerAdapter", bound="BrokerAdapter")


class BrokerAdapter(Broker, Protocol):
    """Broker extension that adds lifecycle hooks and config helpers."""

    def connect(self) -> None:
        """Establish the external connection."""

    def disconnect(self) -> None:
        """Clean up any network handles or threads."""

    @classmethod
    def from_config(cls: type[T_BrokerAdapter], config: BrokerConfig) -> T_BrokerAdapter:
        """Instantiate adapter from BrokerConfig."""
        ...


