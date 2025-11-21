"""Search space definitions for strategy discovery."""

from dataclasses import dataclass
from typing import Any, Sequence, Iterator
import numpy as np
from itertools import product


@dataclass
class Param:
    """Discrete parameter with a list of values."""

    name: str
    values: Sequence[Any]

    def __post_init__(self):
        """Validate parameter."""
        if not self.values:
            raise ValueError(f"Param '{self.name}' must have at least one value")


@dataclass
class ContinuousParam:
    """Continuous parameter with a range."""

    name: str
    low: float
    high: float
    log: bool = False

    def __post_init__(self):
        """Validate parameter."""
        if self.low >= self.high:
            raise ValueError(
                f"ContinuousParam '{self.name}': low ({self.low}) must be < high ({self.high})"
            )

    def sample(self, n: int, random_state: int | None = None) -> list[float]:
        """
        Sample n values from this parameter's range.

        Args:
            n: Number of samples
            random_state: Random seed for reproducibility

        Returns:
            List of sampled values
        """
        rng = np.random.RandomState(random_state)

        if self.log:
            # Log-uniform sampling
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            samples = np.exp(rng.uniform(log_low, log_high, n))
        else:
            # Uniform sampling
            samples = rng.uniform(self.low, self.high, n)

        return samples.tolist()


@dataclass
class SearchSpace:
    """Search space for strategy parameters."""

    params: Sequence[Param | ContinuousParam]

    def __post_init__(self):
        """Validate search space."""
        if not self.params:
            raise ValueError("SearchSpace must have at least one parameter")

        # Check for duplicate names
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            raise ValueError("SearchSpace parameters must have unique names")

    def grid_combinations(self) -> Iterator[dict[str, Any]]:
        """
        Generate all combinations for discrete parameters.

        Yields:
            Dictionary mapping parameter names to values
        """
        discrete_params = [p for p in self.params if isinstance(p, Param)]
        if not discrete_params:
            # No discrete params, yield empty dict
            yield {}
            return

        param_names = [p.name for p in discrete_params]
        param_values = [p.values for p in discrete_params]

        for combo in product(*param_values):
            yield dict(zip(param_names, combo))

    def sample_random(self, n: int, random_state: int | None = None) -> list[dict[str, Any]]:
        """
        Sample n random configurations from the search space.

        Args:
            n: Number of samples
            random_state: Random seed for reproducibility

        Returns:
            List of parameter dictionaries
        """
        rng = np.random.RandomState(random_state)
        samples = []

        for _ in range(n):
            config = {}

            # Sample discrete params
            for param in self.params:
                if isinstance(param, Param):
                    config[param.name] = rng.choice(param.values)
                elif isinstance(param, ContinuousParam):
                    # Sample single value
                    if param.log:
                        log_low = np.log(param.low)
                        log_high = np.log(param.high)
                        value = np.exp(rng.uniform(log_low, log_high))
                    else:
                        value = rng.uniform(param.low, param.high)
                    config[param.name] = float(value)

            samples.append(config)

        return samples

    def size(self) -> int | None:
        """
        Get the size of the search space (number of combinations).

        Returns:
            Total number of combinations if all discrete, None if any continuous params
        """
        continuous_params = [p for p in self.params if isinstance(p, ContinuousParam)]
        if continuous_params:
            return None  # Infinite/continuous space

        discrete_params = [p for p in self.params if isinstance(p, Param)]
        total = 1
        for param in discrete_params:
            total *= len(param.values)
        return total

