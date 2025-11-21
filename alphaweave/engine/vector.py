"""Vector backtester implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Type, Union
from types import SimpleNamespace

import pandas as pd

from alphaweave.core.frame import Frame
from alphaweave.core.types import Fill, Order, OrderType
from alphaweave.engine.base import BaseBacktester
from alphaweave.engine.portfolio import Portfolio
from alphaweave.engine.risk import RiskLimits
from alphaweave.execution.fees import FeesModel, NoFees
from alphaweave.execution.slippage import SlippageModel, NoSlippage
from alphaweave.execution.volume import VolumeLimitModel
from alphaweave.results.result import BacktestResult
from alphaweave.strategy.base import Strategy


def _get_frame_index(frame: Frame) -> pd.DatetimeIndex:
    pdf = frame.to_pandas()
    idx = pdf.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Frame must have a DatetimeIndex")
    dt_idx = pd.DatetimeIndex(idx)
    return dt_idx.drop_duplicates().sort_values()


def _build_master_calendar(data: Union[Frame, Dict[str, Frame]]) -> pd.DatetimeIndex:
    if isinstance(data, Frame):
        calendar = _get_frame_index(data)
    elif isinstance(data, dict):
        calendar: Optional[pd.DatetimeIndex] = None
        for frame in data.values():
            idx = _get_frame_index(frame)
            calendar = idx if calendar is None else calendar.intersection(idx)
        if calendar is None:
            raise ValueError("No data provided for calendar construction")
    else:
        raise TypeError("data must be a Frame or dict[str, Frame]")

    if calendar.empty:
        raise ValueError("Master calendar is empty; no overlapping dates between data sources")
    return calendar.sort_values()


def _align_frame_to_calendar(frame: Frame, calendar: pd.DatetimeIndex) -> Frame:
    pdf = frame.to_pandas()
    aligned = pdf.reindex(calendar)
    if aligned.isnull().any().any():
        raise ValueError("Data missing entries for master calendar alignment")
    return Frame.from_pandas(aligned)


def _align_data_to_calendar(
    data: Union[Frame, Dict[str, Frame]], calendar: pd.DatetimeIndex
) -> Union[Frame, Dict[str, Frame]]:
    if isinstance(data, Frame):
        return _align_frame_to_calendar(data, calendar)
    elif isinstance(data, dict):
        return {symbol: _align_frame_to_calendar(frame, calendar) for symbol, frame in data.items()}
    else:
        raise TypeError("data must be a Frame or dict[str, Frame]")


class VectorBacktester(BaseBacktester):
    """Vector backtester with portfolio-aware execution."""

    def run(
        self,
        strategy_cls: Type[Strategy],
        data: Union[Frame, Dict[str, Frame]],
        capital: float = 100_000.0,
        fees: Optional[FeesModel] = None,
        slippage: Optional[SlippageModel] = None,
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        volume_limit: Optional[VolumeLimitModel] = None,
        risk_limits: Optional[RiskLimits] = None,
    ) -> BacktestResult:
        """Execute a backtest and return the results."""

        if isinstance(data, Frame):
            data_dict: Dict[str, Frame] = {"_default": data}
        elif isinstance(data, dict):
            if not data:
                raise ValueError("data must contain at least one symbol")
            data_dict = data
        else:
            raise TypeError("data must be a Frame or dict[str, Frame]")

        calendar = _build_master_calendar(data_dict)
        aligned = _align_data_to_calendar(data_dict, calendar)
        if isinstance(aligned, Frame):
            aligned_dict: Dict[str, Frame] = {"_default": aligned}
            strategy_data: Union[Frame, Dict[str, Frame]] = aligned
        else:
            aligned_dict = aligned
            strategy_data = aligned

        data_frames = {symbol: frame.to_pandas() for symbol, frame in aligned_dict.items()}

        fees_model = fees or NoFees()
        slippage_model = slippage or NoSlippage()
        volume_model = volume_limit or VolumeLimitModel()
        risk = risk_limits or RiskLimits()
        kwargs = strategy_kwargs or {}

        strategy = strategy_cls(strategy_data, **kwargs) if kwargs else strategy_cls(strategy_data)
        strategy.init()

        portfolio = Portfolio(capital)
        equity_series: list[float] = []
        trades: list[Fill] = []
        order_id = 0

        def _coerce_order(entry: Any) -> Optional[Order]:
            if isinstance(entry, Order):
                return entry
            if isinstance(entry, (tuple, list)) and len(entry) >= 3:
                kind = entry[0]
                symbol = entry[1]
                value = entry[2]
                if kind == "target_percent":
                    return Order(
                        symbol=symbol,
                        size=0.0,
                        order_type=OrderType.TARGET_PERCENT,
                        target_percent=float(value),
                    )
            return None

        def _apply_fill(
            src_order: Order,
            symbol: str,
            size: float,
            price: float,
            timestamp: pd.Timestamp,
        ) -> None:
            nonlocal order_id
            if abs(size) < 1e-12:
                return
            order_id += 1
            fill = Fill(
                order_id=order_id,
                symbol=symbol,
                size=size,
                price=price,
                datetime=timestamp.to_pydatetime(),
            )
            fee = fees_model.calculate(src_order, fill)
            portfolio.apply_fill(fill)
            portfolio.cash -= fee
            trades.append(fill)

        for i, timestamp in enumerate(calendar):
            prices: Dict[str, float] = {}
            bar_rows: Dict[str, pd.Series] = {}
            for symbol, df in data_frames.items():
                row = df.iloc[i]
                bar_rows[symbol] = row
                close = row.get("close")
                if pd.isna(close):
                    continue
                prices[symbol] = float(close)

            if not prices:
                continue

            strategy._orders = []
            strategy._set_current_index(i)
            strategy._set_current_timestamp(pd.Timestamp(timestamp))
            strategy.next(i)

            normalized_orders = []
            for entry in strategy._orders:
                order = _coerce_order(entry)
                if order is not None:
                    normalized_orders.append(order)

            target_orders = [
                o for o in normalized_orders if o.order_type == OrderType.TARGET_PERCENT
            ]
            explicit_orders = [
                o for o in normalized_orders if o.order_type != OrderType.TARGET_PERCENT
            ]

            # Process target percent orders
            if target_orders:
                portfolio_value = portfolio.total_value(prices)
                if portfolio_value > 0:
                    targets: Dict[str, float] = {}
                    for order in target_orders:
                        if order.target_percent is None:
                            continue
                        targets[order.symbol] = order.target_percent

                    if targets:
                        clamped: Dict[str, float] = {}
                        for symbol, weight in targets.items():
                            max_w = max(risk.max_symbol_weight, 0.0)
                            if max_w > 0:
                                weight = max(min(weight, max_w), -max_w)
                            clamped[symbol] = weight

                        gross = sum(abs(w) for w in clamped.values())
                        max_gross = max(risk.max_gross_leverage, 0.0)
                        if max_gross > 0 and gross > max_gross and gross > 0:
                            scale = max_gross / gross
                            clamped = {sym: w * scale for sym, w in clamped.items()}

                        for symbol, weight in clamped.items():
                            price = prices.get(symbol)
                            if not price or price == 0:
                                continue
                            bar_row = bar_rows.get(symbol)
                            if bar_row is None:
                                continue
                            target_value = weight * portfolio_value
                            current_pos = portfolio.positions.get(symbol)
                            current_size = current_pos.size if current_pos else 0.0
                            target_size = target_value / price
                            desired_change = target_size - current_size
                            bar_volume = float(bar_row.get("volume", 0) or 0.0)
                            size_change = volume_model.clamp_size(desired_change, bar_volume)
                            if size_change > 0:
                                max_affordable = portfolio.cash / price if price > 0 else 0.0
                                size_change = min(size_change, max_affordable)
                            elif size_change < 0:
                                max_sellable = abs(current_size)
                                size_change = max(size_change, -max_sellable)
                            if abs(size_change) <= 1e-12:
                                continue
                            exec_order = Order(
                                symbol=symbol,
                                size=size_change,
                                order_type=OrderType.MARKET,
                            )
                            execution_price = slippage_model.execute(
                                exec_order, SimpleNamespace(close=price)
                            )
                            _apply_fill(exec_order, symbol, size_change, execution_price, timestamp)

            # Process explicit orders
            for order in explicit_orders:
                symbol = order.symbol
                if symbol not in data_frames:
                    if len(data_frames) == 1:
                        symbol = next(iter(data_frames))
                    else:
                        continue
                row = bar_rows.get(symbol)
                if row is None:
                    continue
                close = row.get("close")
                high = row.get("high")
                low = row.get("low")
                volume = float(row.get("volume", 0) or 0.0)
                desired_size = float(order.size)
                price = float(close) if close is not None else None
                if price is None or price == 0:
                    continue

                def clamp_size(size: float) -> float:
                    clamped = volume_model.clamp_size(size, volume)
                    current_pos = portfolio.positions.get(symbol)
                    current_qty = current_pos.size if current_pos else 0.0
                    if clamped > 0:
                        max_affordable = portfolio.cash / price if price > 0 else 0.0
                        clamped = min(clamped, max_affordable)
                    elif clamped < 0:
                        max_sellable = abs(current_qty)
                        clamped = max(clamped, -max_sellable)
                    return clamped

                filled = False
                fill_price = None
                fill_size = 0.0

                if order.order_type == OrderType.MARKET:
                    price = float(close)
                    size = clamp_size(desired_size)
                    if abs(size) > 1e-12:
                        exec_order = Order(symbol=symbol, size=size, order_type=OrderType.MARKET)
                        exec_price = slippage_model.execute(
                            exec_order, SimpleNamespace(close=price)
                        )
                        _apply_fill(exec_order, symbol, size, exec_price, timestamp)
                elif order.order_type == OrderType.LIMIT and order.limit_price is not None:
                    if pd.notna(low) and pd.notna(high):
                        limit = order.limit_price
                        if desired_size > 0 and low <= limit <= high:
                            size = clamp_size(desired_size)
                            if abs(size) > 1e-12:
                                exec_order = Order(
                                    symbol=symbol,
                                    size=size,
                                    order_type=OrderType.LIMIT,
                                    limit_price=limit,
                                )
                                _apply_fill(exec_order, symbol, size, limit, timestamp)
                        elif desired_size < 0 and low <= limit <= high:
                            size = clamp_size(desired_size)
                            if abs(size) > 1e-12:
                                exec_order = Order(
                                    symbol=symbol,
                                    size=size,
                                    order_type=OrderType.LIMIT,
                                    limit_price=limit,
                                )
                                _apply_fill(exec_order, symbol, size, limit, timestamp)
                elif order.order_type == OrderType.STOP and order.stop_price is not None:
                    stop = order.stop_price
                    if desired_size > 0 and pd.notna(high) and high >= stop:
                        size = clamp_size(desired_size)
                        if abs(size) > 1e-12:
                            exec_order = Order(
                                symbol=symbol,
                                size=size,
                                order_type=OrderType.STOP,
                                stop_price=stop,
                            )
                            _apply_fill(exec_order, symbol, size, stop, timestamp)
                    elif desired_size < 0 and pd.notna(low) and low <= stop:
                        size = clamp_size(desired_size)
                        if abs(size) > 1e-12:
                            exec_order = Order(
                                symbol=symbol,
                                size=size,
                                order_type=OrderType.STOP,
                                stop_price=stop,
                            )
                            _apply_fill(exec_order, symbol, size, stop, timestamp)
                elif (
                    order.order_type == OrderType.STOP_LIMIT
                    and order.stop_price is not None
                    and order.limit_price is not None
                ):
                    triggered = False
                    if desired_size > 0 and pd.notna(high) and high >= order.stop_price:
                        triggered = True
                    elif desired_size < 0 and pd.notna(low) and low <= order.stop_price:
                        triggered = True
                    if triggered and pd.notna(low) and pd.notna(high):
                        limit = order.limit_price
                        if low <= limit <= high:
                            size = clamp_size(desired_size)
                            if abs(size) > 1e-12:
                                exec_order = Order(
                                    symbol=symbol,
                                    size=size,
                                    order_type=OrderType.STOP_LIMIT,
                                    stop_price=order.stop_price,
                                    limit_price=limit,
                                )
                                _apply_fill(exec_order, symbol, size, limit, timestamp)

            equity_value = float(portfolio.total_value(prices))
            equity_series.append(equity_value)

        return BacktestResult(equity_series=equity_series, trades=trades)
