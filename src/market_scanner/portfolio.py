import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class Position:
    symbol: str
    instrument_type: str # 'CE', 'PE', 'FUT', 'EQ'
    strike: Optional[float]
    expiry: Optional[str] # date string 'YYYY-MM-DD'
    qty: int # + for long, - for short
    entry_price: float
    current_price: float = 0.0
    greeks: Dict[str, float] = field(default_factory=lambda: {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0})

    def update_price(self, price: float):
        self.current_price = price

    def update_greeks(self, greeks: Dict[str, float]):
        self.greeks = greeks

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.qty

class Portfolio:
    def __init__(self, initial_capital: float = 100000.0):
        self.positions: List[Position] = []
        self.cash: float = initial_capital
        self.initial_capital = initial_capital

    def add_position(self, pos: Position):
        self.positions.append(pos)
        # Simplified cash deduction (margin not calculated here yet)
        # self.cash -= pos.entry_price * pos.qty 

    def remove_position(self, pos: Position):
        if pos in self.positions:
            self.positions.remove(pos)
    
    def get_total_pnl(self) -> float:
        return sum(p.pnl for p in self.positions)

    def get_net_greeks(self) -> Dict[str, float]:
        net_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for p in self.positions:
            # Weighted Greeks: Greek * Qty
            # Note: For options, greek is per share? Usually yes.
            # So multiply by Qty.
            net_greeks["delta"] += p.greeks.get("delta", 0.0) * p.qty
            net_greeks["gamma"] += p.greeks.get("gamma", 0.0) * p.qty
            net_greeks["vega"] += p.greeks.get("vega", 0.0) * p.qty
            net_greeks["theta"] += p.greeks.get("theta", 0.0) * p.qty
        return net_greeks

    def get_total_exposure(self) -> float:
        # Gross Market Value
        return sum(abs(p.market_value) for p in self.positions)

    def get_capital_utilization(self) -> float:
        # This assumes margin ~= market value for simplicity, or we define margin separately.
        # Let's use get_total_exposure / initial_capital for now as a rough proxy for leverage/risk.
        if self.initial_capital == 0: return 0.0
        return self.get_total_exposure() / self.initial_capital
