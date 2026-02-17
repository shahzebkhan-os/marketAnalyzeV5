import pytest
import numpy as np
from monte_carlo import MonteCarloEngine
from portfolio import Portfolio, Position

def test_gbm_simulation_shape():
    mc = MonteCarloEngine(seed=42)
    paths = mc.simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1.0, n_sims=100, n_steps=10)
    assert paths.shape == (100, 11) # n_steps + 1 because of T=0
    assert paths[0, 0] == 100

def test_bs_pricing():
    mc = MonteCarloEngine()
    # Known BS Value: S=100, K=100, T=1, r=0.05, sigma=0.2, Call
    # Approx ~ 10.45
    S = np.array([100.0])
    price = mc.price_option_bs(S, K=100, T=1, r=0.05, sigma=0.2, option_type='CE')
    assert 10.0 < price[0] < 11.0

def test_stress_test():
    p = Portfolio()
    # Long Call: +Delta, +Gamma, +Vega
    # Delta ~ 0.5 * 10 = 5
    # Gamma ~ 0.0 * ... let's set explicit
    pos = Position("NIFTY", "CE", 100, "", 10, 10, 10, {"delta": 0.5, "gamma": 0.02, "vega": 0.1})
    p.add_position(pos)
    
    mc = MonteCarloEngine()
    stress = mc.run_stress_test(p)
    
    # Gap Up: Positive Delta -> Positive PnL
    # Gap Down: Positive Delta -> Negative PnL
    assert stress['gap_up_2pct'] > 0
    assert stress['gap_down_2pct'] < 0
    
    # IV Spike: Positive Vega -> Positive PnL
    assert stress['iv_spike_20pct'] > 0

def test_simulation_runs():
    p = Portfolio()
    p.add_position(Position("NIFTY", "CE", 22500, "", 1, 150, 22500, {"delta": 0.5}))
    
    mc = MonteCarloEngine()
    res = mc.run_portfolio_simulation(p, n_sims=50)
    assert 'var_95' in res
    assert 'var_99' in res
    assert len(res['pnl_sim']) == 50
