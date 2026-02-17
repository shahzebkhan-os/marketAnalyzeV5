import pytest
from portfolio import Portfolio, Position
from risk_engine import RiskEngine

def test_net_greeks_calculation():
    p = Portfolio()
    # Position A: +50 Delta (Long)
    pos1 = Position("NIFTY", "CE", 22000, "2024-03-28", 1, 100, 110, {"delta": 50.0, "gamma": 10.0, "vega": 5.0, "theta": -2.0})
    # Position B: -20 Delta (Short)
    pos2 = Position("NIFTY", "PE", 21000, "2024-03-28", 1, 50, 40, {"delta": -20.0, "gamma": 5.0, "vega": 4.0, "theta": -2.0})
    
    p.add_position(pos1)
    p.add_position(pos2)
    
    net = p.get_net_greeks()
    assert net['delta'] == 30.0 # 50 - 20
    assert net['gamma'] == 15.0 # 10 + 5
    assert net['theta'] == -4.0 # -2 - 2

def test_hedge_suggestion_delta():
    re = RiskEngine()
    
    # Case 1: High Positive Delta -> Sell Futures
    greeks_high_delta = {"delta": 600.0, "gamma": 10.0}
    suggestions = re.propose_hedge(greeks_high_delta)
    assert len(suggestions) == 1
    assert "SELL 600 NIFTY FUT" in suggestions[0]
    
    # Case 2: High Negative Delta -> Buy Futures
    greeks_low_delta = {"delta": -600.0, "gamma": 10.0}
    suggestions = re.propose_hedge(greeks_low_delta)
    assert len(suggestions) == 1
    assert "BUY 600 NIFTY FUT" in suggestions[0]

def test_exposure_limit_breach():
    re = RiskEngine()
    re.MAX_CAPITAL_UTILIZATION = 0.1 # 10%
    
    p = Portfolio(initial_capital=100000.0)
    # create position with high market value
    pos = Position("NIFTY", "FUT", 0, "", 1, 15000, 15000, {}) # 15k exposure
    p.add_position(pos)
    
    # Exposure 15k / 100k = 15% > 10%
    warnings = re.check_exposure_limits(p, current_daily_pnl=0.0)
    assert len(warnings) > 0
    assert "Capital Utilization" in warnings[0]

def test_portfolio_variance_mock():
    re = RiskEngine()
    p = Portfolio()
    p.add_position(Position("NIFTY", "FUT", 0, "", 1, 100, 100, {}))
    
    var = re.calculate_portfolio_variance(p)
    assert var > 0.0
