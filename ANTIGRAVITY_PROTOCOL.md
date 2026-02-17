# Antigravity Protocol: Short-Covering Detection

The **Antigravity Protocol** is a specialized detection logic designed to identify "Short Covering Rallies." These occur when the market moves upward against a heavy resistance level because institutional Call Writers are trapped and forced to unwind their bearish positions.

## 1. The Core Logic

The protocol calculates a score from **0 to 100** based on three weighted factors:

### A. The Trap (Weight: 40%)
- **Objective:** Identify if the "Call Wall" is being breached.
- **Metric:** Finds the strike price with the **Highest Call Open Interest (OI)**.
- **Logic:** Triggered if `Spot Price >= (Wall Strike - 0.5%)`. 
- **Significance:** This indicates that the primary resistance level is under siege, forcing writers to reconsider their exposure.

### B. The Panic (Weight: 40%)
- **Objective:** Detect if writers are actively fleeing their positions.
- **Metric:** Net change in Call OI for strikes within ±2% of the Spot Price.
- **Logic:** Triggered if `Net Change < 0`.
- **Significance:** A negative OI change near the spot price during a rally is a classic signal of a "Short Squeeze," as writers buy back contracts to limit losses.

### C. The Velocity (Weight: 20%)
- **Objective:** Confirm the momentum of the squeeze.
- **Metric:** Price movement relative to Implied Volatility (IV).
- **Logic:** Rising Price + Rising/High IV.
- **Significance:** A true short squeeze often shows rising IV during an uptrend, as the demand for protection (and the panic to cover) spikes.

---

## 2. Technical Thresholds

| Score | Status | Description |
| :--- | :--- | :--- |
| **> 75** | **🚀 SQUEEZE** | High probability of a rapid vertical move as writers capitulate. |
| **50 - 75** | **⚠️ PRESSURE** | Calls are challenged; monitor for an OI shed. |
| **< 50** | **✅ STABLE** | Resistance remains intact or market move is orderly. |

---

## 3. Implementation in Scanner

The Antigravity Score is integrated into the **Market Scanner** table and the **Institutional Metrics** dashboard. When a score exceeds 75, an emergency alert is triggered:

> 🚀 **ANTIGRAVITY ALERT**: Short Covering Detected! Score: 85.0 | Wall: 25100.0 | Shed: 15400

This alert signifies that the prevailing uptrend is likely driven by "involuntary" buying (covering), which can lead to explosive price action.
