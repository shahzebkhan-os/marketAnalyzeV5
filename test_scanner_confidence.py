import asyncio
import pandas as pd
from client import GrowwClient
import structlog

# Mocking enough to run the logic
class MockDashboard:
    async def analyze_candidate(self, symbol):
        async with GrowwClient() as client:
            data = await client.fetch_option_chain(symbol)
            chain = data.get('option_chain', [])
            spot = data.get('spot_price', 0.0)
            if not chain: return None
            
            df = pd.DataFrame(chain)
            best = df.sort_values('oi', ascending=False).iloc[0].to_dict()
            
            max_oi = df['oi'].max()
            oi_score = (best['oi'] / max_oi) if max_oi > 0 else 0
            dist_pct = abs(best['strike'] - spot) / spot if spot > 0 else 1.0
            prox_score = max(0, 1 - (dist_pct / 0.03)) 
            confidence = (oi_score * 0.6 + prox_score * 0.4) * 100
            
            best['confidence_score'] = round(confidence, 1)
            if confidence > 80: best['action'] = "🔥 STRONG BUY"
            elif confidence > 60: best['action'] = "✅ BUY"
            else: best['action'] = "👀 WATCH"
            return best

async def main():
    db = MockDashboard()
    symbol = "NIFTY"
    print(f"Testing confidence scoring for {symbol}...")
    result = await db.analyze_candidate(symbol)
    if result:
        print(f"Strike: {result['strike']} {result['type']}")
        print(f"Confidence: {result['confidence_score']}%")
        print(f"Action: {result['action']}")
    else:
        print("No result found.")

if __name__ == "__main__":
    asyncio.run(main())
