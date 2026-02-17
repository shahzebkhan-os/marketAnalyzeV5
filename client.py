import aiohttp
import logging
import pandas as pd
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import asyncio
try:
    from growwapi import GrowwAPI
except ImportError:
    GrowwAPI = None # Handle missing dependency gracefully for linting/loading until installed

# Configure standard logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_result
from settings import settings


# Module-level cache for instrument metadata (persists across Streamlit logic re-runs)
_INSTRUMENTS_CACHE = None
_LAST_INST_FETCH = 0.0

class GrowwClient:
    def __init__(self):
        self.base_url = "https://groww.in/v1/api" # Placeholder base URL
        self.session = None
        self.token = None
        self.groww_sdk = None
        self.web_provider = None

    async def __aenter__(self):
        # Always create a new session for the current loop's lifecycle
        self.session = aiohttp.ClientSession()
        
        # Only authenticate if we don't have a token/SDK yet
        if not self.token or not self.groww_sdk:
            try:
                await self.authenticate()
            except Exception as e:
                logger.warning(f"Main authentication failed, will rely on Web Fallback: {e}")
                self.groww_sdk = None
        
        # Always create/update web_provider with the current session
        self.web_provider = GrowwWebProvider(self.session)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            try:
                await self.session.close()
            except:
                pass
        self.session = None

    @retry(
        retry=retry_if_exception_type(aiohttp.ClientError),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5)
    )
    async def authenticate(self):
        """
        Authenticates with the Groww API.
        Attempts to exchange GROWW_API_KEY and GROWW_API_SECRET for a session token
        if both are provided. Fallbacks to using the key as a direct token.
        """
        try:
            logger.info("Authenticating with Groww API...")
            
            api_key = settings.GROWW_API_KEY
            secret = settings.GROWW_API_SECRET
            
            if api_key and secret:
                # Use key and secret to get a fresh session token
                # GrowwAPI(any_string) can be used to call get_access_token
                temp_sdk = GrowwAPI(api_key)
                # Run sync call in thread
                import asyncio
                self.token = await asyncio.to_thread(
                    temp_sdk.get_access_token, 
                    api_key=api_key, 
                    secret=secret
                )
                logger.info("Authentication successful via Key/Secret exchange")
            elif api_key and len(api_key) > 50:
                # Direct token
                self.token = api_key
                logger.info("Authentication using provided direct token")
            else:
                self.token = "mock_token"
                logger.warning("Using mock token (Incomplete credentials in .env)")

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    async def fetch_option_chain(self, symbol: str) -> dict:
        """
        Fetches the option chain for a given symbol.
        Uses a robust reconstruction strategy to bypass account restrictions on bulk endpoints.
        """
        # If SDK is missing or not authenticated, immediately try Web Fallback
        if not self.groww_sdk:
             logger.info(f"SDK not initialized, attempting Web Fallback for {symbol}...")
             if not self.web_provider:
                 self.web_provider = GrowwWebProvider(self.session)
             return await self.web_provider.fetch_option_chain(symbol)

        try:
             # Run sync reconstruction in thread
             import asyncio
             return await asyncio.to_thread(self._reconstruct_option_chain, symbol)
        except Exception as e:
            err_msg = str(e).lower()
            logger.warning(f"Groww SDK fetch failed: {err_msg}")
            if "rate limit" in err_msg or "429" in err_msg or "forbidden" in err_msg:
                logger.warning(f"F&O API issues detected. Switching to Web Fallback for {symbol}... Error: {e}")
                if not self.web_provider:
                    self.web_provider = GrowwWebProvider(self.session)
                return await self.web_provider.fetch_option_chain(symbol)
            
            logger.error(f"Failed to fetch option chain for {symbol}: {e}")
            raise

    def _reconstruct_option_chain(self, symbol: str) -> dict:
        """
        Manually builds the option chain from metadata and individual quote/ltp calls.
        Bypasses '403 Forbidden' on get_option_chain for unactivated accounts.
        """
        try:
            exchange = "NSE"
            
            # 1. Fetch Spot Price (CASH quote usually allowed)
            spot_quote = self.groww_sdk.get_quote(trading_symbol=symbol, exchange=exchange, segment="CASH")
            spot_price = spot_quote.get("last_price", 0.0)
            if not spot_price:
                logger.warning(f"Could not fetch spot price for {symbol}, using metadata spot")

            # 2. Get Instruments and filter for symbol + FNO
            df = self.get_instruments()
            fno_df = df[(df['underlying_symbol'] == symbol) & (df['segment'] == 'FNO')].copy()
            if fno_df.empty:
                logger.error(f"No FNO instruments found for symbol {symbol}")
                return {"spot_price": spot_price, "option_chain": []}

            # 3. Find Nearest Expiry
            fno_df['expiry_date'] = fno_df['expiry_date'].astype(str)
            expiries = sorted(fno_df['expiry_date'].dropna().unique())
            if not expiries:
                return {"spot_price": spot_price, "option_chain": []}
            nearest_expiry = expiries[0]

            # 4. Filter for nearest expiry and extract strike info
            target_df = fno_df[fno_df['expiry_date'] == nearest_expiry].copy()
            target_df['strike_price'] = pd.to_numeric(target_df['strike_price'], errors='coerce')
            
            # Find strikes around spot (Window: +/- 10 strikes)
            unique_strikes = list(sorted(target_df['strike_price'].unique()))
            atm_strike = float(min(unique_strikes, key=lambda x: abs(float(x) - spot_price)))
            idx = unique_strikes.index(atm_strike)
            
            start_idx = int(max(0, idx - 10))
            end_idx = int(min(len(unique_strikes), idx + 11))
            window_strikes = unique_strikes[start_idx:end_idx]
            
            # 5. Batch Fetch LTPs for these strikes
            symbols_to_fetch = []
            strike_map = {} # strike -> {'CE': sym, 'PE': sym}
            for strike in window_strikes:
                ce_rows = target_df[(target_df['strike_price'] == strike) & (target_df['instrument_type'] == 'CE')]
                pe_rows = target_df[(target_df['strike_price'] == strike) & (target_df['instrument_type'] == 'PE')]
                if not ce_rows.empty and not pe_rows.empty:
                    ce_sym = ce_rows.iloc[0]['trading_symbol']
                    pe_sym = pe_rows.iloc[0]['trading_symbol']
                    symbols_to_fetch.extend([f"NSE_{ce_sym}", f"NSE_{pe_sym}"])
                    strike_map[strike] = {'CE': ce_sym, 'PE': pe_sym}

            if not symbols_to_fetch:
                logger.warning(f"No FNO symbols found for window for {symbol}")
                return {"spot_price": spot_price, "option_chain": []}

            # Bulk request LTPs (more efficient, avoids rate limits)
            ltp_results = self.groww_sdk.get_ltp(exchange_trading_symbols=tuple(symbols_to_fetch), segment="FNO")
            
            # 6. Build the flattened chain
            flat_chain = []
            local_idx = window_strikes.index(atm_strike)
            for strike in window_strikes:
                if strike not in strike_map: continue
                
                ce_sym = strike_map[strike]['CE']
                pe_sym = strike_map[strike]['PE']
                
                ce_ltp = ltp_results.get(f"NSE_{ce_sym}", 0.0)
                pe_ltp = ltp_results.get(f"NSE_{pe_sym}", 0.0)
                
                # Fetch detailed quote (including OI) only for strikes near ATM to avoid rate limits
                ce_oi, pe_oi = 0, 0
                if strike in window_strikes[max(0, local_idx-3):min(len(window_strikes), local_idx+4)]: # +/- 3 strikes from ATM
                    try:
                        import time
                        # Minimal sleep to avoid triggering rate limit during sequential calls
                        time.sleep(0.1) 
                        ce_q = self.groww_sdk.get_quote(trading_symbol=ce_sym, exchange=exchange, segment="FNO")
                        pe_q = self.groww_sdk.get_quote(trading_symbol=pe_sym, exchange=exchange, segment="FNO")
                        ce_oi = ce_q.get('open_interest', 0)
                        pe_oi = pe_q.get('open_interest', 0)
                    except:
                        pass # Silently fail back to 0 OI if throttled

                # Add CE
                flat_chain.append({
                    "strike": strike, "type": "CE", "ltp": ce_ltp, "oi": ce_oi,
                    "expiry": nearest_expiry, "call_oi": ce_oi, "put_oi": pe_oi
                })
                # Add PE
                flat_chain.append({
                    "strike": strike, "type": "PE", "ltp": pe_ltp, "oi": pe_oi,
                    "expiry": nearest_expiry, "call_oi": ce_oi, "put_oi": pe_oi
                })

            return {
                "spot_price": spot_price,
                "option_chain": flat_chain
            }

        except Exception as e:
            logger.error(f"Reconstruction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise e

    async def fetch_market_scan(self) -> List[Dict[str, Any]]:
        """
        Async wrapper for market-wide F&O scan.
        """
        # Ensure we have a token before creating the SDK
        if not self.token:
             try:
                 await self.authenticate()
             except Exception as e:
                 logger.warning(f"Scan auth failed: {e}")
        
        # Absolute safety: Ensure token is at least a placeholder string
        token_to_use = str(self.token) if self.token else "no_token_available"

        if not self.groww_sdk:
             logger.info(f"Initializing SDK for market scan, token_exists={bool(self.token)}")
             self.groww_sdk = GrowwAPI(token_to_use)
        
        try:
             import asyncio
             token_preview = token_to_use[:5]
             logger.info(f"Using existing token: {token_preview}...")
             return await asyncio.to_thread(self.scan_fno_market)
        except Exception as e:
            logger.error(f"Main auth error: {e}")
            raise

    def get_instruments(self):
        """
        Helper to get instruments with simple caching. Uses module-level global cache.
        """
        import time
        global _INSTRUMENTS_CACHE, _LAST_INST_FETCH
        
        now = int(time.time())
        # Cache for 1 hour
        if _INSTRUMENTS_CACHE is not None and _LAST_INST_FETCH and (now - int(_LAST_INST_FETCH) < 3600):
            return _INSTRUMENTS_CACHE
            
        logger.info("Fetching full instrument list (may take seconds)...")
        _INSTRUMENTS_CACHE = self.groww_sdk.get_all_instruments()
        _LAST_INST_FETCH = float(now)
        return _INSTRUMENTS_CACHE

    def scan_fno_market(self) -> List[Dict[str, Any]]:
        """
        Scans all F&O underlyings and returns their spot prices.
        """
        try:
            if not self.groww_sdk:
                logger.error("SDK not initialized in scan_fno_market")
                return []
                
            df = self.get_instruments()
            if df is None or df.empty:
                logger.warning("No instruments returned for scan")
                return []

            fno_df = df[df['segment'] == 'FNO'].copy()
            
            # Get unique underlyings, filtering out TEST symbols
            all_underlyings = fno_df['underlying_symbol'].dropna().unique()
            underlyings = sorted([str(s) for s in all_underlyings if "TEST" not in str(s).upper()])
            logger.info(f"Found {len(underlyings)} real F&O underlyings")
            
            # Batch fetch LTP for underlyings from CASH segment
            symbols_to_fetch = [f"NSE_{str(s)}" for s in underlyings if s is not None]
            
            all_ltps = {}
            chunk_size = 50
            for i in range(0, len(symbols_to_fetch), chunk_size):
                chunk = symbols_to_fetch[i:i+chunk_size]
                logger.info(f"Fetching LTP chunk {i//chunk_size + 1}...")
                try:
                    res = self.groww_sdk.get_ltp(exchange_trading_symbols=tuple(chunk), segment="CASH")
                    if res:
                        all_ltps.update(res)
                except Exception as ex:
                    logger.warning(f"LTP fetch failed for chunk: {ex}")
            
            results = []
            for s in underlyings:
                if s is None: continue
                ltp = all_ltps.get(f"NSE_{str(s)}", 0.0)
                if ltp and float(ltp) > 0:
                    results.append({"symbol": str(s), "spot": float(ltp)})
            
            return results
        except Exception as e:
            import traceback
            logger.error(f"Market scan fatal error: {traceback.format_exc()}")
            raise e

class GrowwWebProvider:
    """
    Fallback provider that fetches data directly from groww.in web interface.
    """
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "X-APP-ID": "growwWeb",
            "X-PLATFORM": "web"
        }

    async def fetch_option_chain(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches option chain by scraping the Groww web page and extracting __NEXT_DATA__.
        """
        # Symbol Mapping for Groww URLs
        mapping = {
            "NIFTY": "nifty",
            "BANKNIFTY": "nifty-bank",
            "FINNIFTY": "nifty-financial-services"
        }
        slug = mapping.get(symbol.upper(), str(symbol).lower())
        url = f"https://groww.in/options/{slug}"
        logger.info(f"Fetching via Web Fallback: {url}")
        
        async with self.session.get(url, headers=self.headers) as resp:
            if resp.status != 200:
                raise Exception(f"Web provider failed with status {resp.status}")
            html = await resp.text()

        soup = BeautifulSoup(html, 'html.parser')
        script_tag = soup.find('script', id='__NEXT_DATA__')
        if not script_tag:
            raise Exception("Could not find __NEXT_DATA__ in web page")

        data = json.loads(script_tag.string)
        
        # Navigate the JSON structure based on research
        try:
            page_props_data = data['props']['pageProps']['data']
            oc_data = page_props_data['optionChain']
            contracts = oc_data['optionContracts']
            
            # Find spot price
            spot_price = oc_data.get('underlyingValue') or page_props_data.get('underlyingValue', 0.0)
            if spot_price == 0.0 and 'company' in page_props_data:
                clive = page_props_data['company'].get('liveData', {})
                spot_price = clive.get('ltp') or clive.get('lastPrice', 0.0)
            
            # Flatten the chain to match the internal standardized format
            flattened_chain = []
            for c in contracts:
                strike_price = c.get('strikePrice', 0) / 100.0
                ce = c.get('ce', {})
                pe = c.get('pe', {})
                
                ce_ltp = ce.get('liveData', {}).get('ltp', 0.0)
                ce_oi = ce.get('liveData', {}).get('oi', 0)
                pe_ltp = pe.get('liveData', {}).get('ltp', 0.0)
                pe_oi = pe.get('liveData', {}).get('oi', 0)

                # Add CE row
                flattened_chain.append({
                    "strike": strike_price,
                    "type": "CE",
                    "ltp": ce_ltp,
                    "oi": ce_oi,
                    "call_oi": ce_oi,
                    "put_oi": pe_oi,
                    "source": "web_fallback"
                })
                # Add PE row
                flattened_chain.append({
                    "strike": strike_price,
                    "type": "PE",
                    "ltp": pe_ltp,
                    "oi": pe_oi,
                    "call_oi": ce_oi,
                    "put_oi": pe_oi,
                    "source": "web_fallback"
                })
            
            return {
                "spot_price": spot_price,
                "option_chain": flattened_chain,
                "source": "web_fallback"
            }
        except Exception as e:
            logger.error(f"Web Fallback failed for {symbol}: {e}")
            raise

    async def fetch_batch_prices(self, contract_ids: List[str]) -> Dict[str, float]:
        """
        Fetches latest prices for a batch of contract IDs via the internal web API.
        """
        try:
            url = "https://groww.in/v1/api/stocks_fo_data/v1/tr_live_prices/exchange/NSE/segment/FNO/latest_prices_batch"
            payload = contract_ids
            
            async with self.session.post(url, json=payload, headers=self.headers) as resp:
                if resp.status != 200:
                    logger.warning(f"Web price provider failed with status {resp.status}")
                    return {}
                
                data = await resp.json()
                results = {}
                for item in data:
                    cid = item.get('growwContractId')
                    ltp = item.get('ltp', 0.0)
                    if cid:
                        results[cid] = ltp
                return results
        except Exception as e:
            logger.error(f"Web batch price fetch failed: {e}")
            return {}
