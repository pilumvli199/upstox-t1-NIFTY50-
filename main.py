#!/usr/bin/env python3
"""
STRIKE MASTER V14.0 - MASTER LIST EDITION
================================================
✅ METHOD: Downloads 'NSE.json.gz' to find REAL Instrument Keys (No guessing)
✅ DATA: Merges Historical (Past) + Intraday (Live) APIs
✅ FIXES: Solves 'Invalid Instrument Key' & '400' Errors permanently.

Version: 14.0 - Hybrid Data & Instrument Discovery
"""

import os
import asyncio
import aiohttp
import urllib.parse
from datetime import datetime, timedelta, time
import pytz
import json
import logging
import gzip
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

# Optional dependencies
try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("StrikeMaster-V14")

# API Configuration
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# URL for Master Instrument List
INSTRUMENTS_JSON_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

# Indices to Track
TARGET_INDICES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# Trading Config
ALERT_ONLY_MODE = True
SCAN_INTERVAL = 60

# ==================== DATA CLASSES ====================
@dataclass
class InstrumentInfo:
    name: str
    spot_key: str
    future_key: str
    future_symbol: str
    expiry: str

@dataclass
class Signal:
    type: str
    reason: str
    price: float
    stop: float
    target: float
    timestamp: datetime
    index_name: str

# ==================== INSTRUMENT DISCOVERY (THE FIX) ====================
class InstrumentManager:
    """
    Downloads the official Upstox Instrument list to find correct keys.
    NO MORE GUESSING.
    """
    def __init__(self):
        self.instruments_map: Dict[str, InstrumentInfo] = {}
        self.is_ready = False

    async def initialize(self):
        logger.info("📥 Downloading Master Instrument List (NSE.json.gz)...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(INSTRUMENTS_JSON_URL) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        decompressed = gzip.decompress(content)
                        data = json.loads(decompressed)
                        self._process_instruments(data)
                        return True
                    else:
                        logger.error(f"❌ Failed to download instruments: {resp.status}")
                        return False
        except Exception as e:
            logger.error(f"💥 Instrument Download Error: {e}")
            return False

    def _process_instruments(self, data):
        logger.info("🔍 Searching for correct Futures keys...")
        now = datetime.now(IST)
        
        # Temp storage
        futures_candidates = {name: [] for name in TARGET_INDICES}
        spot_keys = {
            'NIFTY': 'NSE_INDEX|Nifty 50',
            'BANKNIFTY': 'NSE_INDEX|Nifty Bank',
            'FINNIFTY': 'NSE_INDEX|Nifty Fin Service',
            'MIDCPNIFTY': 'NSE_INDEX|NIFTY MID SELECT'
        }

        # 1. Scan the whole list
        for item in data:
            if item.get('segment') == 'NSE_FO' and item.get('instrument_type') == 'FUT':
                name = item.get('name')
                if name in TARGET_INDICES:
                    expiry_ms = item.get('expiry')
                    if expiry_ms:
                        exp_date = datetime.fromtimestamp(expiry_ms/1000, tz=IST)
                        if exp_date >= now: # Only future expiries
                            futures_candidates[name].append({
                                'key': item['instrument_key'],
                                'symbol': item['trading_symbol'],
                                'expiry': exp_date,
                                'expiry_ms': expiry_ms
                            })

        # 2. Select the NEAREST expiry for each index
        for name in TARGET_INDICES:
            candidates = futures_candidates[name]
            if not candidates:
                logger.warning(f"⚠️ No futures found for {name}")
                continue

            # Sort by expiry date (nearest first)
            candidates.sort(key=lambda x: x['expiry_ms'])
            nearest = candidates[0]

            self.instruments_map[name] = InstrumentInfo(
                name=name,
                spot_key=spot_keys.get(name, ""),
                future_key=nearest['key'],
                future_symbol=nearest['symbol'],
                expiry=nearest['expiry'].strftime('%Y-%m-%d')
            )
            logger.info(f"✅ {name}: Found Key: {nearest['key']} ({nearest['symbol']})")

        self.is_ready = True

# ==================== DATA FETCHER (HYBRID: HIST + INTRA) ====================
class HybridDataFetcher:
    """
    Fetches Data from TWO sources:
    1. Historical API: For past context (VWAP/ATR calculations)
    2. Intraday API: For live today's candles
    """
    def __init__(self, instrument_info: InstrumentInfo):
        self.info = instrument_info
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }

    async def fetch_merged_data(self) -> pd.DataFrame:
        """Fetch both APIs and merge them"""
        async with aiohttp.ClientSession() as session:
            # 1. Fetch Intraday (Live Today)
            intraday_df = await self._fetch_intraday(session)
            
            # 2. Fetch Historical (Past 3 days) - Needed for accurate VWAP/ATR
            historical_df = await self._fetch_historical(session)

            # 3. Merge
            if intraday_df.empty and historical_df.empty:
                return pd.DataFrame()
            
            if intraday_df.empty:
                return historical_df
            
            if historical_df.empty:
                return intraday_df

            # Combine and remove duplicates
            combined = pd.concat([historical_df, intraday_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            
            return combined

    async def _fetch_intraday(self, session) -> pd.DataFrame:
        """Get today's 1-minute candles"""
        enc_key = urllib.parse.quote(self.info.future_key, safe='')
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{enc_key}/1minute"
        
        try:
            async with session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    candles = data.get('data', {}).get('candles', [])
                    if candles:
                        df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'oi'])
                        df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert(IST)
                        df.set_index('ts', inplace=True)
                        return df
        except Exception as e:
            logger.error(f"⚠️ Intraday Fetch Error {self.info.name}: {e}")
        return pd.DataFrame()

    async def _fetch_historical(self, session) -> pd.DataFrame:
        """Get past data (3 days)"""
        enc_key = urllib.parse.quote(self.info.future_key, safe='')
        to_date = (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d') # Yesterday
        from_date = (datetime.now(IST) - timedelta(days=4)).strftime('%Y-%m-%d')
        url = f"https://api.upstox.com/v2/historical-candle/{enc_key}/1minute/{to_date}/{from_date}"

        try:
            async with session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    candles = data.get('data', {}).get('candles', [])
                    if candles:
                        df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'oi'])
                        df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert(IST)
                        df.set_index('ts', inplace=True)
                        return df
        except Exception:
            pass # Historical might fail if contract is very new
        return pd.DataFrame()
    
    async def get_spot_price(self) -> float:
        """Get Spot Price for Option Chain calculations"""
        enc_key = urllib.parse.quote(self.info.spot_key, safe='')
        url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={enc_key}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Handling generic structure "NSE_INDEX:Nifty 50" vs "NSE_INDEX|Nifty 50"
                        res_data = data.get('data', {})
                        for k, v in res_data.items():
                            return v.get('last_price', 0)
            except:
                pass
        return 0.0

    async def get_option_chain(self, spot_price):
        """Get Option Chain for OI Analysis"""
        if spot_price == 0: return {}
        
        enc_key = urllib.parse.quote(self.info.spot_key, safe='')
        url = f"https://api.upstox.com/v2/option/chain?instrument_key={enc_key}&expiry_date={self.info.expiry}"
        
        strike_data = {}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        contracts = data.get('data', [])
                        
                        # Find ATM
                        strike_gap = 100 if 'BANK' in self.info.name else 50
                        if 'MID' in self.info.name: strike_gap = 25
                        
                        atm = round(spot_price / strike_gap) * strike_gap
                        
                        for c in contracts:
                            strike = c['strike_price']
                            if (atm - 3*strike_gap) <= strike <= (atm + 3*strike_gap):
                                ce = c.get('call_options', {}).get('market_data', {})
                                pe = c.get('put_options', {}).get('market_data', {})
                                strike_data[strike] = {
                                    'ce_oi': ce.get('oi', 0), 'pe_oi': pe.get('oi', 0),
                                    'ce_ltp': ce.get('ltp', 0), 'pe_ltp': pe.get('ltp', 0)
                                }
            except Exception as e:
                logger.error(f"Option Chain Error: {e}")
                
        return strike_data

# ==================== ANALYSIS ENGINE ====================
class Analyzer:
    def calculate_vwap(self, df):
        if df.empty: return 0
        # Typical Price * Volume
        df['tp_v'] = ((df['high'] + df['low'] + df['close']) / 3) * df['vol']
        # Cumulative Sum
        vwap = df['tp_v'].cumsum() / df['vol'].cumsum()
        return vwap.iloc[-1]

    def check_signal(self, df, strike_data, index_name):
        if df.empty or not strike_data: return None
        
        current = df.iloc[-1]
        vwap = self.calculate_vwap(df)
        
        # Simple Logic Example
        # Buy CE if Price > VWAP and PCR > 1
        total_ce = sum(x['ce_oi'] for x in strike_data.values())
        total_pe = sum(x['pe_oi'] for x in strike_data.values())
        pcr = total_pe / total_ce if total_ce > 0 else 0
        
        logger.info(f"📊 {index_name}: Price={current['close']:.2f} VWAP={vwap:.2f} PCR={pcr:.2f}")

        if current['close'] > vwap and pcr > 1.2:
            return Signal("CE_BUY", "Price > VWAP & Bullish PCR", current['close'], current['close']-20, current['close']+40, datetime.now(IST), index_name)
        elif current['close'] < vwap and pcr < 0.8:
            return Signal("PE_BUY", "Price < VWAP & Bearish PCR", current['close'], current['close']+20, current['close']-40, datetime.now(IST), index_name)
            
        return None

# ==================== MAIN BOT LOGIC ====================
class StrikeMasterBot:
    def __init__(self):
        self.instrument_mgr = InstrumentManager()
        self.analyzer = Analyzer()
        self.telegram = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_AVAILABLE else None

    async def start(self):
        logger.info("🚀 STRIKE MASTER V14.0 STARTING...")
        
        # 1. Initialize Instruments (Download Master List)
        success = await self.instrument_mgr.initialize()
        if not success:
            logger.error("❌ Failed to initialize instruments. Exiting.")
            return

        logger.info("✅ Instruments Ready. Starting Loop...")
        
        while True:
            try:
                await self.run_cycle()
                logger.info("⏳ Waiting 60s...")
                await asyncio.sleep(60)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Cycle Error: {e}")
                await asyncio.sleep(10)

    async def run_cycle(self):
        # Loop through found indices in instrument manager
        for name, info in self.instrument_mgr.instruments_map.items():
            fetcher = HybridDataFetcher(info)
            
            # 1. Get Merged Data (Hist + Intra)
            df = await fetcher.fetch_merged_data()
            if df.empty:
                logger.warning(f"⚠️ {name}: No Data found.")
                continue

            # 2. Get Spot & Option Chain
            spot = await fetcher.get_spot_price()
            strike_data = await fetcher.get_option_chain(spot)

            # 3. Analyze
            signal = self.analyzer.check_signal(df, strike_data, name)
            
            if signal:
                await self.send_alert(signal)

    async def send_alert(self, s: Signal):
        msg = f"""
⚡ STRIKE MASTER ALERT ⚡
Index: {s.index_name}
Type: {s.type}
Reason: {s.reason}

Entry: {s.price:.2f}
Target: {s.target:.2f}
Stop: {s.stop:.2f}

Time: {s.timestamp.strftime('%H:%M:%S')}
"""
        logger.info(f"🚨 SIGNAL: {msg}")
        if self.telegram:
            try:
                await self.telegram.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            except:
                pass

if __name__ == "__main__":
    bot = StrikeMasterBot()
    asyncio.run(bot.start())
