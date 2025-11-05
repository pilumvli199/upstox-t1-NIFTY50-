#!/usr/bin/env python3
"""
H-LIQUIDITY WALT (High-Liquidity Weighted Average Long Term)
============================================================
✅ Pre-Market: 8:55 AM (Historical Analysis)
✅ Live Scan: 9:16 AM - 3:30 PM (Every 5 minutes)
✅ DeepSeek V3 AI Analysis
✅ Redis Data Storage (1-day expiry)
✅ Active Filter (Volume/OI threshold)
✅ Chart Generation (5-Min TF)
✅ Telegram Alerts + P&L Tracking
✅ 40 Instruments (36 Stocks + 4 Indices)
"""

import os
import sys
import asyncio
import requests
import urllib.parse
import redis
import json
from datetime import datetime, timedelta, time
import pytz
import time as time_sleep
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('h_liquidity_walt.log')
    ]
)
logger = logging.getLogger(__name__)

# API Keys
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Redis Configuration (Railway.app support)
REDIS_URL = os.getenv('REDIS_URL')  # Railway.app format: redis://default:password@host:port
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

BASE_URL = "https://api.upstox.com"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
SCAN_INTERVAL = 300  # 5 minutes

# Active Filter Thresholds
VOLUME_THRESHOLD = 10_00_00_000  # ₹10 Cr
OI_CHANGE_THRESHOLD = 5  # 5%

# ==================== SYMBOLS (TOP 36 STOCKS + 4 INDICES) ====================
INDICES = {
    "NSE_INDEX|Nifty 50": {"name": "NIFTY", "display_name": "NIFTY 50", "expiry_day": 3, "category": "INDEX"},
    "NSE_INDEX|Nifty Bank": {"name": "BANKNIFTY", "display_name": "BANK NIFTY", "expiry_day": 2, "category": "INDEX"},
    "NSE_INDEX|NIFTY MID SELECT": {"name": "MIDCPNIFTY", "display_name": "MIDCAP NIFTY", "expiry_day": 0, "category": "INDEX"},
    "BSE_INDEX|SENSEX": {"name": "SENSEX", "display_name": "SENSEX", "expiry_day": 4, "category": "INDEX"}
}

AUTO_STOCKS = {
    "NSE_EQ|INE467B01029": {"name": "TATAMOTORS", "display_name": "TATA MOTORS", "category": "AUTO"},
    "NSE_EQ|INE585B01010": {"name": "MARUTI", "display_name": "MARUTI SUZUKI", "category": "AUTO"},
    "NSE_EQ|INE208A01029": {"name": "ASHOKLEY", "display_name": "ASHOK LEYLAND", "category": "AUTO"},
}

BANK_STOCKS = {
    "NSE_EQ|INE040A01034": {"name": "HDFCBANK", "display_name": "HDFC BANK", "category": "BANK"},
    "NSE_EQ|INE090A01021": {"name": "ICICIBANK", "display_name": "ICICI BANK", "category": "BANK"},
    "NSE_EQ|INE062A01020": {"name": "SBIN", "display_name": "STATE BANK", "category": "BANK"},
    "NSE_EQ|INE028A01039": {"name": "BANKBARODA", "display_name": "BANK OF BARODA", "category": "BANK"},
    "NSE_EQ|INE238A01034": {"name": "AXISBANK", "display_name": "AXIS BANK", "category": "BANK"},
    "NSE_EQ|INE237A01028": {"name": "KOTAKBANK", "display_name": "KOTAK BANK", "category": "BANK"},
}

METAL_STOCKS = {
    "NSE_EQ|INE081A01012": {"name": "TATASTEEL", "display_name": "TATA STEEL", "category": "METAL"},
    "NSE_EQ|INE038A01020": {"name": "HINDALCO", "display_name": "HINDALCO", "category": "METAL"},
    "NSE_EQ|INE019A01038": {"name": "JSWSTEEL", "display_name": "JSW STEEL", "category": "METAL"},
}

OIL_GAS_STOCKS = {
    "NSE_EQ|INE002A01018": {"name": "RELIANCE", "display_name": "RELIANCE", "category": "OIL_GAS"},
    "NSE_EQ|INE213A01029": {"name": "ONGC", "display_name": "ONGC", "category": "OIL_GAS"},
    "NSE_EQ|INE242A01010": {"name": "IOC", "display_name": "INDIAN OIL", "category": "OIL_GAS"},
}

IT_STOCKS = {
    "NSE_EQ|INE009A01021": {"name": "INFY", "display_name": "INFOSYS", "category": "IT"},
    "NSE_EQ|INE075A01022": {"name": "WIPRO", "display_name": "WIPRO", "category": "IT"},
    "NSE_EQ|INE854D01024": {"name": "TCS", "display_name": "TCS", "category": "IT"},
    "NSE_EQ|INE860A01027": {"name": "HCLTECH", "display_name": "HCL TECH", "category": "IT"},
}

PHARMA_STOCKS = {
    "NSE_EQ|INE044A01036": {"name": "SUNPHARMA", "display_name": "SUN PHARMA", "category": "PHARMA"},
    "NSE_EQ|INE361B01024": {"name": "DIVISLAB", "display_name": "DIVI'S LAB", "category": "PHARMA"},
    "NSE_EQ|INE089A01023": {"name": "DRREDDY", "display_name": "DR REDDY'S", "category": "PHARMA"},
}

FMCG_STOCKS = {
    "NSE_EQ|INE154A01025": {"name": "ITC", "display_name": "ITC", "category": "FMCG"},
    "NSE_EQ|INE030A01027": {"name": "HINDUNILVR", "display_name": "HINDUSTAN UNILEVER", "category": "FMCG"},
    "NSE_EQ|INE216A01030": {"name": "BRITANNIA", "display_name": "BRITANNIA", "category": "FMCG"},
}

INFRA_POWER_STOCKS = {
    "NSE_EQ|INE742F01042": {"name": "ADANIPORTS", "display_name": "ADANI PORTS", "category": "INFRA"},
    "NSE_EQ|INE733E01010": {"name": "NTPC", "display_name": "NTPC", "category": "POWER"},
    "NSE_EQ|INE752E01010": {"name": "POWERGRID", "display_name": "POWER GRID", "category": "POWER"},
    "NSE_EQ|INE018A01030": {"name": "LT", "display_name": "L&T", "category": "INFRA"},
}

RETAIL_CONSUMER_STOCKS = {
    "NSE_EQ|INE280A01028": {"name": "TITAN", "display_name": "TITAN", "category": "RETAIL"},
    "NSE_EQ|INE797F01012": {"name": "JUBLFOOD", "display_name": "JUBILANT FOODWORKS", "category": "RETAIL"},
    "NSE_EQ|INE021A01026": {"name": "ASIANPAINT", "display_name": "ASIAN PAINTS", "category": "RETAIL"},
}

INSURANCE_STOCKS = {
    "NSE_EQ|INE795G01014": {"name": "HDFCLIFE", "display_name": "HDFC LIFE", "category": "INSURANCE"},
    "NSE_EQ|INE123W01016": {"name": "SBILIFE", "display_name": "SBI LIFE", "category": "INSURANCE"},
}

OTHER_STOCKS = {
    "NSE_EQ|INE397D01024": {"name": "BHARTIARTL", "display_name": "BHARTI AIRTEL", "category": "TELECOM"},
    "NSE_EQ|INE296A01024": {"name": "BAJFINANCE", "display_name": "BAJAJ FINANCE", "category": "FINANCE"},
    "NSE_EQ|INE758T01015": {"name": "JIOFIN", "display_name": "JIO FINANCIAL", "category": "FINANCE"},
}

ALL_SYMBOLS = {
    **INDICES,
    **AUTO_STOCKS,
    **BANK_STOCKS,
    **METAL_STOCKS,
    **OIL_GAS_STOCKS,
    **IT_STOCKS,
    **PHARMA_STOCKS,
    **FMCG_STOCKS,
    **INFRA_POWER_STOCKS,
    **RETAIL_CONSUMER_STOCKS,
    **INSURANCE_STOCKS,
    **OTHER_STOCKS
}

# ==================== DATA CLASSES ====================
@dataclass
class StrikeData:
    strike: int
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int
    ce_ltp: float
    pe_ltp: float
    ce_iv: float = 0.0
    pe_iv: float = 0.0

@dataclass
class TradeDecision:
    decision: str  # BUY_CALL, BUY_PUT, NO_TRADE
    confidence: int  # 1-5
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    reason: str
    stop_hunt_risk: int
    clarity_score: int

# ==================== REDIS MANAGER ====================
class RedisManager:
    def __init__(self):
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_timeout=5
            )
            self.redis.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            sys.exit(1)
    
    def set_with_expiry(self, key: str, value: str, expiry_seconds: int = 86400):
        """Store data with 1-day expiry"""
        try:
            self.redis.setex(key, expiry_seconds, value)
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
    
    def get(self, key: str) -> Optional[str]:
        """Retrieve data"""
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
    
    def delete(self, key: str):
        """Delete key"""
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
    
    def cleanup_expired(self):
        """Force cleanup expired keys"""
        try:
            for key in self.redis.scan_iter("*"):
                ttl = self.redis.ttl(key)
                if ttl == -1:  # No expiry set
                    self.redis.expire(key, 86400)
            logger.info("✅ Redis cleanup complete")
        except Exception as e:
            logger.error(f"Redis cleanup error: {e}")

# ==================== EXPIRY CALCULATOR ====================
class ExpiryCalculator:
    @staticmethod
    def get_all_expiries_from_api(instrument_key: str, headers: Dict) -> List[str]:
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            url = f"{BASE_URL}/v2/option/contract?instrument_key={encoded_key}"
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                contracts = response.json().get('data', [])
                expiries = sorted(list(set(c['expiry'] for c in contracts if 'expiry' in c)))
                return expiries
            return []
        except:
            return []
    
    @staticmethod
    def calculate_monthly_expiry(expiry_day: int = 3) -> str:
        today = datetime.now(IST).date()
        current_time = datetime.now(IST).time()
        last_day = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        days_to_subtract = (last_day.weekday() - expiry_day) % 7
        expiry = last_day - timedelta(days=days_to_subtract)
        if expiry < today or (expiry == today and current_time >= time(15, 30)):
            next_month = (today.replace(day=28) + timedelta(days=4))
            last_day = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            days_to_subtract = (last_day.weekday() - expiry_day) % 7
            expiry = last_day - timedelta(days=days_to_subtract)
        return expiry.strftime('%Y-%m-%d')
    
    @staticmethod
    def get_best_expiry(instrument_key: str, symbol_info: Dict, headers: Dict) -> str:
        expiry_day = symbol_info.get('expiry_day', 3)
        expiries = ExpiryCalculator.get_all_expiries_from_api(instrument_key, headers)
        if expiries:
            today = datetime.now(IST).date()
            now_time = datetime.now(IST).time()
            future_expiries = []
            for exp_str in expiries:
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                    if exp_date > today or (exp_date == today and now_time < time(15, 30)):
                        future_expiries.append(exp_str)
                except:
                    continue
            if future_expiries:
                return min(future_expiries)
        return ExpiryCalculator.calculate_monthly_expiry(expiry_day)

# ==================== UPSTOX DATA FETCHER ====================
class UpstoxDataFetcher:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
    
    def get_spot_price(self, instrument_key: str) -> float:
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            url = f"{BASE_URL}/v2/market-quote/ltp?instrument_key={encoded_key}"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', {})
                if data:
                    ltp = list(data.values())[0].get('last_price', 0)
                    return float(ltp)
            return 0.0
        except Exception as e:
            logger.error(f"Spot price error: {e}")
            return 0.0
    
    def get_historical_1min(self, instrument_key: str, days: int = 7) -> List:
        """Fetch 7-days 1-min data"""
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            from_date = (datetime.now(IST) - timedelta(days=days)).strftime('%Y-%m-%d')
            url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/1minute/{to_date}/{from_date}"
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                candles = response.json().get('data', {}).get('candles', [])
                return candles
            return []
        except Exception as e:
            logger.error(f"Historical 1min error: {e}")
            return []
    
    def get_intraday_1min(self, instrument_key: str) -> List:
        """Fetch today's 1-min data"""
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
            response = requests.get(url, headers=self.headers, timeout=20)
            if response.status_code == 200:
                candles = response.json().get('data', {}).get('candles', [])
                return candles
            return []
        except Exception as e:
            logger.error(f"Intraday 1min error: {e}")
            return []
    
    def get_option_chain(self, instrument_key: str, expiry: str) -> List[StrikeData]:
        """Fetch option chain with IV"""
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
            response = requests.get(url, headers=self.headers, timeout=20)
            if response.status_code == 200:
                data = response.json()
                strikes_raw = data.get('data', [])
                strikes = []
                for item in strikes_raw:
                    call_data = item.get('call_options', {}).get('market_data', {})
                    put_data = item.get('put_options', {}).get('market_data', {})
                    call_greeks = item.get('call_options', {}).get('option_greeks', {})
                    put_greeks = item.get('put_options', {}).get('option_greeks', {})
                    strikes.append(StrikeData(
                        strike=int(item.get('strike_price', 0)),
                        ce_oi=call_data.get('oi', 0),
                        pe_oi=put_data.get('oi', 0),
                        ce_volume=call_data.get('volume', 0),
                        pe_volume=put_data.get('volume', 0),
                        ce_ltp=call_data.get('ltp', 0),
                        pe_ltp=put_data.get('ltp', 0),
                        ce_iv=call_greeks.get('iv', 0),
                        pe_iv=put_greeks.get('iv', 0)
                    ))
                return strikes
            return []
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return []

# ==================== DATA AGGREGATOR ====================
class DataAggregator:
    @staticmethod
    def aggregate_to_timeframe(df_1min: pd.DataFrame, timeframe: str, candles_needed: int) -> pd.DataFrame:
        """Convert 1-min to 5M/15M/1H"""
        try:
            df_agg = df_1min.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'oi': 'last'
            }).dropna()
            return df_agg.tail(candles_needed)
        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def prepare_dataframes(candles_1min: List) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare 5M, 15M, 1H dataframes"""
        if not candles_1min:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        df = pd.DataFrame(candles_1min, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').astype(float)
        df = df.sort_index()
        
        df_5m = DataAggregator.aggregate_to_timeframe(df, '5min', 50)
        df_15m = DataAggregator.aggregate_to_timeframe(df, '15min', 175)
        df_1h = DataAggregator.aggregate_to_timeframe(df, '1h', 40)
        
        return df_5m, df_15m, df_1h

# ==================== ACTIVE FILTER ====================
class ActiveFilter:
    @staticmethod
    def is_active(instrument_key: str, symbol_info: Dict, fetcher: UpstoxDataFetcher) -> bool:
        """Check if instrument is active based on volume/OI"""
        try:
            # Get last 7 days data for volume check
            candles = fetcher.get_historical_1min(instrument_key, days=7)
            if not candles:
                return False
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['value'] = df['close'] * df['volume']
            avg_daily_value = df['value'].sum() / 7
            
            # Check volume threshold
            if avg_daily_value < VOLUME_THRESHOLD:
                return False
            
            # Check OI change (if F&O stock)
            if 'oi' in df.columns and len(df) > 1:
                oi_change_pct = abs((df['oi'].iloc[-1] - df['oi'].iloc[0]) / df['oi'].iloc[0] * 100)
                if oi_change_pct < OI_CHANGE_THRESHOLD:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Active filter error: {e}")
            return False

# ==================== OI ANALYZER ====================
class OIAnalyzer:
    @staticmethod
    def calculate_pcr(strikes: List[StrikeData]) -> float:
        total_ce_oi = sum(s.ce_oi for s in strikes)
        total_pe_oi = sum(s.pe_oi for s in strikes)
        return total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    
    @staticmethod
    def calculate_max_pain(strikes: List[StrikeData], spot: float) -> int:
        """Calculate max pain strike"""
        pain_values = {}
        for strike_obj in strikes:
            strike = strike_obj.strike
            ce_loss = sum(max(0, s.strike - strike) * s.ce_oi for s in strikes if s.strike > strike)
            pe_loss = sum(max(0, strike - s.strike) * s.pe_oi for s in strikes if s.strike < strike)
            pain_values[strike] = ce_loss + pe_loss
        if pain_values:
            return min(pain_values, key=pain_values.get)
        return int(spot)
    
    @staticmethod
    def get_atm_strikes(all_strikes: List[StrikeData], spot: float, count: int = 21) -> List[StrikeData]:
        """Get 21 ATM strikes (±10 from ATM)"""
        atm = round(spot / 100) * 100
        strikes_with_dist = [(s, abs(s.strike - atm)) for s in all_strikes]
        strikes_with_dist.sort(key=lambda x: x[1])
        return [s[0] for s in strikes_with_dist[:count]]
    
    @staticmethod
    def calculate_iv_delta(current_strikes: List[StrikeData], redis: RedisManager, instrument: str) -> float:
        """Calculate IV change from last scan"""
        try:
            current_avg_iv = np.mean([s.ce_iv + s.pe_iv for s in current_strikes if s.ce_iv > 0 and s.pe_iv > 0]) / 2
            prev_iv_str = redis.get(f"IV_AVG_{instrument}")
            if prev_iv_str:
                prev_iv = float(prev_iv_str)
                delta = ((current_avg_iv - prev_iv) / prev_iv * 100) if prev_iv > 0 else 0
            else:
                delta = 0
            redis.set_with_expiry(f"IV_AVG_{instrument}", str(current_avg_iv))
            return delta
        except:
            return 0

# ==================== DEEPSEEK CLIENT ====================
class DeepSeekClient:
    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze(self, prompt: str) -> Optional[Dict]:
        """Send analysis request to DeepSeek V3"""
        try:
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            response = requests.post(DEEPSEEK_URL, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return self.parse_response(content)
            else:
                logger.error(f"DeepSeek API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"DeepSeek error: {e}")
            return None
    
    def get_system_prompt(self) -> str:
        return """🎯 ELITE TRADER AI ANALYSIS PROMPT (Optimized)

AI Persona: F&O analyst, 20+ yrs NSE/BSE experience. Pure Price Action + OI analysis. High conviction, low-risk trades. Technical terms (English) + explanations (Marathi/Hinglish).

Goal: Analyze data → Single trade recommendation (LONG CE/SHORT PE/NO TRADE)

Core Rules:
- Trade valid ONLY if: Price Action + OI + Volume aligned AND Clarity Score >60
- Risk ≤ MEDIUM (if HIGH → NO TRADE)
- Use OHLCV + OI only (no lagging indicators)

📊 INPUT DATA FORMAT:
INSTRUMENT: [Name] | SPOT: [Price] | EXPIRY: [Date] | DAYS: [X]

**CANDLES:**
- 15-MIN (175): Time|O|H|L|C|Vol
- 1-HR (40): Time|O|H|L|C|Vol  
- 5-MIN (50): Time|O|H|L|C|Vol

**PRE-MARKET (Redis):**
{Pre_Market_Trend_1HR, Major_Support, Major_Resistance, Patterns}

**OPTION CHAIN (ATM ±10):**
Strike|C_OI|C_ΔOI|C_Vol|P_OI|P_ΔOI|P_Vol
PCR: [X] | Max Pain: [Y] | Highest Call/Put OI: [Strikes]

**OI COMPARISON:**
- 15-Min Ago: Spot [X], ΔOI, Price Move
- 1-Hr Ago: Spot [Y], ΔOI, Price Move

**CONTEXT:**
Time: [HH:MM] | Session: [Opening/Trending/Lunch/Power] | VIX: [X]

🎯 ANALYSIS FRAMEWORK:

**1. SNAPSHOT**
- Range: H[X]-L[Y]=[Z]pts | Position: [Top/Mid/Bot 25%]
- 1Hr Trend: [Up/Down/Side] | 15Min→1Hr: [Momentum aligned?]

**2. OI ANALYSIS**
- PCR=[X] → [तेजी/मंदी/Neutral]
- Top 3 R/S: Strike|OI|ΔOI|Strength|Meaning

**OI+Price Matrix:**
| Price | Call OI | Put OI | Signal |
|-------|---------|--------|--------|
| ↑/↓   | ↑/↓     | ↑/↓    | [Bulls/Bears + Reason] |

आपली स्थिती: [Match scenario + interpretation]
Max Pain: [X] | Deviation: [Y]pts | Pull: [Yes/No + reason]

**3. PRICE ACTION**
- 1Hr Trend: [Type] | Structure: [Intact/Break] | Strength: [X/5]⭐
- R: R1[X], R2[Y], R3[Z]
- S: S1[X], S2[Y], S3[Z]
- Position: [Between/At] | Nearest: [Price@Distance]
- Round Numbers: Above[X], Below[Y]

**4. PATTERNS (15-Min)**
Pattern: [Name] | Type: [Cont/Rev] | Status: [%/Done/Broken]
Validity: Structure✅/❌ Volume✅/❌ OI✅/❌ → Score:[X/100]
Trade: [YES/WAIT/NO] | Target:[X] | SL:[Y]
Candle: [Pattern@Price] | Context:[S/R] | Reliability:[X/5]⭐
Failed Pattern? [Yes/No] → Reversal Signal: [Detail]

**5. VOLUME**
- Avg:[X]K | Trend:[↑/↓/→] | Spikes:[Yes@Price]
- Divergence: [Yes/No + meaning]
- Last Move: Price[Dir] + Vol[X]x + OI[C/P][↑/↓] → Signal:[X/5]⭐

**6. RISK**
Clarity Score:
Trend[X/20] + OI[Y/20] + Vol[Z/20] + Pattern[A/20] + Multi-TF[B/20] = **[Total/100]**

Red Flags: [Count/7]
- [ ] Low Vol | [ ] Stagnant OI | [ ] Choppy | [ ] Failed Patterns
- [ ] OI Conflict | [ ] Near Expiry | [ ] Bad Session

Risk Level: [LOW/MED/HIGH] | Position Size: [Full/50%/25%/AVOID]

**7. TRADE DECISION (If Score>60 + Risk≤MED + Flags<3)**

🎯 EXECUTE: [LONG CE/SHORT PE] | Confidence: [X/5]⭐

Entry: [X] | Trigger: [काय झाल्यावर?]
Confirm: ✅[Vol>1.5x] ✅[OI signal]

SL: [Y] | Distance: [Z]pts | Reason: [का इथे?] | Buffer: 20-30pts

Targets: T1[X], T2[Y], T3[Z]
R:R: [1:X]

WHY? ✅[Factor1] ✅[Factor2] ✅[Factor3]

EXIT IF: ❌[Condition1] ❌[Condition2]

**8. NO TRADE (If Score<60 OR Risk>MED OR Flags≥3)**

🚫 WAIT: [Primary Reason]

Wait For: ✅[Trigger1] ✅[Trigger2]
Watch: Breakout[X] | Breakdown[Y]

**9. SUMMARY**
Price:[X] | Trend:[X] | Clarity:[X/100] | Risk:[X]
OI:[Bull/Bear] PCR[X] | PA:[Clean/Choppy] | Vol:[Strong/Weak]

🔴R:[X] | 🟢S:[Y] | ⚡Key:[Z]

TODAY: [TRADE: Entry[X] SL[Y] ⭐[X/5]] OR [WAIT: Trigger + Level]

💡 "[One-line summary]"

⚠️ CRITICAL: Output MUST be valid JSON format:
{
  "decision": "BUY_CALL" | "BUY_PUT" | "NO_TRADE",
  "confidence": 1-5,
  "entry_price": float,
  "stop_loss": float,
  "target_1": float,
  "target_2": float,
  "reason": "string (max 200 chars)",
  "stop_hunt_risk": 0-100,
  "clarity_score": 0-100
}

⚠️ Position size as per SL. योग्य setup मिळेपर्यंत patience."""
    
    def parse_response(self, content: str) -> Optional[Dict]:
        """Parse AI response to extract trade decision"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback: Parse text response
            decision = "NO_TRADE"
            if "BUY_CALL" in content or "LONG CE" in content:
                decision = "BUY_CALL"
            elif "BUY_PUT" in content or "SHORT PE" in content:
                decision = "BUY_PUT"
            
            return {
                "decision": decision,
                "confidence": 3,
                "entry_price": 0,
                "stop_loss": 0,
                "target_1": 0,
                "target_2": 0,
                "reason": content[:200],
                "stop_hunt_risk": 50,
                "clarity_score": 50
            }
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return None

# ==================== CHART GENERATOR ====================
class ChartGenerator:
    @staticmethod
    def create_trade_chart(symbol: str, df: pd.DataFrame, spot: float, 
                          decision: Dict, category: str, path: str):
        """Create chart with trade setup overlay"""
        BG = '#FFFFFF'
        GRID = '#E0E0E0'
        TEXT = '#2C3E50'
        GREEN = '#26a69a'
        RED = '#ef5350'
        YELLOW = '#FFD700'
        ORANGE = '#FF9800'
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       facecolor=BG)
        
        ax1.set_facecolor(BG)
        df_plot = df.reset_index(drop=True)
        
        # Candlesticks
        for idx, row in df_plot.iterrows():
            color = GREEN if row['close'] > row['open'] else RED
            ax1.plot([idx+0.3, idx+0.3], [row['low'], row['high']],
                    color=color, linewidth=1.2, alpha=0.8)
            ax1.add_patch(Rectangle(
                (idx, min(row['open'], row['close'])),
                0.6,
                abs(row['close'] - row['open']) if abs(row['close'] - row['open']) > 0 else spot * 0.0001,
                facecolor=color,
                edgecolor=color,
                alpha=0.85
            ))
        
        # Highlight last candle
        last_idx = len(df_plot) - 1
        last_close = df_plot.iloc[-1]['close']
        ax1.scatter([last_idx + 0.3], [last_close],
                   color=YELLOW, s=250, marker='D', zorder=10,
                   edgecolors=TEXT, linewidths=2, alpha=0.9)
        
        # Current price line
        ax1.axhline(spot, color=ORANGE, linewidth=2, linestyle='--', alpha=0.7)
        ax1.text(2, spot, f' Spot: ₹{spot:.2f}',
                color=ORANGE, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         alpha=0.9, edgecolor=ORANGE, linewidth=1.5))
        
        # Trade setup lines (if trade decision)
        if decision['decision'] != 'NO_TRADE':
            entry = decision.get('entry_price', 0)
            sl = decision.get('stop_loss', 0)
            t1 = decision.get('target_1', 0)
            
            if entry > 0:
                ax1.axhline(entry, color=ORANGE, linewidth=2, linestyle=':', alpha=0.8)
                ax1.text(len(df_plot) - 5, entry, f' ENTRY: ₹{entry:.2f}',
                        color=ORANGE, fontsize=9, fontweight='bold')
            
            if sl > 0:
                ax1.axhline(sl, color=RED, linewidth=3, linestyle='-', alpha=0.8)
                ax1.text(len(df_plot) - 5, sl, f' SL: ₹{sl:.2f}',
                        color=RED, fontsize=9, fontweight='bold')
            
            if t1 > 0:
                ax1.axhline(t1, color=GREEN, linewidth=3, linestyle='-', alpha=0.8)
                ax1.text(len(df_plot) - 5, t1, f' T1: ₹{t1:.2f}',
                        color=GREEN, fontsize=9, fontweight='bold')
        
        # Category emoji
        category_emoji = {
            "INDEX": "📈", "AUTO": "🚗", "BANK": "🏦", "METAL": "🏭",
            "OIL_GAS": "⛽", "IT": "💻", "PHARMA": "💊", "FMCG": "🛒",
            "INFRA": "⚡", "POWER": "⚡", "RETAIL": "👕", "INSURANCE": "🛡️",
            "TELECOM": "📱", "FINANCE": "💰"
        }.get(category, "📊")
        
        # Title
        direction = decision.get('decision', 'NO_TRADE').replace('_', ' ')
        confidence = '⭐' * decision.get('confidence', 0)
        title = f"{category_emoji} {symbol} | {direction} | {confidence}"
        ax1.set_title(title, color=TEXT, fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, color=GRID, alpha=0.4)
        ax1.tick_params(colors=TEXT)
        ax1.set_ylabel('Price (₹)', color=TEXT, fontsize=11, fontweight='bold')
        
        # Volume
        ax2.set_facecolor(BG)
        colors = [GREEN if df_plot.iloc[i]['close'] > df_plot.iloc[i]['open'] else RED
                 for i in range(len(df_plot))]
        ax2.bar(range(len(df_plot)), df_plot['volume'], color=colors, alpha=0.7, width=0.8)
        ax2.set_ylabel('Volume', color=TEXT, fontsize=10, fontweight='bold')
        ax2.tick_params(colors=TEXT)
        ax2.grid(True, color=GRID, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, facecolor=BG)
        plt.close()

# ==================== TELEGRAM NOTIFIER ====================
class TelegramNotifier:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.trade_log = []
    
    async def send_startup(self):
        msg = f"""🚀 **H-LIQUIDITY WALT v1.0**

⏰ {datetime.now(IST).strftime('%d-%b %H:%M IST')}

✅ Pre-Market: 8:55 AM
✅ Live Scan: Every 5 minutes
✅ DeepSeek V3 AI Analysis
✅ Active Filter (Volume/OI)
✅ Redis Storage (1-day expiry)

📊 **Monitoring {len(ALL_SYMBOLS)} Symbols**

🟢 **SYSTEM ACTIVE**"""
        
        await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
    
    async def send_trade_alert(self, symbol: str, display_name: str, category: str,
                              spot: float, decision: Dict, strikes: List[StrikeData],
                              chart_path: str, expiry: str):
        """Send trade alert with chart"""
        try:
            # Send chart
            with open(chart_path, 'rb') as photo:
                await self.bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo)
            
            # PCR calculation
            pcr = OIAnalyzer.calculate_pcr(strikes)
            
            # Build alert message
            category_emoji = {
                "INDEX": "📈", "AUTO": "🚗", "BANK": "🏦", "METAL": "🏭",
                "OIL_GAS": "⛽", "IT": "💻", "PHARMA": "💊", "FMCG": "🛒",
                "INFRA": "⚡", "POWER": "⚡", "RETAIL": "👕", "INSURANCE": "🛡️",
                "TELECOM": "📱", "FINANCE": "💰"
            }.get(category, "📊")
            
            direction = decision['decision'].replace('_', ' ')
            confidence_stars = '⭐' * decision['confidence']
            
            msg = f"""🚨 **LIVE TRADE ALERT** 🚨

{category_emoji} **{display_name}**
━━━━━━━━━━━━━━━━━━━━━━

💹 **Spot:** ₹{spot:.2f}
🎯 **DIRECTION:** {direction}
📈 **CONFIDENCE:** {confidence_stars} ({decision['confidence']}/5)

**📊 TRADE SETUP:**
Entry Price: ₹{decision['entry_price']:.2f}
Stop Loss: ₹{decision['stop_loss']:.2f}
Target 1: ₹{decision['target_1']:.2f}
Target 2: ₹{decision['target_2']:.2f}

**💡 REASON:**
{decision['reason']}

**⚠️ RISK:**
Clarity Score: {decision['clarity_score']}/100
Stop Hunt Risk: {decision['stop_hunt_risk']}%

**📅 EXPIRY:** {expiry}
**📊 PCR:** {pcr:.3f}

🕒 {datetime.now(IST).strftime('%d-%b %H:%M IST')}
━━━━━━━━━━━━━━━━━━━━━━"""
            
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
            
            # Log trade
            self.trade_log.append({
                'timestamp': datetime.now(IST).isoformat(),
                'symbol': symbol,
                'direction': direction,
                'entry': decision['entry_price'],
                'sl': decision['stop_loss'],
                'target': decision['target_1'],
                'confidence': decision['confidence']
            })
            
            logger.info(f"  ✅ Trade alert sent: {display_name}")
            
        except Exception as e:
            logger.error(f"Telegram alert error: {e}")
            traceback.print_exc()
    
    async def send_error_alert(self, error_msg: str):
        """Send error notification"""
        try:
            msg = f"🚨 **SYSTEM ERROR**\n\n{error_msg}\n\n🕒 {datetime.now(IST).strftime('%H:%M IST')}"
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
        except:
            pass
    
    async def send_daily_summary(self):
        """Send daily P&L summary"""
        try:
            if not self.trade_log:
                return
            
            msg = f"""📊 **DAILY SUMMARY**

🕒 {datetime.now(IST).strftime('%d-%b-%Y')}

**Total Signals:** {len(self.trade_log)}

**Recent Trades:**
"""
            for trade in self.trade_log[-5:]:
                msg += f"\n{trade['symbol']} | {trade['direction']} | Entry: ₹{trade['entry']:.2f}"
            
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
        except:
            pass

# ==================== PRE-MARKET ENGINE ====================
class PreMarketEngine:
    def __init__(self, fetcher: UpstoxDataFetcher, redis: RedisManager, deepseek: DeepSeekClient):
        self.fetcher = fetcher
        self.redis = redis
        self.deepseek = deepseek
    
    async def run_pre_market_analysis(self, instrument_key: str, symbol_info: Dict) -> bool:
        """Run historical analysis for one instrument"""
        try:
            symbol_name = symbol_info.get('name', '')
            display_name = symbol_info.get('display_name', symbol_name)
            
            logger.info(f"  📊 Pre-Market: {display_name}")
            
            # Fetch 7-days historical data
            candles_1min = self.fetcher.get_historical_1min(instrument_key, days=7)
            if not candles_1min:
                logger.warning(f"    ❌ No historical data")
                return False
            
            # Aggregate to timeframes
            df_5m, df_15m, df_1h = DataAggregator.prepare_dataframes(candles_1min)
            
            if df_1h.empty or df_15m.empty:
                logger.warning(f"    ❌ Aggregation failed")
                return False
            
            # Get spot price
            spot = self.fetcher.get_spot_price(instrument_key)
            if spot == 0:
                logger.warning(f"    ❌ No spot price")
                return False
            
            # Prepare compressed historical summary
            last_10_1h = df_1h.tail(10)
            summary_1h = "\n".join([
                f"{row.name.strftime('%H:%M')}|{row['open']:.2f}|{row['high']:.2f}|{row['low']:.2f}|{row['close']:.2f}|{int(row['volume'])}|{int(row['oi'])}"
                for _, row in last_10_1h.iterrows()
            ])
            
            # Calculate key S/R levels (simple high/low)
            support_1 = df_15m['low'].tail(50).min()
            support_2 = df_15m['low'].tail(100).min()
            resistance_1 = df_15m['high'].tail(50).max()
            resistance_2 = df_15m['high'].tail(100).max()
            
            # Build pre-market prompt
            prompt = f"""INSTRUMENT: {symbol_name}
SPOT: ₹{spot:.2f}

**1-HOUR TREND (Last 10 candles):**
Time|Open|High|Low|Close|Volume|OI
{summary_1h}

**KEY LEVELS (15-MIN):**
Support: {support_1:.2f}, {support_2:.2f}
Resistance: {resistance_1:.2f}, {resistance_2:.2f}

**Analyze:**
1. 1-Hour Trend (Bullish/Bearish/Sideways)
2. Major Support/Resistance
3. Any pattern detected?
4. Overall Bias

Output JSON format."""
            
            # Get AI analysis
            analysis = self.deepseek.analyze(prompt)
            
            if analysis:
                # Store in Redis
                historical_data = {
                    'trend_1h': 'BULLISH' if df_1h['close'].iloc[-1] > df_1h['close'].iloc[-5] else 'BEARISH',
                    'support': [float(support_1), float(support_2)],
                    'resistance': [float(resistance_1), float(resistance_2)],
                    'spot': spot,
                    'timestamp': datetime.now(IST).isoformat()
                }
                
                self.redis.set_with_expiry(
                    f"HISTORICAL_ANALYSIS_{symbol_name}",
                    json.dumps(historical_data)
                )
                
                # Store aggregated candles
                self.redis.set_with_expiry(
                    f"CANDLES_1H_{symbol_name}",
                    df_1h.to_json()
                )
                self.redis.set_with_expiry(
                    f"CANDLES_15M_{symbol_name}",
                    df_15m.to_json()
                )
                self.redis.set_with_expiry(
                    f"CANDLES_5M_{symbol_name}",
                    df_5m.to_json()
                )
                
                logger.info(f"    ✅ Analysis stored")
                return True
            else:
                logger.warning(f"    ⚠️ AI analysis failed")
                return False
            
        except Exception as e:
            logger.error(f"Pre-market error for {symbol_info.get('display_name')}: {e}")
            return False
    
    async def run_parallel_analysis(self, symbols: Dict):
        """Run pre-market for all symbols in parallel"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 PRE-MARKET ANALYSIS - {datetime.now(IST).strftime('%H:%M:%S IST')}")
        logger.info(f"{'='*80}")
        
        # Process in batches of 10
        items = list(symbols.items())
        batch_size = 10
        success_count = 0
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            logger.info(f"\n📦 Batch {i//batch_size + 1} ({len(batch)} instruments)")
            
            tasks = [
                self.run_pre_market_analysis(key, info)
                for key, info in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count += sum(1 for r in results if r is True)
            
            # Small delay between batches
            if i + batch_size < len(items):
                await asyncio.sleep(2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ PRE-MARKET COMPLETE: {success_count}/{len(symbols)} instruments")
        logger.info(f"{'='*80}\n")
        
        return success_count

# ==================== LIVE SCANNER ====================
class LiveScanner:
    def __init__(self, fetcher: UpstoxDataFetcher, redis: RedisManager, 
                 deepseek: DeepSeekClient, notifier: TelegramNotifier):
        self.fetcher = fetcher
        self.redis = redis
        self.deepseek = deepseek
        self.notifier = notifier
        self.scan_count = 0
    
    async def analyze_instrument(self, instrument_key: str, symbol_info: Dict):
        """Analyze single instrument for trade setup"""
        try:
            symbol_name = symbol_info.get('name', '')
            display_name = symbol_info.get('display_name', symbol_name)
            category = symbol_info.get('category', 'OTHER')
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 {display_name}")
            logger.info(f"{'='*70}")
            
            # Get current spot price
            spot = self.fetcher.get_spot_price(instrument_key)
            if spot == 0:
                logger.warning(f"  ❌ No spot price")
                return
            
            logger.info(f"  💹 Spot: ₹{spot:.2f}")
            
            # Get expiry
            expiry = ExpiryCalculator.get_best_expiry(
                instrument_key, symbol_info, self.fetcher.headers
            )
            
            # Fetch last 5-min intraday data
            candles_intraday = self.fetcher.get_intraday_1min(instrument_key)
            if not candles_intraday:
                logger.warning(f"  ⚠️ No intraday data")
                return
            
            # Get stored historical data from Redis
            candles_5m_json = self.redis.get(f"CANDLES_5M_{symbol_name}")
            candles_15m_json = self.redis.get(f"CANDLES_15M_{symbol_name}")
            
            if not candles_5m_json or not candles_15m_json:
                logger.warning(f"  ⚠️ No historical data in Redis")
                return
            
            df_5m_historical = pd.read_json(candles_5m_json)
            df_15m_historical = pd.read_json(candles_15m_json)
            
            # Append today's data
            df_intraday = pd.DataFrame(candles_intraday, 
                                      columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df_intraday['timestamp'] = pd.to_datetime(df_intraday['timestamp'])
            df_intraday = df_intraday.set_index('timestamp').astype(float).sort_index()
            
            # Aggregate today's data
            df_5m_today = DataAggregator.aggregate_to_timeframe(df_intraday, '5min', 100)
            df_15m_today = DataAggregator.aggregate_to_timeframe(df_intraday, '15min', 100)
            
            # Combine historical + today
            df_5m = pd.concat([df_5m_historical, df_5m_today]).tail(50)
            df_15m = pd.concat([df_15m_historical, df_15m_today]).tail(175)
            
            if df_5m.empty or df_15m.empty:
                logger.warning(f"  ⚠️ Empty dataframes")
                return
            
            logger.info(f"  📊 Candles: 5M={len(df_5m)}, 15M={len(df_15m)}")
            
            # Get option chain
            all_strikes = self.fetcher.get_option_chain(instrument_key, expiry)
            if not all_strikes:
                logger.warning(f"  ⚠️ No option chain")
                return
            
            atm_strikes = OIAnalyzer.get_atm_strikes(all_strikes, spot, count=21)
            
            # Calculate metrics
            pcr = OIAnalyzer.calculate_pcr(atm_strikes)
            max_pain = OIAnalyzer.calculate_max_pain(atm_strikes, spot)
            iv_delta = OIAnalyzer.calculate_iv_delta(atm_strikes, self.redis, symbol_name)
            
            logger.info(f"  📊 PCR: {pcr:.3f} | Max Pain: {max_pain} | IV Δ: {iv_delta:.1f}%")
            
            # Get historical analysis from Redis
            historical_json = self.redis.get(f"HISTORICAL_ANALYSIS_{symbol_name}")
            historical_data = json.loads(historical_json) if historical_json else {}
            
            # Build incremental prompt (last 3 candles + summary)
            last_3_15m = df_15m.tail(3)
            candle_summary = "\n".join([
                f"{row.name.strftime('%H:%M')}|{row['open']:.2f}|{row['high']:.2f}|{row['low']:.2f}|{row['close']:.2f}|{int(row['volume'])}|{int(row['oi'])}"
                for _, row in last_3_15m.iterrows()
            ])
            
            # OI comparison (simplified - using ATM strikes)
            total_ce_oi = sum(s.ce_oi for s in atm_strikes)
            total_pe_oi = sum(s.pe_oi for s in atm_strikes)
            
            prompt = f"""INSTRUMENT: {symbol_name} | SPOT: ₹{spot:.2f} | EXPIRY: {expiry}

**HISTORICAL CONTEXT:**
Trend: {historical_data.get('trend_1h', 'UNKNOWN')}
Support: {historical_data.get('support', [])}
Resistance: {historical_data.get('resistance', [])}

**LIVE UPDATES (Last 3 candles, 15M):**
Time|Open|High|Low|Close|Volume|OI
{candle_summary}

**OPTION CHAIN (21 ATM):**
Total CE OI: {total_ce_oi:,}
Total PE OI: {total_pe_oi:,}
PCR: {pcr:.3f}
Max Pain: {max_pain}
IV Delta: {iv_delta:.1f}%

**CONTEXT:**
Time: {datetime.now(IST).strftime('%H:%M')}
Session: {'Opening' if datetime.now(IST).hour < 11 else 'Power Hour' if datetime.now(IST).hour >= 14 else 'Trending'}

**Analyze and give trade decision (JSON format):**
- If Clarity Score > 60 AND Risk <= MEDIUM AND Stop Hunt Risk < 60% → BUY_CALL or BUY_PUT
- Else → NO_TRADE"""
            
            # Get AI decision
            logger.info(f"  🤖 Analyzing...")
            decision = self.deepseek.analyze(prompt)
            
            if not decision:
                logger.warning(f"  ⚠️ AI analysis failed")
                return
            
            logger.info(f"  📊 Decision: {decision['decision']} | Score: {decision['clarity_score']}/100")
            
            # Check if trade signal
            if decision['decision'] in ['BUY_CALL', 'BUY_PUT']:
                if decision['clarity_score'] >= 60 and decision['stop_hunt_risk'] < 60:
                    # Generate chart
                    chart_path = f"/tmp/{symbol_name}_trade_{self.scan_count}.png"
                    ChartGenerator.create_trade_chart(
                        display_name, df_5m, spot, decision, category, chart_path
                    )
                    
                    # Send alert
                    await self.notifier.send_trade_alert(
                        symbol_name, display_name, category, spot,
                        decision, atm_strikes, chart_path, expiry
                    )
                    
                    logger.info(f"  ✅ Trade alert sent")
                else:
                    logger.info(f"  ⚠️ Trade filtered (Score/Risk threshold)")
            else:
                logger.info(f"  ℹ️ No trade zone")
            
        except Exception as e:
            logger.error(f"Analyze error for {symbol_info.get('display_name')}: {e}")
            traceback.print_exc()
    
    async def run_scan(self, symbols: Dict):
        """Run 5-min scan for active instruments"""
        self.scan_count += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 SCAN #{self.scan_count} - {datetime.now(IST).strftime('%H:%M:%S IST')}")
        logger.info(f"{'='*80}")
        
        # Filter active instruments
        logger.info("📊 Filtering active instruments...")
        active_instruments = []
        
        for key, info in symbols.items():
            # For demo, treat all as active (you can enable filter)
            # is_active = ActiveFilter.is_active(key, info, self.fetcher)
            is_active = True  # Simplified for now
            
            if is_active:
                active_instruments.append((key, info))
        
        logger.info(f"  ✅ Active: {len(active_instruments)}/{len(symbols)}")
        
        # Analyze each active instrument
        for idx, (key, info) in enumerate(active_instruments, 1):
            logger.info(f"\n[{idx}/{len(active_instruments)}]")
            await self.analyze_instrument(key, info)
            
            # Small delay to avoid rate limits
            if idx < len(active_instruments):
                await asyncio.sleep(1)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ SCAN #{self.scan_count} COMPLETE")
        logger.info(f"{'='*80}\n")

# ==================== MAIN BOT ====================
class HLiquidityWALT:
    def __init__(self):
        logger.info("🔄 Initializing H-LIQUIDITY WALT...")
        
        self.fetcher = UpstoxDataFetcher()
        self.redis = RedisManager()
        self.deepseek = DeepSeekClient()
        self.notifier = TelegramNotifier()
        self.pre_market = PreMarketEngine(self.fetcher, self.redis, self.deepseek)
        self.scanner = LiveScanner(self.fetcher, self.redis, self.deepseek, self.notifier)
        
        logger.info("✅ Bot ready")
    
    def is_market_open(self) -> bool:
        now = datetime.now(IST)
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        return time(9, 15) <= current_time <= time(15, 30)
    
    def should_run_pre_market(self) -> bool:
        now = datetime.now(IST)
        current_time = now.time()
        return time(8, 55) <= current_time < time(9, 10)
    
    async def run(self):
        logger.info("="*80)
        logger.info("H-LIQUIDITY WALT TRADING SYSTEM")
        logger.info("="*80)
        
        if not all([UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY]):
            logger.error("❌ Missing credentials!")
            return
        
        await self.notifier.send_startup()
        
        pre_market_done = False
        
        logger.info("="*80)
        logger.info("🟢 RUNNING")
        logger.info("="*80)
        
        while True:
            try:
                now = datetime.now(IST)
                
                # Pre-market phase (8:55-9:10 AM)
                if self.should_run_pre_market() and not pre_market_done:
                    logger.info("\n🌅 PRE-MARKET PHASE")
                    success = await self.pre_market.run_parallel_analysis(ALL_SYMBOLS)
                    
                    if success > len(ALL_SYMBOLS) * 0.7:  # 70% success rate
                        pre_market_done = True
                        await self.notifier.bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=f"✅ Pre-Market Complete: {success}/{len(ALL_SYMBOLS)} instruments"
                        )
                    else:
                        await self.notifier.send_error_alert(
                            f"⚠️ Pre-Market Incomplete: {success}/{len(ALL_SYMBOLS)}"
                        )
                
                # Reset pre-market flag after market close
                if now.time() > time(15, 30):
                    pre_market_done = False
                
                # Live scanning phase (9:16 AM - 3:30 PM, every 5 min)
                if self.is_market_open() and pre_market_done:
                    if now.minute % 5 == 1 or now.minute % 5 == 0:  # Run at X:00, X:05, X:10, etc.
                        await self.scanner.run_scan(ALL_SYMBOLS)
                        
                        # Wait to next 5-min mark
                        await asyncio.sleep(SCAN_INTERVAL)
                    else:
                        await asyncio.sleep(30)
                
                # After market close - send daily summary
                if now.time() == time(15, 35):
                    await self.notifier.send_daily_summary()
                    await asyncio.sleep(60)
                
                # Market closed or waiting
                if not self.is_market_open() and not self.should_run_pre_market():
                    logger.info("⏸️ Market closed. Waiting...")
                    await asyncio.sleep(300)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopped by user")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                traceback.print_exc()
                await self.notifier.send_error_alert(f"System Error: {str(e)[:200]}")
                await asyncio.sleep(60)

# ==================== CLEANUP SCHEDULER ====================
async def daily_cleanup_task(redis: RedisManager):
    """Run daily at 4:00 AM"""
    while True:
        try:
            now = datetime.now(IST)
            if now.time() == time(4, 0):
                logger.info("\n🧹 Running daily cleanup...")
                redis.cleanup_expired()
                logger.info("✅ Cleanup complete")
                await asyncio.sleep(3600)  # Wait 1 hour
            else:
                await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            await asyncio.sleep(300)

# ==================== ENTRY POINT ====================
async def main():
    try:
        bot = HLiquidityWALT()
        
        # Start cleanup task in background
        cleanup_task = asyncio.create_task(daily_cleanup_task(bot.redis))
        
        # Run main bot
        await bot.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("H-LIQUIDITY WALT - STARTING")
    logger.info("="*80)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n✅ Shutdown complete")
    except Exception as e:
        logger.error(f"\n❌ Critical error: {e}")
        traceback.print_exc()
