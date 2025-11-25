#!/usr/bin/env python3
"""
NIFTY OPTIONS BOT V12.0 - STRIKE MASTER
========================================
Multi-Factor Strike Analysis Bot with FIXED Futures Data Fetching

Author: Data Monster Team
Version: 12.0 - Production Ready with Intraday + Historical API
"""

import os
import asyncio
import aiohttp
import urllib.parse
from datetime import datetime, timedelta, time
import pytz
import json
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np
from calendar import monthrange
import gzip
from io import BytesIO

# Optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not installed. Using RAM-only mode.")

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("Telegram not installed. Alerts disabled.")

# Configuration
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("StrikeMaster-V12")

# API Configuration
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Upstox Instruments JSON URL
INSTRUMENTS_JSON_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

# Instrument Symbols
INDICES = {
    'NIFTY': {
        'spot': "NSE_INDEX|Nifty 50",
        'name': 'NIFTY 50',
        'expiry_type': 'weekly',
        'expiry_day': 1,
        'strike_gap': 50
    },
    'BANKNIFTY': {
        'spot': "NSE_INDEX|Nifty Bank",
        'name': 'BANK NIFTY',
        'expiry_type': 'monthly',
        'expiry_day': 1,
        'strike_gap': 100
    },
    'FINNIFTY': {
        'spot': "NSE_INDEX|Nifty Fin Service",
        'name': 'FIN NIFTY',
        'expiry_type': 'monthly',
        'expiry_day': 1,
        'strike_gap': 50
    },
    'MIDCPNIFTY': {
        'spot': "NSE_INDEX|NIFTY MID SELECT",
        'name': 'MIDCAP NIFTY',
        'expiry_type': 'monthly',
        'expiry_day': 1,
        'strike_gap': 25
    }
}

ACTIVE_INDEX = os.getenv('ACTIVE_INDEX', 'NIFTY')
ALERT_ONLY_MODE = True
SCAN_INTERVAL = 60

# Strategy Thresholds
OI_THRESHOLD_STRONG = 8.0
OI_THRESHOLD_MEDIUM = 5.0
ATM_OI_THRESHOLD = 5.0
VOL_SPIKE_2X = 2.0
VOL_SPIKE_3X = 3.0
PCR_EXTREME_BULLISH = 1.15
PCR_BULLISH = 1.08
PCR_BEARISH = 0.92
PCR_EXTREME_BEARISH = 0.85
MIN_CANDLE_SIZE = 8
VWAP_BUFFER = 5

# Time Filters
AVOID_OPENING = (time(9, 15), time(9, 30))
AVOID_CLOSING = (time(15, 0), time(15, 30))

# ATR Configuration
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 2.5

@dataclass
class Signal:
    type: str
    reason: str
    confidence: int
    spot_price: float
    futures_price: float
    strike: int
    target_points: int
    stop_loss_points: int
    pcr: float
    candle_color: str
    volume_surge: float
    oi_5m: float
    oi_15m: float
    atm_ce_change: float
    atm_pe_change: float
    atr: float
    timestamp: datetime

# ==================== INSTRUMENTS CACHE ====================
class InstrumentsCache:
    """Global cache for instruments data"""
    _instance = None
    _instruments_map = {}
    _last_update = None
    _update_interval = timedelta(hours=6)  # Refresh every 6 hours
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def load_instruments(self) -> bool:
        """Download and cache instruments data"""
        now = datetime.now(IST)
        
        # Check if we need to update
        if self._last_update and (now - self._last_update) < self._update_interval:
            if self._instruments_map:
                logger.info(f"Using cached instruments (last updated: {self._last_update.strftime('%H:%M')})")
                return True
        
        logger.info("📥 Downloading Upstox instruments...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(INSTRUMENTS_JSON_URL) as resp:
                    if resp.status == 200:
                        compressed = await resp.read()
                        decompressed = gzip.decompress(compressed)
                        instruments = json.loads(decompressed)
                        
                        # Build futures map
                        self._instruments_map = {}
                        
                        for instrument in instruments:
                            if instrument.get('segment') != 'NSE_FO':
                                continue
                            
                            if instrument.get('instrument_type') != 'FUT':
                                continue
                            
                            name = instrument.get('name', '')
                            if name not in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']:
                                continue
                            
                            expiry_ms = instrument.get('expiry', 0)
                            if not expiry_ms:
                                continue
                            
                            expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=IST)
                            
                            # Skip expired
                            if expiry_dt < now:
                                continue
                            
                            # Store nearest expiry for each index
                            if name not in self._instruments_map:
                                self._instruments_map[name] = {
                                    'instrument_key': instrument.get('instrument_key'),
                                    'trading_symbol': instrument.get('trading_symbol'),
                                    'expiry': expiry_dt.strftime('%d-%b-%Y'),
                                    'expiry_timestamp': expiry_ms
                                }
                            else:
                                # Keep nearest expiry
                                if expiry_ms < self._instruments_map[name]['expiry_timestamp']:
                                    self._instruments_map[name] = {
                                        'instrument_key': instrument.get('instrument_key'),
                                        'trading_symbol': instrument.get('trading_symbol'),
                                        'expiry': expiry_dt.strftime('%d-%b-%Y'),
                                        'expiry_timestamp': expiry_ms
                                    }
                        
                        self._last_update = now
                        
                        logger.info(f"✅ Loaded futures instruments:")
                        for name, info in self._instruments_map.items():
                            logger.info(f"   {name}: {info['instrument_key']} (Expiry: {info['expiry']})")
                        
                        return True
                    else:
                        logger.error(f"❌ HTTP {resp.status}")
                        return False
            
            except Exception as e:
                logger.error(f"💥 Download failed: {e}")
                return False
    
    def get_futures_key(self, index_name: str) -> Optional[str]:
        """Get instrument key for index futures"""
        if index_name in self._instruments_map:
            return self._instruments_map[index_name]['instrument_key']
        return None

def get_expiry_date(index_name: str = 'NIFTY') -> str:
    now = datetime.now(IST)
    today = now.date()
    index_config = INDICES[index_name]
    
    if index_config['expiry_type'] == 'weekly':
        days_until_tuesday = (1 - today.weekday() + 7) % 7
        if days_until_tuesday == 0 and now.time() > time(15, 30):
            expiry = today + timedelta(days=7)
        else:
            expiry = today + timedelta(days=days_until_tuesday if days_until_tuesday > 0 else 7)
    else:
        year = now.year
        month = now.month
        last_day = monthrange(year, month)[1]
        last_date = datetime(year, month, last_day).date()
        days_to_tuesday = (last_date.weekday() - 1) % 7
        last_tuesday = last_date - timedelta(days=days_to_tuesday)
        
        if (today > last_tuesday) or (today == last_tuesday and now.time() > time(15, 30)):
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            last_day = monthrange(year, month)[1]
            last_date = datetime(year, month, last_day).date()
            days_to_tuesday = (last_date.weekday() - 1) % 7
            expiry = last_date - timedelta(days=days_to_tuesday)
        else:
            expiry = last_tuesday
    
    return expiry.strftime('%Y-%m-%d')

def is_data_collection_time() -> bool:
    now = datetime.now(IST).time()
    return time(9, 15) <= now <= time(15, 30)

def is_tradeable_time() -> bool:
    now = datetime.now(IST).time()
    if not (time(9, 15) <= now <= time(15, 30)):
        return False
    if AVOID_OPENING[0] <= now <= AVOID_OPENING[1]:
        logger.info("Opening window - Data collection only")
        return False
    if AVOID_CLOSING[0] <= now <= AVOID_CLOSING[1]:
        logger.info("Closing window - Avoiding manipulation")
        return False
    return True

class RedisBrain:
    def __init__(self):
        self.client = None
        self.memory = {}
        
        if REDIS_AVAILABLE:
            try:
                self.client = redis.from_url(REDIS_URL, decode_responses=True)
                self.client.ping()
                logger.info("✅ Redis Connected")
            except Exception as e:
                logger.warning(f"Redis Failed: {e}. Using RAM Mode")
                self.client = None
        else:
            logger.info("💾 RAM-only mode")
    
    def save_strike_snapshot(self, strike_data: Dict[int, dict]):
        now = datetime.now(IST)
        timestamp = now.replace(second=0, microsecond=0)
        
        for strike, data in strike_data.items():
            key = f"strike:{strike}:{timestamp.strftime('%H%M')}"
            value = json.dumps(data)
            
            if self.client:
                try:
                    self.client.setex(key, 3600, value)
                except Exception as e:
                    logger.debug(f"Redis save error: {e}")
                    self.memory[key] = value
            else:
                self.memory[key] = value
    
    def get_strike_oi_change(self, strike: int, current_data: dict, minutes_ago: int = 15) -> Tuple[float, float]:
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        timestamp = now.replace(second=0, microsecond=0)
        key = f"strike:{strike}:{timestamp.strftime('%H%M')}"
        
        past_data_str = None
        if self.client:
            try:
                past_data_str = self.client.get(key)
            except:
                past_data_str = self.memory.get(key)
        else:
            past_data_str = self.memory.get(key)
        
        if not past_data_str:
            return 0.0, 0.0
        
        try:
            past = json.loads(past_data_str)
            ce_chg = ((current_data['ce_oi'] - past['ce_oi']) / past['ce_oi'] * 100 if past['ce_oi'] > 0 else 0)
            pe_chg = ((current_data['pe_oi'] - past['pe_oi']) / past['pe_oi'] * 100 if past['pe_oi'] > 0 else 0)
            return ce_chg, pe_chg
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return 0.0, 0.0
    
    def save_total_oi_snapshot(self, ce_total: int, pe_total: int):
        now = datetime.now(IST)
        slot = now.replace(second=0, microsecond=0)
        key = f"total_oi:{slot.strftime('%H%M')}"
        data = json.dumps({"ce": ce_total, "pe": pe_total})
        
        if self.client:
            try:
                self.client.setex(key, 3600, data)
            except:
                self.memory[key] = data
        else:
            self.memory[key] = data
    
    def get_total_oi_change(self, current_ce: int, current_pe: int, minutes_ago: int = 15) -> Tuple[float, float]:
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        slot = now.replace(second=0, microsecond=0)
        key = f"total_oi:{slot.strftime('%H%M')}"
        
        past_data = None
        if self.client:
            try:
                past_data = self.client.get(key)
            except:
                past_data = self.memory.get(key)
        else:
            past_data = self.memory.get(key)
        
        if not past_data:
            return 0.0, 0.0
        
        try:
            past = json.loads(past_data)
            ce_chg = ((current_ce - past['ce']) / past['ce'] * 100 if past['ce'] > 0 else 0)
            pe_chg = ((current_pe - past['pe']) / past['pe'] * 100 if past['pe'] > 0 else 0)
            return ce_chg, pe_chg
        except:
            return 0.0, 0.0

class StrikeDataFeed:
    def __init__(self, index_name: str = 'NIFTY'):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.spot_symbol = self.index_config['spot']
        self.headers = {"Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}", "Accept": "application/json"}
        self.retry_count = 3
        self.base_retry_delay = 2
        self.instruments_cache = InstrumentsCache.get_instance()
        logger.info(f"🎯 Initialized {self.index_config['name']}")
    
    async def fetch_with_retry(self, url: str, session: aiohttp.ClientSession):
        for attempt in range(self.retry_count):
            try:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait_time = (2 ** attempt) * self.base_retry_delay
                        logger.warning(f"⏳ Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ HTTP {resp.status}")
                        await asyncio.sleep(self.base_retry_delay)
            except Exception as e:
                logger.error(f"💥 Attempt {attempt+1}: {e}")
                await asyncio.sleep(self.base_retry_delay * (attempt + 1))
        return None
    
    async def fetch_futures_data(self, instrument_key: str, session: aiohttp.ClientSession) -> pd.DataFrame:
        """
        Fetch futures data using DUAL API APPROACH:
        1. Try INTRADAY API first (live today's data)
        2. Fallback to HISTORICAL API if intraday fails
        """
        now = datetime.now(IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        is_market_hours = market_open <= now <= market_close
        
        enc_key = urllib.parse.quote(instrument_key)
        
        # Try INTRADAY API first (only returns today's data)
        if is_market_hours:
            intraday_url = f"https://api.upstox.com/v2/historical-candle/intraday/{enc_key}/1minute"
            logger.info(f"📊 Trying INTRADAY API for live data...")
            
            intraday_data = await self.fetch_with_retry(intraday_url, session)
            
            if intraday_data and intraday_data.get('status') == 'success':
                candles = intraday_data.get('data', {}).get('candles', [])
                
                if candles and len(candles) > 0:
                    df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'oi'])
                    df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert(IST)
                    df = df.sort_values('ts').set_index('ts')
                    df = df.tail(100)  # Last 100 candles
                    
                    latest_time = df.index[-1]
                    data_age = (now - latest_time).total_seconds() / 60
                    
                    logger.info(f"✅ INTRADAY: {len(df)} candles | Latest: {latest_time.strftime('%I:%M %p')} ({data_age:.1f}m ago)")
                    return df
        
        # Fallback to HISTORICAL API
        logger.info(f"📈 Using HISTORICAL API...")
        to_date = now.strftime('%Y-%m-%d')
        from_date = (now - timedelta(days=3)).strftime('%Y-%m-%d')
        historical_url = f"https://api.upstox.com/v2/historical-candle/{enc_key}/1minute/{to_date}/{from_date}"
        
        historical_data = await self.fetch_with_retry(historical_url, session)
        
        if historical_data and historical_data.get('status') == 'success':
            candles = historical_data.get('data', {}).get('candles', [])
            
            if candles:
                df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'oi'])
                df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert(IST)
                df = df.sort_values('ts').set_index('ts')
                
                # Filter to today's data
                today = now.date()
                df = df[df.index.date == today]
                df = df.tail(100)
                
                if not df.empty:
                    latest_time = df.index[-1]
                    logger.info(f"✅ HISTORICAL: {len(df)} candles | Latest: {latest_time.strftime('%I:%M %p')}")
                    return df
        
        logger.warning(f"⚠️ No futures data available")
        return pd.DataFrame()
    
    async def get_market_data(self) -> Tuple[pd.DataFrame, Dict[int, dict], str, float, float, float]:
        """Fetch complete market data with improved futures fetching"""
        
        # Load instruments cache
        await self.instruments_cache.load_instruments()
        
        async with aiohttp.ClientSession() as session:
            spot_price = 0
            futures_price = 0
            df = pd.DataFrame()
            strike_data = {}
            total_options_volume = 0
            
            # 1. Fetch Spot Price
            logger.info("📍 Fetching Spot Price...")
            enc_spot = urllib.parse.quote(self.spot_symbol)
            ltp_url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={enc_spot}"
            
            ltp_data = await self.fetch_with_retry(ltp_url, session)
            if ltp_data and 'data' in ltp_data:
                if self.spot_symbol in ltp_data['data']:
                    spot_price = ltp_data['data'][self.spot_symbol].get('last_price', 0)
                    logger.info(f"✅ Spot: ₹{spot_price:.2f}")
            
            if spot_price == 0:
                logger.error("❌ Failed to fetch spot price")
                return df, strike_data, "", 0, 0, 0
            
            # 2. Fetch Futures Data (Using Instruments Cache)
            futures_key = self.instruments_cache.get_futures_key(self.index_name)
            
            if futures_key:
                logger.info(f"🔍 Fetching Futures: {futures_key}")
                df = await self.fetch_futures_data(futures_key, session)
                
                if not df.empty:
                    futures_price = df['close'].iloc[-1]
                    logger.info(f"✅ Futures: ₹{futures_price:.2f}")
                else:
                    logger.warning(f"⚠️ No futures data - using spot price")
                    futures_price = spot_price
            else:
                logger.warning(f"⚠️ Futures instrument not found - using spot price")
                futures_price = spot_price
            
            # 3. Fetch Option Chain
            logger.info("🔗 Fetching Option Chain...")
            expiry = get_expiry_date(self.index_name)
            chain_url = f"https://api.upstox.com/v2/option/chain?instrument_key={enc_spot}&expiry_date={expiry}"
            
            strike_gap = self.index_config['strike_gap']
            atm_strike = round(spot_price / strike_gap) * strike_gap
            min_strike = atm_strike - (2 * strike_gap)
            max_strike = atm_strike + (2 * strike_gap)
            logger.info(f"📊 {self.index_name} ATM: {atm_strike} | Range: {min_strike}-{max_strike}")
            
            chain_data = await self.fetch_with_retry(chain_url, session)
            if chain_data and chain_data.get('status') == 'success':
                for option in chain_data.get('data', []):
                    strike = option.get('strike_price', 0)
                    if min_strike <= strike <= max_strike:
                        call_data = option.get('call_options', {}).get('market_data', {})
                        put_data = option.get('put_options', {}).get('market_data', {})
                        strike_data[strike] = {
                            'ce_oi': call_data.get('oi', 0),
                            'pe_oi': put_data.get('oi', 0),
                            'ce_vol': call_data.get('volume', 0),
                            'pe_vol': put_data.get('volume', 0),
                            'ce_ltp': call_data.get('ltp', 0),
                            'pe_ltp': put_data.get('ltp', 0)
                        }
                        total_options_volume += (call_data.get('volume', 0) + put_data.get('volume', 0))
                logger.info(f"✅ Collected {len(strike_data)} strikes")
            
            return df, strike_data, expiry, spot_price, futures_price, total_options_volume

class StrikeAnalyzer:
    def __init__(self):
        self.volume_history = []
    
    def calculate_vwap(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0
        df_copy = df.copy()
        df_copy['tp'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
        df_copy['vol_price'] = df_copy['tp'] * df_copy['vol']
        total_vol = df_copy['vol'].sum()
        if total_vol == 0:
            return df_copy['close'].iloc[-1]
        vwap = df_copy['vol_price'].cumsum() / df_copy['vol'].cumsum()
        return vwap.iloc[-1]
    
    def calculate_atr(self, df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
        if len(df) < period:
            return 30
        df_copy = df.tail(period).copy()
        df_copy['h-l'] = df_copy['high'] - df_copy['low']
        df_copy['h-pc'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['l-pc'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        df_copy['tr'] = df_copy[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return df_copy['tr'].mean()
    
    def get_candle_info(self, df: pd.DataFrame) -> Tuple[str, float]:
        if df.empty:
            return 'NEUTRAL', 0
        last = df.iloc[-1]
        candle_size = abs(last['close'] - last['open'])
        if last['close'] > last['open']:
            color = 'GREEN'
        elif last['close'] < last['open']:
            color = 'RED'
        else:
            color = 'DOJI'
        return color, candle_size
    
    def check_volume_surge(self, current_vol: float) -> Tuple[bool, float]:
        now = datetime.now(IST)
        self.volume_history.append({'time': now, 'volume': current_vol})
        cutoff = now - timedelta(minutes=20)
        self.volume_history = [x for x in self.volume_history if x['time'] > cutoff]
        if len(self.volume_history) < 5:
            return False, 0
        past_volumes = [x['volume'] for x in self.volume_history[:-1]]
        avg_vol = sum(past_volumes) / len(past_volumes)
        if avg_vol == 0:
            return False, 0
        multiplier = current_vol / avg_vol
        return multiplier >= VOL_SPIKE_2X, multiplier
    
    def calculate_focused_pcr(self, strike_data: Dict[int, dict]) -> float:
        total_ce = sum(data['ce_oi'] for data in strike_data.values())
        total_pe = sum(data['pe_oi'] for data in strike_data.values())
        return total_pe / total_ce if total_ce > 0 else 1.0
    
    def analyze_atm_battle(self, strike_data: Dict[int, dict], atm_strike: int, redis_brain) -> Tuple[float, float]:
        if atm_strike not in strike_data:
            return 0, 0
        current = strike_data[atm_strike]
        ce_15m, pe_15m = redis_brain.get_strike_oi_change(atm_strike, current, 15)
        logger.info(f"⚔️ ATM {atm_strike}: CE={ce_15m:+.1f}% | PE={pe_15m:+.1f}%")
        return ce_15m, pe_15m
    
    def check_momentum(self, df: pd.DataFrame, direction: str = 'bullish') -> bool:
        if df.empty or len(df) < 3:
            return False
        last_3 = df.tail(3)
        if direction == 'bullish':
            return sum(last_3['close'] > last_3['open']) >= 2
        else:
            return sum(last_3['close'] < last_3['open']) >= 2

class StrikeMasterBot:
    def __init__(self, index_name: str = 'NIFTY'):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.feed = StrikeDataFeed(index_name)
        self.redis = RedisBrain()
        self.analyzer = StrikeAnalyzer()
        self.telegram = None
        self.last_alert_time = None
        self.alert_cooldown = 300
        
        if TELEGRAM_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                self.telegram = Bot(token=TELEGRAM_BOT_TOKEN)
                logger.info("✅ Telegram Ready")
            except Exception as e:
                logger.warning(f"⚠️ Telegram: {e}")
    
    async def run_cycle(self):
        if not is_data_collection_time():
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 STRIKE MASTER SCAN - {self.index_config['name']}")
        logger.info(f"{'='*60}")
        
        df, strike_data, expiry, spot, futures, vol = await self.feed.get_market_data()
        
        if df.empty or not strike_data or spot ==
