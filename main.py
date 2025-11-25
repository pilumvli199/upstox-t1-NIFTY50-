#!/usr/bin/env python3
"""
NIFTY OPTIONS BOT V12.0 - STRIKE MASTER (FIXED WITH PROPER SYMBOLS)
===================================================================
Multi-Factor Strike Analysis Bot with Correct API Integration

Author: Data Monster Team
Version: 12.1 - Fixed Symbols + 2 Sec Delays
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
        'expiry_day': 1,  # Tuesday = 1
        'strike_gap': 50
    },
    'BANKNIFTY': {
        'spot': "NSE_INDEX|Nifty Bank",
        'name': 'BANK NIFTY',
        'expiry_type': 'weekly',
        'expiry_day': 2,  # Wednesday = 2
        'strike_gap': 100
    },
    'FINNIFTY': {
        'spot': "NSE_INDEX|Nifty Fin Service",
        'name': 'FIN NIFTY',
        'expiry_type': 'weekly',
        'expiry_day': 1,  # Tuesday = 1
        'strike_gap': 50
    },
    'MIDCPNIFTY': {
        'spot': "NSE_INDEX|NIFTY MID SELECT",
        'name': 'MIDCAP NIFTY',
        'expiry_type': 'monthly',
        'expiry_day': 1,  # Last Tuesday
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
PCR_BULLISH = 1.08
PCR_BEARISH = 0.92
MIN_CANDLE_SIZE = 8
VWAP_BUFFER = 5

# Time Filters
AVOID_OPENING = (time(9, 15), time(9, 30))
AVOID_CLOSING = (time(15, 0), time(15, 30))

# ATR Configuration
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 2.5

# API DELAY
API_DELAY = 2  # 2 seconds between requests

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
    oi_15m: float
    atm_ce_change: float
    atm_pe_change: float
    atr: float
    timestamp: datetime

class InstrumentsCache:
    """Global cache for instruments data"""
    _instance = None
    _instruments_map = {}
    _last_update = None
    _update_interval = timedelta(hours=6)
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def load_instruments(self) -> bool:
        """Download and cache instruments data"""
        now = datetime.now(IST)
        
        if self._last_update and (now - self._last_update) < self._update_interval:
            if self._instruments_map:
                logger.info(f"Using cached instruments (last updated: {self._last_update.strftime('%H:%M')})")
                return True
        
        logger.info("📥 Downloading Upstox instruments...")
        
        retry_count = 3
        for attempt in range(retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(INSTRUMENTS_JSON_URL, timeout=30) as resp:
                        if resp.status == 200:
                            compressed = await resp.read()
                            decompressed = gzip.decompress(compressed)
                            instruments = json.loads(decompressed)
                            
                            self._instruments_map = {}
                            
                            for instrument in instruments:
                                try:
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
                                    
                                    if expiry_dt < now:
                                        continue
                                    
                                    if name not in self._instruments_map:
                                        self._instruments_map[name] = {
                                            'instrument_key': instrument.get('instrument_key'),
                                            'trading_symbol': instrument.get('trading_symbol'),
                                            'expiry': expiry_dt.strftime('%d-%b-%Y'),
                                            'expiry_timestamp': expiry_ms
                                        }
                                    else:
                                        if expiry_ms < self._instruments_map[name]['expiry_timestamp']:
                                            self._instruments_map[name] = {
                                                'instrument_key': instrument.get('instrument_key'),
                                                'trading_symbol': instrument.get('trading_symbol'),
                                                'expiry': expiry_dt.strftime('%d-%b-%Y'),
                                                'expiry_timestamp': expiry_ms
                                            }
                                except Exception as e:
                                    continue
                            
                            self._last_update = now
                            
                            logger.info(f"✅ Loaded futures instruments:")
                            for name, info in self._instruments_map.items():
                                logger.info(f"   {name}: {info['instrument_key']} (Expiry: {info['expiry']})")
                            
                            return True
                        else:
                            logger.warning(f"HTTP {resp.status} on attempt {attempt + 1}")
                            if attempt < retry_count - 1:
                                await asyncio.sleep(5)
            
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(5)
        
        return False
    
    def get_futures_key(self, index_name: str) -> Optional[str]:
        """Get instrument key for index futures"""
        return self._instruments_map.get(index_name, {}).get('instrument_key')

def get_expiry_date(index_name: str = 'NIFTY') -> str:
    """Calculate next expiry date (ORIGINAL LOGIC FROM YOUR CODE)"""
    now = datetime.now(IST)
    today = now.date()
    index_config = INDICES[index_name]
    
    if index_config['expiry_type'] == 'weekly':
        # Weekly expiry - Tuesday for NIFTY, Wednesday for BANKNIFTY
        days_until_tuesday = (1 - today.weekday() + 7) % 7
        if days_until_tuesday == 0 and now.time() > time(15, 30):
            expiry = today + timedelta(days=7)
        else:
            expiry = today + timedelta(days=days_until_tuesday if days_until_tuesday > 0 else 7)
    else:
        # Monthly expiry - Last Tuesday of month
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
    """Check if current time is within market hours"""
    now = datetime.now(IST).time()
    return time(9, 15) <= now <= time(15, 30)

def is_tradeable_time() -> bool:
    """Check if current time is suitable for trading signals"""
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
    """Memory system for historical data storage"""
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
        """Save current strike data"""
        now = datetime.now(IST)
        timestamp = now.replace(second=0, microsecond=0)
        
        for strike, data in strike_data.items():
            key = f"strike:{strike}:{timestamp.strftime('%H%M')}"
            value = json.dumps(data)
            
            if self.client:
                try:
                    self.client.setex(key, 3600, value)
                except:
                    self.memory[key] = value
            else:
                self.memory[key] = value
    
    def get_strike_oi_change(self, strike: int, current_data: dict, minutes_ago: int = 15) -> Tuple[float, float]:
        """Calculate OI change"""
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
        except:
            return 0.0, 0.0
    
    def save_total_oi_snapshot(self, ce_total: int, pe_total: int):
        """Save total OI"""
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
        """Calculate total OI change"""
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
    """Data fetching engine with proper symbol handling"""
    def __init__(self, index_name: str = 'NIFTY'):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.spot_symbol = self.index_config['spot']
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.instruments_cache = InstrumentsCache.get_instance()
        logger.info(f"🎯 Initialized {self.index_config['name']}")
    
    async def fetch_with_retry(self, url: str, session: aiohttp.ClientSession, timeout: int = 15):
        """Fetch with retry logic"""
        for attempt in range(3):
            try:
                async with session.get(url, headers=self.headers, timeout=timeout) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait_time = 2 ** attempt * 2
                        logger.warning(f"⏳ Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"HTTP {resp.status}")
                        await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Request error: {e}")
                await asyncio.sleep(2)
        return None
    
    async def fetch_futures_data(self, instrument_key: str, session: aiohttp.ClientSession) -> pd.DataFrame:
        """Fetch futures candle data"""
        now = datetime.now(IST)
        enc_key = urllib.parse.quote(instrument_key, safe='')
        
        # Try INTRADAY first
        intraday_url = f"https://api.upstox.com/v2/historical-candle/intraday/{enc_key}/1minute"
        logger.info(f"📊 Fetching futures data...")
        
        data = await self.fetch_with_retry(intraday_url, session)
        
        if data and data.get('status') == 'success':
            candles = data.get('data', {}).get('candles', [])
            if candles:
                try:
                    df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'oi'])
                    df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert(IST)
                    df = df.sort_values('ts').set_index('ts')
                    df = df.tail(100)
                    logger.info(f"✅ Futures: {len(df)} candles")
                    return df
                except Exception as e:
                    logger.error(f"Processing error: {e}")
        
        # Fallback to historical
        to_date = now.strftime('%Y-%m-%d')
        from_date = (now - timedelta(days=5)).strftime('%Y-%m-%d')
        hist_url = f"https://api.upstox.com/v2/historical-candle/{enc_key}/1minute/{to_date}/{from_date}"
        
        data = await self.fetch_with_retry(hist_url, session)
        
        if data and data.get('status') == 'success':
            candles = data.get('data', {}).get('candles', [])
            if candles:
                try:
                    df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'oi'])
                    df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert(IST)
                    df = df.sort_values('ts').set_index('ts')
                    today = now.date()
                    df = df[df.index.date == today]
                    df = df.tail(100)
                    if not df.empty:
                        logger.info(f"✅ Futures (historical): {len(df)} candles")
                        return df
                except Exception as e:
                    logger.error(f"Processing error: {e}")
        
        logger.warning("⚠️ No futures data")
        return pd.DataFrame()
    
    async def get_market_data(self) -> Tuple[pd.DataFrame, Dict[int, dict], str, float, float, float]:
        """
        Fetch market data with PROPER SYMBOLS:
        1. Spot Price: NSE_INDEX|Nifty 50
        2. Futures: NSE_FO|43650 (from instruments)
        3. Options: NSE_INDEX|Nifty 50
        
        WITH 2 SECOND DELAYS BETWEEN REQUESTS
        """
        await self.instruments_cache.load_instruments()
        
        async with aiohttp.ClientSession() as session:
            spot_price = 0
            futures_price = 0
            df = pd.DataFrame()
            strike_data = {}
            total_options_volume = 0
            
            # ===== 1. FETCH SPOT PRICE =====
            logger.info("📍 Fetching Spot Price...")
            enc_spot = urllib.parse.quote(self.spot_symbol, safe='')
            ltp_url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={enc_spot}"
            
            ltp_data = await self.fetch_with_retry(ltp_url, session)
            
            if ltp_data and ltp_data.get('status') == 'success':
                data = ltp_data.get('data', {})
                
                # Correct extraction - response structure is:
                # {"data": {"NSE_INDEX|Nifty 50": {"last_price": 23900.50}}}
                if self.spot_symbol in data:
                    spot_data = data[self.spot_symbol]
                    spot_price = spot_data.get('last_price', 0)
                    logger.info(f"✅ Spot: ₹{spot_price:.2f}")
                else:
                    logger.error(f"❌ Spot symbol '{self.spot_symbol}' not found")
                    logger.error(f"Available keys: {list(data.keys())}")
                    # Try first available key as fallback
                    if data:
                        first_key = list(data.keys())[0]
                        spot_data = data[first_key]
                        spot_price = spot_data.get('last_price', 0)
                        logger.warning(f"⚠️ Using fallback key: {first_key} = ₹{spot_price:.2f}")
            else:
                logger.error(f"❌ LTP API failed")
                if ltp_data:
                    logger.error(f"Response: {json.dumps(ltp_data, indent=2)}")
            
            if spot_price == 0:
                logger.error("❌ Cannot proceed without spot price")
                return df, strike_data, "", 0, 0, 0
            
            # 2 SECOND DELAY
            await asyncio.sleep(API_DELAY)
            
            # ===== 2. FETCH FUTURES DATA =====
            futures_key = self.instruments_cache.get_futures_key(self.index_name)
            
            if futures_key:
                logger.info(f"🔍 Fetching Futures: {futures_key}")
                df = await self.fetch_futures_data(futures_key, session)
                
                if not df.empty:
                    futures_price = df['close'].iloc[-1]
                    logger.info(f"✅ Futures: ₹{futures_price:.2f}")
                else:
                    logger.warning(f"⚠️ Using spot as futures")
                    futures_price = spot_price
            else:
                logger.warning(f"⚠️ Futures key not found, using spot")
                futures_price = spot_price
            
            # 2 SECOND DELAY
            await asyncio.sleep(API_DELAY)
            
            # ===== 3. FETCH OPTION CHAIN =====
            logger.info("🔗 Fetching Option Chain...")
            expiry = get_expiry_date(self.index_name)
            chain_url = f"https://api.upstox.com/v2/option/chain?instrument_key={enc_spot}&expiry_date={expiry}"
            
            strike_gap = self.index_config['strike_gap']
            atm_strike = round(spot_price / strike_gap) * strike_gap
            min_strike = atm_strike - (2 * strike_gap)
            max_strike = atm_strike + (2 * strike_gap)
            logger.info(f"📊 ATM: {atm_strike} | Range: {min_strike}-{max_strike} | Expiry: {expiry}")
            
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
            else:
                logger.error(f"❌ Option chain failed: {chain_data}")
            
            return df, strike_data, expiry, spot_price, futures_price, total_options_volume

class StrikeAnalyzer:
    """Technical analysis engine"""
    def __init__(self):
        self.volume_history = []
    
    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate VWAP"""
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
        """Calculate ATR"""
        if len(df) < period:
            return 30
        df_copy = df.tail(period).copy()
        df_copy['h-l'] = df_copy['high'] - df_copy['low']
        df_copy['h-pc'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['l-pc'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        df_copy['tr'] = df_copy[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return df_copy['tr'].mean()
    
    def get_candle_info(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Get candle color and size"""
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
        """Detect volume spikes"""
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
        """Calculate PCR"""
        total_ce = sum(data['ce_oi'] for data in strike_data.values())
        total_pe = sum(data['pe_oi'] for data in strike_data.values())
        return total_pe / total_ce if total_ce > 0 else 1.0
    
    def analyze_atm_battle(self, strike_data: Dict[int, dict], atm_strike: int, redis_brain) -> Tuple[float, float]:
        """Analyze ATM OI changes"""
        if atm_strike not in strike_data:
            return 0, 0
        current = strike_data[atm_strike]
        ce_15m, pe_15m = redis_brain.get_strike_oi_change(atm_strike, current, 15)
        logger.info(f"⚔️ ATM {atm_strike}: CE={ce_15m:+.1f}% | PE={pe_15m:+.1f}%")
        return ce_15m, pe_15m
    
    def check_momentum(self, df: pd.DataFrame, direction: str = 'bullish') -> bool:
        """Check momentum"""
        if df.empty or len(df) < 3:
            return False
        last_3 = df.tail(3)
        if direction == 'bullish':
            return sum(last_3['close'] > last_3['open']) >= 2
        else:
            return sum(last_3['close'] < last_3['open']) >= 2

class StrikeMasterBot:
    """Main bot orchestrator"""
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
        """Execute one analysis cycle"""
        if not is_data_collection_time():
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 STRIKE MASTER SCAN - {self.index_config['name']}")
        logger.info(f"{'='*60}")
        
        try:
            df, strike_data, expiry, spot, futures, vol = await self.feed.get_market_data()
            
            if df.empty or not strike_data or spot == 0:
                logger.warning("⚠️ Incomplete data - skipping cycle")
                return
            
            vwap = self.analyzer.calculate_vwap(df)
            atr = self.analyzer.calculate_atr(df)
            pcr = self.analyzer.calculate_focused_pcr(strike_data)
            candle_color, candle_size = self.analyzer.get_candle_info(df)
            has_vol_spike, vol_mult = self.analyzer.check_volume_surge(vol)
            
            strike_gap = self.index_config['strike_gap']
            atm_strike = round(spot / strike_gap) * strike_gap
            atm_ce_15m, atm_pe_15m = self.analyzer.analyze_atm_battle(strike_data, atm_strike, self.redis)
            
            total_ce = sum(d['ce_oi'] for d in strike_data.values())
            total_pe = sum(d['pe_oi'] for d in strike_data.values())
            ce_total_15m, pe_total_15m = self.redis.get_total_oi_change(total_ce, total_pe, 15)
            
            self.redis.save_strike_snapshot(strike_data)
            self.redis.save_total_oi_snapshot(total_ce, total_pe)
            
            logger.info(f"📊 Spot: ₹{spot:.2f} | Futures: ₹{futures:.2f} | PCR: {pcr:.2f}")
            logger.info(f"📈 VWAP: ₹{vwap:.2f} | ATR: {atr:.1f} | Candle: {candle_color}")
            logger.info(f"📉 OI 15m: CE={ce_total_15m:+.1f}% PE={pe_total_15m:+.1f}%")
            
            if is_tradeable_time():
                signal = self.generate_signal(
                    spot, futures, vwap, abs(futures - vwap), pcr, atr,
                    ce_total_15m, pe_total_15m,
                    atm_ce_15m, atm_pe_15m,
                    candle_color, candle_size,
                    has_vol_spike, vol_mult, df
                )
                if signal:
                    await self.send_alert(signal)
            else:
                logger.info("📊 Data collected (avoid window)")
            
            logger.info(f"{'='*60}\n")
        
        except Exception as e:
            logger.error(f"💥 Error in cycle: {e}")
            logger.exception("Full traceback:")
    
    def generate_signal(self, spot_price, futures_price, vwap, vwap_distance, pcr, atr,
                       ce_total_15m, pe_total_15m,
                       atm_ce_change, atm_pe_change, candle_color, candle_size,
                       has_vol_spike, vol_mult, df) -> Optional[Signal]:
        """Generate trading signals"""
        
        strike_gap = self.index_config['strike_gap']
        strike = round(spot_price / strike_gap) * strike_gap
        stop_loss_points = int(atr * ATR_SL_MULTIPLIER)
        target_points = int(atr * ATR_TARGET_MULTIPLIER)
        
        # Dynamic targets
        if abs(ce_total_15m) >= OI_THRESHOLD_STRONG or abs(atm_ce_change) >= OI_THRESHOLD_STRONG:
            target_points = max(target_points, 80)
        elif abs(ce_total_15m) >= OI_THRESHOLD_MEDIUM or abs(atm_ce_change) >= OI_THRESHOLD_MEDIUM:
            target_points = max(target_points, 50)
        
        # BULLISH SIGNAL: CE Unwinding
        if ce_total_15m < -OI_THRESHOLD_MEDIUM or atm_ce_change < -ATM_OI_THRESHOLD:
            checks = {
                "CE OI Unwinding": ce_total_15m < -OI_THRESHOLD_MEDIUM,
                "ATM CE Unwinding": atm_ce_change < -ATM_OI_THRESHOLD,
                "Price > VWAP": futures_price > vwap,
                "GREEN Candle": candle_color == 'GREEN'
            }
            
            bonus = {
                "Big Candle": candle_size >= MIN_CANDLE_SIZE,
                "Far from VWAP": vwap_distance >= VWAP_BUFFER,
                "Bullish PCR": pcr > PCR_BULLISH,
                "Volume Spike": has_vol_spike,
                "Momentum": self.analyzer.check_momentum(df, 'bullish')
            }
            
            passed = sum(checks.values())
            bonus_passed = sum(bonus.values())
            
            if passed == 4:
                confidence = 75 + (bonus_passed * 4)
                logger.info(f"🚀 CE SIGNAL! Confidence: {confidence}%")
                
                return Signal(
                    type="CE_BUY",
                    reason=f"Call Unwinding (ATM: {atm_ce_change:.1f}%)",
                    confidence=min(confidence, 95),
                    spot_price=spot_price,
                    futures_price=futures_price,
                    strike=strike,
                    target_points=target_points,
                    stop_loss_points=stop_loss_points,
                    pcr=pcr,
                    candle_color=candle_color,
                    volume_surge=vol_mult,
                    oi_15m=ce_total_15m,
                    atm_ce_change=atm_ce_change,
                    atm_pe_change=atm_pe_change,
                    atr=atr,
                    timestamp=datetime.now(IST)
                )
        
        # BEARISH SIGNAL: PE Unwinding
        if pe_total_15m < -OI_THRESHOLD_MEDIUM or atm_pe_change < -ATM_OI_THRESHOLD:
            if abs(pe_total_15m) >= OI_THRESHOLD_STRONG or abs(atm_pe_change) >= OI_THRESHOLD_STRONG:
                target_points = max(target_points, 80)
            
            checks = {
                "PE OI Unwinding": pe_total_15m < -OI_THRESHOLD_MEDIUM,
                "ATM PE Unwinding": atm_pe_change < -ATM_OI_THRESHOLD,
                "Price < VWAP": futures_price < vwap,
                "RED Candle": candle_color == 'RED'
            }
            
            bonus = {
                "Big Candle": candle_size >= MIN_CANDLE_SIZE,
                "Far from VWAP": vwap_distance >= VWAP_BUFFER,
                "Bearish PCR": pcr < PCR_BEARISH,
                "Volume Spike": has_vol_spike,
                "Momentum": self.analyzer.check_momentum(df, 'bearish')
            }
            
            passed = sum(checks.values())
            bonus_passed = sum(bonus.values())
            
            if passed == 4:
                confidence = 75 + (bonus_passed * 4)
                logger.info(f"🔻 PE SIGNAL! Confidence: {confidence}%")
                
                return Signal(
                    type="PE_BUY",
                    reason=f"Put Unwinding (ATM: {atm_pe_change:.1f}%)",
                    confidence=min(confidence, 95),
                    spot_price=spot_price,
                    futures_price=futures_price,
                    strike=strike,
                    target_points=target_points,
                    stop_loss_points=stop_loss_points,
                    pcr=pcr,
                    candle_color=candle_color,
                    volume_surge=vol_mult,
                    oi_15m=pe_total_15m,
                    atm_ce_change=atm_ce_change,
                    atm_pe_change=atm_pe_change,
                    atr=atr,
                    timestamp=datetime.now(IST)
                )
        
        return None
    
    async def send_alert(self, s: Signal):
        """Send Telegram alert"""
        if self.last_alert_time:
            elapsed = (datetime.now(IST) - self.last_alert_time).seconds
            if elapsed < self.alert_cooldown:
                logger.info(f"⏸️ Cooldown: {self.alert_cooldown - elapsed}s remaining")
                return
        
        self.last_alert_time = datetime.now(IST)
        
        emoji = "📞" if s.type == "CE_BUY" else "📉"
        
        if s.type == "CE_BUY":
            entry = s.spot_price
            target = entry + s.target_points
            stop_loss = entry - s.stop_loss_points
        else:
            entry = s.spot_price
            target = entry - s.target_points
            stop_loss = entry + s.stop_loss_points
        
        mode = "🔔 ALERT ONLY" if ALERT_ONLY_MODE else "🟢 LIVE"
        timestamp_str = s.timestamp.strftime('%d-%b %I:%M %p')
        
        msg = f"""
{emoji} {self.index_config['name']} STRIKE MASTER V12
{mode}

📊 SIGNAL: {s.type}
💰 Entry: ₹{entry:.1f}
🎯 Target: ₹{target:.1f} (+{s.target_points})
🛑 Stop Loss: ₹{stop_loss:.1f}
⚡ Strike: {s.strike}

📈 Reason: {s.reason}
🎲 Confidence: {s.confidence}%
📊 PCR: {s.pcr:.2f}
🕐 Time: {timestamp_str}

🔥 ATR: {s.atr:.1f} | Candle: {s.candle_color}
"""
        
        logger.info(f"🚨 SIGNAL: {s.type} @ ₹{entry:.1f}")
        
        if self.telegram:
            try:
                await self.telegram.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                logger.info("✅ Telegram alert sent")
            except Exception as e:
                logger.error(f"❌ Telegram error: {e}")
    
    async def send_startup_message(self):
        """Send startup notification"""
        now = datetime.now(IST)
        startup_time = now.strftime('%d-%b %I:%M %p')
        mode = "🔔 ALERT ONLY" if ALERT_ONLY_MODE else "🟢 LIVE TRADING"
        
        msg = f"""
🚀 STRIKE MASTER V12.1 ONLINE

⏰ {startup_time}
📊 {self.index_config['name']}
{mode}

⚙️ Config:
• Strike Gap: {self.index_config['strike_gap']}
• Expiry: {self.index_config['expiry_type']}
• Scan Interval: {SCAN_INTERVAL}s
• API Delay: {API_DELAY}s

🔥 Features:
✅ Proper Symbol Handling
✅ 2 Sec API Delays
✅ Real-time OI Analysis
✅ Multi-factor Signals

Ready to scan! 🎯
"""
        
        if self.telegram:
            try:
                await self.telegram.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                logger.info("✅ Startup message sent")
            except Exception as e:
                logger.error(f"❌ Startup error: {e}")

async def main():
    """Main entry point"""
    if ACTIVE_INDEX not in INDICES:
        logger.error(f"❌ Invalid ACTIVE_INDEX: {ACTIVE_INDEX}")
        return
    
    bot = StrikeMasterBot(ACTIVE_INDEX)
    
    logger.info("=" * 80)
    logger.info("🚀 STRIKE MASTER V12.1 - FIXED SYMBOLS + 2 SEC DELAYS")
    logger.info("=" * 80)
    logger.info(f"📊 Index: {bot.index_config['name']}")
    logger.info(f"🔔 Mode: {'ALERT ONLY' if ALERT_ONLY_MODE else 'LIVE TRADING'}")
    logger.info(f"⏱️ Scan Interval: {SCAN_INTERVAL}s")
    logger.info(f"⏳ API Delay: {API_DELAY}s between requests")
    logger.info(f"🔥 Features: Proper Symbols | Sequential Fetch | Original Expiry Logic")
    logger.info("=" * 80)
    logger.info("")
    
    await bot.send_startup_message()
    
    iteration = 0
    
    while True:
        try:
            now = datetime.now(IST).time()
            
            if time(9, 15) <= now <= time(15, 30):
                iteration += 1
                logger.info(f"🔄 Cycle #{iteration}")
                
                await bot.run_cycle()
                await asyncio.sleep(SCAN_INTERVAL)
            else:
                logger.info("😴 Market closed - waiting...")
                await asyncio.sleep(300)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped by user")
            break
        
        except Exception as e:
            logger.error(f"💥 Error in main loop: {e}")
            logger.exception("Full traceback:")
            logger.info("⏳ Retrying in 30s...")
            await asyncio.sleep(30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Shutdown complete")
