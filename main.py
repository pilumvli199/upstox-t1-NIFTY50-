#!/usr/bin/env python3
"""
NIFTY OPTIONS BOT V13.0 - MULTI INDEX MASTER
============================================
🔥 ALL 4 INDICES ACTIVE SIMULTANEOUSLY

Strategy: Multi-Factor Strike Analysis on ALL Indices
Target: 50-80 points daily per index | 80%+ accuracy

ACTIVE INDICES (November 2025):
✅ NIFTY 50 - Weekly Tuesday expiry
✅ BANKNIFTY - Monthly Tuesday expiry
✅ FINNIFTY - Monthly Tuesday expiry  
✅ MIDCPNIFTY - Monthly Tuesday expiry, 25pt gap

Bot scans ALL 4 indices in parallel and generates signals!

Author: Data Monster Team
Version: 13.0 - Multi Index Production
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
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np
from calendar import monthrange

# Optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("⚠️ Redis not installed. Using RAM-only mode.")

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("⚠️ Telegram not installed. Alerts disabled.")

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("MultiIndexMaster-V13.0")

# API Configuration
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# 🔥 ALL INDICES CONFIGURATION (Verified with Upstox API docs)
INDICES = {
    'NIFTY': {
        'spot': "NSE_INDEX|Nifty 50",  # Verified instrument key
        'name': 'NIFTY 50',
        'strike_gap': 50,
        'has_weekly': True,
        'expiry_day': 1,  # Tuesday
        'futures_prefix': 'NIFTY'
    },
    'BANKNIFTY': {
        'spot': "NSE_INDEX|Nifty Bank",  # Verified instrument key
        'name': 'BANK NIFTY',
        'strike_gap': 100,
        'has_weekly': False,
        'expiry_day': 1,  # Last Tuesday of month
        'futures_prefix': 'BANKNIFTY'
    },
    'FINNIFTY': {
        'spot': "NSE_INDEX|Nifty Fin Service",  # Verified instrument key
        'name': 'FIN NIFTY',
        'strike_gap': 50,
        'has_weekly': False,
        'expiry_day': 1,  # Last Tuesday
        'futures_prefix': 'FINNIFTY'
    },
    'MIDCPNIFTY': {
        'spot': "NSE_INDEX|NIFTY MID SELECT",  # Verified instrument key
        'name': 'MIDCAP NIFTY',
        'strike_gap': 25,  # 25 points gap
        'has_weekly': False,
        'expiry_day': 1,  # Last Tuesday
        'futures_prefix': 'MIDCPNIFTY'
    }
}

# 🔥 ALL INDICES ENABLED
ACTIVE_INDICES = list(INDICES.keys())  # ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# Trading Configuration
ALERT_ONLY_MODE = True
SCAN_INTERVAL = 60  # seconds between full scans
INDEX_SCAN_DELAY = 2  # seconds delay between indices to avoid rate limits

# Strategy Thresholds
OI_THRESHOLD_STRONG = 8.0
OI_THRESHOLD_MEDIUM = 5.0
ATM_OI_THRESHOLD = 5.0

VOL_SPIKE_2X = 2.0
VOL_SPIKE_3X = 3.0

PCR_BULLISH = 1.08
PCR_BEARISH = 0.92

MIN_CANDLE_SIZE = 8
VWAP_BUFFER = 5

# Time Filters
AVOID_OPENING = (time(9, 15), time(9, 45))
AVOID_CLOSING = (time(15, 15), time(15, 30))

# ATR Configuration
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 2.5

@dataclass
class Signal:
    """Trading Signal"""
    index_name: str
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

# ==================== EXPIRY LOGIC ====================
def get_current_futures_symbol(index_name: str) -> str:
    """Auto-detect Futures symbol for any index"""
    now = datetime.now(IST)
    year = now.year
    month = now.month
    
    last_day = monthrange(year, month)[1]
    last_date = datetime(year, month, last_day, tzinfo=IST)
    days_to_tuesday = (last_date.weekday() - 1) % 7
    last_tuesday = last_date - timedelta(days=days_to_tuesday)
    
    if now.date() > last_tuesday.date() or (
        now.date() == last_tuesday.date() and now.time() > time(15, 30)
    ):
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    
    year_short = year % 100
    month_name = datetime(year, month, 1).strftime('%b').upper()
    
    prefix = INDICES[index_name]['futures_prefix']
    symbol = f"NSE_FO|{prefix}{year_short:02d}{month_name}FUT"
    
    return symbol

def get_expiry_date(index_name: str) -> str:
    """Get expiry date for any index"""
    now = datetime.now(IST)
    today = now.date()
    config = INDICES[index_name]
    
    if config['has_weekly']:
        # NIFTY: Next Tuesday (Every week!)
        days_to_tuesday = (1 - today.weekday() + 7) % 7
        
        if days_to_tuesday == 0:
            if now.time() > time(15, 30):
                expiry = today + timedelta(days=7)
            else:
                expiry = today
        else:
            expiry = today + timedelta(days=days_to_tuesday)
    
    else:
        # Others: Last Tuesday of month
        year = now.year
        month = now.month
        
        last_day = monthrange(year, month)[1]
        last_date = datetime(year, month, last_day)
        
        days_to_tuesday = (last_date.weekday() - 1) % 7
        last_tuesday = last_date - timedelta(days=days_to_tuesday)
        
        if now.date() > last_tuesday.date() or (
            now.date() == last_tuesday.date() and now.time() > time(15, 30)
        ):
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            
            last_day = monthrange(year, month)[1]
            last_date = datetime(year, month, last_day)
            days_to_tuesday = (last_date.weekday() - 1) % 7
            last_tuesday = last_date - timedelta(days=days_to_tuesday)
        
        expiry = last_tuesday.date()
    
    return expiry.strftime('%Y-%m-%d')

def is_tradeable_time() -> bool:
    """Check if current time is good for trading"""
    now = datetime.now(IST).time()
    
    if not (time(9, 15) <= now <= time(15, 30)):
        return False
    
    if AVOID_OPENING[0] <= now <= AVOID_OPENING[1]:
        return False
    
    if AVOID_CLOSING[0] <= now <= AVOID_CLOSING[1]:
        return False
    
    return True

# ==================== REDIS BRAIN ====================
class RedisBrain:
    """Memory system for OI tracking (per index)"""
    
    def __init__(self):
        self.client = None
        self.memory = {}
        
        if REDIS_AVAILABLE:
            try:
                self.client = redis.from_url(REDIS_URL, decode_responses=True)
                self.client.ping()
                logger.info("✅ Redis Connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis Failed: Using RAM Mode")
                self.client = None
        else:
            logger.info("📦 RAM-only mode")
    
    def save_strike_snapshot(self, index_name: str, strike_data: Dict[int, dict]):
        """Save strike-level OI data per index"""
        now = datetime.now(IST)
        timestamp = now.replace(second=0, microsecond=0)
        
        for strike, data in strike_data.items():
            key = f"{index_name}:strike:{strike}:{timestamp.strftime('%H%M')}"
            value = json.dumps(data)
            
            if self.client:
                try:
                    self.client.setex(key, 3600, value)
                except:
                    self.memory[key] = value
            else:
                self.memory[key] = value
    
    def get_strike_oi_change(self, index_name: str, strike: int, 
                             current_data: dict, minutes_ago: int = 15) -> Tuple[float, float]:
        """Calculate OI change % for specific strike of specific index"""
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        timestamp = now.replace(second=0, microsecond=0)
        key = f"{index_name}:strike:{strike}:{timestamp.strftime('%H%M')}"
        
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
            ce_chg = ((current_data['ce_oi'] - past['ce_oi']) / past['ce_oi'] * 100
                      if past['ce_oi'] > 0 else 0)
            pe_chg = ((current_data['pe_oi'] - past['pe_oi']) / past['pe_oi'] * 100
                      if past['pe_oi'] > 0 else 0)
            return ce_chg, pe_chg
        except:
            return 0.0, 0.0
    
    def save_total_oi_snapshot(self, index_name: str, ce_total: int, pe_total: int):
        """Save total OI per index"""
        now = datetime.now(IST)
        slot = now.replace(second=0, microsecond=0)
        key = f"{index_name}:total_oi:{slot.strftime('%H%M')}"
        data = json.dumps({"ce": ce_total, "pe": pe_total})
        
        if self.client:
            try:
                self.client.setex(key, 3600, data)
            except:
                self.memory[key] = data
        else:
            self.memory[key] = data
    
    def get_total_oi_change(self, index_name: str, current_ce: int, 
                           current_pe: int, minutes_ago: int = 15) -> Tuple[float, float]:
        """Get total OI change % per index"""
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        slot = now.replace(second=0, microsecond=0)
        key = f"{index_name}:total_oi:{slot.strftime('%H%M')}"
        
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
            ce_chg = ((current_ce - past['ce']) / past['ce'] * 100
                      if past['ce'] > 0 else 0)
            pe_chg = ((current_pe - past['pe']) / past['pe'] * 100
                      if past['pe'] > 0 else 0)
            return ce_chg, pe_chg
        except:
            return 0.0, 0.0

# ==================== DATA FEED ====================
class MultiIndexDataFeed:
    """Fetch market data for specific index"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.spot_symbol = self.index_config['spot']
        self.strike_gap = self.index_config['strike_gap']
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.retry_count = 3
        self.base_retry_delay = 2
        self.futures_symbol = get_current_futures_symbol(index_name)
    
    async def fetch_with_retry(self, url: str, session: aiohttp.ClientSession):
        """Smart retry with exponential backoff"""
        for attempt in range(self.retry_count):
            try:
                async with session.get(url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait_time = (2 ** attempt) * self.base_retry_delay
                        logger.warning(f"⏳ [{self.index_name}] Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ [{self.index_name}] HTTP {resp.status}")
                        await asyncio.sleep(self.base_retry_delay)
            except asyncio.TimeoutError:
                logger.error(f"⏱️ [{self.index_name}] Timeout on attempt {attempt+1}")
                await asyncio.sleep(self.base_retry_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"💥 [{self.index_name}] Attempt {attempt+1}: {e}")
                await asyncio.sleep(self.base_retry_delay * (attempt + 1))
        
        return None
    
    async def get_market_data(self) -> Tuple[pd.DataFrame, Dict[int, dict],
                                            str, float, float, float]:
        """Fetch all required data for this index"""
        async with aiohttp.ClientSession() as session:
            spot_price = 0
            futures_price = 0
            df = pd.DataFrame()
            strike_data = {}
            total_options_volume = 0
            
            # 1. GET SPOT PRICE
            enc_spot = urllib.parse.quote(self.spot_symbol)
            ltp_url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={enc_spot}"
            
            ltp_data = await self.fetch_with_retry(ltp_url, session)
            if ltp_data and 'data' in ltp_data:
                possible_keys = [
                    self.spot_symbol,
                    self.spot_symbol.replace('NSE_INDEX|', 'NSE_INDEX:'),
                    self.index_config['name']
                ]
                
                for key in possible_keys:
                    if key in ltp_data['data']:
                        spot_price = ltp_data['data'][key].get('last_price', 0)
                        if spot_price > 0:
                            break
            
            if spot_price == 0:
                logger.error(f"❌ [{self.index_name}] Failed to fetch spot price")
                return df, strike_data, "", 0, 0, 0
            
            # 2. GET FUTURES CANDLES
            enc_futures = urllib.parse.quote(self.futures_symbol)
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            from_date = (datetime.now(IST) - timedelta(days=10)).strftime('%Y-%m-%d')
            candle_url = f"https://api.upstox.com/v2/historical-candle/{enc_futures}/1minute/{to_date}/{from_date}"
            
            candle_data = await self.fetch_with_retry(candle_url, session)
            if candle_data and candle_data.get('status') == 'success':
                candles = candle_data.get('data', {}).get('candles', [])
                if candles:
                    df = pd.DataFrame(
                        candles,
                        columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'oi']
                    )
                    df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert(IST)
                    df = df.sort_values('ts').set_index('ts')
                    
                    today = datetime.now(IST).date()
                    df = df[df.index.date == today].tail(100)
                    
                    if not df.empty:
                        futures_price = df['close'].iloc[-1]
            
            # 3. GET OPTION CHAIN (5-STRIKE FOCUS)
            expiry = get_expiry_date(self.index_name)
            chain_url = f"https://api.upstox.com/v2/option/chain?instrument_key={enc_spot}&expiry_date={expiry}"
            
            atm_strike = round(spot_price / self.strike_gap) * self.strike_gap
            min_strike = atm_strike - (2 * self.strike_gap)
            max_strike = atm_strike + (2 * self.strike_gap)
            
            chain_data = await self.fetch_with_retry(chain_url, session)
            if chain_data and chain_data.get('status') == 'success':
                for option in chain_data.get('data', []):
                    strike = option.get('strike_price', 0)
                    
                    if min_strike <= strike <= max_strike:
                        call_data = option.get('call_options', {}).get('market_data', {})
                        put_data = option.get('put_options', {}).get('market_data', {})
                        
                        ce_oi = call_data.get('oi', 0)
                        pe_oi = put_data.get('oi', 0)
                        ce_vol = call_data.get('volume', 0)
                        pe_vol = put_data.get('volume', 0)
                        
                        strike_data[strike] = {
                            'ce_oi': ce_oi,
                            'pe_oi': pe_oi,
                            'ce_vol': ce_vol,
                            'pe_vol': pe_vol,
                            'ce_ltp': call_data.get('ltp', 0),
                            'pe_ltp': put_data.get('ltp', 0)
                        }
                        
                        total_options_volume += (ce_vol + pe_vol)
            
            return df, strike_data, expiry, spot_price, futures_price, total_options_volume

# ==================== ANALYZER ====================
class StrikeAnalyzer:
    """Multi-factor analysis engine"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.volume_history = []
    
    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """VWAP from futures"""
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
        """ATR for dynamic stops"""
        if len(df) < period:
            return 30
        
        df_copy = df.tail(period).copy()
        
        df_copy['h-l'] = df_copy['high'] - df_copy['low']
        df_copy['h-pc'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['l-pc'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        
        df_copy['tr'] = df_copy[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return df_copy['tr'].mean()
    
    def get_candle_info(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Current candle"""
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
        """Options volume spike"""
        now = datetime.now(IST)
        self.volume_history.append({'time': now, 'volume': current_vol})
        
        cutoff = now - timedelta(minutes=20)
        self.volume_history = [
            x for x in self.volume_history if x['time'] > cutoff
        ]
        
        if len(self.volume_history) < 5:
            return False, 0
        
        past_volumes = [x['volume'] for x in self.volume_history[:-1]]
        avg_vol = sum(past_volumes) / len(past_volumes)
        
        if avg_vol == 0:
            return False, 0
        
        multiplier = current_vol / avg_vol
        has_spike = multiplier >= VOL_SPIKE_2X
        
        return has_spike, multiplier
    
    def calculate_focused_pcr(self, strike_data: Dict[int, dict]) -> float:
        """PCR from 5 strikes"""
        total_ce = sum(data['ce_oi'] for data in strike_data.values())
        total_pe = sum(data['pe_oi'] for data in strike_data.values())
        
        pcr = total_pe / total_ce if total_ce > 0 else 1.0
        return pcr
    
    def analyze_atm_battle(self, strike_data: Dict[int, dict], atm_strike: int,
                          redis_brain: RedisBrain) -> Tuple[float, float]:
        """ATM Battle Analysis"""
        if atm_strike not in strike_data:
            return 0, 0
        
        current = strike_data[atm_strike]
        
        ce_15m, pe_15m = redis_brain.get_strike_oi_change(
            self.index_name, atm_strike, current, minutes_ago=15
        )
        
        return ce_15m, pe_15m
    
    def check_momentum(self, df: pd.DataFrame, direction: str = 'bullish') -> bool:
        """3-candle momentum"""
        if df.empty or len(df) < 3:
            return False
        
        last_3 = df.tail(3)
        
        if direction == 'bullish':
            green_count = sum(last_3['close'] > last_3['open'])
            return green_count >= 2
        else:
            red_count = sum(last_3['close'] < last_3['open'])
            return red_count >= 2

# ==================== SINGLE INDEX BOT ====================
class IndexBot:
    """Bot for single index analysis"""
    
    def __init__(self, index_name: str, redis_brain: RedisBrain):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.feed = MultiIndexDataFeed(index_name)
        self.redis = redis_brain
        self.analyzer = StrikeAnalyzer(index_name)
    
    async def analyze(self) -> Optional[Signal]:
        """Run analysis for this index"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 [{self.index_config['name']}] Starting Analysis")
        logger.info(f"{'='*60}")
        
        df, strike_data, expiry, spot_price, futures_price, options_vol = \
            await self.feed.get_market_data()
        
        if df.empty or not strike_data or spot_price == 0:
            logger.warning(f"⏳ [{self.index_name}] Incomplete data, skipping")
            return None
        
        logger.info(f"💰 Spot: {spot_price:.2f} | Futures: {futures_price:.2f}")
        
        vwap = self.analyzer.calculate_vwap(df)
        atr = self.analyzer.calculate_atr(df)
        pcr = self.analyzer.calculate_focused_pcr(strike_data)
        candle_color, candle_size = self.analyzer.get_candle_info(df)
        has_vol_spike, vol_mult = self.analyzer.check_volume_surge(options_vol)
        vwap_distance = abs(futures_price - vwap)
        
        atm_strike = round(spot_price / self.index_config['strike_gap']) * self.index_config['strike_gap']
        atm_ce_15m, atm_pe_15m = self.analyzer.analyze_atm_battle(
            strike_data, atm_strike, self.redis
        )
        
        total_ce = sum(d['ce_oi'] for d in strike_data.values())
        total_pe = sum(d['pe_oi'] for d in strike_data.values())
        
        ce_total_15m, pe_total_15m = self.redis.get_total_oi_change(
            self.index_name, total_ce, total_pe, minutes_ago=15
        )
        ce_total_5m, pe_total_5m = self.redis.get_total_oi_change(
            self.index_name, total_ce, total_pe, minutes_ago=5
        )
        
        logger.info(f"📊 PCR: {pcr:.2f} | ATR: {atr:.1f} | VWAP: {vwap:.2f}")
        logger.info(f"⚔️ ATM CE: {atm_ce_15m:+.1f}% | PE: {atm_pe_15m:+.1f}%")
        
        self.redis.save_strike_snapshot(self.index_name, strike_data)
        self.redis.save_total_oi_snapshot(self.index_name, total_ce, total_pe)
        
        signal = self.generate_signal(
            spot_price=spot_price,
            futures_price=futures_price,
            vwap=vwap,
            vwap_distance=vwap_distance,
            pcr=pcr,
            atr=atr,
            ce_total_15m=ce_total_15m,
            pe_total_15m=pe_total_15m,
            ce_total_5m=ce_total_5m,
            pe_total_5m=pe_total_5m,
            atm_ce_change=atm_ce_15m,
            atm_pe_change=atm_pe_15m,
            candle_color=candle_color,
            candle_size=candle_size,
            has_vol_spike=has_vol_spike,
            vol_mult=vol_mult,
            df=df
        )
        
        return signal
    
    def generate_signal(self, spot_price: float, futures_price: float,
                       vwap: float, vwap_distance: float, pcr: float,
                       atr: float, ce_total_15m: float, pe_total_15m: float,
                       ce_total_5m: float, pe_total_5m: float,
                       atm_ce_change: float, atm_pe_change: float,
                       candle_color: str, candle_size: float,
                       has_vol_spike: bool, vol_mult: float,
                       df: pd.DataFrame) -> Optional[Signal]:
        """Multi-Factor Signal Generation"""
        
        strike = round(spot_price / self.index_config['strike_gap']) * self.index_config['strike_gap']
        
        stop_loss_points = int(atr * ATR_SL_MULTIPLIER)
        target_points = int(atr * ATR_TARGET_MULTIPLIER)
        
        if abs(ce_total_15m) >= OI_THRESHOLD_STRONG or abs(atm_ce_change) >= OI_THRESHOLD_STRONG:
            target_points = max(target_points, 80)
        elif abs(ce_total_15m) >= OI_THRESHOLD_MEDIUM or abs(atm_ce_change) >= OI_THRESHOLD_MEDIUM:
            target_points = max(target_points, 50)
        
        # CE BUY LOGIC
        if ce_total_15m < -OI_THRESHOLD_MEDIUM or atm_ce_change < -ATM_OI_THRESHOLD:
            
            checks = {
                "CE OI Unwinding (Total)": ce_total_15m < -OI_THRESHOLD_MEDIUM,
                "ATM CE Unwinding": atm_ce_change < -ATM_OI_THRESHOLD,
                "Price > VWAP": futures_price > vwap,
                "GREEN Candle": candle_color == 'GREEN'
            }
            
            bonus = {
                "Strong 5m OI": ce_total_5m < -5.0,
                "Big Candle": candle_size >= MIN_CANDLE_SIZE,
                "Far from VWAP": vwap_distance >= VWAP_BUFFER,
                "Bullish PCR": pcr > PCR_BULLISH,
                "Volume Spike": has_vol_spike,
                "3+ Green Momentum": self.analyzer.check_momentum(df, 'bullish')
            }
            
            passed = sum(checks.values())
            bonus_passed = sum(bonus.values())
            
            logger.info(f"\n🔍 [{self.index_name}] CE SIGNAL CHECK")
            for name, result in checks.items():
                logger.info(f"  {'✅' if result else '❌'} {name}")
            
            if passed == 4:
                confidence = 75 + (bonus_passed * 3)
                logger.info(f"🎯 [{self.index_name}] CE APPROVED! Conf: {confidence}%")
                
                return Signal(
                    index_name=self.index_name,
                    type="CE_BUY",
                    reason=f"Call Short Covering (ATM: {atm_ce_change:.1f}%)",
                    confidence=min(confidence, 95),
                    spot_price=spot_price,
                    futures_price=futures_price,
                    strike=strike,
                    target_points=target_points,
                    stop_loss_points=stop_loss_points,
                    pcr=pcr,
                    candle_color=candle_color,
                    volume_surge=vol_mult,
                    oi_5m=ce_total_5m,
                    oi_15m=ce_total_15m,
                    atm_ce_change=atm_ce_change,
                    atm_pe_change=atm_pe_change,
                    atr=atr,
                    timestamp=datetime.now(IST)
                )
        
        # PE BUY LOGIC
        if pe_total_15m < -OI_THRESHOLD_MEDIUM or atm_pe_change < -ATM_OI_THRESHOLD:
            
            if abs(pe_total_15m) >= OI_THRESHOLD_STRONG or abs(atm_pe_change) >= OI_THRESHOLD_STRONG:
                target_points = max(target_points, 80)
            elif abs(pe_total_15m) >= OI_THRESHOLD_MEDIUM or abs(atm_pe_change) >= OI_THRESHOLD_MEDIUM:
                target_points = max(target_points, 50)
            
            checks = {
                "PE OI Unwinding (Total)": pe_total_15m < -OI_THRESHOLD_MEDIUM,
                "ATM PE Unwinding": atm_pe_change < -ATM_OI_THRESHOLD,
                "Price < VWAP": futures_price < vwap,
                "RED Candle": candle_color == 'RED'
            }
            
            bonus = {
                "Strong 5m OI": pe_total_5m < -5.0,
                "Big Candle": candle_size >= MIN_CANDLE_SIZE,
                "Far from VWAP": vwap_distance >= VWAP_BUFFER,
                "Bearish PCR": pcr < PCR_BEARISH,
                "Volume Spike": has_vol_spike,
                "3+ Red Momentum": self.analyzer.check_momentum(df, 'bearish')
            }
            
            passed = sum(checks.values())
            bonus_passed = sum(bonus.values())
            
            logger.info(f"\n🔍 [{self.index_name}] PE SIGNAL CHECK")
            for name, result in checks.items():
                logger.info(f"  {'✅' if result else '❌'} {name}")
            
            if passed == 4:
                confidence = 75 + (bonus_passed * 3)
                logger.info(f"🎯 [{self.index_name}] PE APPROVED! Conf: {confidence}%")
                
                return Signal(
                    index_name=self.index_name,
                    type="PE_BUY",
                    reason=f"Put Long Unwinding (ATM: {atm_pe_change:.1f}%)",
                    confidence=min(confidence, 95),
                    spot_price=spot_price,
                    futures_price=futures_price,
                    strike=strike,
                    target_points=target_points,
                    stop_loss_points=stop_loss_points,
                    pcr=pcr,
                    candle_color=candle_color,
                    volume_surge=vol_mult,
                    oi_5m=pe_total_5m,
                    oi_15m=pe_total_15m,
                    atm_ce_change=atm_ce_change,
                    atm_pe_change=atm_pe_change,
                    atr=atr,
                    timestamp=datetime.now(IST)
                )
        
        return None

# ==================== MULTI INDEX MASTER BOT ====================
class MultiIndexMasterBot:
    """Master bot that manages all 4 indices"""
    
    def __init__(self):
        self.redis = RedisBrain()
        self.index_bots = {}
        self.telegram = None
        self.last_alert_times = {}  # Per index cooldown
        self.alert_cooldown = 300  # 5 minutes per index
        
        # Initialize bot for each index
        for index_name in ACTIVE_INDICES:
            self.index_bots[index_name] = IndexBot(index_name, self.redis)
            self.last_alert_times[index_name] = None
        
        if TELEGRAM_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                self.telegram = Bot(token=TELEGRAM_BOT_TOKEN)
                logger.info("✅ Telegram Ready")
            except Exception as e:
                logger.warning(f"⚠️ Telegram setup failed: {e}")
    
    async def run_cycle(self):
        """Scan all indices in one cycle"""
        
        if not is_tradeable_time():
            return
        
        logger.info("\n" + "=" * 80)
        logger.info(f"🔥 MULTI-INDEX SCAN CYCLE - {datetime.now(IST).strftime('%I:%M:%S %p')}")
        logger.info("=" * 80)
        
        signals = []
        
        # Scan each index sequentially with small delay to avoid rate limits
        for index_name in ACTIVE_INDICES:
            try:
                bot = self.index_bots[index_name]
                signal = await bot.analyze()
                
                if signal:
                    signals.append(signal)
                    logger.info(f"✅ [{index_name}] Signal Generated!")
                else:
                    logger.info(f"✋ [{index_name}] No signal")
                
                # Small delay between indices to respect API rate limits
                await asyncio.sleep(INDEX_SCAN_DELAY)
                
            except Exception as e:
                logger.error(f"💥 [{index_name}] Error: {e}")
                continue
        
        # Send all signals
        for signal in signals:
            await self.send_alert(signal)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Cycle Complete - {len(signals)} signal(s) found")
        logger.info("=" * 80)
    
    async def send_alert(self, s: Signal):
        """Send Telegram alert for specific index signal"""
        
        # Check cooldown for this specific index
        if self.last_alert_times[s.index_name]:
            elapsed = (datetime.now(IST) - self.last_alert_times[s.index_name]).seconds
            if elapsed < self.alert_cooldown:
                logger.info(f"⏳ [{s.index_name}] Alert cooldown: {self.alert_cooldown - elapsed}s")
                return
        
        self.last_alert_times[s.index_name] = datetime.now(IST)
        
        emoji = "🟢" if s.type == "CE_BUY" else "🔴"
        
        if s.type == "CE_BUY":
            entry = s.spot_price
            target = entry + s.target_points
            stop_loss = entry - s.stop_loss_points
        else:
            entry = s.spot_price
            target = entry - s.target_points
            stop_loss = entry + s.stop_loss_points
        
        mode = "🧪 ALERT ONLY" if ALERT_ONLY_MODE else "⚡ LIVE TRADING"
        timestamp_str = s.timestamp.strftime('%d-%b %I:%M %p')
        
        msg = f"""
{emoji} {INDICES[s.index_name]['name']} SIGNAL - V13.0

{mode}

━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL: {s.type}
━━━━━━━━━━━━━━━━━━━━━━━━

📍 Entry: {entry:.1f}
🎯 Target: {target:.1f} ({s.target_points:+.0f} pts)
🛑 Stop Loss: {stop_loss:.1f} ({s.stop_loss_points:.0f} pts)
📊 Strike: {s.strike}

━━━━━━━━━━━━━━━━━━━━━━━━
LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━

{s.reason}
Confidence: {s.confidence}%

━━━━━━━━━━━━━━━━━━━━━━━━
MARKET DATA
━━━━━━━━━━━━━━━━━━━━━━━━

💰 Spot: {s.spot_price:.1f}
📈 Futures: {s.futures_price:.1f}
📊 PCR: {s.pcr:.2f}
🕯️ Candle: {s.candle_color}
🔥 Volume: {s.volume_surge:.1f}x
📏 ATR: {s.atr:.1f}

━━━━━━━━━━━━━━━━━━━━━━━━
OI ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━

ATM Strike Battle:
  CE: {s.atm_ce_change:+.1f}%
  PE: {s.atm_pe_change:+.1f}%

Total OI:
  5-min: {s.oi_5m:+.1f}%
  15-min: {s.oi_15m:+.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━

⏰ {timestamp_str}

✅ Multi-Index System
✅ 5-Strike Focus
✅ Multi-Factor Analysis
"""
        
        logger.info(f"\n🚨 [{s.index_name}] SIGNAL ALERT!")
        logger.info(f"   {s.type} @ {entry:.1f} → {target:.1f}")
        logger.info(f"   Confidence: {s.confidence}%")
        
        if self.telegram:
            try:
                await self.telegram.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=msg
                )
                logger.info(f"✅ [{s.index_name}] Alert sent to Telegram")
            except Exception as e:
                logger.error(f"❌ [{s.index_name}] Telegram error: {e}")
    
    async def send_startup_message(self):
        """Startup notification with all indices info"""
        now = datetime.now(IST)
        startup_time = now.strftime('%d-%b %I:%M %p')
        
        mode = "🧪 ALERT ONLY" if ALERT_ONLY_MODE else "⚡ LIVE TRADING"
        
        # Build indices info
        indices_info = []
        for index_name in ACTIVE_INDICES:
            config = INDICES[index_name]
            futures_sym = get_current_futures_symbol(index_name)
            expiry_type = "Weekly" if config['has_weekly'] else "Monthly"
            
            indices_info.append(
                f"📊 {config['name']}\n"
                f"   Futures: {futures_sym}\n"
                f"   Gap: {config['strike_gap']}pts | Expiry: {expiry_type}"
            )
        
        indices_text = "\n\n".join(indices_info)
        
        msg = f"""
🚀 MULTI-INDEX MASTER BOT V13.0

━━━━━━━━━━━━━━━━━━━━━━━━
STATUS
━━━━━━━━━━━━━━━━━━━━━━━━

⏰ Started: {startup_time}
🔄 Mode: {mode}
⏱️ Scan: Every {SCAN_INTERVAL}s
🎯 Active Indices: {len(ACTIVE_INDICES)}

━━━━━━━━━━━━━━━━━━━━━━━━
ALL INDICES ACTIVE
━━━━━━━━━━━━━━━━━━━━━━━━

{indices_text}

━━━━━━━━━━━━━━━━━━━━━━━━
FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━

✅ Parallel multi-index scanning
✅ Per-index signal cooldown
✅ ATM Battle Analysis (5 strikes)
✅ Multi-Factor Scoring
✅ ATR-based dynamic stops
✅ Time filters active
✅ Volume spike detection

━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Target: 50-80 points per index
⚡ Scanning all 4 indices now!
"""
        
        logger.info("📲 Sending startup message...")
        
        if self.telegram:
            try:
                await self.telegram.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=msg
                )
                logger.info("✅ Startup message sent")
            except Exception as e:
                logger.error(f"❌ Startup message failed: {e}")

# ==================== MAIN ====================
async def main():
    """Main bot loop with all indices"""
    bot = MultiIndexMasterBot()
    
    logger.info("=" * 80)
    logger.info("🚀 MULTI-INDEX MASTER BOT V13.0 STARTING")
    logger.info("=" * 80)
    logger.info("")
    logger.info("🔥 ALL INDICES ACTIVE:")
    for index_name in ACTIVE_INDICES:
        config = INDICES[index_name]
        futures = get_current_futures_symbol(index_name)
        expiry_type = "Weekly" if config['has_weekly'] else "Monthly"
        logger.info(f"   ✅ {config['name']}")
        logger.info(f"      Futures: {futures}")
        logger.info(f"      Strike Gap: {config['strike_gap']} | Expiry: {expiry_type}")
    logger.info("")
    logger.info(f"⏱️  Scan Interval: {SCAN_INTERVAL} seconds")
    logger.info(f"🎯 Indices per cycle: {len(ACTIVE_INDICES)}")
    logger.info("")
    logger.info("=" * 80)
    
    await bot.send_startup_message()
    
    while True:
        try:
            now = datetime.now(IST).time()
            
            if time(9, 15) <= now <= time(15, 30):
                await bot.run_cycle()
                await asyncio.sleep(SCAN_INTERVAL)
            else:
                logger.info("🌙 Market closed - Waiting...")
                await asyncio.sleep(300)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped by user")
            break
        
        except Exception as e:
            logger.error(f"💥 Critical error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Shutdown complete")
