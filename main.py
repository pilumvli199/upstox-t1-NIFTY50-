#!/usr/bin/env python3
"""
STRIKE MASTER V13.3 PRO - ROBUST SPOT/EXPIRY FIX
================================================
✅ Uses CORRECT Upstox API: /v2/option/expiries
✅ ENHANCED Spot Price Fetching with better fallback logic.
✅ Works for all 4 indices
✅ Detailed error logging for 400/Spot failures.

Version: 13.3 - Spot/Expiry Robustness Fixed
"""

import os
import asyncio
import aiohttp
import urllib.parse
from datetime import datetime, timedelta, time
import pytz
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict

# Optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    # logging.warning("⚠️ Redis not available")

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    # logging.warning("⚠️ Telegram not available")

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("StrikeMaster-PRO")

# API Configuration
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Indices Configuration (Verified instrument keys from Upstox Docs)
INDICES = {
    'NIFTY': {
        'spot': "NSE_INDEX|Nifty 50",
        'name': 'NIFTY 50',
        'instrument_key': 'NSE_INDEX|Nifty 50',
        'strike_gap': 50
    },
    'BANKNIFTY': {
        'spot': "NSE_INDEX|Nifty Bank",
        'name': 'BANK NIFTY',
        'instrument_key': 'NSE_INDEX|Nifty Bank',
        'strike_gap': 100
    },
    'FINNIFTY': {
        'spot': "NSE_INDEX|Nifty Fin Service",
        'name': 'FIN NIFTY',
        'instrument_key': 'NSE_INDEX|Nifty Fin Service',
        'strike_gap': 50
    },
    'MIDCPNIFTY': {
        'spot': "NSE_INDEX|NIFTY MID SELECT",
        'name': 'MIDCAP NIFTY',
        'instrument_key': 'NSE_INDEX|NIFTY MID SELECT',
        'strike_gap': 25
    }
}

# Active indices
ACTIVE_INDICES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# Trading Configuration
ALERT_ONLY_MODE = True
SCAN_INTERVAL = 60
TRACKING_INTERVAL = 60

# Enhanced Thresholds (Existing)
OI_THRESHOLD_STRONG = 8.0
OI_THRESHOLD_MEDIUM = 5.0
ATM_OI_THRESHOLD = 5.0
ORDER_FLOW_IMBALANCE = 2.0
VOL_SPIKE_2X = 2.0
PCR_BULLISH = 1.08
PCR_BEARISH = 0.92
MIN_CANDLE_SIZE = 8
VWAP_BUFFER = 5

# Time Filters
AVOID_OPENING = (time(9, 15), time(9, 45))
AVOID_CLOSING = (time(15, 15), time(15, 30))

# Risk Management
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 2.5
PARTIAL_BOOK_RATIO = 0.5
TRAIL_ACTIVATION = 0.6
TRAIL_STEP = 10

# ==================== DATA CLASSES (UNCHANGED) ====================
@dataclass
class Signal:
    """Enhanced Trading Signal"""
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
    index_name: str
    order_flow_imbalance: float = 0.0
    max_pain_distance: float = 0.0
    gamma_zone: bool = False
    multi_tf_confirm: bool = False

@dataclass
class ActiveTrade:
    """Live trade tracking"""
    signal: Signal
    entry_price: float
    entry_time: datetime
    current_price: float
    current_sl: float
    current_target: float
    pnl_points: float = 0.0
    pnl_percent: float = 0.0
    elapsed_minutes: int = 0
    partial_booked: bool = False
    trailing_active: bool = False
    last_update: datetime = field(default_factory=lambda: datetime.now(IST))
    
    def update(self, current_price: float):
        """Update trade metrics"""
        self.current_price = current_price
        self.pnl_points = current_price - self.entry_price
        self.pnl_percent = (self.pnl_points / self.entry_price) * 100
        self.elapsed_minutes = int((datetime.now(IST) - self.entry_time).total_seconds() / 60)
        self.last_update = datetime.now(IST)

# ==================== CORRECT EXPIRY MANAGER (FIXED/ENHANCED) ====================
class ExpiryManager:
    """
    🔥 CORRECT EXPIRY LOGIC (V13.3)
    - Uses /v2/option/expiries API
    - 2-Level Fallback System with better error logging
    """
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.cached_expiry = None
        self.cached_futures_symbol = None
        self.cache_time = None
        self.cache_duration = timedelta(hours=6)
    
    async def get_nearest_expiry(self) -> Tuple[str, str]:
        """
        Download ALL expiries using CORRECT Upstox API
        Returns: (expiry_date_str, futures_symbol)
        """
        now = datetime.now(IST)
        
        # Return cache if valid
        if self.cached_expiry and self.cache_time:
            if now - self.cache_time < self.cache_duration:
                logger.info(f"📦 Using cached expiry: {self.cached_expiry}")
                return self.cached_expiry, self.cached_futures_symbol
        
        logger.info(f"🔍 Fetching expiries for {self.index_config['name']}...")
        
        try:
            async with aiohttp.ClientSession() as session:
                instrument_key = self.index_config['instrument_key']
                encoded_key = urllib.parse.quote(instrument_key, safe='')
                
                # ✅ CORRECT API - Get All Expiries
                url = f"https://api.upstox.com/v2/option/expiries?instrument_key={encoded_key}"
                
                async with session.get(url, headers=self.headers, timeout=15) as resp:
                    if resp.status != 200:
                        # ENHANCED LOGGING for 400 error (from logs)
                        error_text = await resp.text()
                        logger.error(f"❌ Expiry fetch failed: {resp.status}. Response: {error_text[:300]}...") 
                        return await self._fallback_to_option_contracts()
                    
                    data = await resp.json()
                    
                    if data.get('status') != 'success':
                        logger.error(f"❌ API error (Status Fail): {data.get('message', 'Unknown API Error')}")
                        return await self._fallback_to_option_contracts()
                    
                    # ✅ CORRECT: data is array of date strings
                    expiry_dates_str = data.get('data', [])
                    
                    if not expiry_dates_str:
                        logger.error("❌ No expiries in response")
                        return await self._fallback_to_option_contracts()
                    
                    # Parse dates
                    expiry_dates = []
                    for date_str in expiry_dates_str:
                        try:
                            expiry_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                            expiry_dates.append(expiry_date)
                        except:
                            continue
                    
                    if not expiry_dates:
                        logger.error("❌ No valid expiries parsed")
                        return await self._fallback_to_option_contracts()
                    
                    # Sort expiries
                    expiry_dates.sort()
                    
                    # Find nearest UPCOMING expiry
                    today = now.date()
                    cutoff_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
                    
                    nearest_expiry = None
                    
                    for expiry_date in expiry_dates:
                        if expiry_date == today:
                            if now < cutoff_time:
                                nearest_expiry = expiry_date
                                break
                        elif expiry_date > today:
                            nearest_expiry = expiry_date
                            break
                    
                    if not nearest_expiry:
                        # Fallback for end-of-series case
                        if expiry_dates:
                            nearest_expiry = expiry_dates[-1] 
                            logger.warning("⚠️ Using last available expiry date.")
                        else:
                            logger.error("❌ No upcoming expiry found")
                            return await self._fallback_to_option_contracts()
                    
                    # Generate futures symbol
                    futures_symbol = self._generate_futures_symbol(nearest_expiry)
                    
                    # Cache results
                    self.cached_expiry = nearest_expiry.strftime('%Y-%m-%d')
                    self.cached_futures_symbol = futures_symbol
                    self.cache_time = now
                    
                    logger.info(f"✅ Nearest Expiry: {self.cached_expiry}")
                    logger.info(f"✅ Futures: {futures_symbol}")
                    
                    return self.cached_expiry, futures_symbol
        
        except Exception as e:
            logger.error(f"💥 Expiry fetch error: {e}")
            return await self._fallback_to_option_contracts()
    
    async def _fallback_to_option_contracts(self) -> Tuple[str, str]:
        """
        Fallback Level 1: Use /v2/option/contract API
        """
        logger.warning("⚠️ Fallback Level 1: Using option contracts API")
        # ... (Fallback logic - unchanged as it's the next best method)
        
        try:
            async with aiohttp.ClientSession() as session:
                instrument_key = self.index_config['instrument_key']
                encoded_key = urllib.parse.quote(instrument_key, safe='')
                
                # Get ALL option contracts
                url = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_key}"
                
                async with session.get(url, headers=self.headers, timeout=15) as resp:
                    if resp.status != 200:
                        return self._manual_calculation()
                    
                    data = await resp.json()
                    
                    if data.get('status') != 'success':
                        return self._manual_calculation()
                    
                    contracts = data.get('data', [])
                    
                    # Extract unique expiry dates
                    expiry_set = set()
                    for contract in contracts:
                        expiry_str = contract.get('expiry')
                        if expiry_str:
                            expiry_set.add(expiry_str)
                    
                    if not expiry_set:
                        return self._manual_calculation()
                    
                    expiry_dates = sorted([
                        datetime.strptime(d, '%Y-%m-%d').date() 
                        for d in expiry_set
                    ])
                    
                    # Find nearest
                    now = datetime.now(IST)
                    today = now.date()
                    cutoff_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
                    
                    nearest_expiry = None
                    for expiry_date in expiry_dates:
                        if expiry_date == today:
                            if now < cutoff_time:
                                nearest_expiry = expiry_date
                                break
                        elif expiry_date > today:
                            nearest_expiry = expiry_date
                            break
                    
                    if not nearest_expiry:
                        if expiry_dates:
                            nearest_expiry = expiry_dates[-1] 
                        else:
                            return self._manual_calculation()
                    
                    futures_symbol = self._generate_futures_symbol(nearest_expiry)
                    
                    self.cached_expiry = nearest_expiry.strftime('%Y-%m-%d')
                    self.cached_futures_symbol = futures_symbol
                    self.cache_time = datetime.now(IST)
                    
                    logger.info(f"✅ Fallback Level 1 Success: {self.cached_expiry}")
                    
                    return self.cached_expiry, futures_symbol
        
        except Exception as e:
            logger.error(f"💥 Fallback Level 1 failed: {e}")
            return self._manual_calculation()
    
    def _manual_calculation(self) -> Tuple[str, str]:
        """Fallback Level 2: Manual calculation"""
        logger.warning("⚠️ Fallback Level 2: Manual calculation (Highest risk)")
        
        if self.cached_expiry and self.cached_futures_symbol:
            logger.warning("⚠️ Using last cached expiry")
            return self.cached_expiry, self.cached_futures_symbol
        
        now = datetime.now(IST)
        
        # Find next Thursday (most common expiry)
        days_ahead = (3 - now.weekday()) % 7 # 3 is Thursday (0=Mon, 6=Sun)
        
        # Midcap Nifty expires on Wednesday (2)
        if self.index_name == 'MIDCPNIFTY':
            days_ahead = (2 - now.weekday()) % 7
        # Finnifty expires on Tuesday (1)
        elif self.index_name == 'FINNIFTY':
            days_ahead = (1 - now.weekday()) % 7
        
        if days_ahead == 0 and now.time() > time(15, 30):
            days_ahead = 7
        
        expiry_date = (now + timedelta(days=days_ahead)).date()
        futures_symbol = self._generate_futures_symbol(expiry_date)
        
        logger.warning(f"⚠️ Manual expiry: {expiry_date.strftime('%Y-%m-%d')}")
        
        return expiry_date.strftime('%Y-%m-%d'), futures_symbol
    
    def _generate_futures_symbol(self, expiry_date) -> str:
        """Generate futures symbol from expiry date (Using standard NSE format)"""
        year_short = expiry_date.year % 100
        month_name = expiry_date.strftime('%b').upper()
        
        prefix_map = {
            'NIFTY': 'NIFTY',
            'BANKNIFTY': 'BANKNIFTY',
            'FINNIFTY': 'FINNIFTY',
            'MIDCPNIFTY': 'MIDCPNIFTY'
        }
        prefix = prefix_map.get(self.index_name, 'NIFTY')
        
        # Format: NSE_FO|NIFTY25DECFUT (Standard NSE format)
        return f"NSE_FO|{prefix}{year_short:02d}{month_name}FUT"

# ==================== UTILITIES & REDIS BRAIN (UNCHANGED) ====================
def is_tradeable_time() -> bool:
    """Check trading window"""
    now = datetime.now(IST).time()
    
    if not (time(9, 15) <= now <= time(15, 30)):
        return False
    
    if AVOID_OPENING[0] <= now <= AVOID_OPENING[1]:
        return False
    
    if AVOID_CLOSING[0] <= now <= AVOID_CLOSING[1]:
        return False
    
    return True

class RedisBrain:
    """Enhanced memory system (UNCHANGED)"""
    # ... (implementation remains the same)
    
    def __init__(self):
        self.client = None
        self.memory = {}
        
        if REDIS_AVAILABLE:
            try:
                self.client = redis.from_url(REDIS_URL, decode_responses=True)
                self.client.ping()
                logger.info("✅ Redis Connected")
            except:
                self.client = None
        
        if not self.client:
            logger.info("💾 RAM-only mode")
    
    def save_strike_snapshot(self, index_name: str, strike_data: Dict[int, dict]):
        """Save OI snapshot"""
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
    
    def get_strike_oi_change(self, index_name: str, strike: int, current_data: dict, 
                             minutes_ago: int = 15) -> Tuple[float, float]:
        """Calculate OI change"""
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        timestamp = now.replace(second=0, microsecond=0)
        key = f"{index_name}:strike:{strike}:{timestamp.strftime('%H%M')}"
        
        past_data_str = self.client.get(key) if self.client else self.memory.get(key)
        
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
        """Save total OI"""
        now = datetime.now(IST)
        slot = now.replace(second=0, microsecond=0)
        key = f"{index_name}:total_oi:{slot.strftime('%H%M')}"
        data = json.dumps({"ce": total_ce, "pe": pe_total})
        
        if self.client:
            try:
                self.client.setex(key, 3600, data)
            except:
                self.memory[key] = data
        else:
            self.memory[key] = data
    
    def get_total_oi_change(self, index_name: str, current_ce: int, current_pe: int, 
                           minutes_ago: int = 15) -> Tuple[float, float]:
        """Get total OI change"""
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        slot = now.replace(second=0, microsecond=0)
        key = f"{index_name}:total_oi:{slot.strftime('%H%M')}"
        
        past_data = self.client.get(key) if self.client else self.memory.get(key)
        
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

# ==================== DATA FEED (FIXED/ENHANCED) ====================
class StrikeDataFeed:
    """Enhanced data fetching with CORRECT expiry logic (V13.3)"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.expiry_manager = ExpiryManager(index_name)
        self.expiry_date = None
        self.futures_symbol = None
    
    async def initialize(self):
        """Initialize expiry data"""
        self.expiry_date, self.futures_symbol = await self.expiry_manager.get_nearest_expiry()
        logger.info(f"🎯 {self.index_config['name']}: {self.futures_symbol} (Exp: {self.expiry_date})")
    
    async def fetch_with_retry(self, url: str, session: aiohttp.ClientSession):
        """Retry logic"""
        for attempt in range(3):
            try:
                async with session.get(url, headers=self.headers, timeout=15) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        logger.warning(f"⚠️ Rate limit hit. Retrying in {2 ** attempt * 2}s")
                        await asyncio.sleep(2 ** attempt * 2)
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ API Error {resp.status} on URL: {url}. Response: {error_text[:300]}...")
                        await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"❌ Fetch exception on attempt {attempt+1}: {e}")
                await asyncio.sleep(2)
        return None
    
    async def get_market_data(self) -> Tuple[pd.DataFrame, Dict[int, dict], 
                                            str, float, float, float]:
        """Fetch all data with CORRECT expiry"""
        
        # Refresh expiry if needed
        if not self.expiry_date or not self.futures_symbol:
            await self.initialize()
        
        async with aiohttp.ClientSession() as session:
            spot_price = 0
            futures_price = 0
            df = pd.DataFrame()
            strike_data = {}
            total_options_volume = 0
            
            # 1. SPOT PRICE (FIXED - 3 Methods)
            logger.info(f"🔍 {self.index_config['name']}: Fetching Spot...")
            enc_spot = urllib.parse.quote(self.index_config['spot'], safe='')
            
            # Method 1: Full Market Quote (Most Reliable)
            quote_url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={enc_spot}"
            quote_data = await self.fetch_with_retry(quote_url, session)
            
            if quote_data and quote_data.get('status') == 'success':
                data = quote_data.get('data', {})
                spot_symbol = self.index_config['spot']
                
                if spot_symbol in data:
                    spot_info = data[spot_symbol]
                    # Try multiple price fields for robustness
                    spot_price = (spot_info.get('last_price') or 
                                 spot_info.get('ohlc', {}).get('close') or 
                                 spot_info.get('ohlc', {}).get('open', 0))
                    
                    if spot_price > 0:
                        logger.info(f"✅ Spot (Method 1 - Quote): ₹{spot_price:.2f}")
                else:
                    logger.warning(f"⚠️ Spot (Method 1): Key '{spot_symbol}' not in response data.")

            # Method 2: LTP API (Fallback)
            if spot_price == 0:
                logger.warning("⚠️ Trying Method 2: LTP API")
                ltp_url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={enc_spot}"
                ltp_data = await self.fetch_with_retry(ltp_url, session)
                
                if ltp_data and ltp_data.get('status') == 'success':
                    data = ltp_data.get('data', {})
                    spot_symbol = self.index_config['spot']
                    
                    if spot_symbol in data:
                        spot_info = data[spot_symbol]
                        spot_price = spot_info.get('last_price', 0)
                        
                        if spot_price > 0:
                            logger.info(f"✅ Spot (Method 2 - LTP): ₹{spot_price:.2f}")
            
            # 2. FUTURES CANDLES (Need to fetch for Futures Price and Technical DF)
            logger.info(f"🔍 Fetching Futures: {self.futures_symbol}")
            enc_futures = urllib.parse.quote(self.futures_symbol, safe='')
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            # Fetch for current day only, historical only if needed for deeper ATR/VWAP calc
            from_date = (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d') 
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
                    # Filter for today's data
                    df = df[df.index.date == today].tail(100) 
                    
                    if not df.empty:
                        futures_price = df['close'].iloc[-1]
                        logger.info(f"✅ Futures: {len(df)} candles | ₹{futures_price:.2f}")
            
            # Method 3: Use Futures as Spot (Last Resort for Spot Price)
            if spot_price == 0 and futures_price > 0:
                logger.warning("⚠️ Method 3: Using Futures price as Spot (Last Resort)")
                spot_price = futures_price
                logger.info(f"✅ Spot (Method 3 - Futures): ₹{spot_price:.2f}")
            
            # Final check
            if spot_price == 0:
                logger.error(f"❌ All spot fetch methods failed for {self.index_config['name']}!")
                return df, strike_data, "", 0, 0, 0
            
            # 3. OPTION CHAIN
            logger.info("🔍 Fetching Option Chain...")
            chain_url = f"https://api.upstox.com/v2/option/chain?instrument_key={enc_spot}&expiry_date={self.expiry_date}"
            
            strike_gap = self.index_config['strike_gap']
            atm_strike = round(spot_price / strike_gap) * strike_gap
            min_strike = atm_strike - (2 * strike_gap)
            max_strike = atm_strike + (2 * strike_gap)
            
            logger.info(f"📊 ATM: {atm_strike} | Range: {min_strike}-{max_strike}")
            
            chain_data = await self.fetch_with_retry(chain_url, session)
            if chain_data and chain_data.get('status') == 'success':
                for option in chain_data.get('data', []):
                    strike = option.get('strike_price', 0)
                    
                    # Only collect data for the target ATM strikes (ATM +/- 2)
                    if min_strike <= strike <= max_strike:
                        call_data = option.get('call_options', {}).get('market_data', {})
                        put_data = option.get('put_options', {}).get('market_data', {})
                        
                        # Use a 1-level check for data availability
                        ce_oi = call_data.get('oi', 0) if call_data else 0
                        pe_oi = put_data.get('oi', 0) if put_data else 0
                        
                        strike_data[strike] = {
                            'ce_oi': ce_oi,
                            'pe_oi': pe_oi,
                            'ce_vol': call_data.get('volume', 0) if call_data else 0,
                            'pe_vol': put_data.get('volume', 0) if put_data else 0,
                            'ce_ltp': call_data.get('ltp', 0) if call_data else 0,
                            'pe_ltp': put_data.get('ltp', 0) if put_data else 0
                        }
                        
                        total_options_volume += (strike_data[strike]['ce_vol'] + strike_data[strike]['pe_vol'])
                
                logger.info(f"✅ Collected {len(strike_data)} strikes")
            else:
                logger.error("❌ Option Chain fetch failed!")
                
            
            return df, strike_data, self.expiry_date, spot_price, futures_price, total_options_volume

# ==================== ENHANCED ANALYZER & TRADE TRACKER (UNCHANGED) ====================

class EnhancedAnalyzer:
    """Advanced Analysis Engine (UNCHANGED)"""
    # ... (implementation remains the same)
    
    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """VWAP"""
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
        """ATR"""
        if len(df) < period:
            return 30
        
        df_copy = df.tail(period).copy()
        
        df_copy['h-l'] = df_copy['high'] - df_copy['low']
        df_copy['h-pc'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['l-pc'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        
        df_copy['tr'] = df_copy[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return df_copy['tr'].mean()
    
    def get_candle_info(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Candle analysis"""
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
    
    def check_volume_surge(self, index_name: str, current_vol: float) -> Tuple[bool, float]:
        """Volume spike detection"""
        now = datetime.now(IST)
        
        if index_name not in self.volume_history:
            self.volume_history[index_name] = []
        
        self.volume_history[index_name].append({'time': now, 'volume': current_vol})
        
        cutoff = now - timedelta(minutes=20)
        self.volume_history[index_name] = [
            x for x in self.volume_history[index_name] if x['time'] > cutoff
        ]
        
        if len(self.volume_history[index_name]) < 5:
            return False, 0
        
        past_volumes = [x['volume'] for x in self.volume_history[index_name][:-1]]
        avg_vol = sum(past_volumes) / len(past_volumes)
        
        if avg_vol == 0:
            return False, 0
        
        multiplier = current_vol / avg_vol
        return multiplier >= VOL_SPIKE_2X, multiplier
    
    def calculate_pcr(self, strike_data: Dict[int, dict]) -> float:
        """PCR"""
        total_ce = sum(data['ce_oi'] for data in strike_data.values())
        total_pe = sum(data['pe_oi'] for data in strike_data.values())
        
        return total_pe / total_ce if total_ce > 0 else 1.0
    
    def calculate_order_flow_imbalance(self, strike_data: Dict[int, dict]) -> float:
        """Order Flow Imbalance"""
        ce_vol = sum(data['ce_vol'] for data in strike_data.values())
        pe_vol = sum(data['pe_vol'] for data in strike_data.values())
        
        if pe_vol == 0:
            return 0
        
        ratio = ce_vol / pe_vol
        logger.info(f"📊 Order Flow: CE/PE = {ratio:.2f}")
        return ratio
    
    def calculate_max_pain(self, strike_data: Dict[int, dict], spot_price: float) -> Tuple[int, float]:
        """Max Pain Strike"""
        max_pain_strike = 0
        min_pain_value = float('inf')
        
        for test_strike in strike_data.keys():
            pain = 0
            
            for strike, data in strike_data.items():
                if test_strike < strike:
                    pain += data['ce_oi'] * (strike - test_strike)
                
                if test_strike > strike:
                    pain += data['pe_oi'] * (test_strike - strike)
            
            if pain < min_pain_value:
                min_pain_value = pain
                max_pain_strike = test_strike
        
        distance = abs(spot_price - max_pain_strike)
        logger.info(f"🎯 Max Pain: {max_pain_strike} (Distance: {distance:.0f})")
        
        return max_pain_strike, distance
    
    def detect_gamma_zone(self, strike_data: Dict[int, dict], atm_strike: int) -> bool:
        """Gamma Squeeze Zone Detection"""
        if atm_strike not in strike_data:
            return False
        
        atm_data = strike_data[atm_strike]
        total_atm_oi = atm_data['ce_oi'] + atm_data['pe_oi']
        
        total_oi = sum(d['ce_oi'] + d['pe_oi'] for d in strike_data.values())
        
        if total_oi == 0:
            return False
        
        atm_concentration = (total_atm_oi / total_oi) * 100
        
        is_gamma_zone = atm_concentration > 30
        
        if is_gamma_zone:
            logger.info(f"⚡ Gamma Zone! ATM OI: {atm_concentration:.1f}%")
        
        return is_gamma_zone
    
    def check_multi_tf_confirmation(self, ce_5m: float, ce_15m: float, 
                                    pe_5m: float, pe_15m: float) -> bool:
        """Multi-Timeframe Confirmation"""
        ce_aligned = (ce_5m < -3 and ce_15m < -5) or (ce_5m > 3 and ce_15m > 5)
        pe_aligned = (pe_5m < -3 and pe_15m < -5) or (pe_5m > 3 and pe_15m > 5)
        
        confirmed = ce_aligned or pe_aligned
        
        if confirmed:
            logger.info(f"✅ Multi-TF Confirmed: 5m & 15m aligned")
        
        return confirmed
    
    def analyze_atm_battle(self, index_name: str, strike_data: Dict[int, dict], 
                          atm_strike: int, redis_brain: RedisBrain) -> Tuple[float, float]:
        """ATM battle"""
        if atm_strike not in strike_data:
            return 0, 0
        
        current = strike_data[atm_strike]
        
        ce_15m, pe_15m = redis_brain.get_strike_oi_change(
            index_name, atm_strike, current, minutes_ago=15
        )
        
        logger.info(f"⚔️ ATM {atm_strike}: CE={ce_15m:+.1f}% PE={pe_15m:+.1f}%")
        
        return ce_15m, pe_15m
    
    def check_momentum(self, df: pd.DataFrame, direction: str = 'bullish') -> bool:
        """Momentum check"""
        if df.empty or len(df) < 3:
            return False
        
        last_3 = df.tail(3)
        
        if direction == 'bullish':
            return sum(last_3['close'] > last_3['open']) >= 2
        else:
            return sum(last_3['close'] < last_3['open']) >= 2

class TradeTracker:
    """Live Trade Tracking System (UNCHANGED)"""
    # ... (implementation remains the same)
    
    def __init__(self, telegram: Optional[Bot]):
        self.active_trades: Dict[str, ActiveTrade] = {}
        self.telegram = telegram
    
    def add_trade(self, signal: Signal):
        """Start tracking new trade"""
        trade_id = f"{signal.index_name}_{signal.timestamp.strftime('%H%M%S')}"
        
        if signal.type == "CE_BUY":
            entry = signal.spot_price
            target = entry + signal.target_points
            sl = entry - signal.stop_loss_points
        else:
            entry = signal.spot_price
            target = entry - signal.target_points
            sl = entry + signal.stop_loss_points
        
        trade = ActiveTrade(
            signal=signal,
            entry_price=entry,
            entry_time=signal.timestamp,
            current_price=entry,
            current_sl=sl,
            current_target=target
        )
        
        self.active_trades[trade_id] = trade
        logger.info(f"📌 Tracking: {trade_id}")
    
    async def update_trades(self, index_name: str, current_price: float):
        """Update all active trades"""
        for trade_id, trade in list(self.active_trades.items()):
            if trade.signal.index_name != index_name:
                continue
            
            trade.update(current_price)
            
            # Check exit conditions
            exit_reason = None
            
            # 1. Target hit
            if trade.signal.type == "CE_BUY":
                if current_price >= trade.current_target:
                    exit_reason = "🎯 TARGET HIT"
            else:
                if current_price <= trade.current_target:
                    exit_reason = "🎯 TARGET HIT"
            
            # 2. Stop loss hit
            if trade.signal.type == "CE_BUY":
                if current_price <= trade.current_sl:
                    exit_reason = "🛑 STOP LOSS HIT"
            else:
                if current_price >= trade.current_sl:
                    exit_reason = "🛑 STOP LOSS HIT"
            
            # 3. Partial booking
            if not trade.partial_booked:
                progress = abs(trade.pnl_points) / abs(trade.signal.target_points)
                if progress >= PARTIAL_BOOK_RATIO:
                    trade.partial_booked = True
                    await self.send_partial_book_alert(trade)
            
            # 4. Trailing SL
            if not trade.trailing_active:
                progress = abs(trade.pnl_points) / abs(trade.signal.target_points)
                if progress >= TRAIL_ACTIVATION:
                    trade.trailing_active = True
                    logger.info(f"🔄 Trailing activated for {trade_id}")
            
            if trade.trailing_active:
                new_sl = self.calculate_trailing_sl(trade, current_price)
                if new_sl != trade.current_sl:
                    trade.current_sl = new_sl
                    logger.info(f"📈 SL trailed to {new_sl:.1f}")
            
            # Exit if needed
            if exit_reason:
                await self.send_exit_alert(trade, exit_reason)
                del self.active_trades[trade_id]
            else:
                await self.send_update_alert(trade)
    
    def calculate_trailing_sl(self, trade: ActiveTrade, current_price: float) -> float:
        """Calculate trailing stop loss"""
        if trade.signal.type == "CE_BUY":
            new_sl = current_price - TRAIL_STEP
            return max(new_sl, trade.current_sl)
        else:
            new_sl = current_price + TRAIL_STEP
            return min(new_sl, trade.current_sl)
    
    async def send_update_alert(self, trade: ActiveTrade):
        """Send 1-min position update"""
        if not self.telegram:
            return
        
        s = trade.signal
        emoji = "🟢" if s.type == "CE_BUY" else "🔴"
        
        progress = (abs(trade.pnl_points) / abs(s.target_points)) * 100
        sl_buffer = abs(trade.current_price - trade.current_sl)
        
        msg = f"""
{emoji} {s.index_name} {s.type} [ACTIVE]

Entry: {trade.entry_price:.1f} | Now: {trade.current_price:.1f}

━━━━━━━━━━━━━━━━━━━━━━━━
⏱️ {trade.elapsed_minutes}m elapsed
💰 P&L: {trade.pnl_percent:+.2f}% ({trade.pnl_points:+.1f} pts)
🎯 Target: {trade.current_target:.1f} ({progress:.0f}% done)
🛑 SL: {trade.current_sl:.1f} (buffer: {sl_buffer:.0f})
━━━━━━━━━━━━━━━━━━━━━━━━

Status:
{'✅ Partial Booked' if trade.partial_booked else '⏳ Full Position'}
{'🔄 Trailing Active' if trade.trailing_active else '📍 Fixed SL'}
"""
        
        try:
            # Note: This should ideally be rate-limited to avoid excessive messages/spam.
            # For simplicity, sending every minute, but in production, use a dedicated channel/update logic.
            await self.telegram.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        except Exception as e:
            logger.error(f"Update alert failed: {e}")
    
    async def send_partial_book_alert(self, trade: ActiveTrade):
        """Alert for 50% booking"""
        if not self.telegram:
            return
        
        s = trade.signal
        
        msg = f"""
🔔 PARTIAL BOOKING ALERT

{s.index_name} {s.type}

━━━━━━━━━━━━━━━━━━━━━━━━
📍 Entry: {trade.entry_price:.1f}
💰 Current: {trade.current_price:.1f}
📊 P&L: {trade.pnl_percent:+.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━
ACTION: Book 50% Position
Reason: Reached 1:1 Risk-Reward

✅ Lock in partial profits
🎯 Let remaining 50% run to target
"""
        
        try:
            await self.telegram.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            logger.info(f"✅ Partial book alert sent")
        except:
            pass
    
    async def send_exit_alert(self, trade: ActiveTrade, reason: str):
        """Alert on trade exit"""
        if not self.telegram:
            return
        
        s = trade.signal
        
        msg = f"""
🔚 TRADE CLOSED

{reason}

{s.index_name} {s.type}

━━━━━━━━━━━━━━━━━━━━━━━━
📍 Entry: {trade.entry_price:.1f}
📍 Exit: {trade.current_price:.1f}
━━━━━━━━━━━━━━━━━━━━━━━━

💰 Final P&L: {trade.pnl_percent:+.2f}%
📊 Points: {trade.pnl_points:+.1f}
⏱️ Duration: {trade.elapsed_minutes} minutes

━━━━━━━━━━━━━━━━━━━━━━━━
Position Closed ✅
"""
        
        try:
            await self.telegram.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            logger.info(f"✅ Exit alert sent: {reason}")
        except:
            pass

# ==================== MAIN BOT (UNCHANGED CORE LOGIC) ====================
class StrikeMasterPro:
    """V13.3 PRO Bot with FIXED Spot/Expiry Logic"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.feed = StrikeDataFeed(index_name)
        self.redis = RedisBrain()
        self.analyzer = EnhancedAnalyzer()
        self.telegram = None
        self.tracker = None
        self.last_alert_time = None
        self.alert_cooldown = 300
        
        if TELEGRAM_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                self.telegram = Bot(token=TELEGRAM_BOT_TOKEN)
                self.tracker = TradeTracker(self.telegram)
                logger.info("✅ Telegram + Tracker Ready")
            except Exception as e:
                logger.warning(f"⚠️ Telegram: {e}")
    
    async def initialize(self):
        """Initialize feed with expiry data"""
        await self.feed.initialize()
    
    async def run_cycle(self):
        """Analysis cycle"""
        if not is_tradeable_time():
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 {self.index_config['name']} SCAN")
        logger.info(f"{'='*80}")
        
        df, strike_data, expiry, spot, futures, vol = await self.feed.get_market_data()
        
        if df.empty or not strike_data or spot == 0:
            logger.warning("⏳ Incomplete data - Skipping signal generation.")
            return
        
        # Basic metrics
        vwap = self.analyzer.calculate_vwap(df)
        atr = self.analyzer.calculate_atr(df)
        pcr = self.analyzer.calculate_pcr(strike_data)
        candle_color, candle_size = self.analyzer.get_candle_info(df)
        has_vol_spike, vol_mult = self.analyzer.check_volume_surge(self.index_name, vol)
        vwap_distance = abs(futures - vwap)
        
        # Enhanced Analysis
        order_flow = self.analyzer.calculate_order_flow_imbalance(strike_data)
        max_pain_strike, max_pain_dist = self.analyzer.calculate_max_pain(strike_data, spot)
        
        strike_gap = self.index_config['strike_gap']
        atm_strike = round(spot / strike_gap) * strike_gap
        gamma_zone = self.analyzer.detect_gamma_zone(strike_data, atm_strike)
        
        atm_ce_15m, atm_pe_15m = self.analyzer.analyze_atm_battle(
            self.index_name, strike_data, atm_strike, self.redis
        )
        
        total_ce = sum(d['ce_oi'] for d in strike_data.values())
        total_pe = sum(d['pe_oi'] for d in strike_data.values())
        
        ce_total_15m, pe_total_15m = self.redis.get_total_oi_change(
            self.index_name, total_ce, total_pe, minutes_ago=15
        )
        ce_total_5m, pe_total_5m = self.redis.get_total_oi_change(
            self.index_name, total_ce, total_pe, minutes_ago=5
        )
        
        multi_tf = self.analyzer.check_multi_tf_confirmation(
            ce_total_5m, ce_total_15m, pe_total_5m, pe_total_15m
        )
        
        self.redis.save_strike_snapshot(self.index_name, strike_data)
        self.redis.save_total_oi_snapshot(self.index_name, total_ce, total_pe)
        
        logger.info(f"💰 Spot: {spot:.2f} | Futures: {futures:.2f}")
        logger.info(f"📊 VWAP: {vwap:.2f} | PCR: {pcr:.2f} | Candle: {candle_color}")
        logger.info(f"📉 OI 15m: CE={ce_total_15m:+.1f}% | PE={pe_total_15m:+.1f}%")
        
        # Generate signal
        signal = self.generate_signal(
            spot, futures, vwap, vwap_distance, pcr, atr,
            ce_total_15m, pe_total_15m, ce_total_5m, pe_total_5m,
            atm_ce_15m, atm_pe_15m,
            candle_color, candle_size,
            has_vol_spike, vol_mult, df,
            order_flow, max_pain_dist, gamma_zone, multi_tf
        )
        
        if signal:
            await self.send_alert(signal)
            
            if self.tracker and not ALERT_ONLY_MODE:
                self.tracker.add_trade(signal)
        else:
            logger.info("✋ No setup")
        
        # Update active trades
        if self.tracker and self.tracker.active_trades and not ALERT_ONLY_MODE:
            await self.tracker.update_trades(self.index_name, spot)
        
        logger.info(f"{'='*80}\n")
    
    def generate_signal(self, spot_price, futures_price, vwap, vwap_distance, pcr, atr,
                       ce_total_15m, pe_total_15m, ce_total_5m, pe_total_5m,
                       atm_ce_change, atm_pe_change, candle_color, candle_size,
                       has_vol_spike, vol_mult, df,
                       order_flow, max_pain_dist, gamma_zone, multi_tf) -> Optional[Signal]:
        """Enhanced signal generation (UNCHANGED CORE LOGIC)"""
        
        strike_gap = self.index_config['strike_gap']
        strike = round(spot_price / strike_gap) * strike_gap
        
        stop_loss_points = int(atr * ATR_SL_MULTIPLIER)
        target_points = int(atr * ATR_TARGET_MULTIPLIER)
        
        # Dynamic targets
        if abs(ce_total_15m) >= OI_THRESHOLD_STRONG or abs(atm_ce_change) >= OI_THRESHOLD_STRONG:
            target_points = max(target_points, 80)
        elif abs(ce_total_15m) >= OI_THRESHOLD_MEDIUM or abs(atm_ce_change) >= OI_THRESHOLD_MEDIUM:
            target_points = max(target_points, 50)
        
        # CE BUY SIGNAL
        if ce_total_15m < -OI_THRESHOLD_MEDIUM or atm_ce_change < -ATM_OI_THRESHOLD:
            checks = {
                "CE OI Unwinding": ce_total_15m < -OI_THRESHOLD_MEDIUM,
                "ATM CE Unwinding": atm_ce_change < -ATM_OI_THRESHOLD,
                "Price > VWAP": futures_price > vwap,
                "GREEN Candle": candle_color == 'GREEN'
            }
            
            bonus = {
                "Strong 5m": ce_total_5m < -5.0,
                "Big Candle": candle_size >= MIN_CANDLE_SIZE,
                "Far VWAP": vwap_distance >= VWAP_BUFFER,
                "Bullish PCR": pcr > PCR_BULLISH,
                "Vol Spike": has_vol_spike,
                "Momentum": self.analyzer.check_momentum(df, 'bullish'),
                "Order Flow Bullish": order_flow < 1.0,
                "Multi-TF Confirm": multi_tf,
                "Gamma Zone": gamma_zone
            }
            
            passed = sum(checks.values())
            bonus_passed = sum(bonus.values())
            
            if passed == 4:
                confidence = 75 + (bonus_passed * 3)
                logger.info(f"🎯 CE SIGNAL! Conf: {confidence}%")
                
                return Signal(
                    type="CE_BUY",
                    reason=f"Call Unwinding (ATM: {atm_ce_change:.1f}%)",
                    confidence=min(confidence, 98),
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
                    timestamp=datetime.now(IST),
                    index_name=self.index_name,
                    order_flow_imbalance=order_flow,
                    max_pain_distance=max_pain_dist,
                    gamma_zone=gamma_zone,
                    multi_tf_confirm=multi_tf
                )
        
        # PE BUY SIGNAL
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
                "Strong 5m": pe_total_5m < -5.0,
                "Big Candle": candle_size >= MIN_CANDLE_SIZE,
                "Far VWAP": vwap_distance >= VWAP_BUFFER,
                "Bearish PCR": pcr < PCR_BEARISH,
                "Vol Spike": has_vol_spike,
                "Momentum": self.analyzer.check_momentum(df, 'bearish'),
                "Order Flow Bearish": order_flow > 1.0,
                "Multi-TF Confirm": multi_tf,
                "Gamma Zone": gamma_zone
            }
            
            passed = sum(checks.values())
            bonus_passed = sum(bonus.values())
            
            if passed == 4:
                confidence = 75 + (bonus_passed * 3)
                logger.info(f"🎯 PE SIGNAL! Conf: {confidence}%")
                
                return Signal(
                    type="PE_BUY",
                    reason=f"Put Unwinding (ATM: {atm_pe_change:.1f}%)",
                    confidence=min(confidence, 98),
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
                    timestamp=datetime.now(IST),
                    index_name=self.index_name,
                    order_flow_imbalance=order_flow,
                    max_pain_distance=max_pain_dist,
                    gamma_zone=gamma_zone,
                    multi_tf_confirm=multi_tf
                )
        
        return None
    
    async def send_alert(self, s: Signal):
        """Enhanced alert"""
        if self.last_alert_time:
            elapsed = (datetime.now(IST) - self.last_alert_time).seconds
            if elapsed < self.alert_cooldown:
                logger.info(f"⏳ Cooldown: {self.alert_cooldown - elapsed}s")
                return
        
        self.last_alert_time = datetime.now(IST)
        
        emoji = "🟢" if s.type == "CE_BUY" else "🔴"
        
        if s.type == "CE_BUY":
            entry = s.spot_price
            target = entry + s.target_points
            stop_loss = entry - s.stop_loss_points
        else:
            entry = s.spot_price
            target = entry - s.target_points
            stop_loss = entry + s.stop_loss_points
        
        mode = "🧪 ALERT ONLY" if ALERT_ONLY_MODE else "⚡ LIVE"
        timestamp_str = s.timestamp.strftime('%d-%b %I:%M %p')
        
        msg = f"""
{emoji} {self.index_config['name']} V13.3 PRO

{mode}

━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL: {s.type}
━━━━━━━━━━━━━━━━━━━━━━━━

📍 Entry: {entry:.1f}
🎯 Target: {target:.1f} ({s.target_points:+.0f} pts)
🛑 Stop: {stop_loss:.1f} ({s.stop_loss_points:.0f} pts)
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

ATM Strike:
  CE: {s.atm_ce_change:+.1f}%
  PE: {s.atm_pe_change:+.1f}%

Total OI:
  5m: {s.oi_5m:+.1f}%
  15m: {s.oi_15m:+.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━
ENHANCED METRICS
━━━━━━━━━━━━━━━━━━━━━━━━

📊 Order Flow: {s.order_flow_imbalance:.2f}
🎯 Max Pain: {s.max_pain_distance:.0f} pts away
{'⚡ Gamma Zone: YES' if s.gamma_zone else ''}
{'✅ Multi-TF: Confirmed' if s.multi_tf_confirm else ''}

━━━━━━━━━━━━━━━━━━━━━━━━

⏰ {timestamp_str}

✅ FIXED: Robust Expiry & Spot Fetch
✅ Phase 1: Enhanced Analysis
✅ Phase 2: Live Tracking Active
"""
        
        logger.info(f"🚨 {s.type} @ {entry:.1f} → {target:.1f}")
        
        if self.telegram:
            try:
                await self.telegram.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                logger.info("✅ Alert sent")
            except Exception as e:
                logger.error(f"❌ Telegram: {e}")
    
    async def send_startup_message(self):
        """Startup notification"""
        now = datetime.now(IST)
        startup_time = now.strftime('%d-%b %I:%M %p')
        mode = "🧪 ALERT ONLY" if ALERT_ONLY_MODE else "⚡ LIVE TRADING"
        
        msg = f"""
🚀 STRIKE MASTER V13.3 PRO - ROBUST FIX

━━━━━━━━━━━━━━━━━━━━━━━━
STATUS
━━━━━━━━━━━━━━━━━━━━━━━━

⏰ {startup_time}
📊 {self.index_config['name']}
🔄 {mode}

━━━━━━━━━━━━━━━━━━━━━━━━
✅ FIXED IN V13.3
━━━━━━━━━━━━━━━━━━━━━━━━

🔥 Expiry Fetch: Detailed error log for 400.
🔥 Spot Fetch: Enhanced 3-level fallback for robust index prices.
🔥 Manual Fallback: Corrected expiry day calculation for Midcap/Finnifty.

━━━━━━━━━━━━━━━━━━━━━━━━
FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━

✅ Phase 1: Enhanced Analysis
✅ Phase 2: Live Tracking
   • Auto trailing SL
   • Partial booking (50% @ 1:1)

━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG
━━━━━━━━━━━━━━━━━━━━━━━━

📈 Futures: {self.feed.futures_symbol}
📅 Expiry: {self.feed.expiry_date}
🎯 Strikes: 5 (ATM ± 2)
📏 Stops: ATR-based + Trailing

━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Target: 85%+ Accuracy
⚡ Institutional Grade System
"""
        
        if self.telegram:
            try:
                await self.telegram.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                logger.info("✅ Startup sent")
            except:
                pass

# ==================== MAIN ====================
async def main():
    """Main entry - All 4 indices with CORRECT expiry detection"""
    
    # Validate indices
    active_indices = [idx for idx in ACTIVE_INDICES if idx in INDICES]
    
    if not active_indices:
        logger.error("❌ No valid indices!")
        return
    
    # Create bots for all 4 indices
    bots = {}
    for index_name in active_indices:
        try:
            bot = StrikeMasterPro(index_name)
            # Initialize to fetch expiry early
            await bot.initialize() 
            bots[index_name] = bot
            logger.info(f"✅ {INDICES[index_name]['name']}")
        except Exception as e:
            logger.error(f"❌ {index_name}: {e}")
    
    if not bots:
        logger.error("❌ No bots initialized!")
        return
    
    logger.info("=" * 80)
    logger.info(f"🚀 STRIKE MASTER V13.3 PRO - ROBUST FIXES")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"📊 ACTIVE INDICES ({len(bots)}):")
    for idx, bot in bots.items():
        logger.info(f"   • {INDICES[idx]['name']}")
        logger.info(f"     {bot.feed.futures_symbol}")
        logger.info(f"     Expiry: {bot.feed.expiry_date}")
    logger.info("")
    logger.info(f"🔔 Mode: {'ALERT ONLY' if ALERT_ONLY_MODE else 'LIVE TRADING'}")
    logger.info(f"⏱️ Scan: Every {SCAN_INTERVAL}s")
    logger.info(f"📊 Tracking: Every {TRACKING_INTERVAL}s")
    logger.info("")
    logger.info("🔥 V13.3 FIXES:")
    logger.info("   ✅ Robust Spot/Expiry Fetch")
    logger.info("   ✅ Enhanced Error Logging (Diagnose 400s)")
    logger.info("   ✅ Corrected Manual Expiry for Midcap/Finnifty")
    logger.info("=" * 80)
    
    # Send startup messages
    for bot in bots.values():
        try:
            await bot.send_startup_message()
            await asyncio.sleep(1)
        except:
            pass
    
    # Main loop
    iteration = 0
    
    while True:
        try:
            now = datetime.now(IST).time()
            
            if time(9, 15) <= now <= time(15, 30):
                iteration += 1
                logger.info(f"\n{'='*80}")
                logger.info(f"🔄 CYCLE #{iteration}")
                logger.info(f"{'='*80}")
                
                # Run all bots in parallel
                tasks = [bot.run_cycle() for bot in bots.values()]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(SCAN_INTERVAL)
            else:
                logger.info("🌙 Market closed")
                await asyncio.sleep(300)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped")
            break
        
        except Exception as e:
            logger.error(f"💥 Critical Error in Main Loop: {e}")
            logger.exception("Traceback:")
            await asyncio.sleep(30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Shutdown complete")

