"""
🚀 UPSTOX OPTIONS BOT v8.0 - FULLY FIXED & ENHANCED
====================================================
All bugs from v7.0 fixed:
✅ PCR Interpretation Corrected (Rising PCR = Bullish)
✅ 15-Minute OI Data Properly Populated
✅ Signal Threshold Actually Enforced
✅ Multi-Timeframe Data Stored Correctly
✅ Historical API Format Fixed
✅ Cache Interval Lookup Improved
✅ Rate Limiting Protection Added
✅ Retry Logic Implemented
✅ Timezone Consistency Fixed
✅ Pattern Deduplication Added
✅ S/R Confluence Matching Improved
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, timedelta
import json
import logging
import gzip
from typing import Dict, List, Optional, Tuple
from telegram import Bot
from telegram.error import TelegramError
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import mplfinance as mpf
from io import BytesIO
import pytz
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import os

# ======================== CONFIGURATION ========================
UPSTOX_API_URL = "https://api.upstox.com/v2"
UPSTOX_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

# Trading params
ANALYSIS_INTERVAL = 5 * 60  # 5 minutes
CANDLES_COUNT = 200  # 200 candles for chart
ATM_RANGE = 3  # ±3 strikes

# ✅ FIXED: Signal Thresholds
SIGNAL_THRESHOLDS = {
    "MIN_OI_CHANGE_PCT": 5.0,      # Minimum 5% OI change to consider significant
    "MIN_PCR_CHANGE": 0.15,         # Minimum PCR change to consider significant
    "MIN_PRICE_CHANGE_PCT": 0.1,    # Minimum 0.1% price change
    "MIN_CONFIDENCE_SCORE": 3,      # Minimum score to generate signal (out of 10)
    "STRONG_PCR_BULLISH": 2.5,      # PCR > 2.5 = Strong Support
    "STRONG_PCR_BEARISH": 0.5,      # PCR < 0.5 = Strong Resistance
    "MODERATE_PCR_BULLISH": 1.5,    # PCR > 1.5 = Moderate Support
    "MODERATE_PCR_BEARISH": 0.7,    # PCR < 0.7 = Moderate Resistance
}

# ✅ API Rate Limiting
API_DELAY = 0.15  # 150ms between API calls (safer)
MAX_RETRIES = 3   # Retry failed requests

# Market hours (IST)
MARKET_START = dt_time(9, 15)
MARKET_END = dt_time(15, 30)
IST = pytz.timezone('Asia/Kolkata')

# All Major Indices
INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]

# ✅ EXPIRY DAY MAPPING (All on TUESDAY as per your preference)
EXPIRY_DAYS = {
    "NIFTY": 1,       # Tuesday
    "BANKNIFTY": 1,   # Tuesday
    "FINNIFTY": 1,    # Tuesday
    "MIDCPNIFTY": 1,  # Tuesday
}

# Strike Intervals
STRIKE_INTERVALS = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
}

# Lot Sizes
LOT_SIZES = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 40,
    "MIDCPNIFTY": 75,
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ======================== INDIAN NUMBER FORMATTING ========================
def format_indian_number(num: float, decimal_places: int = 2) -> str:
    """Format number in Indian style (Lakh/Crore)"""
    if num is None or num == 0:
        return "0"
    
    abs_num = abs(num)
    sign = "-" if num < 0 else ""
    
    if abs_num >= 10000000:  # 1 Crore
        value = abs_num / 10000000
        if value >= 100:
            return f"{sign}{value:.0f}Cr"
        elif value >= 10:
            return f"{sign}{value:.1f}Cr"
        else:
            return f"{sign}{value:.2f}Cr"
    elif abs_num >= 100000:  # 1 Lakh
        value = abs_num / 100000
        if value >= 100:
            return f"{sign}{value:.0f}L"
        elif value >= 10:
            return f"{sign}{value:.1f}L"
        else:
            return f"{sign}{value:.2f}L"
    elif abs_num >= 1000:  # Thousands
        value = abs_num / 1000
        return f"{sign}{value:.1f}K"
    else:
        return f"{sign}{abs_num:.{decimal_places}f}"


def format_indian_number_full(num: float) -> str:
    """Format with full Indian comma separation"""
    if num is None:
        return "0"
    
    num = int(num)
    sign = "-" if num < 0 else ""
    num = abs(num)
    
    s = str(num)
    if len(s) <= 3:
        return sign + s
    
    result = s[-3:]
    s = s[:-3]
    
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]
    
    return sign + result


# ======================== ENUMS FOR SIGNALS ========================
class SignalType(Enum):
    STRONG_BULLISH = "🟢🟢 STRONG BULLISH"
    BULLISH = "🟢 BULLISH"
    WEAK_BULLISH = "🟡 WEAK BULLISH"
    NEUTRAL = "⚪ NEUTRAL"
    WEAK_BEARISH = "🟡 WEAK BEARISH"
    BEARISH = "🔴 BEARISH"
    STRONG_BEARISH = "🔴🔴 STRONG BEARISH"


class ActionType(Enum):
    BUY_AGGRESSIVE = "BUY AGGRESSIVELY"
    BUY = "BUY"
    BUY_DIP = "BUY DIP"
    HOLD = "HOLD"
    EXIT_LONGS = "EXIT LONGS"
    SELL = "SELL"
    SELL_AGGRESSIVE = "SELL AGGRESSIVELY"
    WAIT = "WAIT"


# ======================== DATA CLASSES ========================
@dataclass
class StrikeData:
    """Data for a single strike"""
    strike: int
    ce_oi: int = 0
    pe_oi: int = 0
    ce_ltp: float = 0.0
    pe_ltp: float = 0.0
    ce_volume: int = 0
    pe_volume: int = 0
    pcr: float = 0.0
    timestamp: datetime = None


@dataclass
class SymbolSnapshot:
    """Complete snapshot of a symbol at a point in time"""
    symbol: str
    timestamp: datetime
    spot_price: float
    atm_strike: int
    total_ce_oi: int = 0
    total_pe_oi: int = 0
    overall_pcr: float = 0.0
    strikes_data: Dict[int, StrikeData] = field(default_factory=dict)


# ======================== IN-MEMORY CACHE (ENHANCED) ========================
class InMemoryCache:
    """
    ✅ ENHANCED IN-MEMORY CACHE
    - Stores snapshots with proper timestamp tracking
    - Supports multi-timeframe lookups (5min, 15min, 30min)
    """
    
    def __init__(self, max_snapshots: int = 100):
        self.max_snapshots = max_snapshots
        self._cache: Dict[str, deque] = {}
        self._lock = asyncio.Lock()
    
    async def add_snapshot(self, snapshot: SymbolSnapshot):
        """Add a new snapshot to cache"""
        async with self._lock:
            if snapshot.symbol not in self._cache:
                self._cache[snapshot.symbol] = deque(maxlen=self.max_snapshots)
            self._cache[snapshot.symbol].append(snapshot)
            logger.debug(f"📦 Cached snapshot for {snapshot.symbol} at {snapshot.timestamp}")
    
    async def get_previous_snapshot(self, symbol: str, minutes_ago: int = 5) -> Optional[SymbolSnapshot]:
        """Get snapshot from approximately N minutes ago"""
        async with self._lock:
            if symbol not in self._cache or len(self._cache[symbol]) < 2:
                return None
            
            target_time = datetime.now(IST) - timedelta(minutes=minutes_ago)
            
            # Find closest snapshot to target time
            best_snapshot = None
            min_diff = float('inf')
            
            for snapshot in self._cache[symbol]:
                diff = abs((snapshot.timestamp - target_time).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    best_snapshot = snapshot
            
            # ✅ FIXED: More lenient time matching (within 3 minutes)
            if best_snapshot and min_diff <= 180:
                return best_snapshot
            
            # Fallback: return second-to-last if available
            if len(self._cache[symbol]) >= 2:
                return self._cache[symbol][-2]
            
            return None
    
    async def get_snapshot_at_interval(self, symbol: str, minutes_ago: int) -> Optional[SymbolSnapshot]:
        """
        ✅ FIXED: Get snapshot closest to specified minutes ago
        With proper fallback and tolerance
        """
        async with self._lock:
            if symbol not in self._cache or len(self._cache[symbol]) == 0:
                return None
            
            target_time = datetime.now(IST) - timedelta(minutes=minutes_ago)
            best_snapshot = None
            min_diff = float('inf')
            
            for snapshot in self._cache[symbol]:
                diff = abs((snapshot.timestamp - target_time).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    best_snapshot = snapshot
            
            # ✅ FIXED: More lenient - within half the interval window
            max_tolerance = (minutes_ago / 2) * 60  # Half the interval in seconds
            if best_snapshot and min_diff <= max(180, max_tolerance):
                return best_snapshot
            
            # If no match, try to return oldest available if we don't have enough history
            snapshots_list = list(self._cache[symbol])
            if snapshots_list:
                return snapshots_list[0]  # Return oldest
            
            return None
    
    async def get_latest_snapshot(self, symbol: str) -> Optional[SymbolSnapshot]:
        """Get most recent snapshot"""
        async with self._lock:
            if symbol not in self._cache or len(self._cache[symbol]) == 0:
                return None
            return self._cache[symbol][-1]
    
    async def get_all_snapshots(self, symbol: str) -> List[SymbolSnapshot]:
        """Get all cached snapshots for a symbol"""
        async with self._lock:
            if symbol not in self._cache:
                return []
            return list(self._cache[symbol])
    
    async def get_snapshots_in_range(self, symbol: str, minutes_back: int) -> List[SymbolSnapshot]:
        """Get all snapshots within the last N minutes"""
        async with self._lock:
            if symbol not in self._cache:
                return []
            
            cutoff_time = datetime.now(IST) - timedelta(minutes=minutes_back)
            return [s for s in self._cache[symbol] if s.timestamp >= cutoff_time]
    
    def get_cache_size(self, symbol: str) -> int:
        """Get number of cached snapshots for a symbol"""
        if symbol not in self._cache:
            return 0
        return len(self._cache[symbol])


# ======================== CANDLESTICK PATTERN DETECTION ========================
class CandlestickPatterns:
    """
    ✅ TOP 6 STRONGEST CANDLESTICK PATTERNS
    With deduplication and improved detection
    """
    
    @staticmethod
    def is_hammer(row, prev_row=None) -> Tuple[bool, str, str]:
        """Hammer - Strong Bullish Reversal"""
        body = abs(row['close'] - row['open'])
        upper_wick = row['high'] - max(row['open'], row['close'])
        lower_wick = min(row['open'], row['close']) - row['low']
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, "", ""
        
        if (lower_wick > body * 2 and 
            upper_wick < body * 0.3 and 
            body < total_range * 0.35):
            return True, "🔨 HAMMER", "BULLISH"
        return False, "", ""
    
    @staticmethod
    def is_shooting_star(row, prev_row=None) -> Tuple[bool, str, str]:
        """Shooting Star - Strong Bearish Reversal"""
        body = abs(row['close'] - row['open'])
        upper_wick = row['high'] - max(row['open'], row['close'])
        lower_wick = min(row['open'], row['close']) - row['low']
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, "", ""
        
        if (upper_wick > body * 2 and 
            lower_wick < body * 0.3 and 
            body < total_range * 0.35):
            return True, "⭐ SHOOTING STAR", "BEARISH"
        return False, "", ""
    
    @staticmethod
    def is_engulfing(row, prev_row) -> Tuple[bool, str, str]:
        """Bullish/Bearish Engulfing - Very Strong"""
        if prev_row is None:
            return False, "", ""
        
        curr_body = abs(row['close'] - row['open'])
        prev_body = abs(prev_row['close'] - prev_row['open'])
        
        if prev_body == 0:
            return False, "", ""
        
        # Bullish engulfing
        if (row['close'] > row['open'] and 
            prev_row['close'] < prev_row['open'] and
            row['open'] <= prev_row['close'] and
            row['close'] >= prev_row['open'] and
            curr_body > prev_body * 1.1):
            return True, "🟢 BULL ENGULF", "BULLISH"
        
        # Bearish engulfing
        if (row['close'] < row['open'] and 
            prev_row['close'] > prev_row['open'] and
            row['open'] >= prev_row['close'] and
            row['close'] <= prev_row['open'] and
            curr_body > prev_body * 1.1):
            return True, "🔴 BEAR ENGULF", "BEARISH"
        
        return False, "", ""
    
    @staticmethod
    def is_morning_star(df, idx) -> Tuple[bool, str, str]:
        """Morning Star - Strong Bullish Reversal (3 candle pattern)"""
        if idx < 2:
            return False, "", ""
        
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        
        first_body = abs(first['close'] - first['open'])
        second_body = abs(second['close'] - second['open'])
        
        if first_body == 0:
            return False, "", ""
        
        first_red = first['close'] < first['open']
        second_small = second_body < first_body * 0.3
        third_green = third['close'] > third['open']
        third_closes_high = third['close'] > (first['open'] + first['close']) / 2
        
        if first_red and second_small and third_green and third_closes_high:
            return True, "🌅 MORNING STAR", "BULLISH"
        return False, "", ""
    
    @staticmethod
    def is_evening_star(df, idx) -> Tuple[bool, str, str]:
        """Evening Star - Strong Bearish Reversal (3 candle pattern)"""
        if idx < 2:
            return False, "", ""
        
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        
        first_body = abs(first['close'] - first['open'])
        second_body = abs(second['close'] - second['open'])
        
        if first_body == 0:
            return False, "", ""
        
        first_green = first['close'] > first['open']
        second_small = second_body < first_body * 0.3
        third_red = third['close'] < third['open']
        third_closes_low = third['close'] < (first['open'] + first['close']) / 2
        
        if first_green and second_small and third_red and third_closes_low:
            return True, "🌆 EVENING STAR", "BEARISH"
        return False, "", ""
    
    @staticmethod
    def is_doji(row, prev_row=None) -> Tuple[bool, str, str]:
        """Doji - Indecision/Reversal Warning"""
        body = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, "", ""
        
        if body < total_range * 0.1:
            return True, "✖️ DOJI", "NEUTRAL"
        return False, "", ""
    
    @staticmethod
    def detect_all_patterns(df: pd.DataFrame, volume_data: Dict = None) -> List[Dict]:
        """
        ✅ FIXED: Detect all patterns with DEDUPLICATION
        """
        patterns = []
        detected_indices = set()  # ✅ Track already detected indices
        
        for i in range(len(df)):
            # ✅ Skip if pattern already detected at this index
            if i in detected_indices:
                continue
            
            row = df.iloc[i]
            prev_row = df.iloc[i-1] if i > 0 else None
            
            # Get volume (if available)
            candle_volume = 0
            if volume_data:
                candle_volume = volume_data.get(row.name, 0)
            
            avg_volume = np.mean(list(volume_data.values())) if volume_data and len(volume_data) > 0 else 0
            high_volume = candle_volume > avg_volume * 1.2 if avg_volume > 0 else False
            
            pattern_found = False
            
            # Check patterns in order of strength (strongest first)
            
            # 1. Engulfing (strongest)
            if not pattern_found:
                is_pat, name, bias = CandlestickPatterns.is_engulfing(row, prev_row)
                if is_pat:
                    patterns.append({
                        "index": i,
                        "time": row.name,
                        "pattern": name,
                        "type": bias.lower(),
                        "price": row['close'],
                        "high": row['high'],
                        "low": row['low'],
                        "high_volume": high_volume,
                        "strength": 5
                    })
                    detected_indices.add(i)
                    pattern_found = True
            
            # 2. Morning Star
            if not pattern_found:
                is_pat, name, bias = CandlestickPatterns.is_morning_star(df, i)
                if is_pat:
                    patterns.append({
                        "index": i,
                        "time": row.name,
                        "pattern": name,
                        "type": bias.lower(),
                        "price": row['close'],
                        "high": row['high'],
                        "low": row['low'],
                        "high_volume": high_volume,
                        "strength": 5
                    })
                    detected_indices.add(i)
                    detected_indices.add(i-1)
                    detected_indices.add(i-2)
                    pattern_found = True
            
            # 3. Evening Star
            if not pattern_found:
                is_pat, name, bias = CandlestickPatterns.is_evening_star(df, i)
                if is_pat:
                    patterns.append({
                        "index": i,
                        "time": row.name,
                        "pattern": name,
                        "type": bias.lower(),
                        "price": row['close'],
                        "high": row['high'],
                        "low": row['low'],
                        "high_volume": high_volume,
                        "strength": 5
                    })
                    detected_indices.add(i)
                    detected_indices.add(i-1)
                    detected_indices.add(i-2)
                    pattern_found = True
            
            # 4. Hammer
            if not pattern_found:
                is_pat, name, bias = CandlestickPatterns.is_hammer(row, prev_row)
                if is_pat:
                    patterns.append({
                        "index": i,
                        "time": row.name,
                        "pattern": name,
                        "type": bias.lower(),
                        "price": row['close'],
                        "high": row['high'],
                        "low": row['low'],
                        "high_volume": high_volume,
                        "strength": 4
                    })
                    detected_indices.add(i)
                    pattern_found = True
            
            # 5. Shooting Star
            if not pattern_found:
                is_pat, name, bias = CandlestickPatterns.is_shooting_star(row, prev_row)
                if is_pat:
                    patterns.append({
                        "index": i,
                        "time": row.name,
                        "pattern": name,
                        "type": bias.lower(),
                        "price": row['close'],
                        "high": row['high'],
                        "low": row['low'],
                        "high_volume": high_volume,
                        "strength": 4
                    })
                    detected_indices.add(i)
                    pattern_found = True
            
            # 6. Doji (only if no other pattern)
            if not pattern_found:
                is_pat, name, bias = CandlestickPatterns.is_doji(row, prev_row)
                if is_pat:
                    patterns.append({
                        "index": i,
                        "time": row.name,
                        "pattern": name,
                        "type": bias.lower(),
                        "price": row['close'],
                        "high": row['high'],
                        "low": row['low'],
                        "high_volume": high_volume,
                        "strength": 3
                    })
                    detected_indices.add(i)
        
        return patterns


# ======================== SUPPORT/RESISTANCE WITH OI CONFLUENCE ========================
class SupportResistanceAnalyzer:
    """
    ✅ FIXED: S/R Detection using BOTH Price Action + OI Data
    With improved confluence matching
    """
    
    @staticmethod
    def find_pivot_points(df: pd.DataFrame, window: int = 5) -> Tuple[List, List]:
        """Find pivot highs and lows from price data"""
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(df) - window):
            is_pivot_high = True
            is_pivot_low = True
            
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            for j in range(i - window, i + window + 1):
                if j == i:
                    continue
                if df.iloc[j]['high'] >= current_high:
                    is_pivot_high = False
                if df.iloc[j]['low'] <= current_low:
                    is_pivot_low = False
            
            if is_pivot_high:
                pivot_highs.append({
                    "index": i,
                    "time": df.index[i],
                    "price": current_high,
                    "type": "resistance"
                })
            
            if is_pivot_low:
                pivot_lows.append({
                    "index": i,
                    "time": df.index[i],
                    "price": current_low,
                    "type": "support"
                })
        
        return pivot_highs, pivot_lows
    
    @staticmethod
    def cluster_levels(levels: List[float], tolerance_pct: float = 0.3) -> List[Dict]:
        """Cluster nearby price levels into zones"""
        if not levels:
            return []
        
        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            cluster_avg = np.mean(current_cluster)
            if abs(level - cluster_avg) / cluster_avg * 100 < tolerance_pct:
                current_cluster.append(level)
            else:
                clusters.append({
                    "price": np.mean(current_cluster),
                    "touches": len(current_cluster),
                    "strength": len(current_cluster)
                })
                current_cluster = [level]
        
        clusters.append({
            "price": np.mean(current_cluster),
            "touches": len(current_cluster),
            "strength": len(current_cluster)
        })
        
        return clusters
    
    @staticmethod
    def identify_oi_based_levels(strikes_data: Dict, strike_interval: int) -> Dict:
        """
        ✅ Identify S/R levels from OI concentration
        High PE OI = Support, High CE OI = Resistance
        """
        support_levels = []
        resistance_levels = []
        
        max_pe_oi = max([s.pe_oi for s in strikes_data.values()]) if strikes_data else 1
        max_ce_oi = max([s.ce_oi for s in strikes_data.values()]) if strikes_data else 1
        
        for strike, data in strikes_data.items():
            pe_oi_ratio = data.pe_oi / max_pe_oi if max_pe_oi > 0 else 0
            ce_oi_ratio = data.ce_oi / max_ce_oi if max_ce_oi > 0 else 0
            
            # High PE OI = Strong Support
            if pe_oi_ratio > 0.7:
                support_levels.append({
                    "price": strike,
                    "oi": data.pe_oi,
                    "pcr": data.pcr,
                    "oi_strength": pe_oi_ratio,
                    "source": "OI"
                })
            
            # High CE OI = Strong Resistance
            if ce_oi_ratio > 0.7:
                resistance_levels.append({
                    "price": strike,
                    "oi": data.ce_oi,
                    "pcr": data.pcr,
                    "oi_strength": ce_oi_ratio,
                    "source": "OI"
                })
        
        return {
            "oi_supports": sorted(support_levels, key=lambda x: x['oi'], reverse=True),
            "oi_resistances": sorted(resistance_levels, key=lambda x: x['oi'], reverse=True)
        }
    
    @staticmethod
    def combine_price_and_oi_levels(
        price_supports: List[Dict],
        price_resistances: List[Dict],
        oi_supports: List[Dict],
        oi_resistances: List[Dict],
        current_price: float,
        strike_interval: int
    ) -> Dict:
        """
        ✅ FIXED: COMBINE Price Action S/R with OI-based S/R
        With improved confluence matching tolerance
        """
        final_supports = []
        final_resistances = []
        
        # ✅ FIXED: Use 1.5x strike interval for better matching
        tolerance = strike_interval * 1.5
        
        # Process supports
        for ps in price_supports:
            oi_match = None
            for os in oi_supports:
                if abs(ps['price'] - os['price']) <= tolerance:
                    oi_match = os
                    break
            
            if oi_match:
                final_supports.append({
                    "price": (ps['price'] + oi_match['price']) / 2,
                    "strength": ps['strength'] + 3,
                    "touches": ps['touches'],
                    "oi": oi_match.get('oi', 0),
                    "pcr": oi_match.get('pcr', 0),
                    "source": "CONFLUENCE",
                    "confluence": True
                })
            else:
                final_supports.append({
                    "price": ps['price'],
                    "strength": ps['strength'],
                    "touches": ps['touches'],
                    "oi": 0,
                    "pcr": 0,
                    "source": "PRICE",
                    "confluence": False
                })
        
        # Add OI-only supports not matched
        for os in oi_supports:
            matched = False
            for fs in final_supports:
                if abs(fs['price'] - os['price']) <= tolerance:
                    matched = True
                    break
            
            if not matched:
                final_supports.append({
                    "price": os['price'],
                    "strength": int(os['oi_strength'] * 3),
                    "touches": 0,
                    "oi": os['oi'],
                    "pcr": os['pcr'],
                    "source": "OI",
                    "confluence": False
                })
        
        # Process resistances
        for pr in price_resistances:
            oi_match = None
            for or_ in oi_resistances:
                if abs(pr['price'] - or_['price']) <= tolerance:
                    oi_match = or_
                    break
            
            if oi_match:
                final_resistances.append({
                    "price": (pr['price'] + oi_match['price']) / 2,
                    "strength": pr['strength'] + 3,
                    "touches": pr['touches'],
                    "oi": oi_match.get('oi', 0),
                    "pcr": oi_match.get('pcr', 0),
                    "source": "CONFLUENCE",
                    "confluence": True
                })
            else:
                final_resistances.append({
                    "price": pr['price'],
                    "strength": pr['strength'],
                    "touches": pr['touches'],
                    "oi": 0,
                    "pcr": 0,
                    "source": "PRICE",
                    "confluence": False
                })
        
        # Add OI-only resistances
        for or_ in oi_resistances:
            matched = False
            for fr in final_resistances:
                if abs(fr['price'] - or_['price']) <= tolerance:
                    matched = True
                    break
            
            if not matched:
                final_resistances.append({
                    "price": or_['price'],
                    "strength": int(or_['oi_strength'] * 3),
                    "touches": 0,
                    "oi": or_['oi'],
                    "pcr": or_['pcr'],
                    "source": "OI",
                    "confluence": False
                })
        
        # Filter and sort
        final_supports = [s for s in final_supports if s['price'] < current_price]
        final_resistances = [r for r in final_resistances if r['price'] > current_price]
        
        final_supports = sorted(final_supports, key=lambda x: (x['confluence'], x['strength']), reverse=True)[:3]
        final_resistances = sorted(final_resistances, key=lambda x: (x['confluence'], x['strength']), reverse=True)[:3]
        
        return {
            "supports": final_supports,
            "resistances": final_resistances
        }


# ======================== OI CHANGE ANALYZER (FIXED) ========================
class OIChangeAnalyzer:
    """
    ✅ FIXED: OI CHANGE ANALYSIS
    - Properly calculates 5min and 15min changes
    - Stores historical strike data for comparison
    """
    
    def __init__(self, cache: InMemoryCache):
        self.cache = cache
    
    async def calculate_oi_changes(self, symbol: str, current: SymbolSnapshot) -> Dict:
        """
        ✅ FIXED: Calculate OI changes from previous snapshots
        Including 5min, 15min data for each strike
        """
        previous_5m = await self.cache.get_snapshot_at_interval(symbol, 5)
        previous_15m = await self.cache.get_snapshot_at_interval(symbol, 15)
        
        if not previous_5m:
            logger.info(f"⚠️ No previous snapshot for {symbol}, first run")
            return {
                "has_previous": False,
                "price_change": 0,
                "price_change_pct": 0,
                "total_ce_oi_change": 0,
                "total_pe_oi_change": 0,
                "pcr_change": 0,
                "strike_changes": {}
            }
        
        # Calculate overall changes from 5min ago
        price_change = current.spot_price - previous_5m.spot_price
        price_change_pct = (price_change / previous_5m.spot_price * 100) if previous_5m.spot_price > 0 else 0
        
        total_ce_oi_change = current.total_ce_oi - previous_5m.total_ce_oi
        total_pe_oi_change = current.total_pe_oi - previous_5m.total_pe_oi
        
        ce_oi_change_pct = (total_ce_oi_change / previous_5m.total_ce_oi * 100) if previous_5m.total_ce_oi > 0 else 0
        pe_oi_change_pct = (total_pe_oi_change / previous_5m.total_pe_oi * 100) if previous_5m.total_pe_oi > 0 else 0
        
        pcr_change = current.overall_pcr - previous_5m.overall_pcr
        
        # ✅ FIXED: Calculate per-strike changes INCLUDING 15min data
        strike_changes = {}
        for strike, curr_data in current.strikes_data.items():
            prev_5m_data = previous_5m.strikes_data.get(strike) if previous_5m else None
            prev_15m_data = previous_15m.strikes_data.get(strike) if previous_15m else None
            
            # 5-minute ago data
            if prev_5m_data:
                ce_change = curr_data.ce_oi - prev_5m_data.ce_oi
                pe_change = curr_data.pe_oi - prev_5m_data.pe_oi
                pcr_strike_change = curr_data.pcr - prev_5m_data.pcr
                
                ce_change_pct = (ce_change / prev_5m_data.ce_oi * 100) if prev_5m_data.ce_oi > 0 else 0
                pe_change_pct = (pe_change / prev_5m_data.pe_oi * 100) if prev_5m_data.pe_oi > 0 else 0
                
                prev_ce_oi = prev_5m_data.ce_oi
                prev_pe_oi = prev_5m_data.pe_oi
                prev_pcr = prev_5m_data.pcr
            else:
                ce_change = 0
                pe_change = 0
                ce_change_pct = 0
                pe_change_pct = 0
                pcr_strike_change = 0
                prev_ce_oi = curr_data.ce_oi
                prev_pe_oi = curr_data.pe_oi
                prev_pcr = curr_data.pcr
            
            # ✅ FIXED: 15-minute ago data
            if prev_15m_data:
                ce_15m = prev_15m_data.ce_oi
                pe_15m = prev_15m_data.pe_oi
            else:
                # Fallback to 5min data or current
                ce_15m = prev_ce_oi
                pe_15m = prev_pe_oi
            
            strike_changes[strike] = {
                "ce_oi_change": ce_change,
                "pe_oi_change": pe_change,
                "ce_oi_change_pct": ce_change_pct,
                "pe_oi_change_pct": pe_change_pct,
                "pcr_change": pcr_strike_change,
                "prev_ce_oi": prev_ce_oi,
                "prev_pe_oi": prev_pe_oi,
                "prev_pcr": prev_pcr,
                "curr_ce_oi": curr_data.ce_oi,
                "curr_pe_oi": curr_data.pe_oi,
                # ✅ FIXED: Include 15min data
                "ce_15m": ce_15m,
                "pe_15m": pe_15m,
            }
        
        return {
            "has_previous": True,
            "time_diff_seconds": (current.timestamp - previous_5m.timestamp).total_seconds(),
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "total_ce_oi_change": total_ce_oi_change,
            "total_pe_oi_change": total_pe_oi_change,
            "ce_oi_change_pct": ce_oi_change_pct,
            "pe_oi_change_pct": pe_oi_change_pct,
            "pcr_change": pcr_change,
            "prev_pcr": previous_5m.overall_pcr,
            "curr_pcr": current.overall_pcr,
            "strike_changes": strike_changes,
            "has_15m_data": previous_15m is not None
        }
    
    def analyze_scenario(self, oi_changes: Dict) -> Dict:
        """
        ✅ IMPLEMENT 9 SCENARIOS FROM PDF
        Based on Price Movement + Put OI Change + Call OI Change
        """
        if not oi_changes.get("has_previous"):
            return {
                "scenario": 0,
                "signal": SignalType.NEUTRAL,
                "action": ActionType.WAIT,
                "description": "Waiting for data...",
                "confidence": 0
            }
        
        price_pct = oi_changes["price_change_pct"]
        pe_oi_change_pct = oi_changes["pe_oi_change_pct"]
        ce_oi_change_pct = oi_changes["ce_oi_change_pct"]
        
        # Thresholds
        price_up = price_pct > SIGNAL_THRESHOLDS["MIN_PRICE_CHANGE_PCT"]
        price_down = price_pct < -SIGNAL_THRESHOLDS["MIN_PRICE_CHANGE_PCT"]
        price_sideways = not price_up and not price_down
        
        pe_oi_up = pe_oi_change_pct > SIGNAL_THRESHOLDS["MIN_OI_CHANGE_PCT"]
        pe_oi_down = pe_oi_change_pct < -SIGNAL_THRESHOLDS["MIN_OI_CHANGE_PCT"]
        pe_oi_same = not pe_oi_up and not pe_oi_down
        
        ce_oi_up = ce_oi_change_pct > SIGNAL_THRESHOLDS["MIN_OI_CHANGE_PCT"]
        ce_oi_down = ce_oi_change_pct < -SIGNAL_THRESHOLDS["MIN_OI_CHANGE_PCT"]
        ce_oi_same = not ce_oi_up and not ce_oi_down
        
        # ========== 9 SCENARIOS FROM PDF ==========
        
        # Scenario 1: Price ⬆️ + Put OI ⬇️ + Call OI Same = STRONG BULLISH (Put Unwinding)
        if price_up and pe_oi_down and ce_oi_same:
            return {
                "scenario": 1,
                "signal": SignalType.STRONG_BULLISH,
                "action": ActionType.BUY_AGGRESSIVE,
                "description": "PUT UNWINDING - Bulls winning, shorts covering",
                "confidence": 9,
                "details": f"Price +{price_pct:.2f}% | Put OI {pe_oi_change_pct:.1f}%"
            }
        
        # Scenario 2: Price ⬆️ + Put OI Same + Call OI ⬇️ = STRONG BULLISH (Call Unwinding)
        if price_up and pe_oi_same and ce_oi_down:
            return {
                "scenario": 2,
                "signal": SignalType.STRONG_BULLISH,
                "action": ActionType.BUY,
                "description": "CALL UNWINDING - Resistance broken, bears exiting",
                "confidence": 8,
                "details": f"Price +{price_pct:.2f}% | Call OI {ce_oi_change_pct:.1f}%"
            }
        
        # Scenario 3: Price ⬆️ + Put OI Same + Call OI ⬆️ = BEARISH (Resistance Building)
        if price_up and pe_oi_same and ce_oi_up:
            return {
                "scenario": 3,
                "signal": SignalType.BEARISH,
                "action": ActionType.EXIT_LONGS,
                "description": "CALL WRITING - Resistance building as price rises",
                "confidence": 6,
                "details": f"Price +{price_pct:.2f}% | Call OI +{ce_oi_change_pct:.1f}%"
            }
        
        # Scenario 4: Price ⬇️ + Put OI ⬆️ + Call OI Same = BULLISH (Support Building)
        if price_down and pe_oi_up and ce_oi_same:
            return {
                "scenario": 4,
                "signal": SignalType.BULLISH,
                "action": ActionType.BUY_DIP,
                "description": "PUT WRITING - Support building at lower levels",
                "confidence": 7,
                "details": f"Price {price_pct:.2f}% | Put OI +{pe_oi_change_pct:.1f}%"
            }
        
        # Scenario 5: Price ⬇️ + Put OI Same + Call OI ⬇️ = STRONG BEARISH
        if price_down and pe_oi_same and ce_oi_down:
            return {
                "scenario": 5,
                "signal": SignalType.STRONG_BEARISH,
                "action": ActionType.SELL,
                "description": "CALL UNWINDING - Bulls losing, more downside expected",
                "confidence": 8,
                "details": f"Price {price_pct:.2f}% | Call OI {ce_oi_change_pct:.1f}%"
            }
        
        # Scenario 6: Price ⬇️ + Put OI ⬇️ + Call OI Same = STRONG BEARISH (Panic)
        if price_down and pe_oi_down and ce_oi_same:
            return {
                "scenario": 6,
                "signal": SignalType.STRONG_BEARISH,
                "action": ActionType.SELL_AGGRESSIVE,
                "description": "PUT UNWINDING - Panic selling, bears winning",
                "confidence": 9,
                "details": f"Price {price_pct:.2f}% | Put OI {pe_oi_change_pct:.1f}%"
            }
        
        # Scenario 7: Price ⬆️ + Put OI ⬆️ + Call OI Same = WEAK BULLISH (Hedging)
        if price_up and pe_oi_up and ce_oi_same:
            return {
                "scenario": 7,
                "signal": SignalType.WEAK_BULLISH,
                "action": ActionType.HOLD,
                "description": "PUT BUYING - Rise with hedging/protection",
                "confidence": 4,
                "details": f"Price +{price_pct:.2f}% | Put OI +{pe_oi_change_pct:.1f}%"
            }
        
        # Scenario 8: Price Sideways + Put OI ⬆️⬆️ = SUPPORT ZONE
        if price_sideways and pe_oi_up and pe_oi_change_pct > 10:
            return {
                "scenario": 8,
                "signal": SignalType.BULLISH,
                "action": ActionType.BUY,
                "description": "MAJOR SUPPORT ZONE - Heavy put writing",
                "confidence": 7,
                "details": f"Put OI +{pe_oi_change_pct:.1f}% (Support Building)"
            }
        
        # Scenario 9: Price Sideways + Call OI ⬆️⬆️ = RESISTANCE ZONE
        if price_sideways and ce_oi_up and ce_oi_change_pct > 10:
            return {
                "scenario": 9,
                "signal": SignalType.BEARISH,
                "action": ActionType.SELL,
                "description": "MAJOR RESISTANCE ZONE - Heavy call writing",
                "confidence": 7,
                "details": f"Call OI +{ce_oi_change_pct:.1f}% (Resistance Building)"
            }
        
        # Both OI increasing with price up
        if price_up and pe_oi_up and ce_oi_up:
            return {
                "scenario": 10,
                "signal": SignalType.NEUTRAL,
                "action": ActionType.HOLD,
                "description": "MIXED - Both sides adding positions",
                "confidence": 3,
                "details": "Wait for clarity"
            }
        
        # Both OI decreasing
        if pe_oi_down and ce_oi_down:
            direction = SignalType.BULLISH if price_up else (SignalType.BEARISH if price_down else SignalType.NEUTRAL)
            return {
                "scenario": 11,
                "signal": direction,
                "action": ActionType.HOLD,
                "description": "UNWINDING - Both sides closing, follow price",
                "confidence": 5,
                "details": f"Price {price_pct:+.2f}% | Overall unwinding"
            }
        
        # Default
        return {
            "scenario": 0,
            "signal": SignalType.NEUTRAL,
            "action": ActionType.WAIT,
            "description": "No clear signal - Wait for confirmation",
            "confidence": 2,
            "details": f"Price {price_pct:+.2f}%"
        }


# ======================== MULTI-TIMEFRAME ANALYZER ========================
class MultiTimeframeAnalyzer:
    """✅ MULTI-TIMEFRAME OI ANALYSIS"""
    
    def __init__(self, cache: InMemoryCache):
        self.cache = cache
    
    async def analyze_timeframes(self, symbol: str, current: SymbolSnapshot) -> Dict:
        """Analyze OI changes across multiple timeframes"""
        
        snapshot_5min = await self.cache.get_snapshot_at_interval(symbol, 5)
        snapshot_15min = await self.cache.get_snapshot_at_interval(symbol, 15)
        snapshot_30min = await self.cache.get_snapshot_at_interval(symbol, 30)
        
        result = {
            "5min": self._calculate_tf_change(current, snapshot_5min, "5min"),
            "15min": self._calculate_tf_change(current, snapshot_15min, "15min"),
            "30min": self._calculate_tf_change(current, snapshot_30min, "30min"),
            "trend_alignment": "NEUTRAL",
            "trend_strength": 0
        }
        
        # Analyze trend alignment
        bullish_count = 0
        bearish_count = 0
        
        for tf in ["5min", "15min", "30min"]:
            if result[tf]["available"]:
                if result[tf]["pcr_trend"] == "BULLISH":
                    bullish_count += 1
                elif result[tf]["pcr_trend"] == "BEARISH":
                    bearish_count += 1
        
        if bullish_count >= 2:
            result["trend_alignment"] = "BULLISH"
            result["trend_strength"] = bullish_count
        elif bearish_count >= 2:
            result["trend_alignment"] = "BEARISH"
            result["trend_strength"] = bearish_count
        else:
            result["trend_alignment"] = "MIXED"
            result["trend_strength"] = 0
        
        return result
    
    def _calculate_tf_change(self, current: SymbolSnapshot, previous: Optional[SymbolSnapshot], tf_name: str) -> Dict:
        """Calculate changes for a specific timeframe"""
        if not previous:
            return {
                "available": False,
                "timeframe": tf_name,
                "price_change": 0,
                "ce_oi_change_pct": 0,
                "pe_oi_change_pct": 0,
                "pcr_change": 0,
                "pcr_trend": "UNKNOWN"
            }
        
        price_change = current.spot_price - previous.spot_price
        price_change_pct = (price_change / previous.spot_price * 100) if previous.spot_price > 0 else 0
        
        ce_change_pct = ((current.total_ce_oi - previous.total_ce_oi) / previous.total_ce_oi * 100) if previous.total_ce_oi > 0 else 0
        pe_change_pct = ((current.total_pe_oi - previous.total_pe_oi) / previous.total_pe_oi * 100) if previous.total_pe_oi > 0 else 0
        
        pcr_change = current.overall_pcr - previous.overall_pcr
        
        # ✅ FIXED: Correct PCR trend interpretation
        # Rising PCR = More PE relative to CE = Bullish (support building)
        # Falling PCR = More CE relative to PE = Bearish (resistance building)
        if pcr_change > 0.1:
            pcr_trend = "BULLISH"  # ✅ FIXED: Rising PCR = Bullish
        elif pcr_change < -0.1:
            pcr_trend = "BEARISH"  # ✅ FIXED: Falling PCR = Bearish
        else:
            pcr_trend = "NEUTRAL"
        
        return {
            "available": True,
            "timeframe": tf_name,
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "ce_oi_change_pct": ce_change_pct,
            "pe_oi_change_pct": pe_change_pct,
            "pcr_change": pcr_change,
            "prev_pcr": previous.overall_pcr,
            "curr_pcr": current.overall_pcr,
            "pcr_trend": pcr_trend
        }


# ======================== PCR MOMENTUM ANALYZER (FIXED) ========================
class PCRMomentumAnalyzer:
    """
    ✅ FIXED: PCR CHANGE MOMENTUM
    Correct interpretation: Rising PCR = Bullish (more puts = support)
    """
    
    def __init__(self, cache: InMemoryCache):
        self.cache = cache
    
    async def analyze_pcr_momentum(self, symbol: str, current_pcr: float) -> Dict:
        """Analyze PCR momentum over time"""
        snapshots = await self.cache.get_all_snapshots(symbol)
        
        if len(snapshots) < 3:
            return {
                "momentum": "UNKNOWN",
                "direction": "NEUTRAL",
                "strength": 0,
                "pcr_history": []
            }
        
        # Get last N PCR values
        pcr_history = [(s.timestamp, s.overall_pcr) for s in snapshots[-10:]]
        
        if len(pcr_history) >= 2:
            recent_pcr = pcr_history[-1][1]
            older_pcr = pcr_history[-3][1] if len(pcr_history) >= 3 else pcr_history[0][1]
            
            pcr_change = recent_pcr - older_pcr
            
            # ✅ FIXED: CORRECT PCR MOMENTUM INTERPRETATION
            # Rising PCR = More puts being written/bought = Support building = BULLISH
            # Falling PCR = More calls being written/bought = Resistance building = BEARISH
            if pcr_change > 0.3:
                momentum = "STRONG_RISING"
                direction = "BULLISH"  # ✅ FIXED
                strength = min(10, int(abs(pcr_change) * 10))
            elif pcr_change > 0.1:
                momentum = "RISING"
                direction = "BULLISH"  # ✅ FIXED
                strength = min(7, int(abs(pcr_change) * 10))
            elif pcr_change < -0.3:
                momentum = "STRONG_FALLING"
                direction = "BEARISH"  # ✅ FIXED
                strength = min(10, int(abs(pcr_change) * 10))
            elif pcr_change < -0.1:
                momentum = "FALLING"
                direction = "BEARISH"  # ✅ FIXED
                strength = min(7, int(abs(pcr_change) * 10))
            else:
                momentum = "STABLE"
                direction = "NEUTRAL"
                strength = 3
            
            # Check for extremes
            if current_pcr > 3.0:
                momentum = "OVERSOLD_EXTREME"
                direction = "BULLISH"  # Extreme high PCR = Strong support
                strength = 8
            elif current_pcr < 0.3:
                momentum = "OVERBOUGHT_EXTREME"
                direction = "BEARISH"  # Extreme low PCR = Strong resistance
                strength = 8
            
            return {
                "momentum": momentum,
                "direction": direction,
                "strength": strength,
                "pcr_change": pcr_change,
                "current_pcr": current_pcr,
                "pcr_history": pcr_history[-5:]
            }
        
        return {
            "momentum": "UNKNOWN",
            "direction": "NEUTRAL",
            "strength": 0,
            "pcr_history": pcr_history
        }


# ======================== SIGNAL GENERATOR ========================
class SignalGenerator:
    """✅ COMPREHENSIVE SIGNAL GENERATOR"""
    
    def generate_signal(
        self,
        oi_scenario: Dict,
        mtf_analysis: Dict,
        pcr_momentum: Dict,
        current_pcr: float,
        current_price: float,
        atm_strike: int,
        symbol: str
    ) -> Dict:
        """Generate final trading signal with confidence score"""
        
        confidence_score = 0
        reasons = []
        
        # 1. OI Scenario contribution (0-3 points)
        scenario_confidence = oi_scenario.get("confidence", 0)
        if scenario_confidence >= 7:
            confidence_score += 3
            reasons.append(f"Strong OI signal: {oi_scenario['description']}")
        elif scenario_confidence >= 5:
            confidence_score += 2
            reasons.append(f"Moderate OI signal: {oi_scenario['description']}")
        elif scenario_confidence >= 3:
            confidence_score += 1
            reasons.append(f"Weak OI signal: {oi_scenario['description']}")
        
        # 2. Multi-timeframe alignment (0-3 points)
        if mtf_analysis["trend_alignment"] != "MIXED":
            tf_strength = mtf_analysis["trend_strength"]
            if tf_strength >= 3:
                confidence_score += 3
                reasons.append(f"All timeframes {mtf_analysis['trend_alignment']}")
            elif tf_strength >= 2:
                confidence_score += 2
                reasons.append(f"2/3 timeframes {mtf_analysis['trend_alignment']}")
        
        # 3. PCR Zone contribution (0-2 points)
        if current_pcr > SIGNAL_THRESHOLDS["STRONG_PCR_BULLISH"]:
            confidence_score += 2
            reasons.append(f"Strong Support Zone (PCR {current_pcr:.2f})")
        elif current_pcr > SIGNAL_THRESHOLDS["MODERATE_PCR_BULLISH"]:
            confidence_score += 1
            reasons.append(f"Support Zone (PCR {current_pcr:.2f})")
        elif current_pcr < SIGNAL_THRESHOLDS["STRONG_PCR_BEARISH"]:
            confidence_score += 2
            reasons.append(f"Strong Resistance Zone (PCR {current_pcr:.2f})")
        elif current_pcr < SIGNAL_THRESHOLDS["MODERATE_PCR_BEARISH"]:
            confidence_score += 1
            reasons.append(f"Resistance Zone (PCR {current_pcr:.2f})")
        
        # 4. PCR Momentum contribution (0-2 points)
        if pcr_momentum["strength"] >= 7:
            confidence_score += 2
            reasons.append(f"Strong PCR momentum: {pcr_momentum['momentum']}")
        elif pcr_momentum["strength"] >= 4:
            confidence_score += 1
            reasons.append(f"PCR momentum: {pcr_momentum['momentum']}")
        
        # Determine final signal
        base_signal = oi_scenario.get("signal", SignalType.NEUTRAL)
        base_action = oi_scenario.get("action", ActionType.WAIT)
        
        # Check for confluence
        oi_direction = "BULLISH" if "BULLISH" in base_signal.value else ("BEARISH" if "BEARISH" in base_signal.value else "NEUTRAL")
        mtf_direction = mtf_analysis["trend_alignment"]
        pcr_direction = pcr_momentum["direction"]
        
        directions = [oi_direction, mtf_direction, pcr_direction]
        bullish_count = directions.count("BULLISH")
        bearish_count = directions.count("BEARISH")
        
        if bullish_count >= 2:
            final_bias = "BULLISH"
            if confidence_score >= 7:
                final_signal = SignalType.STRONG_BULLISH
                final_action = ActionType.BUY_AGGRESSIVE
            elif confidence_score >= 5:
                final_signal = SignalType.BULLISH
                final_action = ActionType.BUY
            else:
                final_signal = SignalType.WEAK_BULLISH
                final_action = ActionType.HOLD
        elif bearish_count >= 2:
            final_bias = "BEARISH"
            if confidence_score >= 7:
                final_signal = SignalType.STRONG_BEARISH
                final_action = ActionType.SELL_AGGRESSIVE
            elif confidence_score >= 5:
                final_signal = SignalType.BEARISH
                final_action = ActionType.SELL
            else:
                final_signal = SignalType.WEAK_BEARISH
                final_action = ActionType.EXIT_LONGS
        else:
            final_bias = "NEUTRAL"
            final_signal = base_signal
            final_action = base_action
        
        # Calculate entry/exit levels
        interval = STRIKE_INTERVALS.get(symbol, 50)
        
        if final_bias == "BULLISH":
            entry_strike = atm_strike
            sl_strike = atm_strike - interval
            target_strike = atm_strike + (2 * interval)
            option_type = "CE"
        elif final_bias == "BEARISH":
            entry_strike = atm_strike
            sl_strike = atm_strike + interval
            target_strike = atm_strike - (2 * interval)
            option_type = "PE"
        else:
            entry_strike = atm_strike
            sl_strike = atm_strike
            target_strike = atm_strike
            option_type = "WAIT"
        
        should_alert = confidence_score >= SIGNAL_THRESHOLDS["MIN_CONFIDENCE_SCORE"]
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now(IST),
            "spot_price": current_price,
            "atm_strike": atm_strike,
            "signal": final_signal,
            "action": final_action,
            "bias": final_bias,
            "confidence_score": confidence_score,
            "max_score": 10,
            "option_type": option_type,
            "entry_strike": entry_strike,
            "stop_loss_strike": sl_strike,
            "target_strike": target_strike,
            "oi_scenario": oi_scenario,
            "mtf_analysis": mtf_analysis,
            "pcr_momentum": pcr_momentum,
            "current_pcr": current_pcr,
            "reasons": reasons,
            "should_alert": should_alert,
            "alert_reason": "Confidence score meets threshold" if should_alert else f"Low confidence ({confidence_score}/10)"
        }


# ======================== UPSTOX CLIENT (ENHANCED) ========================
class UpstoxClient:
    """
    ✅ ENHANCED: With retry logic and better error handling
    """
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.instruments_cache = None
        self.futures_keys = {}
    
    async def create_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=timeout)
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    def get_instrument_key(self, symbol: str) -> str:
        mapping = {
            "NIFTY": "NSE_INDEX|Nifty 50",
            "BANKNIFTY": "NSE_INDEX|Nifty Bank",
            "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
            "FINNIFTY": "NSE_INDEX|Nifty Fin Service"
        }
        return mapping.get(symbol, f"NSE_EQ|{symbol}")
    
    async def _request_with_retry(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """✅ NEW: Request with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                async with getattr(self.session, method)(url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"⚠️ Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"⚠️ Request failed: {response.status}")
                        return None
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Timeout on attempt {attempt + 1}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"❌ Request error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None
                await asyncio.sleep(1)
        
        return None
    
    async def download_instruments(self):
        """Download instrument master"""
        try:
            logger.info("📡 Downloading instruments...")
            url = UPSTOX_INSTRUMENTS_URL
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"❌ Download failed: {response.status}")
                    return None
                
                content = await response.read()
                json_text = gzip.decompress(content).decode('utf-8')
                instruments = json.loads(json_text)
                
                logger.info(f"✅ Downloaded {len(instruments)} instruments")
                self.instruments_cache = instruments
                
                now = datetime.now(IST)
                
                for symbol in INDICES:
                    for instrument in instruments:
                        if instrument.get('segment') != 'NSE_FO':
                            continue
                        if instrument.get('instrument_type') != 'FUT':
                            continue
                        if instrument.get('name') != symbol:
                            continue
                        
                        expiry = instrument.get('expiry')
                        if expiry:
                            expiry_dt = datetime.fromtimestamp(expiry / 1000, tz=IST)
                            if expiry_dt > now:
                                self.futures_keys[symbol] = instrument.get('instrument_key')
                                logger.info(f"✅ {symbol} Futures: {self.futures_keys[symbol]}")
                                break
                
                return instruments
                
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return None
    
    async def get_available_expiries(self, symbol: str) -> List[str]:
        """Get available expiries for a symbol"""
        try:
            if not self.instruments_cache:
                instruments = await self.download_instruments()
                if not instruments:
                    return []
            else:
                instruments = self.instruments_cache
            
            now = datetime.now(IST)
            expiries_set = set()
            
            for instrument in instruments:
                if instrument.get('segment') != 'NSE_FO':
                    continue
                if instrument.get('instrument_type') not in ['CE', 'PE']:
                    continue
                if instrument.get('name', '') != symbol:
                    continue
                
                expiry_ms = instrument.get('expiry')
                if not expiry_ms:
                    continue
                
                try:
                    expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=IST)
                    if expiry_dt > now:
                        expiry_str = expiry_dt.strftime('%Y-%m-%d')
                        expiries_set.add(expiry_str)
                except:
                    continue
            
            if expiries_set:
                return sorted(list(expiries_set))
            return []
                
        except Exception as e:
            logger.error(f"❌ Error getting expiries for {symbol}: {e}")
            return []
    
    def get_nearest_expiry_for_symbol(self, symbol: str, expiries: List[str]) -> Optional[str]:
        """Get nearest expiry on preferred weekday (Tuesday)"""
        if not expiries:
            return None
        
        now = datetime.now(IST)
        preferred_weekday = EXPIRY_DAYS.get(symbol, 1)  # Tuesday = 1
        
        for expiry_str in expiries:
            try:
                expiry_dt = datetime.strptime(expiry_str, '%Y-%m-%d')
                expiry_dt = IST.localize(expiry_dt)
                
                if expiry_dt > now and expiry_dt.weekday() == preferred_weekday:
                    return expiry_str
            except:
                continue
        
        return expiries[0] if expiries else None
    
    async def get_full_quote(self, instrument_key: str) -> Dict:
        """Get full quote with retry"""
        try:
            url = f"{UPSTOX_API_URL}/market-quote/quotes"
            params = {"instrument_key": instrument_key}
            
            data = await self._request_with_retry('get', url, params=params)
            
            if data and data.get("status") == "success" and data.get("data"):
                for key, value in data["data"].items():
                    return {
                        "ltp": value.get("last_price", 0.0),
                        "volume": value.get("volume", 0),
                        "oi": value.get("oi", 0)
                    }
            
            return {"ltp": 0.0, "volume": 0, "oi": 0}
        except Exception as e:
            logger.error(f"❌ Quote error: {e}")
            return {"ltp": 0.0, "volume": 0, "oi": 0}
    
    async def get_ltp(self, instrument_key: str) -> float:
        quote = await self.get_full_quote(instrument_key)
        return quote["ltp"]
    
    async def get_historical_candles(self, instrument_key: str) -> pd.DataFrame:
        """
        ✅ FIXED: Get historical candles with proper API format
        """
        try:
            all_candles = []
            today = datetime.now(IST).date()
            
            # Get historical data for previous days
            for days_back in range(1, 6):
                historical_date = today - timedelta(days=days_back)
                
                # Skip weekends
                if historical_date.weekday() >= 5:
                    continue
                
                to_date = historical_date.strftime('%Y-%m-%d')
                from_date = historical_date.strftime('%Y-%m-%d')
                
                # ✅ FIXED: Correct V2 API format
                url = f"{UPSTOX_API_URL}/historical-candle/{instrument_key}/5minute/{to_date}/{from_date}"
                
                try:
                    data = await self._request_with_retry('get', url)
                    
                    if data and data.get("status") == "success" and data.get("data", {}).get("candles"):
                        hist_candles = data["data"]["candles"]
                        
                        for candle in hist_candles:
                            all_candles.append({
                                'timestamp': pd.to_datetime(candle[0]),
                                'open': candle[1],
                                'high': candle[2],
                                'low': candle[3],
                                'close': candle[4],
                                'volume': candle[5] if len(candle) > 5 else 0,
                                'oi': candle[6] if len(candle) > 6 else 0
                            })
                        
                        logger.debug(f"✅ Historical {to_date}: {len(hist_candles)} candles")
                except Exception as e:
                    logger.warning(f"⚠️ Historical data for {to_date} failed: {e}")
                
                await asyncio.sleep(API_DELAY)
            
            # Get intraday data for today
            url_intraday = f"{UPSTOX_API_URL}/historical-candle/intraday/{instrument_key}/5minute"
            
            data = await self._request_with_retry('get', url_intraday)
            
            if data and data.get("status") == "success" and data.get("data", {}).get("candles"):
                intraday_candles = data["data"]["candles"]
                
                for candle in intraday_candles:
                    all_candles.append({
                        'timestamp': pd.to_datetime(candle[0]),
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5] if len(candle) > 5 else 0,
                        'oi': candle[6] if len(candle) > 6 else 0
                    })
                
                logger.info(f"✅ Intraday: {len(intraday_candles)} candles")
            
            if not all_candles:
                logger.warning(f"⚠️ No candles found for {instrument_key}")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_candles)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]
            
            # Keep last 200 candles
            if len(df) > CANDLES_COUNT:
                df = df.tail(CANDLES_COUNT)
            
            logger.info(f"✅ Total candles loaded: {len(df)}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching candles: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    async def get_option_contracts(self, symbol: str, expiry: str) -> List[Dict]:
        """Get option contracts for symbol and expiry"""
        try:
            instrument_key = self.get_instrument_key(symbol)
            url = f"{UPSTOX_API_URL}/option/contract"
            
            params = {
                "instrument_key": instrument_key,
                "expiry_date": expiry
            }
            
            data = await self._request_with_retry('get', url, params=params)
            
            if data and data.get("status") == "success":
                contracts = data.get("data", [])
                if contracts:
                    logger.info(f"✅ Fetched {len(contracts)} option contracts for {symbol}")
                    return contracts
                return []
            return []
                    
        except Exception as e:
            logger.error(f"❌ Error fetching contracts for {symbol}: {e}")
            return []


# ======================== OPTION ANALYZER ========================
class OptionAnalyzer:
    """Main analysis orchestrator"""
    
    def __init__(self, client: UpstoxClient, cache: InMemoryCache):
        self.client = client
        self.cache = cache
        self.oi_analyzer = OIChangeAnalyzer(cache)
        self.mtf_analyzer = MultiTimeframeAnalyzer(cache)
        self.pcr_momentum = PCRMomentumAnalyzer(cache)
        self.signal_generator = SignalGenerator()
    
    def get_strike_interval(self, symbol: str) -> int:
        return STRIKE_INTERVALS.get(symbol, 50)
    
    async def filter_atm_strikes(self, contracts: List[Dict], current_price: float, symbol: str) -> Dict:
        interval = self.get_strike_interval(symbol)
        atm = round(current_price / interval) * interval
        
        min_strike = atm - (ATM_RANGE * interval)
        max_strike = atm + (ATM_RANGE * interval)
        
        logger.info(f"🎯 ATM: {atm}, Range: {min_strike} to {max_strike} (±{ATM_RANGE} strikes)")
        
        ce_contracts = {}
        pe_contracts = {}
        
        for contract in contracts:
            strike = contract.get("strike_price")
            if min_strike <= strike <= max_strike:
                instrument_key = contract.get("instrument_key")
                option_type = contract.get("instrument_type")
                
                contract_data = {
                    "strike": strike,
                    "instrument_key": instrument_key,
                    "trading_symbol": contract.get("trading_symbol"),
                    "ltp": 0, "oi": 0, "volume": 0
                }
                
                if option_type == "CE":
                    ce_contracts[strike] = contract_data
                elif option_type == "PE":
                    pe_contracts[strike] = contract_data
        
        return {
            "ce": ce_contracts,
            "pe": pe_contracts,
            "strikes": sorted(set(list(ce_contracts.keys()) + list(pe_contracts.keys())))
        }
    
    async def fetch_option_prices(self, contracts_data: Dict):
        """Fetch prices with proper rate limiting"""
        for strike, contract in contracts_data["ce"].items():
            quote = await self.client.get_full_quote(contract["instrument_key"])
            contract["ltp"] = quote["ltp"]
            contract["oi"] = quote["oi"]
            contract["volume"] = quote["volume"]
            await asyncio.sleep(API_DELAY)  # ✅ FIXED: Use configurable delay
        
        for strike, contract in contracts_data["pe"].items():
            quote = await self.client.get_full_quote(contract["instrument_key"])
            contract["ltp"] = quote["ltp"]
            contract["oi"] = quote["oi"]
            contract["volume"] = quote["volume"]
            await asyncio.sleep(API_DELAY)
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Main analysis method"""
        try:
            logger.info(f"\n📊 Analyzing {symbol}...")
            
            instrument_key = self.client.get_instrument_key(symbol)
            current_price = await self.client.get_ltp(instrument_key)
            
            if current_price == 0:
                logger.warning(f"⚠️ Could not fetch price for {symbol}")
                return None
            
            logger.info(f"💰 {symbol} Spot: ₹{current_price:,.2f}")
            
            # Get candles
            candles = await self.client.get_historical_candles(instrument_key)
            
            if candles.empty:
                logger.warning(f"⚠️ No candle data for {symbol}")
                return None
            
            logger.info(f"📈 Loaded {len(candles)} candles for {symbol}")
            
            # Get expiry
            expiries = await self.client.get_available_expiries(symbol)
            if not expiries:
                logger.warning(f"⚠️ No expiries found for {symbol}")
                return None
            
            expiry = self.client.get_nearest_expiry_for_symbol(symbol, expiries)
            
            if not expiry:
                return None
            
            contracts = await self.client.get_option_contracts(symbol, expiry)
            
            if not contracts:
                return None
            
            contracts_data = await self.filter_atm_strikes(contracts, current_price, symbol)
            
            if not contracts_data["strikes"]:
                return None
            
            await self.fetch_option_prices(contracts_data)
            
            # Calculate totals
            total_ce_oi = 0
            total_pe_oi = 0
            
            strikes_data = {}
            
            for strike in contracts_data["strikes"]:
                ce = contracts_data["ce"].get(strike, {"oi": 0, "ltp": 0, "volume": 0})
                pe = contracts_data["pe"].get(strike, {"oi": 0, "ltp": 0, "volume": 0})
                
                total_ce_oi += ce.get("oi", 0)
                total_pe_oi += pe.get("oi", 0)
                
                strike_pcr = pe.get("oi", 0) / ce.get("oi", 1) if ce.get("oi", 0) > 0 else 0
                
                strikes_data[strike] = StrikeData(
                    strike=strike,
                    ce_oi=ce.get("oi", 0),
                    pe_oi=pe.get("oi", 0),
                    ce_ltp=ce.get("ltp", 0),
                    pe_ltp=pe.get("ltp", 0),
                    ce_volume=ce.get("volume", 0),
                    pe_volume=pe.get("volume", 0),
                    pcr=strike_pcr,
                    timestamp=datetime.now(IST)
                )
            
            overall_pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            atm_strike = round(current_price / self.get_strike_interval(symbol)) * self.get_strike_interval(symbol)
            
            # Detect patterns
            volume_data = {candles.index[i]: candles.iloc[i].get('volume', 0) for i in range(len(candles))}
            patterns = CandlestickPatterns.detect_all_patterns(candles, volume_data)
            logger.info(f"🎯 Found {len(patterns)} candlestick patterns")
            
            # S/R Analysis
            strike_interval = self.get_strike_interval(symbol)
            pivot_highs, pivot_lows = SupportResistanceAnalyzer.find_pivot_points(candles, window=5)
            
            price_supports = SupportResistanceAnalyzer.cluster_levels([p['price'] for p in pivot_lows])
            price_resistances = SupportResistanceAnalyzer.cluster_levels([p['price'] for p in pivot_highs])
            
            oi_levels = SupportResistanceAnalyzer.identify_oi_based_levels(strikes_data, strike_interval)
            
            sr_levels = SupportResistanceAnalyzer.combine_price_and_oi_levels(
                price_supports,
                price_resistances,
                oi_levels['oi_supports'],
                oi_levels['oi_resistances'],
                current_price,
                strike_interval
            )
            
            logger.info(f"📊 S/R Levels: {len(sr_levels['supports'])} supports, {len(sr_levels['resistances'])} resistances")
            
            # Create snapshot
            current_snapshot = SymbolSnapshot(
                symbol=symbol,
                timestamp=datetime.now(IST),
                spot_price=current_price,
                atm_strike=atm_strike,
                total_ce_oi=total_ce_oi,
                total_pe_oi=total_pe_oi,
                overall_pcr=overall_pcr,
                strikes_data=strikes_data
            )
            
            # OI Change Analysis
            oi_changes = await self.oi_analyzer.calculate_oi_changes(symbol, current_snapshot)
            oi_scenario = self.oi_analyzer.analyze_scenario(oi_changes)
            
            # Multi-Timeframe Analysis
            mtf_analysis = await self.mtf_analyzer.analyze_timeframes(symbol, current_snapshot)
            
            # PCR Momentum
            pcr_momentum_data = await self.pcr_momentum.analyze_pcr_momentum(symbol, overall_pcr)
            
            # Generate Signal
            final_signal = self.signal_generator.generate_signal(
                oi_scenario=oi_scenario,
                mtf_analysis=mtf_analysis,
                pcr_momentum=pcr_momentum_data,
                current_pcr=overall_pcr,
                current_price=current_price,
                atm_strike=atm_strike,
                symbol=symbol
            )
            
            # Save snapshot
            await self.cache.add_snapshot(current_snapshot)
            
            logger.info(f"✅ {symbol}: Signal={final_signal['signal'].value}, Confidence={final_signal['confidence_score']}/10")
            
            # Prepare response
            ce_data = []
            pe_data = []
            
            for s in contracts_data["strikes"]:
                ce = contracts_data["ce"].get(s, {"strike": s, "ltp": 0, "oi": 0, "volume": 0})
                pe = contracts_data["pe"].get(s, {"strike": s, "ltp": 0, "oi": 0, "volume": 0})
                
                strike_data = strikes_data.get(s)
                ce["pcr"] = strike_data.pcr if strike_data else 0
                pe["pcr"] = strike_data.pcr if strike_data else 0
                
                ce_data.append(ce)
                pe_data.append(pe)
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "expiry": expiry,
                "candles": candles,
                "strikes": contracts_data["strikes"],
                "ce_data": ce_data,
                "pe_data": pe_data,
                "atm_strike": atm_strike,
                "total_ce_oi": total_ce_oi,
                "total_pe_oi": total_pe_oi,
                "overall_pcr": overall_pcr,
                "lot_size": LOT_SIZES.get(symbol, 25),
                "patterns": patterns,
                "sr_levels": sr_levels,
                "oi_changes": oi_changes,
                "oi_scenario": oi_scenario,
                "mtf_analysis": mtf_analysis,
                "pcr_momentum": pcr_momentum_data,
                "final_signal": final_signal,
                "cache_size": self.cache.get_cache_size(symbol)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


# ======================== CHART GENERATOR ========================
class ChartGenerator:
    """✅ FIXED Chart Generator with all improvements"""
    
    @staticmethod
    def create_combined_chart(analysis: Dict) -> BytesIO:
        """Create comprehensive chart with all analysis"""
        symbol = analysis["symbol"]
        candles = analysis["candles"]
        current_price = analysis["current_price"]
        overall_pcr = analysis["overall_pcr"]
        final_signal = analysis["final_signal"]
        oi_changes = analysis.get("oi_changes", {})
        mtf_analysis = analysis.get("mtf_analysis", {})
        pcr_momentum = analysis.get("pcr_momentum", {})
        patterns = analysis.get("patterns", [])
        sr_levels = analysis.get("sr_levels", {"supports": [], "resistances": []})
        lot_size = analysis.get("lot_size", 25)
        
        now_time = datetime.now(IST).strftime('%H:%M:%S IST')
        
        # Signal color
        signal_type = final_signal["signal"]
        if "BULLISH" in signal_type.value:
            trend_color = "#26a69a"
        elif "BEARISH" in signal_type.value:
            trend_color = "#ef5350"
        else:
            trend_color = "#757575"
        
        # Create figure
        fig = plt.figure(figsize=(36, 26), facecolor='#0d1117')
        gs = GridSpec(6, 2, height_ratios=[3.0, 0.6, 0.6, 0.6, 0.6, 2.2], width_ratios=[1.5, 1], hspace=0.25, wspace=0.15)
        
        # ========== CANDLESTICK CHART ==========
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor('#161b22')
        
        mc = mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge='inherit',
            wick={'up': '#26a69a', 'down': '#ef5350'},
            volume='in', alpha=0.9
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc, gridstyle='--', gridcolor='#30363d',
            facecolor='#161b22', figcolor='#0d1117', y_on_right=False
        )
        
        candles_display = candles.tail(100).copy()
        
        if not candles_display.empty:
            mpf.plot(candles_display, type='candle', style=s, ax=ax1, volume=False, show_nontrading=False)
        
        # Draw Support Levels
        supports = sr_levels.get('supports', [])
        for i, support in enumerate(supports[:3]):
            support_price = support['price']
            confluence = support.get('confluence', False)
            source = support.get('source', 'PRICE')
            oi_val = support.get('oi', 0)
            
            if confluence:
                linestyle, linewidth, alpha = '-', 3.5, 0.9
            elif source == 'OI':
                linestyle, linewidth, alpha = '--', 2.5, 0.8
            else:
                linestyle, linewidth, alpha = ':', 2, 0.7
            
            ax1.axhline(y=support_price, color='#00ff88', linestyle=linestyle, linewidth=linewidth, alpha=alpha)
            
            label = f"S{i+1}: ₹{support_price:,.0f}"
            if oi_val > 0:
                label += f" ({format_indian_number(oi_val)})"
            if confluence:
                label = f"★ {label}"
            
            ax1.text(1.01, support_price, label, transform=ax1.get_yaxis_transform(),
                    fontsize=11, fontweight='bold', color='#00ff88', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', edgecolor='#00ff88', alpha=0.95, linewidth=2))
        
        # Draw Resistance Levels
        resistances = sr_levels.get('resistances', [])
        for i, resistance in enumerate(resistances[:3]):
            resistance_price = resistance['price']
            confluence = resistance.get('confluence', False)
            source = resistance.get('source', 'PRICE')
            oi_val = resistance.get('oi', 0)
            
            if confluence:
                linestyle, linewidth, alpha = '-', 3.5, 0.9
            elif source == 'OI':
                linestyle, linewidth, alpha = '--', 2.5, 0.8
            else:
                linestyle, linewidth, alpha = ':', 2, 0.7
            
            ax1.axhline(y=resistance_price, color='#ff6b6b', linestyle=linestyle, linewidth=linewidth, alpha=alpha)
            
            label = f"R{i+1}: ₹{resistance_price:,.0f}"
            if oi_val > 0:
                label += f" ({format_indian_number(oi_val)})"
            if confluence:
                label = f"★ {label}"
            
            ax1.text(1.01, resistance_price, label, transform=ax1.get_yaxis_transform(),
                    fontsize=11, fontweight='bold', color='#ff6b6b', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', edgecolor='#ff6b6b', alpha=0.95, linewidth=2))
        
        # Mark Patterns
        if patterns and not candles_display.empty:
            visible_times = set(candles_display.index)
            recent_patterns = [p for p in patterns if p['time'] in visible_times][-10:]
            
            for pattern in recent_patterns:
                candle_time = pattern['time']
                pattern_type = pattern['type']
                pattern_name = pattern['pattern']
                
                try:
                    candle_idx = candles_display.index.get_loc(candle_time)
                    candle_high = candles_display.iloc[candle_idx]['high']
                    candle_low = candles_display.iloc[candle_idx]['low']
                except:
                    continue
                
                if pattern_type == 'bullish':
                    color, y_pos, offset_y = '#00ff88', candle_low, -35
                elif pattern_type == 'bearish':
                    color, y_pos, offset_y = '#ff6b6b', candle_high, 35
                else:
                    color, y_pos, offset_y = '#ffd700', candle_high, 30
                
                vol_mark = "📊" if pattern.get('high_volume') else ""
                
                short_names = {
                    "🔨 HAMMER": "🔨", "⭐ SHOOTING STAR": "⭐",
                    "🟢 BULL ENGULF": "🟢ENG", "🔴 BEAR ENGULF": "🔴ENG",
                    "🌅 MORNING STAR": "🌅", "🌆 EVENING STAR": "🌆", "✖️ DOJI": "✖️"
                }
                short_name = short_names.get(pattern_name, pattern_name[:6])
                
                ax1.annotate(
                    f"{short_name}{vol_mark}", xy=(candle_time, y_pos),
                    xytext=(0, offset_y), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=color, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', edgecolor=color, alpha=0.95, linewidth=1.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=color, lw=1.5)
                )
        
        # Title
        title_text = f"{symbol} | ₹{current_price:,.2f} | {signal_type.value} | Conf: {final_signal['confidence_score']}/10 | PCR: {overall_pcr:.2f} | {now_time}"
        ax1.set_title(title_text, fontsize=20, fontweight='bold', pad=15, color='white')
        ax1.tick_params(colors='white', labelsize=10)
        for spine in ax1.spines.values():
            spine.set_color('#30363d')
        
        # ========== SIGNAL BOX ==========
        ax_signal = fig.add_subplot(gs[1, 0])
        ax_signal.axis('off')
        
        signal_text = f"🎯 TRADING SIGNAL\n{'='*40}\n\n"
        signal_text += f"Signal: {final_signal['signal'].value}\n"
        signal_text += f"Action: {final_signal['action'].value}\n"
        signal_text += f"Confidence: {final_signal['confidence_score']}/10\n"
        
        if final_signal['option_type'] != "WAIT":
            signal_text += f"\n📊 Trade: {final_signal['entry_strike']} {final_signal['option_type']}\n"
            signal_text += f"SL: {final_signal['stop_loss_strike']} | TGT: {final_signal['target_strike']}"
        
        sig_bg = '#1b4332' if "BULLISH" in signal_type.value else ('#4a1c1c' if "BEARISH" in signal_type.value else '#21262d')
        
        ax_signal.text(0.02, 0.95, signal_text, transform=ax_signal.transAxes,
                      fontsize=11, verticalalignment='top', fontfamily='monospace', color='white',
                      bbox=dict(boxstyle='round,pad=0.6', facecolor=sig_bg, edgecolor=trend_color, alpha=0.95, linewidth=3))
        
        # ========== OI CHANGE ==========
        ax_oi = fig.add_subplot(gs[1, 1])
        ax_oi.axis('off')
        
        oi_text = f"📈 OI ANALYSIS\n{'='*35}\n\n"
        
        if oi_changes.get("has_previous"):
            oi_text += f"Price: {oi_changes['price_change']:+.2f} ({oi_changes['price_change_pct']:+.2f}%)\n"
            oi_text += f"CE OI: {oi_changes['ce_oi_change_pct']:+.1f}%\n"
            oi_text += f"PE OI: {oi_changes['pe_oi_change_pct']:+.1f}%\n"
            oi_text += f"PCR Δ: {oi_changes['pcr_change']:+.3f}\n\n"
            scenario = analysis.get("oi_scenario", {})
            oi_text += f"#{scenario.get('scenario', 0)}: {scenario.get('description', 'N/A')[:30]}"
        else:
            oi_text += "⏳ Building OI history...\n(Need 2+ cycles)"
        
        ax_oi.text(0.02, 0.95, oi_text, transform=ax_oi.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace', color='white',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e3a5f', edgecolor='#4a90d9', alpha=0.95, linewidth=2))
        
        # ========== S/R INFO ==========
        ax_sr = fig.add_subplot(gs[2, 0])
        ax_sr.axis('off')
        
        sr_text = f"📍 SUPPORT / RESISTANCE\n{'='*40}\n\n"
        sr_text += "SUPPORT (🟢):\n"
        for s_level in supports[:3]:
            src = "★CONF" if s_level.get('confluence') else s_level.get('source', 'PRC')[:3]
            sr_text += f"  ₹{s_level['price']:,.0f} [{src}]"
            if s_level.get('oi', 0) > 0:
                sr_text += f" OI:{format_indian_number(s_level['oi'])}"
            sr_text += "\n"
        
        sr_text += "\nRESISTANCE (🔴):\n"
        for r_level in resistances[:3]:
            src = "★CONF" if r_level.get('confluence') else r_level.get('source', 'PRC')[:3]
            sr_text += f"  ₹{r_level['price']:,.0f} [{src}]"
            if r_level.get('oi', 0) > 0:
                sr_text += f" OI:{format_indian_number(r_level['oi'])}"
            sr_text += "\n"
        
        ax_sr.text(0.02, 0.95, sr_text, transform=ax_sr.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace', color='white',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d1f3d', edgecolor='#9c27b0', alpha=0.95, linewidth=2))
        
        # ========== PATTERNS ==========
        ax_pat = fig.add_subplot(gs[2, 1])
        ax_pat.axis('off')
        
        pat_text = f"🕯️ CANDLESTICK PATTERNS\n{'='*35}\n\n"
        
        recent_pats = patterns[-6:] if patterns else []
        if recent_pats:
            for p in recent_pats:
                time_str = p['time'].strftime('%H:%M')
                vol = "📊" if p.get('high_volume') else ""
                pat_text += f"{time_str} {p['pattern'][:18]} {vol}\n"
        else:
            pat_text += "No patterns detected yet\n"
        
        ax_pat.text(0.02, 0.95, pat_text, transform=ax_pat.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace', color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a3a1a', edgecolor='#4caf50', alpha=0.95, linewidth=2))
        
        # ========== MTF ==========
        ax_mtf = fig.add_subplot(gs[3, 0])
        ax_mtf.axis('off')
        
        mtf_text = f"⏱️ MULTI-TIMEFRAME\n{'='*40}\n\n"
        mtf_text += f"Alignment: {mtf_analysis.get('trend_alignment', 'N/A')}\n\n"
        
        for tf in ["5min", "15min", "30min"]:
            tf_data = mtf_analysis.get(tf, {})
            if tf_data.get("available"):
                emoji = "🟢" if tf_data['pcr_trend'] == "BULLISH" else ("🔴" if tf_data['pcr_trend'] == "BEARISH" else "⚪")
                mtf_text += f"{tf}: {emoji} PCR {tf_data.get('prev_pcr', 0):.2f}→{tf_data.get('curr_pcr', 0):.2f}\n"
            else:
                mtf_text += f"{tf}: ⏳ Building...\n"
        
        ax_mtf.text(0.02, 0.95, mtf_text, transform=ax_mtf.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace', color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#3d2c1e', edgecolor='#ff9800', alpha=0.95, linewidth=2))
        
        # ========== SYSTEM INFO ==========
        ax_info = fig.add_subplot(gs[3, 1])
        ax_info.axis('off')
        
        info_text = f"📊 PCR & SYSTEM\n{'='*35}\n\n"
        info_text += f"PCR: {overall_pcr:.3f}\n"
        info_text += f"Momentum: {pcr_momentum.get('momentum', 'N/A')}\n"
        info_text += f"Direction: {pcr_momentum.get('direction', 'N/A')}\n\n"
        info_text += f"Candles: {len(candles)}\n"
        info_text += f"Cache: {analysis.get('cache_size', 0)}\n"
        info_text += f"Expiry: {analysis['expiry']}\n"
        info_text += f"Lot: {lot_size}"
        
        ax_info.text(0.02, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#3d1a2e', edgecolor='#e91e63', alpha=0.95, linewidth=2))
        
        # ========== OI TABLE ==========
        ax_table = fig.add_subplot(gs[4:, :])
        ax_table.axis('tight')
        ax_table.axis('off')
        
        strike_changes = oi_changes.get("strike_changes", {})
        
        # 14 Column table
        table_data = [[
            "Strike", "PE Now", "PE 5m", "PE 15m", "PE Δ%",
            "CE Now", "CE 5m", "CE 15m", "CE Δ%",
            "PCR", "PCR Δ", "CE₹", "PE₹", "Zone"
        ]]
        
        atm_strike = analysis['atm_strike']
        total_ce_oi = analysis["total_ce_oi"]
        total_pe_oi = analysis["total_pe_oi"]
        
        for i, strike in enumerate(analysis["strikes"]):
            ce = analysis["ce_data"][i]
            pe = analysis["pe_data"][i]
            pcr = pe.get("pcr", 0)
            
            pe_now = pe['oi']
            ce_now = ce['oi']
            ce_ltp = ce.get('ltp', 0)
            pe_ltp = pe.get('ltp', 0)
            
            s_change = strike_changes.get(strike, {})
            pe_5m = s_change.get("prev_pe_oi", pe_now)
            ce_5m = s_change.get("prev_ce_oi", ce_now)
            
            # ✅ FIXED: Now properly populated from cache
            pe_15m = s_change.get("pe_15m", pe_5m)
            ce_15m = s_change.get("ce_15m", ce_5m)
            
            # Calculate changes
            pe_change_pct = ((pe_now - pe_15m) / pe_15m * 100) if pe_15m > 0 else 0
            ce_change_pct = ((ce_now - ce_15m) / ce_15m * 100) if ce_15m > 0 else 0
            
            prev_pcr = s_change.get("prev_pcr", pcr)
            pcr_change = pcr - prev_pcr
            
            # Format changes
            pe_delta = f"↑{pe_change_pct:+.1f}%" if pe_change_pct > 5 else (f"↓{pe_change_pct:+.1f}%" if pe_change_pct < -5 else ("—" if pe_change_pct == 0 else f"{pe_change_pct:+.1f}%"))
            ce_delta = f"↑{ce_change_pct:+.1f}%" if ce_change_pct > 5 else (f"↓{ce_change_pct:+.1f}%" if ce_change_pct < -5 else ("—" if ce_change_pct == 0 else f"{ce_change_pct:+.1f}%"))
            pcr_delta = f"↑{pcr_change:+.2f}" if pcr_change > 0.1 else (f"↓{pcr_change:+.2f}" if pcr_change < -0.1 else ("—" if pcr_change == 0 else f"{pcr_change:+.2f}"))
            
            # Zone
            if pcr > 2.5:
                zone = "🟢🟢SUP"
            elif pcr > 1.5:
                zone = "🟢Sup"
            elif pcr < 0.5:
                zone = "🔴🔴RES"
            elif pcr < 0.7:
                zone = "🔴Res"
            else:
                zone = "⚪Neu"
            
            row = [
                f"₹{strike:,}{'*' if strike == atm_strike else ''}",
                format_indian_number(pe_now), format_indian_number(pe_5m), format_indian_number(pe_15m), pe_delta,
                format_indian_number(ce_now), format_indian_number(ce_5m), format_indian_number(ce_15m), ce_delta,
                f"{pcr:.2f}", pcr_delta, f"₹{ce_ltp:.1f}", f"₹{pe_ltp:.1f}", zone
            ]
            table_data.append(row)
        
        # Total row
        overall_pe_change = oi_changes.get("pe_oi_change_pct", 0)
        overall_ce_change = oi_changes.get("ce_oi_change_pct", 0)
        overall_pcr_change = oi_changes.get("pcr_change", 0)
        
        overall_pe_delta = f"↑{overall_pe_change:+.1f}%" if overall_pe_change > 5 else (f"↓{overall_pe_change:+.1f}%" if overall_pe_change < -5 else (f"{overall_pe_change:+.1f}%" if oi_changes.get("has_previous") else "—"))
        overall_ce_delta = f"↑{overall_ce_change:+.1f}%" if overall_ce_change > 5 else (f"↓{overall_ce_change:+.1f}%" if overall_ce_change < -5 else (f"{overall_ce_change:+.1f}%" if oi_changes.get("has_previous") else "—"))
        overall_pcr_delta = f"↑{overall_pcr_change:+.2f}" if overall_pcr_change > 0.05 else (f"↓{overall_pcr_change:+.2f}" if overall_pcr_change < -0.05 else (f"{overall_pcr_change:+.2f}" if oi_changes.get("has_previous") else "—"))
        
        table_data.append([
            "TOTAL",
            format_indian_number(total_pe_oi), "", "", overall_pe_delta,
            format_indian_number(total_ce_oi), "", "", overall_ce_delta,
            f"{overall_pcr:.2f}", overall_pcr_delta, "", "", final_signal['signal'].value[:8]
        ])
        
        table = ax_table.table(
            cellText=table_data, loc='center', cellLoc='center',
            colWidths=[0.08, 0.065, 0.06, 0.06, 0.065, 0.065, 0.06, 0.06, 0.065, 0.055, 0.065, 0.055, 0.055, 0.08]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(14):
            table[(0, i)].set_facecolor('#0f3460')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=9)
        
        # Style data rows
        for row_idx in range(1, len(table_data)):
            for col_idx in range(14):
                cell = table[(row_idx, col_idx)]
                cell.set_facecolor('#161b22')
                cell.set_text_props(color='white', fontsize=8)
        
        # Style summary row
        summary_row = len(table_data) - 1
        for i in range(14):
            table[(summary_row, i)].set_facecolor('#1b4332')
            table[(summary_row, i)].set_text_props(weight='bold', color='#00ff88', fontsize=9)
        
        # Highlight ATM
        for i, strike in enumerate(analysis["strikes"], 1):
            if strike == atm_strike:
                for j in range(14):
                    table[(i, j)].set_facecolor('#2d4263')
                    table[(i, j)].set_text_props(weight='bold', color='#ffd700', fontsize=8)
        
        # Color change columns
        for row_idx in range(1, len(table_data)):
            # PE Δ
            pe_text = table_data[row_idx][4]
            if '↑' in pe_text:
                table[(row_idx, 4)].set_text_props(color='#00ff88')
            elif '↓' in pe_text:
                table[(row_idx, 4)].set_text_props(color='#ff6b6b')
            
            # CE Δ
            ce_text = table_data[row_idx][8]
            if '↑' in ce_text:
                table[(row_idx, 8)].set_text_props(color='#ff6b6b')
            elif '↓' in ce_text:
                table[(row_idx, 8)].set_text_props(color='#00ff88')
            
            # PCR Δ
            pcr_text = table_data[row_idx][10]
            if '↑' in pcr_text:
                table[(row_idx, 10)].set_text_props(color='#00ff88')
            elif '↓' in pcr_text:
                table[(row_idx, 10)].set_text_props(color='#ff6b6b')
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0d1117', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf


# ======================== TELEGRAM ALERTER ========================
class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
    
    async def send_chart(self, chart_buffer: BytesIO, symbol: str, analysis: Dict):
        try:
            final_signal = analysis.get("final_signal", {})
            oi_changes = analysis.get("oi_changes", {})
            mtf_analysis = analysis.get("mtf_analysis", {})
            
            signal_type = final_signal.get("signal", SignalType.NEUTRAL)
            confidence = final_signal.get("confidence_score", 0)
            
            caption = f"""📊 {symbol} Analysis v8.0 (FIXED)

💰 Spot: ₹{analysis['current_price']:,.2f}
📅 Expiry: {analysis['expiry']}

🎯 SIGNAL: {signal_type.value}
📈 Confidence: {confidence}/10
🎬 Action: {final_signal.get('action', ActionType.WAIT).value}

📊 PCR: {analysis['overall_pcr']:.3f}
📈 CE OI: {format_indian_number(analysis['total_ce_oi'])}
📉 PE OI: {format_indian_number(analysis['total_pe_oi'])}"""
            
            if oi_changes.get("has_previous"):
                caption += f"""

🔄 OI Changes:
   Price: {oi_changes['price_change']:+.2f} ({oi_changes['price_change_pct']:+.2f}%)
   CE OI: {oi_changes['ce_oi_change_pct']:+.1f}%
   PE OI: {oi_changes['pe_oi_change_pct']:+.1f}%
   PCR Δ: {oi_changes['pcr_change']:+.3f}"""
            
            caption += f"""

⏱️ Multi-TF: {mtf_analysis.get('trend_alignment', 'N/A')}"""
            
            if final_signal.get("option_type") != "WAIT":
                caption += f"""

💼 Trade Setup:
   {final_signal['entry_strike']} {final_signal['option_type']}
   SL: {final_signal['stop_loss_strike']}
   TGT: {final_signal['target_strike']}"""
            
            caption += f"""

⏰ {datetime.now(IST).strftime('%d-%b %H:%M IST')}
💾 Cache: {analysis.get('cache_size', 0)} snapshots

✅ Enhanced Bot v8.0 (All Fixes Applied)"""
            
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=chart_buffer,
                caption=caption
            )
            
            logger.info(f"✅ Alert sent for {symbol}")
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"❌ Error sending alert for {symbol}: {e}")


# ======================== MAIN BOT ========================
class UpstoxOptionsBot:
    def __init__(self):
        self.client = UpstoxClient(UPSTOX_ACCESS_TOKEN)
        self.cache = InMemoryCache(max_snapshots=100)  # Increased cache size
        self.analyzer = OptionAnalyzer(self.client, self.cache)
        self.chart_gen = ChartGenerator()
        self.alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    def is_market_open(self) -> bool:
        now = datetime.now(IST).time()
        today = datetime.now(IST).date()
        
        if today.weekday() >= 5:
            return False
            
        return MARKET_START <= now <= MARKET_END
    
    async def process_symbols(self):
        now_time = datetime.now(IST)
        
        logger.info("\n" + "="*70)
        logger.info(f"🔍 FIXED ANALYSIS CYCLE v8.0 - {now_time.strftime('%H:%M:%S IST')}")
        logger.info("="*70)
        
        for symbol in INDICES:
            try:
                analysis = await self.analyzer.analyze_symbol(symbol)
                
                if analysis:
                    final_signal = analysis.get("final_signal", {})
                    
                    # ✅ FIXED: Actually enforce the threshold
                    if final_signal.get("should_alert"):
                        chart = self.chart_gen.create_combined_chart(analysis)
                        await self.alerter.send_chart(chart, symbol, analysis)
                        logger.info(f"✅ {symbol}: ALERT SENT (Confidence {final_signal['confidence_score']}/10)")
                    else:
                        logger.info(f"⏳ {symbol}: Below threshold ({final_signal['confidence_score']}/10), NO ALERT")
                        # ✅ FIXED: Removed duplicate send_chart call!
                else:
                    logger.warning(f"⚠️ {symbol} analysis failed")
                
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("="*70)
        logger.info(f"✅ CYCLE COMPLETE | Cache sizes: {', '.join([f'{s}:{self.cache.get_cache_size(s)}' for s in INDICES])}")
        logger.info("="*70 + "\n")
    
    async def run(self):
        current_time = datetime.now(IST)
        market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
        
        print("\n" + "="*70)
        print("🚀 FIXED UPSTOX OPTIONS BOT v8.0", flush=True)
        print("="*70)
        print(f"📅 {current_time.strftime('%d-%b-%Y %A')}", flush=True)
        print(f"🕐 {current_time.strftime('%H:%M:%S IST')}", flush=True)
        print(f"📊 Market: {market_status}", flush=True)
        print(f"⏱️  Interval: 5 minutes", flush=True)
        print(f"📈 Indices: {', '.join(INDICES)}", flush=True)
        print("="*70)
        print("✅ ALL FIXES APPLIED:", flush=True)
        print("   • PCR Interpretation CORRECTED (Rising = Bullish)", flush=True)
        print("   • 15-Minute OI Data POPULATED", flush=True)
        print("   • Signal Threshold ENFORCED", flush=True)
        print("   • Retry Logic ADDED", flush=True)
        print("   • Rate Limiting PROTECTED", flush=True)
        print("   • Historical API FORMAT FIXED", flush=True)
        print("   • Pattern DEDUPLICATION", flush=True)
        print("   • S/R Confluence IMPROVED", flush=True)
        print("   • Cache Lookup ENHANCED", flush=True)
        print(f"   • API Delay: {API_DELAY*1000:.0f}ms", flush=True)
        print(f"   • Max Retries: {MAX_RETRIES}", flush=True)
        print("="*70 + "\n", flush=True)
        
        await self.client.create_session()
        
        try:
            await self.client.download_instruments()
            
            while True:
                try:
                    await self.process_symbols()
                    
                    next_run = datetime.now(IST) + timedelta(seconds=ANALYSIS_INTERVAL)
                    logger.info(f"⏰ Next cycle: {next_run.strftime('%H:%M:%S')}\n")
                    
                    await asyncio.sleep(ANALYSIS_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}")
                    await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
        
        finally:
            await self.client.close_session()
            logger.info("👋 Session closed")


# ======================== ENTRY POINT ========================
if __name__ == "__main__":
    bot = UpstoxOptionsBot()
    asyncio.run(bot.run())
