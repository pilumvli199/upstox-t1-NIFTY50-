#!/usr/bin/env python3
"""
NIFTY50 HYBRID TRADING BOT - FIXED VERSION
=================================================
Multi-Timeframe Price Action + OI Intelligence
Scan Interval: Every 5 minutes
Target: NIFTY50 Index Only

🔧 FIXES:
✅ Proper expiry date format (YYYY-MM-DD)
✅ URL encoding for instrument keys
✅ API expiry validation
✅ Fallback mechanisms
"""

import os
import asyncio
import requests
import urllib.parse
from datetime import datetime, timedelta, time
import pytz
import time as time_sleep
from telegram import Bot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import traceback
import re
import redis

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nifty50_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# API Keys from Environment
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'your_token')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your_key')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_token')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_chat_id')

# Redis Connection
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5)

# NIFTY50 Symbol Configuration
NIFTY_SYMBOL = "NSE_INDEX|Nifty 50"
NIFTY_NAME = "NIFTY"
NIFTY_DISPLAY = "NIFTY50"

# ==================== DATA CLASSES ====================
@dataclass
class StrikeData:
    strike: int
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int
    ce_price: float
    pe_price: float
    ce_oi_change_15m: int = 0
    ce_oi_change_30m: int = 0
    pe_oi_change_15m: int = 0
    pe_oi_change_30m: int = 0

@dataclass
class OISnapshot:
    """Complete OI snapshot at a point in time"""
    timestamp: datetime
    strikes: List[StrikeData]
    pcr: float
    max_pain: int
    support_strikes: List[int]
    resistance_strikes: List[int]
    total_ce_oi: int
    total_pe_oi: int

@dataclass
class TradeSignal:
    """Trading signal with full details"""
    signal_type: str
    confidence: int
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: str
    recommended_strike: int
    reasoning: str
    price_analysis: str
    oi_analysis: str
    alignment_score: int
    risk_factors: List[str]
    support_levels: List[float]
    resistance_levels: List[float]
    pattern_detected: str
    timeframe_alignment: str

# ==================== EXPIRY CALCULATOR (FIXED) ====================
class ExpiryCalculator:
    @staticmethod
    def get_all_expiries_from_api(instrument_key: str, access_token: str) -> List[str]:
        """Fetch all available expiries from Upstox API"""
        try:
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
            
            # URL encode the instrument key
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            url = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_key}"
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                contracts = response.json().get('data', [])
                # Extract unique expiry dates
                expiries = sorted(list(set(c['expiry'] for c in contracts if 'expiry' in c)))
                logger.info(f"  📅 Found {len(expiries)} expiries from API")
                return expiries
            else:
                logger.warning(f"  ⚠️ Contract API returned {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"  ❌ Expiry API error: {e}")
            return []
    
    @staticmethod
    def get_next_tuesday() -> str:
        """Calculate next Tuesday expiry (YYYY-MM-DD format)"""
        today = datetime.now(IST).date()
        current_time = datetime.now(IST).time()
        
        # Find next Tuesday (weekday = 1)
        days_ahead = 1 - today.weekday()
        
        if days_ahead <= 0:  # Today is Tuesday or past Tuesday
            if today.weekday() == 1 and current_time < time(15, 30):
                # Today is Tuesday and market still open
                expiry = today
            else:
                # Move to next Tuesday
                days_ahead += 7
                expiry = today + timedelta(days=days_ahead)
        else:
            # Upcoming Tuesday this week
            expiry = today + timedelta(days=days_ahead)
        
        return expiry.strftime('%Y-%m-%d')
    
    @staticmethod
    def get_weekly_expiry(access_token: str) -> str:
        """Get NIFTY50 weekly expiry with API validation"""
        # Get all available expiries
        expiries = ExpiryCalculator.get_all_expiries_from_api(NIFTY_SYMBOL, access_token)
        
        if expiries:
            # Filter future expiries
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
                nearest_expiry = min(future_expiries)
                logger.info(f"  ✅ Using API expiry: {nearest_expiry}")
                return nearest_expiry
        
        # Fallback: Calculate next Tuesday
        calculated_expiry = ExpiryCalculator.get_next_tuesday()
        logger.info(f"  ⚠️ Using calculated expiry: {calculated_expiry}")
        return calculated_expiry
    
    @staticmethod
    def days_to_expiry(expiry_str: str) -> int:
        """Calculate days remaining to expiry"""
        try:
            expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
            return (expiry_date - datetime.now(IST).date()).days
        except:
            return 0
    
    @staticmethod
    def format_for_display(expiry_str: str) -> str:
        """Convert YYYY-MM-DD to DDMmmYY format for display"""
        try:
            dt = datetime.strptime(expiry_str, '%Y-%m-%d')
            return dt.strftime('%d%b%y').upper()
        except:
            return expiry_str

# ==================== REDIS OI MANAGER ====================
class RedisOIManager:
    """Manages OI data storage and retrieval with time-based comparison"""
    
    @staticmethod
    def save_oi_snapshot(snapshot: OISnapshot):
        """Save complete OI snapshot to Redis"""
        key = f"oi:nifty50:{snapshot.timestamp.strftime('%Y-%m-%d_%H:%M')}"
        
        data = {
            "timestamp": snapshot.timestamp.isoformat(),
            "pcr": snapshot.pcr,
            "max_pain": snapshot.max_pain,
            "support_strikes": snapshot.support_strikes,
            "resistance_strikes": snapshot.resistance_strikes,
            "total_ce_oi": snapshot.total_ce_oi,
            "total_pe_oi": snapshot.total_pe_oi,
            "strikes": [
                {
                    "strike": s.strike,
                    "ce_oi": s.ce_oi,
                    "pe_oi": s.pe_oi,
                    "ce_volume": s.ce_volume,
                    "pe_volume": s.pe_volume,
                    "ce_price": s.ce_price,
                    "pe_price": s.pe_price
                }
                for s in snapshot.strikes
            ]
        }
        
        # Store for 3 days
        redis_client.setex(key, 259200, json.dumps(data))
        logger.info(f"  💾 Saved OI snapshot: {key}")
    
    @staticmethod
    def get_oi_snapshot(minutes_ago: int) -> Optional[OISnapshot]:
        """Retrieve OI snapshot from N minutes ago"""
        target_time = datetime.now(IST) - timedelta(minutes=minutes_ago)
        
        # Round to nearest 5-minute mark
        target_time = target_time.replace(
            minute=(target_time.minute // 5) * 5,
            second=0,
            microsecond=0
        )
        
        key = f"oi:nifty50:{target_time.strftime('%Y-%m-%d_%H:%M')}"
        data = redis_client.get(key)
        
        if data:
            parsed = json.loads(data)
            return OISnapshot(
                timestamp=datetime.fromisoformat(parsed['timestamp']),
                strikes=[
                    StrikeData(
                        strike=s['strike'],
                        ce_oi=s['ce_oi'],
                        pe_oi=s['pe_oi'],
                        ce_volume=s['ce_volume'],
                        pe_volume=s['pe_volume'],
                        ce_price=s['ce_price'],
                        pe_price=s['pe_price']
                    )
                    for s in parsed['strikes']
                ],
                pcr=parsed['pcr'],
                max_pain=parsed['max_pain'],
                support_strikes=parsed['support_strikes'],
                resistance_strikes=parsed['resistance_strikes'],
                total_ce_oi=parsed['total_ce_oi'],
                total_pe_oi=parsed['total_pe_oi']
            )
        
        return None
    
    @staticmethod
    def save_candle_data(timeframe: str, df: pd.DataFrame):
        """Save candle data to Redis (deleted daily at 3:15 PM)"""
        key = f"candles:nifty50:{timeframe}"
        
        # Convert to JSON
        df_copy = df.copy()
        df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        data = df_copy.to_json(orient='records')
        
        # Calculate seconds until 3:15 PM today (or tomorrow if past 3:15 PM)
        now = datetime.now(IST)
        delete_time = now.replace(hour=15, minute=15, second=0, microsecond=0)
        
        if now.time() >= time(15, 15):
            delete_time += timedelta(days=1)
        
        ttl = int((delete_time - now).total_seconds())
        
        redis_client.setex(key, ttl, data)
        logger.info(f"  💾 Saved {timeframe} candles (expires at 3:15 PM)")
    
    @staticmethod
    def get_candle_data(timeframe: str) -> Optional[pd.DataFrame]:
        """Retrieve candle data from Redis"""
        key = f"candles:nifty50:{timeframe}"
        data = redis_client.get(key)
        
        if data:
            df = pd.DataFrame(json.loads(data))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        return None

# ==================== UPSTOX DATA FETCHER (FIXED) ====================
class UpstoxDataFetcher:
    """Fetches data from Upstox API with proper encoding"""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
    
    def get_historical_data(self, interval: str, days: int = 7) -> pd.DataFrame:
        """Fetch historical candle data"""
        try:
            to_date = datetime.now(IST)
            from_date = to_date - timedelta(days=days)
            
            # URL encode instrument key
            encoded_symbol = urllib.parse.quote(NIFTY_SYMBOL, safe='')
            
            url = f"https://api.upstox.com/v2/historical-candle/{encoded_symbol}/{interval}/{to_date.strftime('%Y-%m-%d')}/{from_date.strftime('%Y-%m-%d')}"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'candles' in data['data']:
                    df = pd.DataFrame(
                        data['data']['candles'],
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    logger.info(f"  ✅ Fetched {len(df)} {interval} candles")
                    return df
            
            logger.warning(f"  ⚠️ Historical data API returned {response.status_code}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"  ❌ Historical data error: {e}")
            return pd.DataFrame()
    
    def get_ltp(self) -> float:
        """Get Last Traded Price for NIFTY50"""
        try:
            # URL encode instrument key
            encoded_symbol = urllib.parse.quote(NIFTY_SYMBOL, safe='')
            
            url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={encoded_symbol}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and NIFTY_SYMBOL in data['data']:
                    ltp = float(data['data'][NIFTY_SYMBOL]['last_price'])
                    logger.info(f"  💹 LTP: ₹{ltp:.2f}")
                    return ltp
            
            logger.warning(f"  ⚠️ LTP API returned {response.status_code}")
            return 0.0
            
        except Exception as e:
            logger.error(f"  ❌ LTP error: {e}")
            return 0.0
    
    def get_option_chain(self, expiry: str) -> List[StrikeData]:
        """Fetch option chain data with proper encoding"""
        try:
            # URL encode instrument key
            encoded_symbol = urllib.parse.quote(NIFTY_SYMBOL, safe='')
            
            # Expiry should already be in YYYY-MM-DD format
            url = f"https://api.upstox.com/v2/option/chain?instrument_key={encoded_symbol}&expiry_date={expiry}"
            
            logger.info(f"  🔗 Option Chain URL: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=30)
            
            logger.info(f"  📡 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' not in data:
                    logger.warning(f"  ⚠️ No 'data' in option chain response")
                    logger.info(f"  Response: {json.dumps(data, indent=2)[:500]}")
                    return []
                
                strikes = []
                for item in data['data']:
                    try:
                        strike_price = int(float(item.get('strike_price', 0)))
                        
                        call_data = item.get('call_options', {}).get('market_data', {})
                        put_data = item.get('put_options', {}).get('market_data', {})
                        
                        strikes.append(StrikeData(
                            strike=strike_price,
                            ce_oi=int(call_data.get('oi', 0)),
                            pe_oi=int(put_data.get('oi', 0)),
                            ce_volume=int(call_data.get('volume', 0)),
                            pe_volume=int(put_data.get('volume', 0)),
                            ce_price=float(call_data.get('ltp', 0)),
                            pe_price=float(put_data.get('ltp', 0))
                        ))
                    except Exception as item_error:
                        logger.warning(f"  ⚠️ Strike parsing error: {item_error}")
                        continue
                
                logger.info(f"  ✅ Fetched {len(strikes)} strikes")
                return strikes
            else:
                # Log error details
                logger.error(f"  ❌ Option chain API error {response.status_code}")
                try:
                    error_data = response.json()
                    logger.error(f"  Error details: {json.dumps(error_data, indent=2)}")
                except:
                    logger.error(f"  Response text: {response.text[:500]}")
                return []
            
        except Exception as e:
            logger.error(f"  ❌ Option chain error: {e}")
            traceback.print_exc()
            return []

# ==================== MULTI-TIMEFRAME PROCESSOR ====================
class MultiTimeframeProcessor:
    """Resamples 1-minute data to higher timeframes"""
    
    @staticmethod
    def resample_to_timeframe(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 1-minute candles to specified timeframe"""
        df = df_1m.copy()
        df.set_index('timestamp', inplace=True)
        
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        
        return resampled
    
    @staticmethod
    def get_timeframe_bias(df: pd.DataFrame) -> Tuple[str, int]:
        """Determine bias (BULLISH/BEARISH/NEUTRAL) with confidence"""
        if len(df) < 20:
            return "NEUTRAL", 50
        
        df_tail = df.tail(20)
        closes = df_tail['close'].values
        
        # SMA comparison
        sma_20 = closes.mean()
        current_price = closes[-1]
        price_vs_sma = ((current_price - sma_20) / sma_20) * 100
        
        # Higher highs/lower lows count
        recent_highs = df_tail['high'].tail(10).values
        hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
        
        recent_lows = df_tail['low'].tail(10).values
        ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])
        
        # Determine bias
        if price_vs_sma > 1 and hh_count >= 6:
            bias = "BULLISH"
            confidence = min(95, 60 + int(price_vs_sma * 5))
        elif price_vs_sma < -1 and ll_count >= 6:
            bias = "BEARISH"
            confidence = min(95, 60 + int(abs(price_vs_sma) * 5))
        else:
            bias = "NEUTRAL"
            confidence = 50
        
        return bias, confidence

# ==================== OI ANALYZER ====================
class OIAnalyzer:
    """Analyzes option chain data"""
    
    @staticmethod
    def calculate_pcr(strikes: List[StrikeData]) -> float:
        """Calculate Put-Call Ratio"""
        total_ce = sum(s.ce_oi for s in strikes)
        total_pe = sum(s.pe_oi for s in strikes)
        return total_pe / total_ce if total_ce > 0 else 0
    
    @staticmethod
    def find_max_pain(strikes: List[StrikeData]) -> int:
        """Calculate Max Pain strike"""
        max_pain_values = {}
        
        for strike_data in strikes:
            strike = strike_data.strike
            total_pain = 0
            
            for s in strikes:
                if s.strike < strike:
                    total_pain += (strike - s.strike) * s.pe_oi
                elif s.strike > strike:
                    total_pain += (s.strike - strike) * s.ce_oi
            
            max_pain_values[strike] = total_pain
        
        return min(max_pain_values, key=max_pain_values.get) if max_pain_values else 0
    
    @staticmethod
    def get_atm_strikes(strikes: List[StrikeData], spot_price: float, count: int = 21) -> List[StrikeData]:
        """Get ATM strikes (±10 strikes from ATM)"""
        atm_strike = round(spot_price / 50) * 50
        
        # Get strikes in range
        strike_range = range(atm_strike - 500, atm_strike + 550, 50)
        relevant = [s for s in strikes if s.strike in strike_range]
        
        # Sort by strike
        relevant = sorted(relevant, key=lambda x: x.strike)
        
        return relevant[:count]
    
    @staticmethod
    def identify_support_resistance(strikes: List[StrikeData]) -> Tuple[List[int], List[int]]:
        """Identify key support and resistance levels based on OI"""
        # Support: High PE OI
        pe_sorted = sorted(strikes, key=lambda x: x.pe_oi, reverse=True)
        support_strikes = [s.strike for s in pe_sorted[:3]]
        
        # Resistance: High CE OI
        ce_sorted = sorted(strikes, key=lambda x: x.ce_oi, reverse=True)
        resistance_strikes = [s.strike for s in ce_sorted[:3]]
        
        return support_strikes, resistance_strikes
    
    @staticmethod
    def create_oi_snapshot(strikes: List[StrikeData], spot_price: float) -> OISnapshot:
        """Create complete OI snapshot"""
        atm_strikes = OIAnalyzer.get_atm_strikes(strikes, spot_price)
        pcr = OIAnalyzer.calculate_pcr(atm_strikes)
        max_pain = OIAnalyzer.find_max_pain(atm_strikes)
        support, resistance = OIAnalyzer.identify_support_resistance(atm_strikes)
        
        total_ce = sum(s.ce_oi for s in atm_strikes)
        total_pe = sum(s.pe_oi for s in atm_strikes)
        
        return OISnapshot(
            timestamp=datetime.now(IST),
            strikes=atm_strikes,
            pcr=pcr,
            max_pain=max_pain,
            support_strikes=support,
            resistance_strikes=resistance,
            total_ce_oi=total_ce,
            total_pe_oi=total_pe
        )

# ==================== AI ANALYZER ====================
class AIAnalyzer:
    """Sends data to DeepSeek for analysis"""
    
    @staticmethod
    def extract_json(content: str) -> Optional[Dict]:
        """Extract JSON from AI response"""
        try:
            content = re.sub(r'```json\s*|\s*```', '', content)
            return json.loads(content)
        except:
            match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
        return None
    
    @staticmethod
    def analyze_with_deepseek(
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        current_price: float,
        current_oi: OISnapshot,
        oi_15m_ago: Optional[OISnapshot],
        oi_30m_ago: Optional[OISnapshot]
    ) -> Optional[TradeSignal]:
        """Send comprehensive analysis request to DeepSeek"""
        
        try:
            df_5m_tail = df_5m.tail(200).copy()
            df_15m_tail = df_15m.tail(175).copy()
            df_1h_tail = df_1h.tail(50).copy()
            
            df_5m_tail['timestamp'] = df_5m_tail['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            df_15m_tail['timestamp'] = df_15m_tail['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            df_1h_tail['timestamp'] = df_1h_tail['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            candles_5m = df_5m_tail[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_json(orient='records')
            candles_15m = df_15m_tail[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_json(orient='records')
            candles_1h = df_1h_tail[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_json(orient='records')
            
            current_oi_text = "\n".join([
                f"{s.strike}: CE: {s.ce_oi:,} | PE: {s.pe_oi:,}"
                for s in current_oi.strikes
            ])
            
            oi_15m_text = "NOT AVAILABLE"
            oi_30m_text = "NOT AVAILABLE"
            oi_velocity_text = "NOT AVAILABLE"
            pcr_comparison = f"Now: {current_oi.pcr:.2f}"
            
            if oi_15m_ago:
                oi_15m_text = "\n".join([
                    f"{s.strike}: CE: {s.ce_oi:,} | PE: {s.pe_oi:,}"
                    for s in oi_15m_ago.strikes
                ])
                pcr_comparison += f" | 15min ago: {oi_15m_ago.pcr:.2f}"
                
                ce_change_15m = current_oi.total_ce_oi - oi_15m_ago.total_ce_oi
                pe_change_15m = current_oi.total_pe_oi - oi_15m_ago.total_pe_oi
                oi_velocity_text = f"Last 15 min: CE {ce_change_15m:+,} | PE {pe_change_15m:+,}"
            
            if oi_30m_ago:
                oi_30m_text = "\n".join([
                    f"{s.strike}: CE: {s.ce_oi:,} | PE: {s.pe_oi:,}"
                    for s in oi_30m_ago.strikes
                ])
                pcr_comparison += f" | 30min ago: {oi_30m_ago.pcr:.2f}"
                
                ce_change_30m = current_oi.total_ce_oi - oi_30m_ago.total_ce_oi
                pe_change_30m = current_oi.total_pe_oi - oi_30m_ago.total_pe_oi
                oi_velocity_text += f"\nLast 30 min: CE {ce_change_30m:+,} | PE {pe_change_30m:+,}"
            
            prompt = f"""You are an elite institutional trader specializing in Price Action and Options Intelligence.

NIFTY50 ANALYSIS:
Current Price: ₹{current_price:.2f}
Timestamp: {datetime.now(IST).strftime('%d-%b-%Y %H:%M:%S')}

MULTI-TIMEFRAME PRICE ACTION:
5-MIN: {candles_5m}
15-MIN: {candles_15m}
1-HOUR: {candles_1h}

OPTION CHAIN DATA:
CURRENT OI: {current_oi_text}
15 MIN AGO: {oi_15m_text}
30 MIN AGO: {oi_30m_text}

OI VELOCITY: {oi_velocity_text}
PCR: {pcr_comparison}
Max Pain: {current_oi.max_pain}
Support: {', '.join(map(str, current_oi.support_strikes))}
Resistance: {', '.join(map(str, current_oi.resistance_strikes))}

Analyze price action + OI fusion and provide trading signal.

OUTPUT JSON:
{{
  "signal_type": "CE_BUY/PE_BUY/NO_TRADE",
  "confidence": 85,
  "entry_price": {current_price:.2f},
  "stop_loss": 0.0,
  "target_1": 0.0,
  "target_2": 0.0,
  "risk_reward": "1:2.5",
  "recommended_strike": {round(current_price/50)*50},
  "reasoning": "Brief reasoning",
  "price_analysis": "Price action analysis",
  "oi_analysis": "OI flow analysis",
  "alignment_score": 8,
  "risk_factors": ["Risk 1", "Risk 2"],
  "support_levels": [0.0, 0.0],
  "resistance_levels": [0.0, 0.0],
  "pattern_detected": "Pattern name or None",
  "timeframe_alignment": "STRONG/MODERATE/WEAK"
}}"""
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an elite institutional F&O trader. Analyze price action + OI fusion. Respond ONLY in JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 3000
                },
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=120
            )
            
            if response.status_code != 200:
                logger.error(f"  ❌ DeepSeek API error: {response.status_code}")
                return None
            
            ai_content = response.json()['choices'][0]['message']['content']
            analysis_dict = AIAnalyzer.extract_json(ai_content)
            
            if not analysis_dict:
                logger.error(f"  ❌ Failed to parse AI response")
                return None
            
            logger.info(f"  🧠 AI Signal: {analysis_dict.get('signal_type')} | Confidence: {analysis_dict.get('confidence')}%")
            
            return TradeSignal(**analysis_dict)
            
        except Exception as e:
            logger.error(f"  ❌ AI analysis error: {e}")
            traceback.print_exc()
            return None

# ==================== CHART GENERATOR ====================
class ChartGenerator:
    """Creates professional trading charts"""
    
    @staticmethod
    def create_chart(
        df_15m: pd.DataFrame,
        signal: TradeSignal,
        spot_price: float,
        save_path: str
    ):
        """Generate professional chart with signal visualization"""
        
        BG = '#131722'
        GRID = '#1e222d'
        TEXT = '#d1d4dc'
        GREEN = '#26a69a'
        RED = '#ef5350'
        YELLOW = '#ffd700'
        BLUE = '#2962ff'
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(18, 10),
            gridspec_kw={'height_ratios': [3, 1]},
            facecolor=BG
        )
        
        ax1.set_facecolor(BG)
        df_plot = df_15m.tail(150).reset_index(drop=True)
        
        for idx, row in df_plot.iterrows():
            color = GREEN if row['close'] > row['open'] else RED
            
            ax1.add_patch(Rectangle(
                (idx, min(row['open'], row['close'])),
                0.6,
                abs(row['close'] - row['open']),
                facecolor=color,
                edgecolor=color,
                alpha=0.8
            ))
            
            ax1.plot(
                [idx+0.3, idx+0.3],
                [row['low'], row['high']],
                color=color,
                linewidth=1,
                alpha=0.6
            )
        
        for support in signal.support_levels:
            ax1.axhline(support, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7)
            ax1.text(
                len(df_plot)*0.98, support,
                f'S:₹{support:.1f}  ',
                color=GREEN,
                fontsize=10,
                ha='right',
                va='bottom',
                bbox=dict(boxstyle='round', facecolor=BG, alpha=0.7)
            )
        
        for resistance in signal.resistance_levels:
            ax1.axhline(resistance, color=RED, linestyle='--', linewidth=1.5, alpha=0.7)
            ax1.text(
                len(df_plot)*0.98, resistance,
                f'R:₹{resistance:.1f}  ',
                color=RED,
                fontsize=10,
                ha='right',
                va='top',
                bbox=dict(boxstyle='round', facecolor=BG, alpha=0.7)
            )
        
        ax1.scatter(
            [len(df_plot)-1],
            [signal.entry_price],
            color=YELLOW,
            s=300,
            marker='D',
            zorder=5,
            edgecolors='white',
            linewidths=2.5
        )
        
        ax1.axhline(signal.stop_loss, color=RED, linewidth=2.5, linestyle=':')
        ax1.axhline(signal.target_1, color=GREEN, linewidth=2, linestyle=':')
        ax1.axhline(signal.target_2, color=GREEN, linewidth=2, linestyle=':')
        
        ax1.text(
            len(df_plot)*0.97, signal.entry_price,
            f'ENTRY: ₹{signal.entry_price:.2f}  ',
            color=YELLOW,
            fontsize=11,
            fontweight='bold',
            ha='right',
            va='center',
            bbox=dict(boxstyle='round', facecolor=BG, edgecolor=YELLOW, linewidth=2)
        )
        
        if signal.pattern_detected and signal.pattern_detected != "None":
            ax1.text(
                len(df_plot)*0.5, df_plot['high'].max() * 0.995,
                signal.pattern_detected.upper(),
                color=YELLOW,
                fontsize=12,
                fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round', facecolor=BG, edgecolor=YELLOW, alpha=0.9)
            )
        
        ax1.text(
            len(df_plot)-1, spot_price,
            f'  CMP: ₹{spot_price:.1f}',
            fontsize=11,
            color='white',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=BLUE, edgecolor='white', linewidth=2),
            va='center'
        )
        
        signal_emoji = "🟢" if signal.signal_type == "CE_BUY" else "🔴" if signal.signal_type == "PE_BUY" else "⚪"
        
        info_text = f"""{signal_emoji} {signal.signal_type} | Confidence: {signal.confidence}%
Alignment: {signal.timeframe_alignment} | Score: {signal.alignment_score}/10

Entry: ₹{signal.entry_price:.1f}
SL: ₹{signal.stop_loss:.1f}
T1: ₹{signal.target_1:.1f}
T2: ₹{signal.target_2:.1f}
RR: {signal.risk_reward}

Strike: {signal.recommended_strike}
Pattern: {signal.pattern_detected}

Reason: {signal.reasoning[:100]}..."""
        
        ax1.text(
            0.01, 0.99,
            info_text,
            transform=ax1.transAxes,
            fontsize=9,
            va='top',
            bbox=dict(boxstyle='round', facecolor=GRID, alpha=0.95, edgecolor=TEXT, linewidth=1),
            color=TEXT,
            family='monospace'
        )
        
        title = f"NIFTY50 | 15-Min | {signal.signal_type} | {signal.timeframe_alignment} Alignment"
        ax1.set_title(title, color=TEXT, fontsize=13, fontweight='bold', pad=15)
        
        ax1.grid(True, color=GRID, alpha=0.3)
        ax1.tick_params(colors=TEXT)
        ax1.set_ylabel('Price (₹)', color=TEXT, fontsize=11)
        
        ax2.set_facecolor(BG)
        colors = [GREEN if df_plot.iloc[i]['close'] > df_plot.iloc[i]['open'] else RED for i in range(len(df_plot))]
        ax2.bar(range(len(df_plot)), df_plot['volume'], color=colors, alpha=0.6, width=0.8)
        ax2.set_ylabel('Volume', color=TEXT, fontsize=11)
        ax2.tick_params(colors=TEXT)
        ax2.grid(True, color=GRID, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor=BG)
        plt.close()
        
        logger.info(f"  📊 Chart saved: {save_path}")

# ==================== MAIN BOT ====================
class Nifty50Bot:
    """Main bot orchestrator"""
    
    def __init__(self):
        self.data_fetcher = UpstoxDataFetcher(UPSTOX_ACCESS_TOKEN)
        self.telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.scan_count = 0
        self.last_signal_time = None
    
    async def send_startup_message(self):
        """Send bot startup notification"""
        expiry = ExpiryCalculator.get_weekly_expiry(UPSTOX_ACCESS_TOKEN)
        expiry_display = ExpiryCalculator.format_for_display(expiry)
        days_left = ExpiryCalculator.days_to_expiry(expiry)
        
        message = f"""
🚀 NIFTY50 TRADING BOT STARTED (FIXED)

⏰ Time: {datetime.now(IST).strftime('%d-%b-%Y %H:%M:%S')}

📊 Configuration:
✅ Symbol: NIFTY50 Index (NSE)
✅ Scan Interval: Every 5 minutes
✅ Market Hours: 9:20 AM - 3:30 PM
✅ Expiry: {expiry_display} ({expiry}) - {days_left} days left

🔧 FIXES APPLIED:
✅ Proper expiry format (YYYY-MM-DD)
✅ URL encoding for API calls
✅ API expiry validation
✅ Enhanced error logging

🧠 Analysis Framework:
✅ Multi-Timeframe (1H + 15M + 5M)
✅ OI Comparison (15m + 30m lookback)
✅ Price Action + OI Fusion
✅ Support/Resistance with OI clusters
✅ Pattern Detection + OI confirmation

🎯 Alert Criteria:
✅ Minimum Confidence: 75%
✅ Alignment Score: 7+/10

🔄 Status: Active & Running
"""
        await self.telegram_bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message
        )
        logger.info("✅ Startup message sent")
    
    async def send_telegram_alert(self, signal: TradeSignal, chart_path: str):
        """Send trading signal to Telegram"""
        try:
            with open(chart_path, 'rb') as photo:
                await self.telegram_bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=photo
                )
            
            signal_emoji = "🟢" if signal.signal_type == "CE_BUY" else "🔴"
            
            reasoning = signal.reasoning[:200].replace('_', ' ').replace('*', ' ')
            price_analysis = signal.price_analysis[:250].replace('_', ' ').replace('*', ' ')
            oi_analysis = signal.oi_analysis[:250].replace('_', ' ').replace('*', ' ')
            pattern = signal.pattern_detected.replace('_', ' ').replace('*', ' ')
            
            message = f"""
{signal_emoji} NIFTY50 {signal.signal_type} SIGNAL

🎯 Confidence: {signal.confidence}%
📊 Alignment: {signal.timeframe_alignment} ({signal.alignment_score}/10)

💡 REASONING:
{reasoning}...

📈 PRICE ANALYSIS:
{price_analysis}...

📊 OI ANALYSIS:
{oi_analysis}...

🎨 PATTERN: {pattern}

💰 TRADE SETUP:
Entry: ₹{signal.entry_price:.2f}
Stop Loss: ₹{signal.stop_loss:.2f}
Target 1: ₹{signal.target_1:.2f}
Target 2: ₹{signal.target_2:.2f}
Risk:Reward → {signal.risk_reward}

📍 Recommended Strike: {signal.recommended_strike}

📊 Support: {', '.join([f'₹{s:.1f}' for s in signal.support_levels])}
📊 Resistance: {', '.join([f'₹{r:.1f}' for r in signal.resistance_levels])}

⚠️ RISK FACTORS:
{chr(10).join(['• ' + rf.replace('_', ' ').replace('*', ' ') for rf in signal.risk_factors[:3]])}

🕐 {datetime.now(IST).strftime('%d-%b %H:%M:%S')}
"""
            
            await self.telegram_bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message
            )
            
            logger.info(f"  ✅ Alert sent: {signal.signal_type}")
            
        except Exception as e:
            logger.error(f"  ❌ Telegram error: {e}")
            traceback.print_exc()
    
    async def run_analysis(self):
        """Run complete analysis cycle"""
        try:
            self.scan_count += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 SCAN #{self.scan_count} - {datetime.now(IST).strftime('%H:%M:%S')}")
            logger.info(f"{'='*70}")
            
            df_1m_cached = RedisOIManager.get_candle_data('1m')
            
            if df_1m_cached is None or len(df_1m_cached) == 0:
                logger.info("  📥 Fetching fresh historical data...")
                df_1m = self.data_fetcher.get_historical_data('1minute', days=7)
                if df_1m.empty:
                    logger.warning("  ⚠️ No historical data available")
                    return
                RedisOIManager.save_candle_data('1m', df_1m)
            else:
                logger.info(f"  ✅ Loaded {len(df_1m_cached)} 1-min candles from Redis")
                df_1m = df_1m_cached
                
                logger.info("  📥 Fetching latest 1-min candle...")
                df_latest = self.data_fetcher.get_historical_data('1minute', days=1)
                if not df_latest.empty:
                    df_1m = pd.concat([df_1m, df_latest]).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                    RedisOIManager.save_candle_data('1m', df_1m)
            
            logger.info("  🔄 Resampling to higher timeframes...")
            df_5m = MultiTimeframeProcessor.resample_to_timeframe(df_1m, '5T')
            df_15m = MultiTimeframeProcessor.resample_to_timeframe(df_1m, '15T')
            df_1h = MultiTimeframeProcessor.resample_to_timeframe(df_1m, '1H')
            
            logger.info(f"  📊 Candles: 5M({len(df_5m)}) | 15M({len(df_15m)}) | 1H({len(df_1h)})")
            
            RedisOIManager.save_candle_data('5m', df_5m)
            RedisOIManager.save_candle_data('15m', df_15m)
            RedisOIManager.save_candle_data('1h', df_1h)
            
            bias_5m, conf_5m = MultiTimeframeProcessor.get_timeframe_bias(df_5m)
            bias_15m, conf_15m = MultiTimeframeProcessor.get_timeframe_bias(df_15m)
            bias_1h, conf_1h = MultiTimeframeProcessor.get_timeframe_bias(df_1h)
            
            logger.info(f"  📊 Bias: 1H {bias_1h}({conf_1h}%) | 15M {bias_15m}({conf_15m}%) | 5M {bias_5m}({conf_5m}%)")
            
            spot_price = self.data_fetcher.get_ltp()
            if spot_price == 0:
                spot_price = df_15m['close'].iloc[-1]
                logger.info(f"  💹 Using last close: ₹{spot_price:.2f}")
            
            expiry = ExpiryCalculator.get_weekly_expiry(UPSTOX_ACCESS_TOKEN)
            expiry_display = ExpiryCalculator.format_for_display(expiry)
            logger.info(f"  📅 Expiry: {expiry_display} ({expiry})")
            
            all_strikes = self.data_fetcher.get_option_chain(expiry)
            if not all_strikes:
                logger.warning("  ⚠️ No option chain data available")
                return
            
            current_oi = OIAnalyzer.create_oi_snapshot(all_strikes, spot_price)
            logger.info(f"  📊 PCR: {current_oi.pcr:.2f} | Max Pain: {current_oi.max_pain}")
            
            RedisOIManager.save_oi_snapshot(current_oi)
            
            oi_15m_ago = RedisOIManager.get_oi_snapshot(15)
            oi_30m_ago = RedisOIManager.get_oi_snapshot(30)
            
            if oi_15m_ago:
                logger.info(f"  ✅ Loaded OI from 15 min ago (PCR: {oi_15m_ago.pcr:.2f})")
            else:
                logger.info(f"  ℹ️ No OI data from 15 min ago")
            
            if oi_30m_ago:
                logger.info(f"  ✅ Loaded OI from 30 min ago (PCR: {oi_30m_ago.pcr:.2f})")
            
            logger.info("  🧠 Sending to DeepSeek AI...")
            signal = AIAnalyzer.analyze_with_deepseek(
                df_5m=df_5m,
                df_15m=df_15m,
                df_1h=df_1h,
                current_price=spot_price,
                current_oi=current_oi,
                oi_15m_ago=oi_15m_ago,
                oi_30m_ago=oi_30m_ago
            )
            
            if not signal:
                logger.info("  ⏸️ No valid signal generated")
                return
            
            if signal.signal_type == "NO_TRADE":
                logger.info(f"  ⏸️ NO_TRADE signal (Confidence: {signal.confidence}%)")
                return
            
            if signal.confidence < 75 or signal.alignment_score < 7:
                logger.info(f"  ⏸️ Below threshold (Conf: {signal.confidence}% | Score: {signal.alignment_score}/10)")
                return
            
            if self.last_signal_time:
                time_since_last = (datetime.now(IST) - self.last_signal_time).total_seconds() / 60
                if time_since_last < 30:
                    logger.info(f"  ⏸️ Cooldown active ({time_since_last:.0f} min since last alert)")
                    return
            
            logger.info(f"  🚨 ALERT! {signal.signal_type} | Conf: {signal.confidence}% | Score: {signal.alignment_score}/10")
            
            chart_path = f"/tmp/nifty50_chart_{datetime.now(IST).strftime('%H%M')}.png"
            ChartGenerator.create_chart(df_15m, signal, spot_price, chart_path)
            
            await self.send_telegram_alert(signal, chart_path)
            self.last_signal_time = datetime.now(IST)
            
        except Exception as e:
            logger.error(f"  ❌ Analysis error: {e}")
            traceback.print_exc()
    
    async def run_scanner(self):
        """Main scanner loop"""
        logger.info("\n" + "="*80)
        logger.info("🚀 NIFTY50 TRADING BOT (FIXED VERSION)")
        logger.info("="*80)
        
        await self.send_startup_message()
        
        while True:
            try:
                now = datetime.now(IST)
                current_time = now.time()
                
                if current_time < time(9, 20) or current_time > time(15, 30):
                    logger.info(f"⏸️ Market closed. Waiting... (Current: {current_time.strftime('%H:%M')})")
                    await asyncio.sleep(300)
                    continue
                
                if now.weekday() >= 5:
                    logger.info(f"📅 Weekend. Pausing...")
                    await asyncio.sleep(3600)
                    continue
                
                await self.run_analysis()
                
                current_minute = now.minute
                next_scan_minute = ((current_minute // 5) + 1) * 5
                if next_scan_minute >= 60:
                    next_scan_minute = 0
                
                next_scan = now.replace(minute=next_scan_minute % 60, second=0, microsecond=0)
                if next_scan_minute == 0:
                    next_scan += timedelta(hours=1)
                
                wait_seconds = (next_scan - now).total_seconds()
                
                logger.info(f"\n✅ Scan complete. Next scan at {next_scan.strftime('%H:%M')} ({wait_seconds:.0f}s)")
                await asyncio.sleep(wait_seconds)
                
            except Exception as e:
                logger.error(f"❌ Scanner error: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    bot = Nifty50Bot()
    asyncio.run(bot.run_scanner())
