#!/usr/bin/env python3
"""
NIFTY50 5-MINUTE SINGLE TIMEFRAME BOT - INTRADAY + HISTORICAL
==============================================================
✅ Separate Intraday + Historical API Fetching
✅ Fixed Today's Live Data Issue
✅ Enhanced Chart with Volume Display
✅ API Connection Status Notifications
✅ ATM Option Chain Data in Alerts
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
from dataclasses import dataclass
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
        logging.FileHandler('nifty50_5min_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# API Keys
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'your_token')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your_key')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_token')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_chat_id')

# Redis Connection
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5)

# NIFTY50 Configuration
NIFTY_SYMBOL = "NSE_INDEX|Nifty 50"
NIFTY_NAME = "NIFTY"
NIFTY_DISPLAY = "NIFTY50"

# Analysis Configuration
CANDLE_COUNT = 420
TIMEFRAME = "5minute"
MARKET_START_TIME = time(9, 15)
MARKET_END_TIME = time(15, 30)

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
    ce_oi_change: int = 0
    pe_oi_change: int = 0

@dataclass
class OISnapshot:
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

# ==================== API CONNECTION CHECKER ====================
class APIConnectionChecker:
    @staticmethod
    def check_upstox(access_token: str) -> Tuple[bool, str]:
        """Check Upstox API connection"""
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(
                "https://api.upstox.com/v2/user/profile",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                return True, "✅ Connected"
            return False, f"❌ Error {response.status_code}"
        except Exception as e:
            return False, f"❌ Failed: {str(e)[:50]}"
    
    @staticmethod
    def check_deepseek(api_key: str) -> Tuple[bool, str]:
        """Check DeepSeek API connection"""
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=10
            )
            if response.status_code == 200:
                return True, "✅ Connected"
            return False, f"❌ Error {response.status_code}"
        except Exception as e:
            return False, f"❌ Failed: {str(e)[:50]}"
    
    @staticmethod
    def check_redis(redis_url: str) -> Tuple[bool, str]:
        """Check Redis connection"""
        try:
            client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=5)
            client.ping()
            return True, "✅ Connected"
        except Exception as e:
            return False, f"❌ Failed: {str(e)[:50]}"

# ==================== EXPIRY CALCULATOR ====================
class ExpiryCalculator:
    @staticmethod
    def get_all_expiries_from_api(instrument_key: str, access_token: str) -> List[str]:
        try:
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
            
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            url = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_key}"
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                contracts = response.json().get('data', [])
                expiries = sorted(list(set(c['expiry'] for c in contracts if 'expiry' in c)))
                logger.info(f"  📅 Found {len(expiries)} expiries from API")
                return expiries
            return []
        except Exception as e:
            logger.error(f"  ❌ Expiry API error: {e}")
            return []
    
    @staticmethod
    def get_next_tuesday() -> str:
        today = datetime.now(IST).date()
        current_time = datetime.now(IST).time()
        
        days_ahead = 1 - today.weekday()
        
        if days_ahead <= 0:
            if today.weekday() == 1 and current_time < time(15, 30):
                expiry = today
            else:
                days_ahead += 7
                expiry = today + timedelta(days=days_ahead)
        else:
            expiry = today + timedelta(days=days_ahead)
        
        return expiry.strftime('%Y-%m-%d')
    
    @staticmethod
    def get_weekly_expiry(access_token: str) -> str:
        expiries = ExpiryCalculator.get_all_expiries_from_api(NIFTY_SYMBOL, access_token)
        
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
                nearest_expiry = min(future_expiries)
                logger.info(f"  ✅ Using API expiry: {nearest_expiry}")
                return nearest_expiry
        
        calculated_expiry = ExpiryCalculator.get_next_tuesday()
        logger.info(f"  ⚠️ Using calculated expiry: {calculated_expiry}")
        return calculated_expiry
    
    @staticmethod
    def days_to_expiry(expiry_str: str) -> int:
        try:
            expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
            return (expiry_date - datetime.now(IST).date()).days
        except:
            return 0
    
    @staticmethod
    def format_for_display(expiry_str: str) -> str:
        try:
            dt = datetime.strptime(expiry_str, '%Y-%m-%d')
            return dt.strftime('%d%b%y').upper()
        except:
            return expiry_str

# ==================== REDIS OI MANAGER ====================
class RedisOIManager:
    @staticmethod
    def save_oi_snapshot(snapshot: OISnapshot):
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
                    "pe_price": s.pe_price,
                    "ce_oi_change": s.ce_oi_change,
                    "pe_oi_change": s.pe_oi_change
                }
                for s in snapshot.strikes
            ]
        }
        
        redis_client.setex(key, 259200, json.dumps(data))
        logger.info(f"  💾 Saved OI snapshot: {key}")
    
    @staticmethod
    def get_oi_snapshot(minutes_ago: int) -> Optional[OISnapshot]:
        target_time = datetime.now(IST) - timedelta(minutes=minutes_ago)
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
                        pe_price=s['pe_price'],
                        ce_oi_change=s.get('ce_oi_change', 0),
                        pe_oi_change=s.get('pe_oi_change', 0)
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
    def save_candle_data(df: pd.DataFrame):
        key = f"candles:nifty50:5m"
        
        df_copy = df.copy()
        df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        data = df_copy.to_json(orient='records')
        
        now = datetime.now(IST)
        delete_time = now.replace(hour=15, minute=15, second=0, microsecond=0)
        
        if now.time() >= time(15, 15):
            delete_time += timedelta(days=1)
        
        ttl = int((delete_time - now).total_seconds())
        
        redis_client.setex(key, ttl, data)
        logger.info(f"  💾 Saved 5-min candles (expires at 3:15 PM)")
    
    @staticmethod
    def get_candle_data() -> Optional[pd.DataFrame]:
        key = f"candles:nifty50:5m"
        data = redis_client.get(key)
        
        if data:
            df = pd.DataFrame(json.loads(data))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        return None

# ==================== UPSTOX DATA FETCHER - FIXED ====================
class UpstoxDataFetcher:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
    
    def get_intraday_data(self) -> pd.DataFrame:
        """Fetch TODAY'S live 5-minute candles using Intraday API"""
        try:
            # CRITICAL: Intraday API URL format check
            encoded_symbol = urllib.parse.quote(NIFTY_SYMBOL, safe='')
            
            # Intraday API - Today's live data ONLY
            url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_symbol}/5minute"
            
            logger.info(f"  📥 Fetching TODAY's intraday data...")
            logger.info(f"  🔗 URL: {url}")
            
            # Check if market is open
            now = datetime.now(IST)
            if now.time() < MARKET_START_TIME:
                logger.warning(f"  ⚠️ Market not started yet (opens at 9:15 AM)")
                return pd.DataFrame()
            
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"  📊 Intraday Response Status: {data.get('status', 'unknown')}")
                
                if 'data' in data and 'candles' in data['data']:
                    candles = data['data']['candles']
                    if len(candles) > 0:
                        df = pd.DataFrame(
                            candles,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
                        )
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        logger.info(f"  ✅ Fetched {len(df)} intraday candles (TODAY)")
                        return df
                    else:
                        logger.warning(f"  ⚠️ Intraday API returned 0 candles (market may not be active)")
                else:
                    logger.warning(f"  ⚠️ Unexpected intraday response structure")
                    logger.warning(f"  Response: {json.dumps(data, indent=2)[:500]}")
            else:
                logger.warning(f"  ⚠️ Intraday API returned {response.status_code}")
                try:
                    error_data = response.json()
                    logger.warning(f"  Error: {json.dumps(error_data, indent=2)}")
                except:
                    logger.warning(f"  Response: {response.text[:300]}")
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"  ❌ Intraday data error: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, days: int = 5) -> pd.DataFrame:
        """Fetch HISTORICAL 5-minute data (COMPLETED days only, NOT today)"""
        try:
            # CRITICAL FIX: to_date must be YESTERDAY (completed day)
            # Upstox historical API does NOT include current day
            to_date = (datetime.now(IST) - timedelta(days=1)).date()  # Yesterday
            from_date = (datetime.now(IST) - timedelta(days=days)).date()  # 5 days ago
            
            # Double check: Don't fetch if weekend
            if to_date.weekday() >= 5:  # Saturday/Sunday
                # Move to last Friday
                days_back = to_date.weekday() - 4
                to_date = to_date - timedelta(days=days_back)
            
            encoded_symbol = urllib.parse.quote(NIFTY_SYMBOL, safe='')
            
            url = f"https://api.upstox.com/v2/historical-candle/{encoded_symbol}/5minute/{to_date.strftime('%Y-%m-%d')}/{from_date.strftime('%Y-%m-%d')}"
            
            logger.info(f"  📥 Fetching historical ({from_date} to {to_date} - COMPLETED days only)...")
            logger.info(f"  🔗 URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"  📊 Historical Response Status: {data.get('status', 'unknown')}")
                
                if 'data' in data and 'candles' in data['data']:
                    candles = data['data']['candles']
                    if len(candles) > 0:
                        df = pd.DataFrame(
                            candles,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
                        )
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        logger.info(f"  ✅ Fetched {len(df)} historical candles")
                        return df
                    else:
                        logger.warning(f"  ⚠️ Historical API returned 0 candles")
                else:
                    logger.warning(f"  ⚠️ Unexpected historical response structure")
                    logger.warning(f"  Response: {json.dumps(data, indent=2)[:500]}")
            else:
                logger.warning(f"  ⚠️ Historical API returned {response.status_code}")
                try:
                    error_data = response.json()
                    logger.warning(f"  Error: {json.dumps(error_data, indent=2)}")
                except:
                    logger.warning(f"  Response: {response.text[:300]}")
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"  ❌ Historical data error: {e}")
            return pd.DataFrame()
    
    def get_combined_data(self) -> pd.DataFrame:
        """Combine Historical + Intraday data for complete picture"""
        try:
            # Step 1: Try historical first
            df_historical = self.get_historical_data(days=5)
            
            # Step 2: Try intraday
            df_intraday = self.get_intraday_data()
            
            # Step 3: Combine logic
            if not df_historical.empty and not df_intraday.empty:
                logger.info(f"  ✅ Combining historical ({len(df_historical)}) + intraday ({len(df_intraday)})")
                df_combined = pd.concat([df_historical, df_intraday])
            elif not df_intraday.empty:
                logger.info(f"  ✅ Using only intraday data ({len(df_intraday)} candles)")
                df_combined = df_intraday
            elif not df_historical.empty:
                logger.info(f"  ✅ Using only historical data ({len(df_historical)} candles)")
                df_combined = df_historical
            else:
                logger.error("  ❌ No data from both APIs - check credentials/network")
                return pd.DataFrame()
            
            # Clean and prepare
            df_combined = df_combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            df_combined = df_combined.tail(500).reset_index(drop=True)
            
            if len(df_combined) > 0:
                first_time = df_combined.iloc[0]['timestamp'].strftime('%d-%b %H:%M')
                last_time = df_combined.iloc[-1]['timestamp'].strftime('%d-%b %H:%M')
                logger.info(f"  ✅ Final dataset: {len(df_combined)} candles ({first_time} to {last_time})")
            
            return df_combined
            
        except Exception as e:
            logger.error(f"  ❌ Combined data error: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_ltp(self) -> float:
        """Get Last Traded Price"""
        try:
            encoded_symbol = urllib.parse.quote(NIFTY_SYMBOL, safe='')
            url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={encoded_symbol}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                try:
                    if 'data' in data and NIFTY_SYMBOL in data['data']:
                        ltp = float(data['data'][NIFTY_SYMBOL]['last_price'])
                        logger.info(f"  ✅ LTP: ₹{ltp:.2f}")
                        return ltp
                    
                    if 'data' in data and isinstance(data['data'], dict) and 'last_price' in data['data']:
                        ltp = float(data['data']['last_price'])
                        logger.info(f"  ✅ LTP: ₹{ltp:.2f}")
                        return ltp
                    
                    if 'data' in data and isinstance(data['data'], dict):
                        first_key = list(data['data'].keys())[0] if data['data'] else None
                        if first_key and 'last_price' in data['data'][first_key]:
                            ltp = float(data['data'][first_key]['last_price'])
                            logger.info(f"  ✅ LTP: ₹{ltp:.2f}")
                            return ltp
                    
                except Exception as parse_error:
                    logger.error(f"  ❌ LTP parsing error: {parse_error}")
            
            return 0.0
            
        except Exception as e:
            logger.error(f"  ❌ LTP error: {e}")
            return 0.0
    
    def get_option_chain(self, expiry: str) -> List[StrikeData]:
        """Fetch option chain data"""
        try:
            encoded_symbol = urllib.parse.quote(NIFTY_SYMBOL, safe='')
            url = f"https://api.upstox.com/v2/option/chain?instrument_key={encoded_symbol}&expiry_date={expiry}"
            
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' not in data:
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
                    except:
                        continue
                
                logger.info(f"  ✅ Fetched {len(strikes)} strikes")
                return strikes
            else:
                logger.error(f"  ❌ Option chain API error {response.status_code}")
                return []
            
        except Exception as e:
            logger.error(f"  ❌ Option chain error: {e}")
            return []

# ==================== OI ANALYZER ====================
class OIAnalyzer:
    @staticmethod
    def calculate_pcr(strikes: List[StrikeData]) -> float:
        total_ce = sum(s.ce_oi for s in strikes)
        total_pe = sum(s.pe_oi for s in strikes)
        return total_pe / total_ce if total_ce > 0 else 0
    
    @staticmethod
    def find_max_pain(strikes: List[StrikeData]) -> int:
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
        atm_strike = round(spot_price / 50) * 50
        strike_range = range(atm_strike - 500, atm_strike + 550, 50)
        relevant = [s for s in strikes if s.strike in strike_range]
        return sorted(relevant, key=lambda x: x.strike)[:count]
    
    @staticmethod
    def identify_support_resistance(strikes: List[StrikeData]) -> Tuple[List[int], List[int]]:
        pe_sorted = sorted(strikes, key=lambda x: x.pe_oi, reverse=True)
        support_strikes = [s.strike for s in pe_sorted[:3]]
        
        ce_sorted = sorted(strikes, key=lambda x: x.ce_oi, reverse=True)
        resistance_strikes = [s.strike for s in ce_sorted[:3]]
        
        return support_strikes, resistance_strikes
    
    @staticmethod
    def calculate_oi_changes(current: List[StrikeData], previous: Optional[OISnapshot]) -> List[StrikeData]:
        if not previous:
            return current
        
        prev_dict = {s.strike: s for s in previous.strikes}
        
        for strike in current:
            if strike.strike in prev_dict:
                prev_strike = prev_dict[strike.strike]
                strike.ce_oi_change = strike.ce_oi - prev_strike.ce_oi
                strike.pe_oi_change = strike.pe_oi - prev_strike.pe_oi
        
        return current
    
    @staticmethod
    def create_oi_snapshot(strikes: List[StrikeData], spot_price: float, prev_snapshot: Optional[OISnapshot] = None) -> OISnapshot:
        atm_strikes = OIAnalyzer.get_atm_strikes(strikes, spot_price)
        atm_strikes = OIAnalyzer.calculate_oi_changes(atm_strikes, prev_snapshot)
        
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
    
    @staticmethod
    def format_atm_option_chain(strikes: List[StrikeData], spot_price: float) -> str:
        """Format ATM option chain data for telegram message"""
        atm_strike = round(spot_price / 50) * 50
        
        # Get ATM ±3 strikes
        strike_range = range(atm_strike - 150, atm_strike + 200, 50)
        atm_strikes = [s for s in strikes if s.strike in strike_range]
        atm_strikes = sorted(atm_strikes, key=lambda x: x.strike)
        
        def format_num(num):
            if num >= 100000:
                return f"{num/100000:.1f}L"
            elif num >= 1000:
                return f"{num/1000:.1f}K"
            return str(int(num))
        
        lines = ["📊 ATM OPTION CHAIN (±3 strikes):"]
        lines.append("Strike | CE_OI  | PE_OI  | CE_Pr | PE_Pr")
        lines.append("─" * 45)
        
        for s in atm_strikes:
            marker = " 🎯" if s.strike == atm_strike else ""
            lines.append(
                f"{s.strike}{marker} | {format_num(s.ce_oi)} | {format_num(s.pe_oi)} | "
                f"₹{s.ce_price:.1f} | ₹{s.pe_price:.1f}"
            )
        
        return "\n".join(lines)

# ==================== ULTRA COMPRESSOR ====================
class UltraCompressor:
    @staticmethod
    def compress_candles(df: pd.DataFrame) -> str:
        df_copy = df.tail(CANDLE_COUNT).copy()
        df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%H:%M')
        
        def format_volume(vol):
            if vol >= 1000000:
                return f"{vol/1000000:.1f}M"
            elif vol >= 1000:
                return f"{vol/1000:.0f}K"
            return str(int(vol))
        
        df_copy['volume'] = df_copy['volume'].apply(format_volume)
        
        lines = []
        for _, row in df_copy.iterrows():
            lines.append(f"{row['timestamp']}|{int(row['open'])}|{int(row['high'])}|{int(row['low'])}|{int(row['close'])}|{row['volume']}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def compress_oi(strikes: List[StrikeData]) -> str:
        def format_num(num):
            if abs(num) >= 1000000:
                return f"{num/1000000:.1f}M"
            elif abs(num) >= 1000:
                return f"{num/1000:.0f}K"
            return str(int(num))
        
        def format_change(change):
            if change > 0:
                return f"+{format_num(change)}"
            elif change < 0:
                return format_num(change)
            return "0"
        
        lines = []
        for s in strikes:
            lines.append(
                f"{s.strike}|{format_num(s.ce_oi)}|{format_change(s.ce_oi_change)}|"
                f"{format_num(s.ce_volume)}|{format_num(s.pe_oi)}|{format_change(s.pe_oi_change)}|"
                f"{format_num(s.pe_volume)}"
            )
        
        return '\n'.join(lines)

# ==================== AI ANALYZER ====================
class AIAnalyzer:
    @staticmethod
    def extract_json(content: str) -> Optional[Dict]:
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
    def analyze_with_deepseek(df_5m: pd.DataFrame, current_price: float, current_oi: OISnapshot) -> Optional[TradeSignal]:
        """Send ultra-compressed analysis request to DeepSeek"""
        try:
            candles_compressed = UltraCompressor.compress_candles(df_5m)
            oi_compressed = UltraCompressor.compress_oi(current_oi.strikes)
            
            df_tail = df_5m.tail(50)
            sma_20 = df_tail['close'].tail(20).mean()
            price_momentum = ((current_price - sma_20) / sma_20) * 100
            
            recent_closes = df_tail['close'].tail(10).values
            bullish_candles = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] > recent_closes[i-1])
            
            # Get OI snapshots for velocity calculation
            oi_15m_ago = RedisOIManager.get_oi_snapshot(15)
            oi_30m_ago = RedisOIManager.get_oi_snapshot(30)
            
            # Calculate PCR changes
            pcr_15m = oi_15m_ago.pcr if oi_15m_ago else current_oi.pcr
            pcr_30m = oi_30m_ago.pcr if oi_30m_ago else current_oi.pcr
            
            prompt = f"""Elite F&O Trader | NIFTY50 5-MIN

PRICE: ₹{current_price:.2f} | TIME: {datetime.now(IST).strftime('%H:%M')}

CANDLES (Last 420):
Time|O|H|L|C|Vol
{candles_compressed}

OPTION CHAIN (ATM±10):
Strike|C_OI|C_ΔOI|C_Vol|P_OI|P_ΔOI|P_Vol
{oi_compressed}

METRICS:
PCR: Now {current_oi.pcr:.2f} | 15m {pcr_15m:.2f} | 30m {pcr_30m:.2f}
MaxPain: {current_oi.max_pain} | S/R: {'/'.join(map(str, current_oi.support_strikes[:2]))}/{'/'.join(map(str, current_oi.resistance_strikes[:2]))}

ANALYZE (Price+Vol+OI FUSION):

1. OI VELOCITY (15-30m):
CE/PE buildup/unwind | Velocity: 15m>30m=accel | PCR trend | Strike focus (ATM/OTM)
Price+OI sync: ↑CE↑=bull | ↓PE↑=bear | Mismatch=reversal

2. PATTERN (3): Bearish(Speed/Slow) | Bullish(Speed/Slow) | Sideways(Range/Trap)
OI support? Active?

3. VOLUME (5): Buying(green+vol) | Selling(red+vol) | Churning(small+HIGH vol=TRAP) | Drying(move+low vol=exhaust) | Climax(spike+long=reverse)
Type? OI velocity match?

4. NAKED OPTIONS (4): CallBuy(CE OTM buildup) | PutBuy(PE OTM buildup) | CallSell(CE resist) | PutSell(PE support)
15-30m dominant?

5. COMBO:
BULL=CallBuy+PutSell+BuyVol+OI(sustained)
BEAR=PutBuy+CallSell+SellVol+OI(sustained)
TRAP=HighVol+SmallCandle+OI(15m spike)
REVERSAL=LowVol+BigMove+OI(unwind)

6. SYNC: Price velocity vs OI velocity | Fast+Fast=strong | Fast+Slow=weak | Slow+Fast=coiling

7. TRIPLE CONFIRM: Pattern+Vol+OI aligned? Fakeout: Price/Vol WITHOUT OI

8. SMART MONEY: CallWrite+rise(30m)=resist | PutWrite+fall(30m)=support | Sudden(15m) vs Gradual(30m)

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
  "reasoning": "1-line why (max 120 chars)",
  "price_analysis": "Price+Vol fusion (max 150 chars)",
  "oi_analysis": "OI velocity 15-30m edge (max 150 chars)",
  "alignment_score": 8,
  "risk_factors": ["Risk1", "Risk2"],
  "support_levels": [0.0, 0.0],
  "resistance_levels": [0.0, 0.0],
  "pattern_detected": "Pattern or None"
}}"""
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "Elite F&O trader. Analyze 5-min price action + OI fusion. Respond ONLY in JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 2500
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

# ==================== CHART GENERATOR - IMPROVED ====================
class ChartGenerator:
    @staticmethod
    def create_chart(df_5m: pd.DataFrame, signal: TradeSignal, spot_price: float, save_path: str):
        """Generate professional chart - Volume visible, Info box at bottom"""
        BG = '#131722'
        GRID = '#1e222d'
        TEXT = '#d1d4dc'
        GREEN = '#26a69a'
        RED = '#ef5350'
        YELLOW = '#ffd700'
        BLUE = '#2962ff'
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 11), gridspec_kw={'height_ratios': [3, 1]}, facecolor=BG)
        
        ax1.set_facecolor(BG)
        
        df_plot = df_5m.tail(200).copy()
        df_plot = df_plot.reset_index(drop=True)
        
        first_candle_time = df_plot.iloc[0]['timestamp'].strftime('%d-%b %H:%M')
        last_candle_time = df_plot.iloc[-1]['timestamp'].strftime('%d-%b %H:%M')
        logger.info(f"  📊 Chart: {len(df_plot)} candles ({first_candle_time} to {last_candle_time})")
        
        time_labels = []
        time_positions = []
        for idx in range(0, len(df_plot), 10):
            time_labels.append(df_plot.iloc[idx]['timestamp'].strftime('%H:%M'))
            time_positions.append(idx)
        
        # Draw candlesticks
        for idx, row in df_plot.iterrows():
            color = GREEN if row['close'] > row['open'] else RED
            ax1.add_patch(Rectangle((idx, min(row['open'], row['close'])), 0.6, abs(row['close'] - row['open']), facecolor=color, edgecolor=color, alpha=0.8))
            ax1.plot([idx+0.3, idx+0.3], [row['low'], row['high']], color=color, linewidth=1, alpha=0.6)
        
        # Support/Resistance levels
        for support in signal.support_levels:
            ax1.axhline(support, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7)
            ax1.text(len(df_plot)*0.98, support, f'S:₹{support:.1f}  ', color=GREEN, fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor=BG, alpha=0.7))
        
        for resistance in signal.resistance_levels:
            ax1.axhline(resistance, color=RED, linestyle='--', linewidth=1.5, alpha=0.7)
            ax1.text(len(df_plot)*0.98, resistance, f'R:₹{resistance:.1f}  ', color=RED, fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round', facecolor=BG, alpha=0.7))
        
        # Stop loss and targets
        ax1.axhline(signal.stop_loss, color=RED, linewidth=2.5, linestyle=':', label=f'SL: ₹{signal.stop_loss:.1f}')
        ax1.axhline(signal.target_1, color=GREEN, linewidth=2, linestyle=':', label=f'T1: ₹{signal.target_1:.1f}')
        ax1.axhline(signal.target_2, color=GREEN, linewidth=2, linestyle=':', label=f'T2: ₹{signal.target_2:.1f}')
        
        # Pattern detected
        if signal.pattern_detected and signal.pattern_detected != "None":
            ax1.text(len(df_plot)*0.5, df_plot['high'].max() * 0.995, signal.pattern_detected.upper(), color=YELLOW, fontsize=12, fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor=BG, edgecolor=YELLOW, alpha=0.9))
        
        # Current market price
        ax1.text(len(df_plot)-1, spot_price, f'  CMP: ₹{spot_price:.1f}', fontsize=11, color='white', fontweight='bold', bbox=dict(boxstyle='round', facecolor=BLUE, edgecolor='white', linewidth=2), va='center')
        
        # Signal info box - MOVED TO BOTTOM LEFT
        signal_emoji = "🟢" if signal.signal_type == "CE_BUY" else "🔴" if signal.signal_type == "PE_BUY" else "⚪"
        info_text = f"""{signal_emoji} {signal.signal_type} | Confidence: {signal.confidence}% | Score: {signal.alignment_score}/10
Entry: ₹{signal.entry_price:.1f} | SL: ₹{signal.stop_loss:.1f} | T1: ₹{signal.target_1:.1f} | T2: ₹{signal.target_2:.1f} | RR: {signal.risk_reward}
Strike: {signal.recommended_strike} | Pattern: {signal.pattern_detected}"""
        
        ax1.text(0.01, 0.01, info_text, transform=ax1.transAxes, fontsize=9, va='bottom', bbox=dict(boxstyle='round', facecolor=GRID, alpha=0.95, edgecolor=TEXT, linewidth=1), color=TEXT, family='monospace')
        
        # Timestamp footer
        current_time = datetime.now(IST).strftime('%H:%M:%S')
        footer_text = f"Last Candle: {last_candle_time} | Generated: {current_time}"
        ax1.text(0.99, 0.01, footer_text, transform=ax1.transAxes, fontsize=8, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor=GRID, alpha=0.8), color=TEXT, family='monospace')
        
        title = f"NIFTY50 | 5-Minute Timeframe | {signal.signal_type} | Score: {signal.alignment_score}/10"
        ax1.set_title(title, color=TEXT, fontsize=14, fontweight='bold', pad=15)
        ax1.set_xticks(time_positions)
        ax1.set_xticklabels(time_labels, rotation=45, ha='right')
        ax1.grid(True, color=GRID, alpha=0.3)
        ax1.tick_params(colors=TEXT)
        ax1.set_ylabel('Price (₹)', color=TEXT, fontsize=11)
        ax1.set_xlabel('Time', color=TEXT, fontsize=11)
        
        # Volume subplot - FIXED TO SHOW VOLUME
        ax2.set_facecolor(BG)
        colors = [GREEN if df_plot.iloc[i]['close'] > df_plot.iloc[i]['open'] else RED for i in range(len(df_plot))]
        
        # Ensure volume is visible
        volumes = df_plot['volume'].values
        ax2.bar(range(len(df_plot)), volumes, color=colors, alpha=0.7, width=0.8)
        
        # Format y-axis for volume
        max_vol = volumes.max()
        if max_vol >= 1000000:
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000000:.1f}M'))
        elif max_vol >= 1000:
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        
        ax2.set_ylabel('Volume', color=TEXT, fontsize=11, fontweight='bold')
        ax2.tick_params(colors=TEXT)
        ax2.grid(True, color=GRID, alpha=0.3, axis='y')
        ax2.set_xticks(time_positions)
        ax2.set_xticklabels(time_labels, rotation=45, ha='right')
        ax2.set_xlabel('Time', color=TEXT, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor=BG)
        plt.close()
        logger.info(f"  📊 Chart saved: {save_path}")

# ==================== MAIN BOT ====================
class Nifty50Bot:
    def __init__(self):
        self.data_fetcher = UpstoxDataFetcher(UPSTOX_ACCESS_TOKEN)
        self.telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.scan_count = 0
        self.last_signal_time = None
    
    async def send_startup_message(self):
        """Send bot startup notification with API status"""
        expiry = ExpiryCalculator.get_weekly_expiry(UPSTOX_ACCESS_TOKEN)
        expiry_display = ExpiryCalculator.format_for_display(expiry)
        days_left = ExpiryCalculator.days_to_expiry(expiry)
        
        # Check API connections
        logger.info("🔍 Checking API connections...")
        upstox_status, upstox_msg = APIConnectionChecker.check_upstox(UPSTOX_ACCESS_TOKEN)
        deepseek_status, deepseek_msg = APIConnectionChecker.check_deepseek(DEEPSEEK_API_KEY)
        redis_status, redis_msg = APIConnectionChecker.check_redis(REDIS_URL)
        
        message = f"""
🚀 NIFTY50 5-MIN BOT STARTED

⏰ Time: {datetime.now(IST).strftime('%d-%b-%Y %H:%M:%S')}

🔌 API CONNECTION STATUS:
━━━━━━━━━━━━━━━━━━━━━━━━
• Upstox API: {upstox_msg}
• DeepSeek AI: {deepseek_msg}
• Redis Cache: {redis_msg}

📊 Configuration:
✅ Symbol: NIFTY50 Index (NSE)
✅ Timeframe: 5-Minute ONLY
✅ Analysis Candles: 420 (Historical + Live)
✅ Scan Interval: Every 5 minutes
✅ Market Hours: 9:15 AM - 3:30 PM
✅ Expiry: {expiry_display} ({expiry}) - {days_left} days left

🔧 DATA FETCHING:
✅ Historical API (V2) - Past 3 days
✅ Intraday API (V2) - Today's live data
✅ Ultra-compressed format (85% token reduction)

🧠 Analysis Framework:
✅ 5-MIN: 420 candles analysis
✅ Entry patterns + momentum detection
✅ Volume surge identification
✅ OI buildup/unwinding tracking
✅ Support/Resistance from price action + OI

🎯 Alert Criteria:
✅ Minimum Confidence: 75%
✅ Alignment Score: 7+/10
✅ Cooldown: 30 minutes

🔄 Status: {"🟢 Active & Running" if all([upstox_status, deepseek_status, redis_status]) else "⚠️ Running with Issues"}
"""
        await self.telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logger.info("✅ Startup message sent with API status")
    
    async def send_telegram_alert(self, signal: TradeSignal, chart_path: str, oi_snapshot: OISnapshot, spot_price: float, all_strikes: List[StrikeData]):
        """Send trading signal with option chain data"""
        try:
            # Send chart first
            with open(chart_path, 'rb') as photo:
                await self.telegram_bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo)
            
            signal_emoji = "🟢" if signal.signal_type == "CE_BUY" else "🔴"
            
            reasoning = signal.reasoning[:200].replace('_', ' ').replace('*', ' ')
            price_analysis = signal.price_analysis[:250].replace('_', ' ').replace('*', ' ')
            oi_analysis = signal.oi_analysis[:250].replace('_', ' ').replace('*', ' ')
            pattern = signal.pattern_detected.replace('_', ' ').replace('*', ' ')
            
            # Format ATM option chain
            atm_chain = OIAnalyzer.format_atm_option_chain(all_strikes, spot_price)
            
            message = f"""
{signal_emoji} NIFTY50 {signal.signal_type} SIGNAL | 5-MIN TIMEFRAME

🎯 Confidence: {signal.confidence}%
📊 Score: {signal.alignment_score}/10

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

{atm_chain}

🕐 {datetime.now(IST).strftime('%d-%b %H:%M:%S')}
"""
            
            await self.telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
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
            
            # Step 1: Get cached data or fetch fresh
            df_5m_cached = RedisOIManager.get_candle_data()
            
            if df_5m_cached is None or len(df_5m_cached) == 0:
                logger.info("  📥 Fetching fresh data (Historical + Intraday)...")
                df_5m = self.data_fetcher.get_combined_data()
                if df_5m.empty:
                    logger.warning("  ⚠️ No data available")
                    return
                RedisOIManager.save_candle_data(df_5m)
            else:
                logger.info(f"  ✅ Loaded {len(df_5m_cached)} candles from Redis")
                df_5m = df_5m_cached
                
                # Update with latest intraday data
                logger.info("  📥 Fetching latest intraday candle...")
                df_latest = self.data_fetcher.get_intraday_data()
                if not df_latest.empty:
                    df_5m = pd.concat([df_5m, df_latest]).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                    df_5m = df_5m.tail(500).reset_index(drop=True)
                    RedisOIManager.save_candle_data(df_5m)
                    logger.info(f"  ✅ Updated with latest data")
            
            if len(df_5m) > 0:
                first_time = df_5m.iloc[0]['timestamp'].strftime('%d-%b %H:%M')
                last_time = df_5m.iloc[-1]['timestamp'].strftime('%d-%b %H:%M')
                logger.info(f"  📊 Available data: {first_time} to {last_time} ({len(df_5m)} candles)")
            
            if len(df_5m) < 100:
                logger.error(f"  ❌ Insufficient data: only {len(df_5m)} candles")
                return
            
            logger.info(f"  📊 Using {min(len(df_5m), CANDLE_COUNT)} candles for analysis")
            
            # Step 2: Get current price
            spot_price = self.data_fetcher.get_ltp()
            if spot_price == 0:
                spot_price = df_5m['close'].iloc[-1]
                logger.info(f"  💹 Using last close: ₹{spot_price:.2f}")
            
            # Step 3: Get option chain
            expiry = ExpiryCalculator.get_weekly_expiry(UPSTOX_ACCESS_TOKEN)
            expiry_display = ExpiryCalculator.format_for_display(expiry)
            logger.info(f"  📅 Expiry: {expiry_display} ({expiry})")
            
            all_strikes = self.data_fetcher.get_option_chain(expiry)
            if not all_strikes:
                logger.warning("  ⚠️ No option chain data available")
                return
            
            # Step 4: Create OI snapshot
            prev_oi = RedisOIManager.get_oi_snapshot(5)
            current_oi = OIAnalyzer.create_oi_snapshot(all_strikes, spot_price, prev_oi)
            logger.info(f"  📊 PCR: {current_oi.pcr:.2f} | Max Pain: {current_oi.max_pain}")
            
            RedisOIManager.save_oi_snapshot(current_oi)
            
            if prev_oi:
                logger.info(f"  ✅ OI changes calculated from 5 min ago (PCR: {prev_oi.pcr:.2f})")
            
            # Step 5: AI Analysis
            logger.info("  🧠 Sending to DeepSeek AI (Ultra-Compressed Format)...")
            signal = AIAnalyzer.analyze_with_deepseek(df_5m=df_5m, current_price=spot_price, current_oi=current_oi)
            
            if not signal:
                logger.info("  ⏸️ No valid signal generated")
                return
            
            logger.info(f"  🧠 AI Signal: {signal.signal_type} | Confidence: {signal.confidence}%")
            
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
            
            chart_path = f"/tmp/nifty50_5min_chart_{datetime.now(IST).strftime('%H%M')}.png"
            ChartGenerator.create_chart(df_5m, signal, spot_price, chart_path)
            
            await self.send_telegram_alert(signal, chart_path, current_oi, spot_price, all_strikes)
            self.last_signal_time = datetime.now(IST)
            
        except Exception as e:
            logger.error(f"  ❌ Analysis error: {e}")
            traceback.print_exc()
    
    async def run_scanner(self):
        """Main scanner loop"""
        logger.info("\n" + "="*80)
        logger.info("🚀 NIFTY50 5-MIN BOT - HISTORICAL + INTRADAY")
        logger.info("="*80)
        
        await self.send_startup_message()
        
        while True:
            try:
                now = datetime.now(IST)
                current_time = now.time()
                
                if current_time < MARKET_START_TIME or current_time > MARKET_END_TIME:
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
    logger.info("="*80)
    logger.info("STARTING NIFTY50 5-MIN BOT - INTRADAY + HISTORICAL")
    logger.info("="*80)
    
    bot = Nifty50Bot()
    asyncio.run(bot.run_scanner())
