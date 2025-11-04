#!/usr/bin/env python3
"""
F&O ANALYSIS BOT WITH DEEPSEEK V3 + REDIS
==========================================
✅ Top 35 Stocks + 2 Indices (BankNifty, MidcapNifty)
✅ Every 15 minutes scan
✅ Redis: Store OI/Volume history (2 hours, 8 scans)
✅ Compare: Current vs Previous OI changes
✅ OI + Candlestick + FII/DII + VIX + News
✅ DeepSeek V3 Analysis
✅ Telegram Alerts (Only Buy Opportunities)
"""

import os
import asyncio
import requests
import urllib.parse
from datetime import datetime, timedelta, time
import pytz
import json
import time as time_sleep
from telegram import Bot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import traceback
import feedparser
import redis

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('fo_analyzer.log')]
)
logger = logging.getLogger(__name__)

# API Keys
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

BASE_URL = "https://api.upstox.com"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
SCAN_INTERVAL = 900  # 15 minutes
REDIS_EXPIRY = 259200  # 3 days in seconds

# ==================== SYMBOLS (37 Total) ====================
INDICES = {
    "NSE_INDEX|Nifty Bank": {"name": "BANKNIFTY", "display_name": "BANK NIFTY", "expiry_day": 2},
    "NSE_INDEX|NIFTY MID SELECT": {"name": "MIDCPNIFTY", "display_name": "MIDCAP NIFTY", "expiry_day": 0}
}

TOP_STOCKS = {
    "NSE_EQ|INE040A01034": {"name": "HDFCBANK", "display_name": "HDFC BANK"},
    "NSE_EQ|INE090A01021": {"name": "ICICIBANK", "display_name": "ICICI BANK"},
    "NSE_EQ|INE062A01020": {"name": "SBIN", "display_name": "STATE BANK"},
    "NSE_EQ|INE238A01034": {"name": "AXISBANK", "display_name": "AXIS BANK"},
    "NSE_EQ|INE237A01028": {"name": "KOTAKBANK", "display_name": "KOTAK BANK"},
    "NSE_EQ|INE028A01039": {"name": "BANKBARODA", "display_name": "BANK OF BARODA"},
    "NSE_EQ|INE476A01014": {"name": "CANBK", "display_name": "CANARA BANK"},
    "NSE_EQ|INE528G01035": {"name": "FEDERALBNK", "display_name": "FEDERAL BANK"},
    "NSE_EQ|INE467B01029": {"name": "TATAMOTORS", "display_name": "TATA MOTORS"},
    "NSE_EQ|INE585B01010": {"name": "MARUTI", "display_name": "MARUTI SUZUKI"},
    "NSE_EQ|INE101A01026": {"name": "M&M", "display_name": "M&M"},
    "NSE_EQ|INE917I01010": {"name": "BAJAJ-AUTO", "display_name": "BAJAJ AUTO"},
    "NSE_EQ|INE002A01018": {"name": "RELIANCE", "display_name": "RELIANCE"},
    "NSE_EQ|INE213A01029": {"name": "ONGC", "display_name": "ONGC"},
    "NSE_EQ|INE242A01010": {"name": "IOC", "display_name": "INDIAN OIL"},
    "NSE_EQ|INE009A01021": {"name": "INFY", "display_name": "INFOSYS"},
    "NSE_EQ|INE075A01022": {"name": "WIPRO", "display_name": "WIPRO"},
    "NSE_EQ|INE854D01024": {"name": "TCS", "display_name": "TCS"},
    "NSE_EQ|INE860A01027": {"name": "HCLTECH", "display_name": "HCL TECH"},
    "NSE_EQ|INE214T01019": {"name": "LTIM", "display_name": "LTI MINDTREE"},
    "NSE_EQ|INE081A01012": {"name": "TATASTEEL", "display_name": "TATA STEEL"},
    "NSE_EQ|INE019A01038": {"name": "JSWSTEEL", "display_name": "JSW STEEL"},
    "NSE_EQ|INE038A01020": {"name": "HINDALCO", "display_name": "HINDALCO"},
    "NSE_EQ|INE044A01036": {"name": "SUNPHARMA", "display_name": "SUN PHARMA"},
    "NSE_EQ|INE361B01024": {"name": "DIVISLAB", "display_name": "DIVI'S LAB"},
    "NSE_EQ|INE089A01023": {"name": "DRREDDY", "display_name": "DR REDDY'S"},
    "NSE_EQ|INE154A01025": {"name": "ITC", "display_name": "ITC"},
    "NSE_EQ|INE030A01027": {"name": "HINDUNILVR", "display_name": "HUL"},
    "NSE_EQ|INE216A01030": {"name": "BRITANNIA", "display_name": "BRITANNIA"},
    "NSE_EQ|INE018A01030": {"name": "LT", "display_name": "L&T"},
    "NSE_EQ|INE742F01042": {"name": "ADANIPORTS", "display_name": "ADANI PORTS"},
    "NSE_EQ|INE397D01024": {"name": "BHARTIARTL", "display_name": "BHARTI AIRTEL"},
    "NSE_EQ|INE296A01024": {"name": "BAJFINANCE", "display_name": "BAJAJ FINANCE"},
    "NSE_EQ|INE280A01028": {"name": "TITAN", "display_name": "TITAN"},
    "NSE_EQ|INE021A01026": {"name": "ASIANPAINT", "display_name": "ASIAN PAINTS"}
}

ALL_SYMBOLS = {**INDICES, **TOP_STOCKS}

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

@dataclass
class OISnapshot:
    timestamp: str
    spot_price: float
    strikes: List[Dict]
    pcr: float
    total_ce_oi: int
    total_pe_oi: int
    total_ce_volume: int
    total_pe_volume: int

@dataclass
class MarketContext:
    fii_buy: float
    fii_sell: float
    fii_net: float
    dii_buy: float
    dii_sell: float
    dii_net: float
    vix: float
    news_headlines: List[str]

# ==================== REDIS MANAGER ====================
class RedisManager:
    def __init__(self):
        try:
            # Timeout added - don't block forever!
            self.redis_client = redis.from_url(
                REDIS_URL, 
                decode_responses=True,
                socket_connect_timeout=5,  # 5 seconds timeout
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            logger.warning("⚠️ Running in NO-REDIS mode (OI comparison disabled)")
            self.redis_client = None
    
    def save_oi_snapshot(self, symbol: str, snapshot: OISnapshot):
        if not self.redis_client:
            return
        
        try:
            key = f"oi:{symbol}:current"
            data = {
                'timestamp': snapshot.timestamp,
                'spot_price': snapshot.spot_price,
                'strikes': json.dumps(snapshot.strikes),
                'pcr': snapshot.pcr,
                'total_ce_oi': snapshot.total_ce_oi,
                'total_pe_oi': snapshot.total_pe_oi,
                'total_ce_volume': snapshot.total_ce_volume,
                'total_pe_volume': snapshot.total_pe_volume
            }
            
            self.redis_client.hset(key, mapping=data)
            self.redis_client.expire(key, REDIS_EXPIRY)
            
            history_key = f"oi:{symbol}:history"
            self.redis_client.lpush(history_key, json.dumps(data))
            self.redis_client.ltrim(history_key, 0, 7)
            self.redis_client.expire(history_key, REDIS_EXPIRY)
            
            logger.info(f"  💾 Redis: Saved {symbol}")
            
        except Exception as e:
            logger.error(f"Redis save error: {e}")
    
    def get_previous_oi(self, symbol: str) -> Optional[Dict]:
        if not self.redis_client:
            return None
        
        try:
            key = f"oi:{symbol}:current"
            data = self.redis_client.hgetall(key)
            
            if data:
                data['strikes'] = json.loads(data['strikes'])
                data['spot_price'] = float(data['spot_price'])
                data['pcr'] = float(data['pcr'])
                data['total_ce_oi'] = int(data['total_ce_oi'])
                data['total_pe_oi'] = int(data['total_pe_oi'])
                data['total_ce_volume'] = int(data['total_ce_volume'])
                data['total_pe_volume'] = int(data['total_pe_volume'])
                return data
            return None
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def get_oi_history(self, symbol: str) -> List[Dict]:
        if not self.redis_client:
            return []
        
        try:
            history_key = f"oi:{symbol}:history"
            history_raw = self.redis_client.lrange(history_key, 0, -1)
            
            history = []
            for item in history_raw:
                data = json.loads(item)
                data['strikes'] = json.loads(data['strikes'])
                history.append(data)
            
            return history
            
        except Exception as e:
            logger.error(f"Redis history error: {e}")
            return []
    
    def compare_oi_changes(self, symbol: str, current: OISnapshot) -> Dict:
        previous = self.get_previous_oi(symbol)
        
        if not previous:
            return {
                'has_previous': False,
                'spot_change': 0,
                'pcr_change': 0,
                'ce_oi_change': 0,
                'pe_oi_change': 0,
                'ce_volume_change': 0,
                'pe_volume_change': 0,
                'strike_changes': []
            }
        
        spot_change = current.spot_price - previous['spot_price']
        pcr_change = current.pcr - previous['pcr']
        ce_oi_change = current.total_ce_oi - previous['total_ce_oi']
        pe_oi_change = current.total_pe_oi - previous['total_pe_oi']
        ce_volume_change = current.total_ce_volume - previous['total_ce_volume']
        pe_volume_change = current.total_pe_volume - previous['total_pe_volume']
        
        strike_changes = []
        prev_strikes = {s['strike']: s for s in previous['strikes']}
        
        for curr_strike in current.strikes:
            strike_val = curr_strike['strike']
            if strike_val in prev_strikes:
                prev = prev_strikes[strike_val]
                strike_changes.append({
                    'strike': strike_val,
                    'ce_oi_change': curr_strike['ce_oi'] - prev['ce_oi'],
                    'pe_oi_change': curr_strike['pe_oi'] - prev['pe_oi'],
                    'ce_volume_change': curr_strike['ce_volume'] - prev['ce_volume'],
                    'pe_volume_change': curr_strike['pe_volume'] - prev['pe_volume']
                })
        
        return {
            'has_previous': True,
            'time_diff': '15 mins',
            'spot_change': spot_change,
            'spot_change_pct': (spot_change / previous['spot_price'] * 100) if previous['spot_price'] > 0 else 0,
            'pcr_change': pcr_change,
            'ce_oi_change': ce_oi_change,
            'pe_oi_change': pe_oi_change,
            'ce_volume_change': ce_volume_change,
            'pe_volume_change': pe_volume_change,
            'strike_changes': strike_changes
        }

# ==================== EXPIRY CALCULATOR ====================
class ExpiryCalculator:
    @staticmethod
    def get_all_expiries_from_api(instrument_key: str) -> List[str]:
        try:
            headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
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
    def get_best_expiry(instrument_key: str, symbol_info: Dict) -> str:
        expiry_day = symbol_info.get('expiry_day', 3)
        expiries = ExpiryCalculator.get_all_expiries_from_api(instrument_key)
        if expiries:
            today = datetime.now(IST).date()
            now_time = datetime.now(IST).time()
            future_expiries = [exp_str for exp_str in expiries 
                             if datetime.strptime(exp_str, '%Y-%m-%d').date() > today 
                             or (datetime.strptime(exp_str, '%Y-%m-%d').date() == today and now_time < time(15, 30))]
            if future_expiries:
                return min(future_expiries)
        return ExpiryCalculator.calculate_monthly_expiry(expiry_day)

# ==================== DATA FETCHERS ====================
class DataFetcher:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}", "Accept": "application/json"}
        self.nse_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        }
    
    def get_spot_price(self, instrument_key: str) -> float:
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            url = f"{BASE_URL}/v2/market-quote/ltp?instrument_key={encoded_key}"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', {})
                if data:
                    return float(list(data.values())[0].get('last_price', 0))
            return 0.0
        except:
            return 0.0
    
    def get_option_chain(self, instrument_key: str, expiry: str) -> List[StrikeData]:
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
            response = requests.get(url, headers=self.headers, timeout=20)
            if response.status_code == 200:
                data = response.json().get('data', [])
                strikes = []
                for item in data:
                    call_data = item.get('call_options', {}).get('market_data', {})
                    put_data = item.get('put_options', {}).get('market_data', {})
                    strikes.append(StrikeData(
                        strike=int(item.get('strike_price', 0)),
                        ce_oi=call_data.get('oi', 0),
                        pe_oi=put_data.get('oi', 0),
                        ce_volume=call_data.get('volume', 0),
                        pe_volume=put_data.get('volume', 0),
                        ce_ltp=call_data.get('ltp', 0),
                        pe_ltp=put_data.get('ltp', 0)
                    ))
                return strikes
            return []
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return []
    
    def get_candlestick_data(self, instrument_key: str) -> Optional[pd.DataFrame]:
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            
            # Try intraday first
            url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/15minute"
            response = requests.get(url, headers=self.headers, timeout=20)
            
            if response.status_code == 200:
                candles = response.json().get('data', {}).get('candles', [])
                if candles and len(candles) > 0:
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp').astype(float).sort_index()
                    return df.tail(8)
            
            # Fallback: Try daily data (for testing when market closed)
            logger.warning(f"  ⚠️ No intraday data, trying daily candles...")
            today = datetime.now(IST).strftime('%Y-%m-%d')
            url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/day/{today}"
            response = requests.get(url, headers=self.headers, timeout=20)
            
            if response.status_code == 200:
                candles = response.json().get('data', {}).get('candles', [])
                if candles and len(candles) > 0:
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp').astype(float).sort_index()
                    return df.tail(8)
            
            return None
        except Exception as e:
            logger.error(f"Candle data error: {e}")
            return None
    
    def get_vix(self) -> float:
        try:
            vix_key = "NSE_INDEX|India VIX"
            return self.get_spot_price(vix_key)
        except:
            return 0.0
    
    def get_fii_dii_data(self) -> Dict:
        try:
            url = "https://www.nseindia.com/api/fiidiiTrading"
            session = requests.Session()
            session.get("https://www.nseindia.com", headers=self.nse_headers, timeout=10)
            time_sleep.sleep(1)
            response = session.get(url, headers=self.nse_headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    fii_data = data[0] if isinstance(data, list) else data
                    return {
                        'fii_buy': float(fii_data.get('fii', {}).get('buyValue', 0)),
                        'fii_sell': float(fii_data.get('fii', {}).get('sellValue', 0)),
                        'fii_net': float(fii_data.get('fii', {}).get('netValue', 0)),
                        'dii_buy': float(fii_data.get('dii', {}).get('buyValue', 0)),
                        'dii_sell': float(fii_data.get('dii', {}).get('sellValue', 0)),
                        'dii_net': float(fii_data.get('dii', {}).get('netValue', 0))
                    }
            return {'fii_buy': 0, 'fii_sell': 0, 'fii_net': 0, 'dii_buy': 0, 'dii_sell': 0, 'dii_net': 0}
        except:
            return {'fii_buy': 0, 'fii_sell': 0, 'fii_net': 0, 'dii_buy': 0, 'dii_sell': 0, 'dii_net': 0}
    
    def get_market_news(self) -> List[str]:
        try:
            headlines = []
            feeds = [
                "https://www.moneycontrol.com/rss/latestnews.xml",
                "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
            ]
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:3]:
                        title = entry.title
                        if any(keyword in title.lower() for keyword in ['nifty', 'sensex', 'market', 'stock', 'rupee', 'rbi', 'fed']):
                            headlines.append(title)
                except:
                    continue
            return headlines[:5]
        except:
            return []

# ==================== CHART GENERATOR ====================
class ChartGenerator:
    @staticmethod
    def create_chart(symbol: str, df: pd.DataFrame, spot_price: float, path: str):
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        ax.set_facecolor('white')
        df_plot = df.reset_index(drop=True)
        
        for idx, row in df_plot.iterrows():
            color = '#26a69a' if row['close'] > row['open'] else '#ef5350'
            ax.plot([idx+0.3, idx+0.3], [row['low'], row['high']], color=color, linewidth=1.2)
            ax.add_patch(Rectangle((idx, min(row['open'], row['close'])), 0.6,
                                  abs(row['close'] - row['open']) if abs(row['close'] - row['open']) > 0 else spot_price * 0.0001,
                                  facecolor=color, edgecolor=color, alpha=0.85))
        
        ax.axhline(spot_price, color='#FF9800', linewidth=2, linestyle='--', alpha=0.7)
        ax.text(1, spot_price, f' ₹{spot_price:.2f}', color='#FF9800', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#FF9800'))
        
        ax.set_title(f"{symbol} | 15M | Last 2 Hours", fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Price (₹)', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path, dpi=120, facecolor='white')
        plt.close()

# ==================== DEEPSEEK ANALYZER ====================
class DeepSeekAnalyzer:
    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.url = DEEPSEEK_URL
    
    def compress_data(self, symbol: str, spot: float, candles: pd.DataFrame, strikes: List[StrikeData], 
                     context: MarketContext, expiry: str, oi_comparison: Dict) -> str:
        candle_str = "|".join([f"{int(row['open'])}/{int(row['high'])}/{int(row['low'])}/{int(row['close'])}:{int(row['volume']/1000)}K" 
                               for _, row in candles.iterrows()])
        
        atm = round(spot / 100) * 100
        atm_strikes = [s for s in strikes if abs(s.strike - atm) <= 1000][:21]
        oi_str = "\n".join([f"{s.strike}|C:{s.ce_oi//1000}K,{s.ce_volume//1000}K|P:{s.pe_oi//1000}K,{s.pe_volume//1000}K" 
                           for s in sorted(atm_strikes, key=lambda x: x.strike)])
        
        total_ce_oi = sum(s.ce_oi for s in atm_strikes)
        total_pe_oi = sum(s.pe_oi for s in atm_strikes)
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        news_str = "\n".join([f"- {h}" for h in context.news_headlines[:3]])
        
        oi_comp_str = ""
        if oi_comparison['has_previous']:
            oi_comp_str = f"""
OI CHANGES (vs 15 mins ago):
Spot: {oi_comparison['spot_change']:+.2f} ({oi_comparison['spot_change_pct']:+.2f}%)
PCR: {oi_comparison['pcr_change']:+.3f}
Total CE OI: {oi_comparison['ce_oi_change']:+,} | Vol: {oi_comparison['ce_volume_change']:+,}
Total PE OI: {oi_comparison['pe_oi_change']:+,} | Vol: {oi_comparison['pe_volume_change']:+,}

Top 5 Strike Changes:
{self._format_strike_changes(oi_comparison['strike_changes'][:5])}
"""
        else:
            oi_comp_str = "OI CHANGES: First scan, no previous data"
        
        data = f"""INSTRUMENT: {symbol}
SPOT: ₹{spot:.2f}
EXPIRY: {expiry}
VIX: {context.vix:.2f}

CANDLES (15M, Last 8): O/H/L/C:Vol
{candle_str}

OPTION CHAIN (21 ATM):
Strike|Call_OI,Vol|Put_OI,Vol
{oi_str}

PCR: {pcr:.3f}
{oi_comp_str}
FII/DII (Today, ₹Cr):
FII: Buy {context.fii_buy:.0f}, Sell {context.fii_sell:.0f}, Net {context.fii_net:+.0f}
DII: Buy {context.dii_buy:.0f}, Sell {context.dii_sell:.0f}, Net {context.dii_net:+.0f}

NEWS:
{news_str if news_str else "No major news"}"""
        
        return data
    
    def _format_strike_changes(self, changes: List[Dict]) -> str:
        lines = []
        for c in changes:
            lines.append(f"{c['strike']}: CE OI {c['ce_oi_change']:+,}, PE OI {c['pe_oi_change']:+,}")
        return "\n".join(lines)
    
    def analyze(self, compressed_data: str) -> Optional[str]:
        try:
            system_prompt = """You are an elite F&O trader with 20+ years experience in Indian markets (NSE). You specialize in pure price action + OI analysis without any lagging indicators.

Follow the framework in the provided data and give concise, actionable analysis. Focus on OI changes (last 15 mins), PCR, price action, volume, and confluence. Only recommend trades when alignment score ≥5/7."""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": compressed_data}
                ],
                "temperature": 0.3,
                "max_tokens": 800
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"DeepSeek error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"DeepSeek analysis error: {e}")
            return None

# ==================== TELEGRAM NOTIFIER ====================
class TelegramNotifier:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    async def send_startup(self):
        msg = f"""🚀 **F&O ANALYSIS BOT + REDIS**

⏰ {datetime.now(IST).strftime('%d-%b %H:%M IST')}

✅ Every 15 minutes
✅ 37 Symbols (2 Indices + 35 Stocks)
✅ Redis: OI/Volume tracking (2 hours)
✅ OI Comparison: Current vs Previous
✅ DeepSeek V3 Analysis
✅ Only Buy Opportunities

🟢 BOT ACTIVE"""
        await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
    
    async def send_opportunity(self, symbol: str, chart_path: str, analysis: str):
        try:
            if "NO TRADE" in analysis or "🚫" in analysis:
                logger.info(f"  ⏭️ No opportunity: {symbol}")
                return
            
            with open(chart_path, 'rb') as photo:
                await self.bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo)
            
            msg = f"""━━━━━━━━━━━━━━━━━━━━━━
💰 **{symbol}** - OPPORTUNITY
━━━━━━━━━━━━━━━━━━━━━━

{analysis}

🕐 {datetime.now(IST).strftime('%d-%b %H:%M IST')}
━━━━━━━━━━━━━━━━━━━━━━"""
            
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
            logger.info(f"  ✅ Opportunity sent: {symbol}")
            
        except Exception as e:
            logger.error(f"Telegram error: {e}")

# ==================== MAIN BOT ====================
class FOAnalyzerBot:
    def __init__(self):
        logger.info("🔄 Initializing F&O Analyzer Bot with Redis...")
        logger.info(f"📍 Redis URL: {REDIS_URL[:20]}..." if REDIS_URL else "❌ No Redis URL")
        
        try:
            self.fetcher = DataFetcher()
            logger.info("✅ DataFetcher initialized")
            
            self.analyzer = DeepSeekAnalyzer()
            logger.info("✅ DeepSeekAnalyzer initialized")
            
            self.notifier = TelegramNotifier()
            logger.info("✅ TelegramNotifier initialized")
            
            self.redis_manager = RedisManager()
            logger.info("✅ RedisManager initialized")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            traceback.print_exc()
            raise
        
        logger.info("✅ Bot ready")
    
    def is_market_open(self) -> bool:
        # Force IST timezone
        now = datetime.now(IST)
        
        # Debug log
        logger.info(f"  🕐 Current IST Time: {now.strftime('%H:%M:%S')} | Day: {now.strftime('%A')}")
        
        # Weekend check
        if now.weekday() >= 5:
            logger.info(f"  ⏸️ Weekend - Market Closed")
            return False
        
        current_time = now.time()
        market_open = time(9, 15) <= current_time <= time(15, 30)
        
        if not market_open:
            logger.info(f"  ⏸️ Market Hours: 9:15 AM - 3:30 PM | Current: {current_time.strftime('%H:%M')}")
        
        return market_open
    
    async def analyze_symbol(self, instrument_key: str, symbol_info: Dict, market_context: MarketContext):
        try:
            symbol_name = symbol_info.get('name', '')
            display_name = symbol_info.get('display_name', symbol_name)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 {display_name}")
            logger.info(f"{'='*60}")
            
            expiry = ExpiryCalculator.get_best_expiry(instrument_key, symbol_info)
            
            spot = self.fetcher.get_spot_price(instrument_key)
            if spot == 0:
                logger.warning(f"  ❌ No spot price")
                return
            logger.info(f"  💹 Spot: ₹{spot:.2f}")
            
            candles = self.fetcher.get_candlestick_data(instrument_key)
            if candles is None or len(candles) == 0:
                logger.warning(f"  ⚠️ No candle data")
                return
            logger.info(f"  📊 Candles: {len(candles)}")
            
            strikes = self.fetcher.get_option_chain(instrument_key, expiry)
            if not strikes:
                logger.warning(f"  ⚠️ No option chain")
                return
            logger.info(f"  🎯 Strikes: {len(strikes)}")
            
            atm = round(spot / 100) * 100
            atm_strikes = [s for s in strikes if abs(s.strike - atm) <= 1000][:21]
            
            total_ce_oi = sum(s.ce_oi for s in atm_strikes)
            total_pe_oi = sum(s.pe_oi for s in atm_strikes)
            total_ce_volume = sum(s.ce_volume for s in atm_strikes)
            total_pe_volume = sum(s.pe_volume for s in atm_strikes)
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            current_snapshot = OISnapshot(
                timestamp=datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                spot_price=spot,
                strikes=[{
                    'strike': s.strike,
                    'ce_oi': s.ce_oi,
                    'pe_oi': s.pe_oi,
                    'ce_volume': s.ce_volume,
                    'pe_volume': s.pe_volume
                } for s in atm_strikes],
                pcr=pcr,
                total_ce_oi=total_ce_oi,
                total_pe_oi=total_pe_oi,
                total_ce_volume=total_ce_volume,
                total_pe_volume=total_pe_volume
            )
            
            oi_comparison = self.redis_manager.compare_oi_changes(symbol_name, current_snapshot)
            
            if oi_comparison['has_previous']:
                logger.info(f"  📊 OI Changes: CE {oi_comparison['ce_oi_change']:+,}, PE {oi_comparison['pe_oi_change']:+,}")
                logger.info(f"  📈 Spot Change: {oi_comparison['spot_change']:+.2f} ({oi_comparison['spot_change_pct']:+.2f}%)")
            else:
                logger.info(f"  📊 First scan - no previous data")
            
            self.redis_manager.save_oi_snapshot(symbol_name, current_snapshot)
            
            chart_path = f"/tmp/{symbol_name}_chart.png"
            ChartGenerator.create_chart(display_name, candles, spot, chart_path)
            
            compressed = self.analyzer.compress_data(
                symbol_name, spot, candles, strikes, market_context, expiry, oi_comparison
            )
            logger.info(f"  🗜️ Data compressed")
            
            logger.info(f"  🤖 Analyzing with DeepSeek V3...")
            analysis = self.analyzer.analyze(compressed)
            
            if analysis:
                logger.info(f"  ✅ Analysis received")
                await self.notifier.send_opportunity(display_name, chart_path, analysis)
            else:
                logger.warning(f"  ❌ Analysis failed")
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol_info.get('display_name')}: {e}")
            traceback.print_exc()
    
    async def run_scan(self):
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 SCAN START - {datetime.now(IST).strftime('%H:%M:%S IST')}")
        logger.info(f"{'='*80}")
        
        logger.info("\n📊 Fetching Market Context...")
        fii_dii = self.fetcher.get_fii_dii_data()
        vix = self.fetcher.get_vix()
        news = self.fetcher.get_market_news()
        
        market_context = MarketContext(
            fii_buy=fii_dii['fii_buy'],
            fii_sell=fii_dii['fii_sell'],
            fii_net=fii_dii['fii_net'],
            dii_buy=fii_dii['dii_buy'],
            dii_sell=fii_dii['dii_sell'],
            dii_net=fii_dii['dii_net'],
            vix=vix,
            news_headlines=news
        )
        
        logger.info(f"  VIX: {vix:.2f}")
        logger.info(f"  FII Net: ₹{fii_dii['fii_net']:.0f} Cr")
        logger.info(f"  DII Net: ₹{fii_dii['dii_net']:.0f} Cr")
        logger.info(f"  News: {len(news)} headlines")
        
        for idx, (key, info) in enumerate(ALL_SYMBOLS.items(), 1):
            logger.info(f"\n[{idx}/{len(ALL_SYMBOLS)}]")
            await self.analyze_symbol(key, info, market_context)
            
            if idx < len(ALL_SYMBOLS):
                await asyncio.sleep(2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ SCAN COMPLETE")
        logger.info(f"{'='*80}\n")
    
    async def run(self):
        logger.info("="*80)
        logger.info("F&O ANALYSIS BOT WITH REDIS + DEEPSEEK V3")
        logger.info("="*80)
        
        if not all([UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY]):
            logger.error("❌ Missing API credentials!")
            logger.error("Required: UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY")
            return
        
        await self.notifier.send_startup()
        
        logger.info("="*80)
        logger.info("🟢 RUNNING - Every 15 minutes with Redis OI tracking")
        logger.info("="*80)
        
        while True:
            try:
                if not self.is_market_open():
                    logger.info("⏸️ Market closed. Waiting...")
                    await asyncio.sleep(300)
                    continue
                
                await self.run_scan()
                
                logger.info(f"⏳ Next scan in 15 minutes...")
                await asyncio.sleep(SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopped by user")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)

# ==================== ENTRY POINT ====================
async def main():
    try:
        bot = FOAnalyzerBot()
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("STARTING F&O ANALYSIS BOT WITH REDIS")
    logger.info("="*80)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n✅ Shutdown complete")
    except Exception as e:
        logger.error(f"\n❌ Critical error: {e}")
        traceback.print_exc()
