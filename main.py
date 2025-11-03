#!/usr/bin/env python3
"""
OPTION CHAIN + CHART MONITOR v1.0
===================================
✅ Every 5 minutes scan
✅ 21 ATM Strikes (OI, Volume, LTP)
✅ 15M Candlestick Chart (Last 400 candles)
✅ Clean PNG Chart Format
✅ Telegram Alerts with Option Chain Table
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
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import traceback

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('option_monitor.log')]
)
logger = logging.getLogger(__name__)

# API Keys
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

BASE_URL = "https://api.upstox.com"
SCAN_INTERVAL = 300  # 5 minutes

# ==================== SYMBOLS ====================
INDICES = {
    "NSE_INDEX|Nifty 50": {"name": "NIFTY", "display_name": "NIFTY 50", "expiry_day": 3},
    "NSE_INDEX|Nifty Bank": {"name": "BANKNIFTY", "display_name": "BANK NIFTY", "expiry_day": 2},
    "NSE_INDEX|NIFTY MID SELECT": {"name": "MIDCPNIFTY", "display_name": "MIDCAP NIFTY", "expiry_day": 0}
}

FO_STOCKS = {
    "NSE_EQ|INE467B01029": {"name": "TATAMOTORS", "display_name": "TATA MOTORS"},
    "NSE_EQ|INE585B01010": {"name": "MARUTI", "display_name": "MARUTI SUZUKI"},
    "NSE_EQ|INE101A01026": {"name": "M&M", "display_name": "M&M"},
    "NSE_EQ|INE917I01010": {"name": "BAJAJ-AUTO", "display_name": "BAJAJ AUTO"},
    "NSE_EQ|INE040A01034": {"name": "HDFCBANK", "display_name": "HDFC BANK"},
    "NSE_EQ|INE090A01021": {"name": "ICICIBANK", "display_name": "ICICI BANK"},
    "NSE_EQ|INE062A01020": {"name": "SBIN", "display_name": "STATE BANK"},
    "NSE_EQ|INE238A01034": {"name": "AXISBANK", "display_name": "AXIS BANK"},
    "NSE_EQ|INE237A01028": {"name": "KOTAKBANK", "display_name": "KOTAK BANK"},
    "NSE_EQ|INE009A01021": {"name": "INFY", "display_name": "INFOSYS"},
    "NSE_EQ|INE854D01024": {"name": "TCS", "display_name": "TCS"},
    "NSE_EQ|INE002A01018": {"name": "RELIANCE", "display_name": "RELIANCE"},
    "NSE_EQ|INE397D01024": {"name": "BHARTIARTL", "display_name": "BHARTI AIRTEL"},
    "NSE_EQ|INE296A01024": {"name": "BAJFINANCE", "display_name": "BAJAJ FINANCE"},
    # ✅ NEW STOCKS ADDED
    "NSE_EQ|INE019A01038": {"name": "ASIANPAINT", "display_name": "ASIAN PAINTS"},
    "NSE_EQ|INE769A01020": {"name": "ATUL", "display_name": "ATUL LTD"},
    "NSE_EQ|INE917I01010": {"name": "AUROPHARMA", "display_name": "AUROBINDO PHARMA"},
    "NSE_EQ|INE917I01010": {"name": "DMART", "display_name": "AVENUE SUPERMARTS"},
    "NSE_EQ|INE917I01010": {"name": "BANDHANBNK", "display_name": "BANDHAN BANK"},
    "NSE_EQ|INE917I01010": {"name": "BANKBARODA", "display_name": "BANK OF BARODA"},
    "NSE_EQ|INE917I01010": {"name": "BERGEPAINT", "display_name": "BERGER PAINTS"},
    "NSE_EQ|INE917I01010": {"name": "BEL", "display_name": "BHARAT ELECTRONICS"},
    "NSE_EQ|INE917I01010": {"name": "BPCL", "display_name": "BHARAT PETROLEUM"},
    "NSE_EQ|INE917I01010": {"name": "BHARATFORG", "display_name": "BHARAT FORGE"},
    "NSE_EQ|INE917I01010": {"name": "BHEL", "display_name": "BHEL"},
    "NSE_EQ|INE917I01010": {"name": "BRITANNIA", "display_name": "BRITANNIA"},
    "NSE_EQ|INE917I01010": {"name": "CANBK", "display_name": "CANARA BANK"},
    "NSE_EQ|INE917I01010": {"name": "CHOLAFIN", "display_name": "CHOLAMANDALAM"},
    "NSE_EQ|INE917I01010": {"name": "CIPLA", "display_name": "CIPLA"},
    "NSE_EQ|INE917I01010": {"name": "COALINDIA", "display_name": "COAL INDIA"},
    "NSE_EQ|INE917I01010": {"name": "COFORGE", "display_name": "COFORGE"},
    "NSE_EQ|INE917I01010": {"name": "COLPAL", "display_name": "COLGATE"},
    "NSE_EQ|INE917I01010": {"name": "DLF", "display_name": "DLF"},
    "NSE_EQ|INE917I01010": {"name": "DABUR", "display_name": "DABUR"},
    "NSE_EQ|INE917I01010": {"name": "DIVISLAB", "display_name": "DIVI'S LAB"},
    "NSE_EQ|INE917I01010": {"name": "LICI", "display_name": "LIC INDIA"},
    "NSE_EQ|INE917I01010": {"name": "DRREDDY", "display_name": "DR REDDY'S"},
    "NSE_EQ|INE917I01010": {"name": "EICHERMOT", "display_name": "EICHER MOTORS"},
    "NSE_EQ|INE917I01010": {"name": "GAIL", "display_name": "GAIL"},
    "NSE_EQ|INE917I01010": {"name": "GRASIM", "display_name": "GRASIM"},
    "NSE_EQ|INE917I01010": {"name": "HCLTECH", "display_name": "HCL TECH"},
    "NSE_EQ|INE917I01010": {"name": "HDFCLIFE", "display_name": "HDFC LIFE"},
    "NSE_EQ|INE917I01010": {"name": "HEROMOTOCO", {"name": "HEROMOTOCO", "display_name": "HERO MOTOCORP"},
    "NSE_EQ|INE917I01010": {"name": "HINDALCO", "display_name": "HINDALCO"},
    "NSE_EQ|INE917I01010": {"name": "HINDPETRO", "display_name": "HINDUSTAN PETRO"},
    "NSE_EQ|INE917I01010": {"name": "HINDUNILVR", "display_name": "HINDUSTAN UNILEVER"},
    "NSE_EQ|INE917I01010": {"name": "IBULHSGFIN", "display_name": "INDIABULLS HOUSING"},
    "NSE_EQ|INE917I01010": {"name": "IDFCFIRSTB", "display_name": "IDFC FIRST BANK"},
    "NSE_EQ|INE917I01010": {"name": "ITC", "display_name": "ITC"},
    "NSE_EQ|INE917I01010": {"name": "INDHOTEL", "display_name": "INDIAN HOTELS"},
    "NSE_EQ|INE917I01010": {"name": "INDUSINDBK", "display_name": "INDUSIND BANK"},
    "NSE_EQ|INE917I01010": {"name": "NAUKRI", "display_name": "INFO EDGE"},
    "NSE_EQ|INE917I01010": {"name": "IRCTC", "display_name": "IRCTC"},
    "NSE_EQ|INE917I01010": {"name": "IGL", "display_name": "IGL"},
    "NSE_EQ|INE917I01010": {"name": "INDUSTOWER", "display_name": "INDUS TOWERS"},
    "NSE_EQ|INE917I01010": {"name": "JSWSTEEL", "display_name": "JSW STEEL"},
    "NSE_EQ|INE917I01010": {"name": "JINDALSTEL", "display_name": "JINDAL STEEL"},
    "NSE_EQ|INE917I01010": {"name": "LT", "display_name": "L&T"},
    "NSE_EQ|INE917I01010": {"name": "LTIM", "display_name": "LTI MINDTREE"},
    "NSE_EQ|INE917I01010": {"name": "LICHSGFIN", "display_name": "LIC HOUSING"},
    "NSE_EQ|INE917I01010": {"name": "LUPIN", "display_name": "LUPIN"},
    "NSE_EQ|INE917I01010": {"name": "MRF", "display_name": "MRF"},
    "NSE_EQ|INE917I01010": {"name": "MPHASIS", "display_name": "MPHASIS"},
    "NSE_EQ|INE917I01010": {"name": "MCX", "display_name": "MCX"},
    "NSE_EQ|INE917I01010": {"name": "NTPC", "display_name": "NTPC"},
    "NSE_EQ|INE917I01010": {"name": "NMDC", "display_name": "NMDC"},
    "NSE_EQ|INE917I01010": {"name": "ONGC", "display_name": "ONGC"},
    "NSE_EQ|INE917I01010": {"name": "OFSS", "display_name": "ORACLE FINANCIAL"},
    "NSE_EQ|INE917I01010": {"name": "PAYTM", "display_name": "PAYTM"},
    "NSE_EQ|INE917I01010": {"name": "PERSISTENT", "display_name": "PERSISTENT"},
    "NSE_EQ|INE917I01010": {"name": "PIDILITIND", "display_name": "PIDILITE"},
    "NSE_EQ|INE917I01010": {"name": "PFC", "display_name": "POWER FINANCE"},
    "NSE_EQ|INE917I01010": {"name": "POWERGRID", "display_name": "POWER GRID"},
    "NSE_EQ|INE917I01010": {"name": "PNB", "display_name": "PUNJAB NATIONAL BANK"},
    "NSE_EQ|INE917I01010": {"name": "RECLTD", "display_name": "REC LIMITED"},
    "NSE_EQ|INE917I01010": {"name": "SBICARD", "display_name": "SBI CARD"},
    "NSE_EQ|INE917I01010": {"name": "SBILIFE", "display_name": "SBI LIFE"},
    "NSE_EQ|INE917I01010": {"name": "SHREECEM", "display_name": "SHREE CEMENT"},
    "NSE_EQ|INE917I01010": {"name": "SIEMENS", "display_name": "SIEMENS"},
    "NSE_EQ|INE917I01010": {"name": "SUNPHARMA", "display_name": "SUN PHARMA"},
    "NSE_EQ|INE917I01010": {"name": "TATACOMM", "display_name": "TATA COMM"},
    "NSE_EQ|INE917I01010": {"name": "TATACONSUM", "display_name": "TATA CONSUMER"},
    "NSE_EQ|INE917I01010": {"name": "TATAMTRDVR", "display_name": "TATA MOTORS DVR"},
    "NSE_EQ|INE917I01010": {"name": "TATAPOWER", "display_name": "TATA POWER"},
    "NSE_EQ|INE917I01010": {"name": "TATASTEEL", "display_name": "TATA STEEL"},
    "NSE_EQ|INE917I01010": {"name": "TECHM", "display_name": "TECH MAHINDRA"},
    "NSE_EQ|INE917I01010": {"name": "TITAN", "display_name": "TITAN"},
    "NSE_EQ|INE917I01010": {"name": "TORNTPHARM", "display_name": "TORRENT PHARMA"},
    "NSE_EQ|INE917I01010": {"name": "TRENT", "display_name": "TRENT"},
    "NSE_EQ|INE917I01010": {"name": "ULTRACEMCO", "display_name": "ULTRATECH CEMENT"},
    "NSE_EQ|INE917I01010": {"name": "MCDOWELL-N", "display_name": "UNITED SPIRITS"},
    "NSE_EQ|INE917I01010": {"name": "UPL", "display_name": "UPL"},
    "NSE_EQ|INE917I01010": {"name": "VBL", "display_name": "VARUN BEVERAGES"},
    "NSE_EQ|INE917I01010": {"name": "VEDL", "display_name": "VEDANTA"},
    "NSE_EQ|INE917I01010": {"name": "VOLTAS", "display_name": "VOLTAS"},
    "NSE_EQ|INE917I01010": {"name": "WIPRO", "display_name": "WIPRO"},
    "NSE_EQ|INE917I01010": {"name": "ZOMATO", "display_name": "ZOMATO"},
    "NSE_EQ|INE917I01010": {"name": "ZYDUSLIFE", "display_name": "ZYDUS LIFESCIENCES"}
}

ALL_SYMBOLS = {**INDICES, **FO_STOCKS}

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

# ==================== EXPIRY CALCULATOR ====================
class ExpiryCalculator:
    @staticmethod
    def get_all_expiries_from_api(instrument_key: str) -> List[str]:
        try:
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
            }
            
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
    def calculate_monthly_expiry(symbol_name: str, expiry_day: int = 3) -> str:
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
        
        return ExpiryCalculator.calculate_monthly_expiry(symbol_info.get('name', ''), expiry_day)
    
    @staticmethod
    def get_display_expiry(expiry_str: str) -> str:
        try:
            dt = datetime.strptime(expiry_str, '%Y-%m-%d')
            return dt.strftime('%d%b%y').upper()
        except:
            return expiry_str

# ==================== DATA FETCHER ====================
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
        except:
            return 0.0
    
    def get_option_chain(self, instrument_key: str, expiry: str) -> List[StrikeData]:
        """Fetch option chain with LTP"""
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
    
    def get_15m_data(self, instrument_key: str, candles_needed: int = 400) -> Optional[pd.DataFrame]:
        """Fetch 15M candles - Last 400 candles"""
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            all_candles = []
            
            # Historical (30 min data, will resample to 15m)
            to_date = (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')
            from_date = (datetime.now(IST) - timedelta(days=30)).strftime('%Y-%m-%d')
            url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/30minute/{to_date}/{from_date}"
            
            response = requests.get(url, headers=self.headers, timeout=20)
            
            if response.status_code == 200:
                candles_30min = response.json().get('data', {}).get('candles', [])
                all_candles.extend(candles_30min)
            
            # Intraday (1 min data, will resample to 15m)
            url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
            response = requests.get(url, headers=self.headers, timeout=20)
            
            if response.status_code == 200:
                candles_1min = response.json().get('data', {}).get('candles', [])
                all_candles.extend(candles_1min)
            
            if not all_candles:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').astype(float)
            df = df.sort_index()
            
            # Resample to 15M
            df_15m = df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'oi': 'last'
            }).dropna()
            
            # Get last 400 candles
            df_15m = df_15m.tail(candles_needed)
            
            return df_15m
            
        except Exception as e:
            logger.error(f"15M data error: {e}")
            return None

# ==================== CHART GENERATOR ====================
class ChartGenerator:
    @staticmethod
    def create_candlestick_chart(symbol: str, df: pd.DataFrame, spot_price: float, path: str):
        """Create clean candlestick chart"""
        
        # Colors
        BG = '#FFFFFF'
        GRID = '#E0E0E0'
        TEXT = '#2C3E50'
        GREEN = '#26a69a'
        RED = '#ef5350'
        YELLOW = '#FFD700'
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       facecolor=BG)
        
        ax1.set_facecolor(BG)
        df_plot = df.reset_index(drop=True)
        
        # Candlesticks
        for idx, row in df_plot.iterrows():
            color = GREEN if row['close'] > row['open'] else RED
            
            # Wick
            ax1.plot([idx+0.3, idx+0.3], [row['low'], row['high']],
                    color=color, linewidth=1.2, alpha=0.8)
            
            # Body
            ax1.add_patch(Rectangle(
                (idx, min(row['open'], row['close'])),
                0.6,
                abs(row['close'] - row['open']) if abs(row['close'] - row['open']) > 0 else spot_price * 0.0001,
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
        ax1.axhline(spot_price, color='#FF9800', linewidth=2, linestyle='--', alpha=0.7)
        ax1.text(2, spot_price, f' Spot: ₹{spot_price:.2f}',
                color='#FF9800', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         alpha=0.9, edgecolor='#FF9800', linewidth=1.5))
        
        # Title
        title = f"{symbol} | 15M Timeframe | {len(df_plot)} Candles"
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
    
    async def send_startup(self):
        msg = f"""🚀 **OPTION CHAIN MONITOR v1.0**

⏰ {datetime.now(IST).strftime('%d-%b %H:%M IST')}

✅ Every 5 minutes scan
✅ 21 ATM Strikes (OI, Volume, LTP)
✅ 15M Chart (Last 400 candles)

📊 Monitoring: {len(ALL_SYMBOLS)} symbols

🟢 **BOT ACTIVE**"""
        
        await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
    
    async def send_option_chain_alert(self, symbol: str, display_name: str,
                                     spot_price: float, strikes: List[StrikeData],
                                     chart_path: str, expiry: str):
        """Send option chain data with chart"""
        try:
            # Send chart first
            with open(chart_path, 'rb') as photo:
                await self.bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo)
            
            # Calculate PCR
            total_ce_oi = sum(s.ce_oi for s in strikes)
            total_pe_oi = sum(s.pe_oi for s in strikes)
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            # Build message
            msg = f"""━━━━━━━━━━━━━━━━━━━━━━
📊 **{display_name}**
━━━━━━━━━━━━━━━━━━━━━━

💹 **Spot:** ₹{spot_price:.2f}
📅 **Expiry:** {expiry}
📈 **PCR:** {pcr:.3f}

━━━━━━━━━━━━━━━━━━━━━━
🔢 **OPTION CHAIN (21 ATM Strikes)**
━━━━━━━━━━━━━━━━━━━━━━

```
Strike | CE-OI    | CE-Vol  | CE-LTP | PE-LTP | PE-Vol  | PE-OI
-------|----------|---------|--------|--------|---------|----------
"""
            
            # Add strike data
            for strike in strikes:
                ce_oi_str = f"{strike.ce_oi:>8,}" if strike.ce_oi > 0 else "       -"
                pe_oi_str = f"{strike.pe_oi:>8,}" if strike.pe_oi > 0 else "       -"
                ce_vol_str = f"{strike.ce_volume:>7,}" if strike.ce_volume > 0 else "      -"
                pe_vol_str = f"{strike.pe_volume:>7,}" if strike.pe_volume > 0 else "      -"
                ce_ltp_str = f"{strike.ce_ltp:>6.2f}" if strike.ce_ltp > 0 else "     -"
                pe_ltp_str = f"{strike.pe_ltp:>6.2f}" if strike.pe_ltp > 0 else "     -"
                
                # Highlight ATM strike
                if abs(strike.strike - spot_price) < 50:
                    strike_str = f"*{strike.strike:>6}*"
                else:
                    strike_str = f"{strike.strike:>6}"
                
                msg += f"{strike_str} | {ce_oi_str} | {ce_vol_str} | {ce_ltp_str} | {pe_ltp_str} | {pe_vol_str} | {pe_oi_str}\n"
            
            msg += f"""```

━━━━━━━━━━━━━━━━━━━━━━
📊 **SUMMARY**
━━━━━━━━━━━━━━━━━━━━━━

**Total CE OI:** {total_ce_oi:,}
**Total PE OI:** {total_pe_oi:,}
**PCR:** {pcr:.3f}

🕐 {datetime.now(IST).strftime('%d-%b %H:%M IST')}
━━━━━━━━━━━━━━━━━━━━━━"""
            
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
            logger.info(f"  ✅ Alert sent: {display_name}")
            
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            traceback.print_exc()

# ==================== MAIN BOT ====================
class OptionMonitorBot:
    def __init__(self):
        logger.info("🔄 Initializing Option Monitor v1.0...")
        
        self.fetcher = UpstoxDataFetcher()
        self.notifier = TelegramNotifier()
        
        logger.info("✅ Bot ready")
    
    def is_market_open(self) -> bool:
        now = datetime.now(IST)
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        return time(9, 15) <= current_time <= time(15, 30)
    
    async def analyze_symbol(self, instrument_key: str, symbol_info: Dict):
        try:
            symbol_name = symbol_info.get('name', '')
            display_name = symbol_info.get('display_name', symbol_name)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 {display_name}")
            logger.info(f"{'='*70}")
            
            # Get expiry
            expiry_api = ExpiryCalculator.get_best_expiry(instrument_key, symbol_info)
            expiry_display = ExpiryCalculator.get_display_expiry(expiry_api)
            logger.info(f"  📅 Expiry: {expiry_display}")
            
            # Get spot price
            spot_price = self.fetcher.get_spot_price(instrument_key)
            if spot_price == 0:
                logger.warning(f"  ❌ No spot price")
                return
            
            logger.info(f"  💹 Spot: ₹{spot_price:.2f}")
            
            # Get 15M chart data (last 400 candles)
            df_15m = self.fetcher.get_15m_data(instrument_key, candles_needed=400)
            if df_15m is None or len(df_15m) == 0:
                logger.warning(f"  ⚠️ No 15M data")
                return
            
            logger.info(f"  📊 15M Candles: {len(df_15m)}")
            
            # Get option chain
            all_strikes = self.fetcher.get_option_chain(instrument_key, expiry_api)
            if not all_strikes:
                logger.warning(f"  ⚠️ No option chain data")
                return
            
            # Get 21 ATM strikes (10 below + ATM + 10 above)
            atm = round(spot_price / 100) * 100
            
            # Define strike range based on symbol type
            if "INDEX" in instrument_key:
                strike_interval = 100
                strikes_count = 10
            else:
                strike_interval = 50 if spot_price < 2000 else 100
                strikes_count = 10
            
            atm_strikes = []
            for i in range(-strikes_count, strikes_count + 1):
                target_strike = atm + (i * strike_interval)
                # Find closest strike
                matching_strike = min(all_strikes, 
                                    key=lambda x: abs(x.strike - target_strike),
                                    default=None)
                if matching_strike and matching_strike not in atm_strikes:
                    atm_strikes.append(matching_strike)
            
            # Sort strikes
            atm_strikes = sorted(atm_strikes, key=lambda x: x.strike)[:21]
            
            if not atm_strikes:
                logger.warning(f"  ⚠️ No ATM strikes found")
                return
            
            logger.info(f"  🎯 ATM Strikes: {len(atm_strikes)}")
            
            # Generate chart
            chart_path = f"/tmp/{symbol_name}_option_monitor.png"
            ChartGenerator.create_candlestick_chart(
                display_name, df_15m, spot_price, chart_path
            )
            logger.info(f"  📊 Chart generated")
            
            # Send alert
            await self.notifier.send_option_chain_alert(
                symbol_name, display_name, spot_price,
                atm_strikes, chart_path, expiry_display
            )
            
            logger.info(f"  ✅ Analysis complete")
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol_info.get('display_name')}: {e}")
            traceback.print_exc()
    
    async def run_scan(self):
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 SCAN - {datetime.now(IST).strftime('%H:%M:%S IST')}")
        logger.info(f"{'='*80}")
        
        for idx, (key, info) in enumerate(ALL_SYMBOLS.items(), 1):
            logger.info(f"\n[{idx}/{len(ALL_SYMBOLS)}]")
            await self.analyze_symbol(key, info)
            
            # Wait between symbols to avoid rate limits
            if idx < len(ALL_SYMBOLS):
                await asyncio.sleep(2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ SCAN COMPLETE")
        logger.info(f"{'='*80}\n")
    
    async def run(self):
        logger.info("="*80)
        logger.info("OPTION CHAIN MONITOR v1.0")
        logger.info("="*80)
        
        if not all([UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
            logger.error("❌ Missing credentials!")
            return
        
        await self.notifier.send_startup()
        
        logger.info("="*80)
        logger.info("🟢 RUNNING - Every 5 minutes")
        logger.info("="*80)
        
        while True:
            try:
                if not self.is_market_open():
                    logger.info("⏸️ Market closed. Waiting...")
                    await asyncio.sleep(300)
                    continue
                
                await self.run_scan()
                
                logger.info(f"⏳ Next scan in 5 minutes...")
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
        bot = OptionMonitorBot()
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("STARTING OPTION CHAIN MONITOR v1.0")
    logger.info("="*80)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n✅ Shutdown complete")
    except Exception as e:
        logger.error(f"\n❌ Critical error: {e}")
        traceback.print_exc()
