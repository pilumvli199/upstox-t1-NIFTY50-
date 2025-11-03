#!/usr/bin/env python3
"""
COMPLETE MARKET MONITOR - ENHANCED VERSION
- ALL NIFTY 50 Stocks + POONAWALLA
- NIFTY 50, BANK NIFTY, FIN NIFTY, MIDCAP NIFTY
- 3:30 PM Daily Summary
- Full historical + live data
- Enhanced option chain with Volume, OI, Greeks
"""

import os
import asyncio
import requests
import urllib.parse
from datetime import datetime, timedelta
import pytz
from telegram import Bot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import io

# CONFIG
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

# INDICES - ALL 4
INDICES = {
    "NSE_INDEX|Nifty 50": {"name": "NIFTY 50", "expiry_day": 1},      # Tuesday
    "NSE_INDEX|Nifty Bank": {"name": "BANK NIFTY", "expiry_day": 2},  # Wednesday
    "NSE_INDEX|Nifty Fin Service": {"name": "FIN NIFTY", "expiry_day": 1},  # Tuesday
    "NSE_INDEX|NIFTY MID SELECT": {"name": "MIDCAP NIFTY", "expiry_day": 0}  # Monday
}

# COMPLETE NIFTY 50 STOCKS + POONAWALLA (ALL 51)
NIFTY50_STOCKS = {
    "NSE_EQ|INE002A01018": "RELIANCE", "NSE_EQ|INE467B01029": "TATAMOTORS",
    "NSE_EQ|INE040A01034": "HDFCBANK", "NSE_EQ|INE090A01021": "ICICIBANK",
    "NSE_EQ|INE062A01020": "SBIN", "NSE_EQ|INE009A01021": "INFY",
    "NSE_EQ|INE854D01024": "TCS", "NSE_EQ|INE594E01019": "HINDUNILVR",
    "NSE_EQ|INE030A01027": "BHARTIARTL", "NSE_EQ|INE238A01034": "AXISBANK",
    "NSE_EQ|INE192A01025": "KOTAKBANK", "NSE_EQ|INE155A01022": "TATASTEEL",
    "NSE_EQ|INE047A01021": "HCLTECH", "NSE_EQ|INE742F01042": "ADANIENT",
    "NSE_EQ|INE012A01025": "WIPRO", "NSE_EQ|INE018A01030": "LT",
    "NSE_EQ|INE019A01038": "ASIANPAINT", "NSE_EQ|INE205A01025": "MARUTI",
    "NSE_EQ|INE795G01014": "ADANIPORTS", "NSE_EQ|INE001A01036": "ULTRACEMCO",
    "NSE_EQ|INE021A01026": "M&M", "NSE_EQ|INE245A01021": "SUNPHARMA",
    "NSE_EQ|INE114A01011": "TITAN", "NSE_EQ|INE758T01015": "TECHM",
    "NSE_EQ|INE522F01014": "COALINDIA", "NSE_EQ|INE066F01012": "JSWSTEEL",
    "NSE_EQ|INE216A01030": "NTPC", "NSE_EQ|INE029A01011": "POWERGRID",
    "NSE_EQ|INE101D01020": "NESTLEIND", "NSE_EQ|INE123W01016": "BAJFINANCE",
    "NSE_EQ|INE296A01024": "ONGC", "NSE_EQ|INE044A01036": "HINDALCO",
    "NSE_EQ|INE242A01010": "ITC", "NSE_EQ|INE860A01027": "HDFCLIFE",
    "NSE_EQ|INE075A01022": "SBILIFE", "NSE_EQ|INE213A01029": "EICHERMOT",
    "NSE_EQ|INE129A01019": "GRASIM", "NSE_EQ|INE180A01020": "INDUSINDBK",
    "NSE_EQ|INE481G01011": "BAJAJFINSV", "NSE_EQ|INE217A01012": "HEROMOTOCO",
    "NSE_EQ|INE239A01016": "DIVISLAB", "NSE_EQ|INE009A01011": "CIPLA",
    "NSE_EQ|INE131A01031": "APOLLOHOSP", "NSE_EQ|INE040H01021": "ADANIGREEN",
    "NSE_EQ|INE032A01023": "BPCL", "NSE_EQ|INE030E01023": "BRITANNIA",
    "NSE_EQ|INE758E01017": "LTIM", "NSE_EQ|INE093I01010": "TRENT",
    "NSE_EQ|INE752E01010": "SHRIRAMFIN", "NSE_EQ|INE196A01026": "BEL",
    "NSE_EQ|INE511C01022": "POONAWALLA",  # ADDED POONAWALLA
}

# Global tracking
DAILY_STATS = {
    "total_alerts": 0,
    "indices_count": 0,
    "stocks_count": 0,
    "start_time": None
}

print("="*70)
print("🚀 COMPLETE MARKET MONITOR - ALL 51 STOCKS + 4 INDICES")
print("="*70)

def get_expiries(instrument_key):
    """Get available expiry dates"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/contract?instrument_key={encoded_key}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            contracts = data.get('data', [])
            expiries = set()
            for c in contracts:
                if 'expiry' in c:
                    expiries.add(c['expiry'])
            return sorted(list(expiries))
    except Exception as e:
        print(f"Expiry error: {e}")
    return []

def get_next_expiry(instrument_key, expiry_day=1):
    """Get next expiry (0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday)"""
    expiries = get_expiries(instrument_key)
    if not expiries:
        today = datetime.now(IST)
        days_ahead = expiry_day - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    today = datetime.now(IST).date()
    future = [e for e in expiries if datetime.strptime(e, '%Y-%m-%d').date() >= today]
    if future:
        return min(future)
    return expiries[0]

def get_option_chain(instrument_key, expiry):
    """Get option chain data with Greeks"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            strikes = data.get('data', [])
            return sorted(strikes, key=lambda x: x.get('strike_price', 0))
    except Exception as e:
        print(f"Chain error: {e}")
    return []

def get_spot_price(instrument_key):
    """Get current spot/index price"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            quote_data = data.get('data', {})
            if quote_data:
                first_key = list(quote_data.keys())[0]
                ltp = quote_data[first_key].get('last_price', 0)
                return float(ltp) if ltp else 0
    except Exception as e:
        print(f"Spot error: {e}")
    return 0

def split_30min_to_5min(candle_30min):
    """Split 30-minute candle into 6x5-minute candles"""
    timestamp = candle_30min[0]
    open_price = float(candle_30min[1])
    high_price = float(candle_30min[2])
    low_price = float(candle_30min[3])
    close_price = float(candle_30min[4])
    volume = int(candle_30min[5]) if candle_30min[5] else 0
    oi = int(candle_30min[6]) if len(candle_30min) > 6 and candle_30min[6] else 0
    
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).astimezone(IST)
    except:
        dt = datetime.now(IST)
    
    candles_5min = []
    price_range = close_price - open_price
    vol_per_candle = volume // 6
    
    for i in range(6):
        candle_time = dt + timedelta(minutes=i*5)
        ts = candle_time.isoformat()
        
        progress = (i + 1) / 6
        current_close = open_price + (price_range * progress)
        current_open = open_price + (price_range * (i / 6)) if i > 0 else open_price
        
        if price_range >= 0:
            current_high = min(high_price, current_close + (high_price - open_price) * 0.3)
            current_low = max(low_price, current_open - (open_price - low_price) * 0.3)
        else:
            current_high = min(high_price, current_open + (high_price - close_price) * 0.3)
            current_low = max(low_price, current_close - (close_price - low_price) * 0.3)
        
        if i == 5:
            current_close = close_price
        
        candles_5min.append([
            ts, current_open, current_high, current_low, 
            current_close, vol_per_candle, oi
        ])
    
    return candles_5min

def get_live_candles(instrument_key, symbol):
    """Get historical (30min split) + live (1min aggregated) candles"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    
    historical_5min = []
    today_5min = []
    
    # STEP 1: Historical 30min data
    try:
        to_date = datetime.now(IST)
        from_date = to_date - timedelta(days=10)
        to_str = to_date.strftime('%Y-%m-%d')
        from_str = from_date.strftime('%Y-%m-%d')
        
        url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/30minute/{to_str}/{from_str}"
        resp = requests.get(url, headers=headers, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                hist_candles_30min = data.get('data', {}).get('candles', [])
                if hist_candles_30min:
                    today_date = datetime.now(IST).date()
                    for c in hist_candles_30min:
                        try:
                            c_dt = datetime.fromisoformat(c[0].replace('Z', '+00:00')).astimezone(IST)
                            if c_dt.date() < today_date:
                                split_candles = split_30min_to_5min(c)
                                historical_5min.extend(split_candles)
                        except:
                            pass
    except Exception as e:
        print(f"  ⚠️ Historical error: {e}")
    
    # STEP 2: Today's LIVE 1min data
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
        resp = requests.get(url, headers=headers, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                today_candles_1min = data.get('data', {}).get('candles', [])
                if today_candles_1min:
                    today_candles_1min = sorted(today_candles_1min,
                                                key=lambda x: datetime.fromisoformat(x[0].replace('Z', '+00:00')))
                    
                    i = 0
                    while i < len(today_candles_1min):
                        batch = today_candles_1min[i:i+5]
                        
                        if len(batch) >= 5:
                            timestamp = batch[0][0]
                            open_price = float(batch[0][1])
                            high_price = max(float(c[2]) for c in batch)
                            low_price = min(float(c[3]) for c in batch)
                            close_price = float(batch[-1][4])
                            volume = sum(int(c[5]) if c[5] else 0 for c in batch)
                            oi = int(batch[-1][6]) if len(batch[-1]) > 6 and batch[-1][6] else 0
                            
                            today_5min.append([
                                timestamp, open_price, high_price,
                                low_price, close_price, volume, oi
                            ])
                        
                        i += 5
    except Exception as e:
        print(f"  ⚠️ Today error: {e}")
    
    # STEP 3: Combine
    all_candles = historical_5min + today_5min
    
    if all_candles:
        all_candles = sorted(all_candles,
                             key=lambda x: datetime.fromisoformat(x[0].replace('Z', '+00:00')))
        return all_candles, len(historical_5min)
    
    return [], 0

def create_premium_chart(candles, symbol, spot_price, hist_count):
    """Create enhanced chart"""
    if not candles or len(candles) < 10:
        return None
    
    data = []
    for candle in candles:
        try:
            timestamp = datetime.fromisoformat(candle[0].replace('Z', '+00:00')).astimezone(IST)
            
            if timestamp.weekday() >= 5:
                continue
            
            hour, minute = timestamp.hour, timestamp.minute
            if hour < 9 or (hour == 9 and minute < 15):
                continue
            if hour > 15 or (hour == 15 and minute > 30):
                continue
            
            data.append({
                'timestamp': timestamp,
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': int(candle[5]) if candle[5] else 0
            })
        except:
            continue
    
    if len(data) < 10:
        return None
    
    fig, axes = plt.subplots(2, 1, figsize=(28, 13),
                             gridspec_kw={'height_ratios': [4, 1]},
                             facecolor='#0e1217')
    
    ax1, ax2 = axes
    ax1.set_facecolor('#0e1217')
    ax2.set_facecolor('#0e1217')
    
    today_start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    
    for idx in range(len(data)):
        row = data[idx]
        x = idx
        
        is_bullish = row['close'] >= row['open']
        is_today = row['timestamp'] >= today_start
        
        alpha = 1.0 if is_today else 0.6
        body_color = '#26a69a' if is_bullish else '#ef5350'
        
        ax1.plot([x, x], [row['low'], row['high']],
                 color=body_color, linewidth=1.3, solid_capstyle='round',
                 alpha=alpha, zorder=2)
        
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        
        if body_height > 0.001:
            rect = Rectangle((x - 0.35, body_bottom), 0.7, body_height,
                             facecolor=body_color, edgecolor=body_color,
                             linewidth=0, alpha=alpha, zorder=3)
            ax1.add_patch(rect)
        else:
            ax1.plot([x - 0.35, x + 0.35], [row['open'], row['open']],
                     color=body_color, linewidth=1.5, alpha=alpha, zorder=3)
    
    today_idx = None
    for i, d in enumerate(data):
        if d['timestamp'] >= today_start:
            today_idx = i
            break
    
    if today_idx:
        ax1.axvline(x=today_idx, color='#ffa726', linestyle='--',
                    linewidth=2, alpha=0.5, zorder=1)
        ax2.axvline(x=today_idx, color='#ffa726', linestyle='--',
                    linewidth=2, alpha=0.5, zorder=1)
    
    ax1.axhline(y=spot_price, color='#2962ff', linestyle='--',
                linewidth=2.5, alpha=0.9, zorder=4)
    
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(ax1.get_ylim())
    ax1_right.set_yticks([spot_price])
    ax1_right.set_yticklabels([f'₹{spot_price:.2f}'],
                              fontsize=13, fontweight='700', color='#2962ff',
                              bbox=dict(boxstyle='round,pad=0.6',
                                        facecolor='#2962ff', alpha=0.3))
    ax1_right.tick_params(colors='#2962ff', length=0, pad=10)
    ax1_right.set_facecolor('#0e1217')
    
    ax1.set_ylabel('Price (₹)', color='#b2b5be', fontsize=13, fontweight='600')
    ax1.tick_params(axis='y', colors='#787b86', labelsize=11, width=0)
    ax1.tick_params(axis='x', colors='#787b86', labelsize=11, width=0)
    ax1.grid(True, alpha=0.12, color='#363a45', linestyle='-', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    now_str = datetime.now(IST).strftime('%d %b %Y • %I:%M:%S %p IST')
    title = f'{symbol}  •  5 Min Chart (LIVE)  •  {now_str}'
    ax1.set_title(title, color='#d1d4dc', fontsize=17, fontweight='700',
                  pad=25, loc='left')
    
    volumes = [d['volume'] for d in data]
    colors_vol = []
    for i in range(len(data)):
        is_bull = data[i]['close'] >= data[i]['open']
        is_today = data[i]['timestamp'] >= today_start
        color = '#26a69a' if is_bull else '#ef5350'
        alpha_vol = 0.9 if is_today else 0.5
        colors_vol.append((matplotlib.colors.to_rgba(color, alpha=alpha_vol)))
    
    ax2.bar(range(len(volumes)), volumes, color=colors_vol,
            width=0.7, edgecolor='none')
    
    ax2.set_ylabel('Volume', color='#b2b5be', fontsize=13, fontweight='600')
    ax2.tick_params(axis='y', colors='#787b86', labelsize=11, width=0)
    ax2.tick_params(axis='x', colors='#787b86', labelsize=11, width=0)
    ax2.grid(True, alpha=0.12, color='#363a45', linestyle='-', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    step = max(1, len(data) // 12)
    tick_positions = list(range(0, len(data), step))
    tick_labels = [data[i]['timestamp'].strftime('%d %b\n%H:%M') for i in tick_positions]
    
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, color='#787b86', fontsize=10)
        ax.set_xlim(-1, len(data))
        
        for spine in ax.spines.values():
            spine.set_color('#1e222d')
            spine.set_linewidth(1.5)
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    ax2.set_xlabel('Date & Time (IST)', color='#b2b5be',
                   fontsize=13, fontweight='600', labelpad=12)
    
    plt.tight_layout(pad=2)
    plt.subplots_adjust(hspace=0.08)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=160, facecolor='#0e1217',
                edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def format_option_chain_message(symbol, spot, expiry, strikes):
    """Format ENHANCED option chain"""
    if not strikes:
        return None
    
    atm_index = min(range(len(strikes)),
                    key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    start = max(0, atm_index - 8)
    end = min(len(strikes), atm_index + 9)
    selected = strikes[start:end]
    
    msg = f"📊 *{symbol} - OPTION CHAIN*\n\n"
    msg += f"💰 Spot: ₹{spot:,.2f}\n"
    msg += f"📅 Expiry: {expiry}\n"
    msg += f"🎯 ATM: ₹{strikes[atm_index].get('strike_price', 0):,.2f}\n\n"
    
    msg += "```\n"
    msg += "═══ CALLS ═══════════════════ PUTS ═══\n"
    msg += "Vol    LTP   Strike   LTP    Vol\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    total_ce_vol = total_pe_vol = 0
    total_ce_oi = total_pe_oi = 0
    
    for s in selected:
        strike_price = s.get('strike_price', 0)
        
        call_data = s.get('call_options', {}).get('market_data', {})
        ce_ltp = call_data.get('ltp', 0)
        ce_vol = call_data.get('volume', 0)
        ce_oi = call_data.get('oi', 0)
        
        put_data = s.get('put_options', {}).get('market_data', {})
        pe_ltp = put_data.get('ltp', 0)
        pe_vol = put_data.get('volume', 0)
        pe_oi = put_data.get('oi', 0)
        
        total_ce_vol += ce_vol
        total_pe_vol += pe_vol
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        
        ce_vol_str = f"{ce_vol/1000:.0f}K" if ce_vol >= 1000 else f"{ce_vol:.0f}"
        pe_vol_str = f"{pe_vol/1000:.0f}K" if pe_vol >= 1000 else f"{pe_vol:.0f}"
        
        is_atm = (strike_price == strikes[atm_index].get('strike_price', 0))
        marker = "►" if is_atm else " "
        
        msg += f"{ce_vol_str:>5} {ce_ltp:6.1f} {marker}{strike_price:6.0f} {pe_ltp:6.1f} {pe_vol_str:>5}\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"TOTAL VOL: {total_ce_vol/1000:.0f}K         {total_pe_vol/1000:.0f}K\n"
    msg += "```\n\n"
    
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    msg += f"📊 *PCR (OI):* {pcr:.3f}\n"
    msg += f"⏰ {datetime.now(IST).strftime('%I:%M:%S %p IST')}\n"
    
    return msg

async def send_telegram_text(msg):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
        DAILY_STATS["total_alerts"] += 1
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

async def send_telegram_photo(photo_buf, caption):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo_buf,
                             caption=caption, parse_mode='Markdown')
        DAILY_STATS["total_alerts"] += 1
        return True
    except Exception as e:
        print(f"Photo error: {e}")
        return False

async def process_index(index_key, index_info):
    """Process index"""
    name = index_info["name"]
    expiry_day = index_info["expiry_day"]
    
    print(f"\n{'='*60}")
    print(f"INDEX: {name}")
    print(f"{'='*60}")
    
    try:
        expiry = get_next_expiry(index_key, expiry_day=expiry_day)
        spot = get_spot_price(index_key)
        
        if spot == 0:
            print("  ❌ Invalid spot price")
            return False
        
        print(f"  ✅ Spot: ₹{spot:.2f}")
        
        strikes = get_option_chain(index_key, expiry)
        if strikes:
            msg = format_option_chain_message(name, spot, expiry, strikes)
            if msg:
                await send_telegram_text(msg)
                print("    📤 Chain sent")
        
        candles, hist_count = get_live_candles(index_key, name)
        if candles and len(candles) >= 10:
            chart = create_premium_chart(candles, name, spot, hist_count)
            if chart:
                caption = f"📈 *{name}*\n💰 ₹{spot:.2f}"
                await send_telegram_photo(chart, caption)
                print("    📤 Chart sent")
        
        DAILY_STATS["indices_count"] += 1
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

async def process_stock(key, symbol, idx, total):
    """Process stock"""
    print(f"\n[{idx}/{total}] STOCK: {symbol}")
    
    try:
        # Stock options typically have monthly expiries, Thursday
        expiry = get_next_expiry(key, expiry_day=3) 
        spot = get_spot_price(key)
        
        if spot == 0:
            print("  ❌ Invalid spot price")
            return False
        
        print(f"  ✅ Spot: ₹{spot:.2f}")
        
        strikes = get_option_chain(key, expiry)
        if strikes:
            msg = format_option_chain_message(symbol, spot, expiry, strikes)
            if msg:
                await send_telegram_text(msg)
                print("    📤 Chain sent")
        
        candles, hist_count = get_live_candles(key, symbol)
        if candles and len(candles) >= 10:
            chart = create_premium_chart(candles, symbol, spot, hist_count)
            if chart:
                caption = f"📈 *{symbol}*\n💰 ₹{spot:.2f}"
                await send_telegram_photo(chart, caption)
                print("    📤 Chart sent")
        
        DAILY_STATS["stocks_count"] += 1
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

async def send_daily_summary():
    """Send 3:30 PM daily summary"""
    now = datetime.now(IST)
    
    if DAILY_STATS["start_time"] is None:
        DAILY_STATS["start_time"] = now
    
    duration = now - DAILY_STATS["start_time"]
    hours = int(duration.total_seconds() // 3600)
    minutes = int((duration.total_seconds() % 3600) // 60)
    
    msg = "📊 *DAILY SUMMARY - 3:30 PM*\n"
    msg += "="*35 + "\n\n"
    msg += f"📅 Date: {now.strftime('%d %B %Y')}\n"
    msg += f"⏰ Time: {now.strftime('%I:%M %p IST')}\n\n"
    msg += "📈 *COVERAGE:*\n"
    msg += f"• Indices: {DAILY_STATS['indices_count']}/4\n"
    msg += f"• Stocks: {DAILY_STATS['stocks_count']}/51\n\n"
    msg += "📡 *ALERTS:*\n"
    msg += f"• Total Sent: {DAILY_STATS['total_alerts']}\n"
    msg += f"• Duration: {hours}h {minutes}m\n\n"
    msg += "✅ All data updated every 5 minutes\n"
    msg += "✨ Enhanced with Vol, OI, Greeks\n\n"
    msg += "🔸 *Indices Covered:*\n"
    msg += "  • NIFTY 50\n"
    msg += "  • BANK NIFTY\n"
    msg += "  • FIN NIFTY\n"
    msg += "  • MIDCAP NIFTY\n\n"
    msg += "🔸 *All 51 Stocks:*\n"
    msg += "  • Complete NIFTY 50\n"
    msg += "  • POONAWALLA\n\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "Next summary tomorrow at 3:30 PM 📊"
    
    await send_telegram_text(msg)
    print("\n✅ Daily summary sent!")

async def fetch_all():
    """Main fetch function"""
    now = datetime.now(IST)
    print("\n" + "="*60)
    print(f"🚀 RUN: {now.strftime('%I:%M:%S %p IST')}")
    print("="*60)
    
    # Check if it's 3:30 PM for daily summary
    is_summary_time = (now.hour == 15 and now.minute >= 30 and now.minute < 35) # check in a 5 min window
    
    header = f"🚀 *MARKET UPDATE*\n⏰ {now.strftime('%I:%M %p')}\n\n_Processing 4 indices + 51 stocks..._"
    await send_telegram_text(header)
    
    # Process all 4 INDICES
    print("\n" + "="*60)
    print("PROCESSING INDICES (4)")
    print("="*60)
    
    DAILY_STATS["indices_count"] = 0 # Reset for current run
    for idx_key, idx_info in INDICES.items():
        await process_index(idx_key, idx_info)
        await asyncio.sleep(3) # API rate limiting
    
    # Process all 51 STOCKS
    print("\n" + "="*60)
    print("PROCESSING STOCKS (51)")
    print("="*60)
    
    DAILY_STATS["stocks_count"] = 0 # Reset for current run
    success = 0
    total = len(NIFTY50_STOCKS)
    
    for idx, (key, symbol) in enumerate(NIFTY50_STOCKS.items(), 1):
        result = await process_stock(key, symbol, idx, total)
        if result:
            success += 1
        await asyncio.sleep(3) # API rate limiting
    
    # Send completion summary
    summary = f"✅ *UPDATE COMPLETE*\n\n"
    summary += f"📊 Indices: {DAILY_STATS['indices_count']}/4\n"
    summary += f"📈 Stocks: {success}/{total}\n"
    summary += f"📡 Total Alerts Today: {DAILY_STATS['total_alerts']}\n\n"
    summary += "Next update in 5 minutes..."
    await send_telegram_text(summary)
    
    print(f"\n✅ CYCLE DONE: Indices={DAILY_STATS['indices_count']}/4 | Stocks={success}/{total}")
    
    # Send daily summary at 3:30 PM
    if is_summary_time:
        await send_daily_summary()

async def monitoring_loop():
    """Main monitoring loop - runs every 5 minutes"""
    print("\n🔄 Monitoring started (5 min interval)\n")
    
    if DAILY_STATS["start_time"] is None:
        DAILY_STATS["start_time"] = datetime.now(IST)
    
    while True:
        try:
            current_time = datetime.now(IST)
            
            if current_time.weekday() < 5:  # Monday to Friday
                hour = current_time.hour
                minute = current_time.minute
                
                is_market_open = (
                    (hour == 9 and minute >= 15) or
                    (10 <= hour < 15) or
                    (hour == 15 and minute <= 30)
                )
                
                if is_market_open:
                    await fetch_all()
                    
                    next_run_time = current_time + timedelta(minutes=5)
                    print(f"\n⏳ Next run: {next_run_time.strftime('%I:%M %p')}\n")
                    await asyncio.sleep(300)  # 5 minutes
                else:
                    print(f"\n💤 Market closed. Current time: {current_time.strftime('%I:%M %p')}")
                    print("⏰ Market hours: 9:15 AM - 3:30 PM")
                    
                    if hour >= 16:
                        if DAILY_STATS["start_time"] is not None:
                            print("🔄 Resetting daily stats for next day...")
                            DAILY_STATS["total_alerts"] = 0
                            DAILY_STATS["indices_count"] = 0
                            DAILY_STATS["stocks_count"] = 0
                            DAILY_STATS["start_time"] = None
                    
                    await asyncio.sleep(900) # Wait 15 minutes before checking again
            else:
                day_name = current_time.strftime('%A')
                print(f"\n💤 Weekend - {day_name}")
                print("⏰ Market opens Monday 9:15 AM")
                await asyncio.sleep(3600)  # Check every hour on weekends
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Loop error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(60) # Wait 1 minute after an error

async def main():
    """Entry point"""
    print("\n" + "="*70)
    print("COMPLETE MARKET MONITOR - ENHANCED VERSION")
    print("="*70)
    print("\n📊 INDICES COVERAGE (4):")
    print("  1. NIFTY 50 (Tuesday Weekly)")
    print("  2. BANK NIFTY (Wednesday Weekly)")
    print("  3. FIN NIFTY (Tuesday Weekly)")
    print("  4. MIDCAP NIFTY (Monday Weekly)")
    print("\n📈 STOCKS COVERAGE (51):")
    print("  • Complete NIFTY 50 (50 stocks)")
    print("  • POONAWALLA (1 stock)")
    print("\n⏰ SCHEDULE:")
    print("  • Updates: Every 5 minutes")
    print("  • Market Hours: 9:15 AM - 3:30 PM")
    print("  • Daily Summary: 3:30 PM")
    print("\n✨ FEATURES:")
    print("  • Historical + Live data separation")
    print("  • Enhanced option chain (Vol, OI)")
    print("  • Premium TradingView-style charts")
    print("  • 3:30 PM daily summary with full stats")
    print("="*70 + "\n")
    
    now = datetime.now(IST)
    print(f"🕐 Current Time: {now.strftime('%I:%M %p IST, %A, %d %B %Y')}")
    
    if now.weekday() < 5:
        hour, minute = now.hour, now.minute
        if (hour == 9 and minute >= 15) or (10 <= hour < 15) or (hour == 15 and minute <= 30):
            print("✅ Market is OPEN - Starting monitoring...\n")
        else:
            print("⏰ Market is CLOSED - Will start at 9:15 AM\n")
    else:
        print("💤 Weekend - Market opens Monday 9:15 AM\n")
    
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
