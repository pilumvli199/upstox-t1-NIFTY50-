#!/usr/bin/env python3
"""
NIFTY 50 INDEX + STOCKS MONITOR
- NIFTY Index: Tuesday expiry (weekly)
- Stocks: Thursday expiry (monthly)
- Option Chain + 15min Charts
- Telegram alerts every 5 minutes
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
import io

# CONFIG
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

# NIFTY INDEX
NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"

# NIFTY 50 STOCKS
NIFTY50_STOCKS = {
    "NSE_EQ|INE002A01018": "RELIANCE",
    "NSE_EQ|INE040A01034": "HDFCBANK",
    "NSE_EQ|INE090A01021": "ICICIBANK",
    "NSE_EQ|INE062A01020": "SBIN",
    "NSE_EQ|INE009A01021": "INFY",
}

print("="*70)
print("🚀 NIFTY 50 MONITOR STARTED")
print("="*70)

def get_expiries(instrument_key):
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

def get_next_expiry(instrument_key, is_nifty_index=False):
    expiries = get_expiries(instrument_key)
    if not expiries:
        today = datetime.now(IST)
        if is_nifty_index:
            days_ahead = 1 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
        else:
            days_ahead = 3 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
        return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    today = datetime.now(IST).date()
    future = [e for e in expiries if datetime.strptime(e, '%Y-%m-%d').date() >= today]
    if future:
        return min(future)
    return expiries[0]

def get_option_chain(instrument_key, expiry):
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

def get_historical_candles(instrument_key, symbol):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    try:
        to_date = datetime.now(IST)
        from_date = to_date - timedelta(days=7)
        to_str = to_date.strftime('%Y-%m-%d')
        from_str = from_date.strftime('%Y-%m-%d')
        url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/30minute/{to_str}/{from_str}"
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                return candles
    except Exception as e:
        print(f"Candle error: {e}")
    return []

def create_candlestick_chart(candles, symbol, spot_price):
    if not candles or len(candles) < 3:
        return None
    dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    for candle in reversed(candles):
        try:
            timestamp = datetime.fromisoformat(candle[0].replace('Z', '+00:00'))
            timestamp = timestamp.astimezone(IST)
            if timestamp.weekday() >= 5:
                continue
            hour, minute = timestamp.hour, timestamp.minute
            if hour < 9 or (hour == 9 and minute < 15):
                continue
            if hour > 15 or (hour == 15 and minute > 30):
                continue
            dates.append(timestamp)
            opens.append(float(candle[1]))
            highs.append(float(candle[2]))
            lows.append(float(candle[3]))
            closes.append(float(candle[4]))
            volumes.append(int(candle[5]) if candle[5] else 0)
        except:
            continue
    if len(dates) < 3:
        return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [4, 1]})
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#fafafa')
    indices = range(len(dates))
    for i in indices:
        is_bullish = closes[i] >= opens[i]
        color = '#089981' if is_bullish else '#f23645'
        ax1.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1.0, zorder=2)
        height = abs(closes[i] - opens[i])
        bottom = min(opens[i], closes[i])
        if height > 0.001:
            rect = Rectangle((i - 0.4, bottom), 0.8, height, facecolor=color, edgecolor=color, zorder=3)
            ax1.add_patch(rect)
    ax1.axhline(y=spot_price, color='#2962FF', linestyle='--', linewidth=1.5, zorder=4)
    ax1.set_title(f'{symbol} - 30min Chart', fontsize=14, fontweight='600', pad=15)
    colors_vol = ['#089981' if closes[i] >= opens[i] else '#f23645' for i in indices]
    ax2.bar(indices, volumes, color=colors_vol, width=0.8, alpha=0.7)
    step = max(1, len(dates) // 8)
    tick_positions = list(range(0, len(dates), step))
    tick_labels = [dates[i].strftime('%d %b\n%H:%M') for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def format_option_chain_message(symbol, spot, expiry, strikes):
    if not strikes:
        return None
    atm_index = min(range(len(strikes)), key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    start = max(0, atm_index - 10)
    end = min(len(strikes), atm_index + 11)
    selected = strikes[start:end]
    msg = f"📊 *{symbol}*\n\n"
    msg += f"💰 Spot: ₹{spot:,.2f}\n"
    msg += f"📅 Expiry: {expiry}\n"
    msg += f"🎯 ATM: ₹{strikes[atm_index].get('strike_price', 0):,.2f}\n\n"
    msg += "```\n"
    msg += "Strike   CE-LTP CE-OI  PE-LTP PE-OI\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    total_ce_oi = total_pe_oi = 0
    for s in selected:
        strike_price = s.get('strike_price', 0)
        call = s.get('call_options', {}).get('market_data', {})
        ce_ltp = call.get('ltp', 0)
        ce_oi = call.get('oi', 0)
        put = s.get('put_options', {}).get('market_data', {})
        pe_ltp = put.get('ltp', 0)
        pe_oi = put.get('oi', 0)
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        msg += f"{strike_price:8.0f} {ce_ltp:6.1f} {ce_oi/1000:5.0f}K {pe_ltp:6.1f} {pe_oi/1000:5.0f}K\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"TOTAL         {total_ce_oi/1000:5.0f}K       {total_pe_oi/1000:5.0f}K\n"
    msg += "```\n"
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    msg += f"📊 PCR: {pcr:.3f}\n"
    msg += f"⏰ {datetime.now(IST).strftime('%I:%M %p')}\n"
    return msg

async def send_telegram_text(msg):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

async def send_telegram_photo(photo_buf, caption):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo_buf, caption=caption, parse_mode='Markdown')
        return True
    except Exception as e:
        print(f"Photo error: {e}")
        return False

async def process_nifty_index():
    print("\n" + "="*50)
    print("NIFTY 50 INDEX")
    print("="*50)
    try:
        expiry = get_next_expiry(NIFTY_INDEX_KEY, is_nifty_index=True)
        spot = get_spot_price(NIFTY_INDEX_KEY)
        if spot == 0:
            print("Invalid spot")
            return False
        print(f"Spot: {spot:.2f} | Expiry: {expiry}")
        strikes = get_option_chain(NIFTY_INDEX_KEY, expiry)
        if not strikes:
            print("No strikes")
            return False
        print(f"Strikes: {len(strikes)}")
        msg = format_option_chain_message("NIFTY 50", spot, expiry, strikes)
        if msg:
            await send_telegram_text(msg)
            print("Chain sent")
        candles = get_historical_candles(NIFTY_INDEX_KEY, "NIFTY")
        if candles:
            chart = create_candlestick_chart(candles, "NIFTY 50", spot)
            if chart:
                await send_telegram_photo(chart, f"NIFTY 50 - ₹{spot:.2f}")
                print("Chart sent")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

async def process_stock(key, symbol, idx, total):
    print(f"\n[{idx}/{total}] {symbol}")
    try:
        expiry = get_next_expiry(key, is_nifty_index=False)
        spot = get_spot_price(key)
        if spot == 0:
            print("  Invalid spot")
            return False
        strikes = get_option_chain(key, expiry)
        if not strikes:
            print("  No strikes")
            return False
        print(f"  Spot: {spot:.2f} | Strikes: {len(strikes)}")
        msg = format_option_chain_message(symbol, spot, expiry, strikes)
        if msg:
            await send_telegram_text(msg)
        candles = get_historical_candles(key, symbol)
        if candles:
            chart = create_candlestick_chart(candles, symbol, spot)
            if chart:
                await send_telegram_photo(chart, f"{symbol} - ₹{spot:.2f}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

async def fetch_all():
    print("\n" + "="*50)
    print(f"RUN: {datetime.now(IST).strftime('%I:%M %p')}")
    print("="*50)
    header = f"🚀 *NIFTY MONITOR*\n⏰ {datetime.now(IST).strftime('%I:%M %p')}\n\n_Starting..._"
    await send_telegram_text(header)
    nifty_ok = await process_nifty_index()
    await asyncio.sleep(2)
    success = 0
    total = len(NIFTY50_STOCKS)
    for idx, (key, symbol) in enumerate(NIFTY50_STOCKS.items(), 1):
        result = await process_stock(key, symbol, idx, total)
        if result:
            success += 1
        await asyncio.sleep(2)
    summary = f"✅ *DONE*\nNIFTY: {'✅' if nifty_ok else '❌'}\nStocks: {success}/{total}"
    await send_telegram_text(summary)
    print(f"\nDONE: NIFTY={'OK' if nifty_ok else 'FAIL'} | Stocks={success}/{total}")

async def monitoring_loop():
    print("\n🔄 Loop started (5 min interval)\n")
    while True:
        try:
            await fetch_all()
            next_time = (datetime.now(IST) + timedelta(minutes=5)).strftime('%I:%M %p')
            print(f"\n⏳ Next: {next_time}\n")
            await asyncio.sleep(300)
        except KeyboardInterrupt:
            print("\n🛑 Stopped")
            break
        except Exception as e:
            print(f"\nLoop error: {e}")
            await asyncio.sleep(60)

async def main():
    print("\n" + "="*50)
    print("NIFTY 50 MONITOR")
    print("="*50)
    print("NIFTY Index + 5 Stocks")
    print("Every 5 minutes")
    print("="*50 + "\n")
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
