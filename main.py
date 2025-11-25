#!/usr/bin/env python3
"""
FUTURES DATA TESTING BOT
========================
Testing Upstox Futures Historical Candles API

Fetches last 10 candles (1-minute) for 4 indices:
- NIFTY 50
- BANK NIFTY
- FIN NIFTY
- MIDCAP NIFTY

Sends JSON data to Telegram every 1 minute
"""

import os
import asyncio
import aiohttp
import urllib.parse
from datetime import datetime, timedelta
import pytz
import json
import logging
from calendar import monthrange

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ Install: pip install python-telegram-bot")
    exit(1)

# ==================== CONFIG ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("FuturesTester")

# API Credentials
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Futures Symbols Config
FUTURES_CONFIG = {
    'NIFTY': {
        'name': 'NIFTY 50',
        'prefix': 'NIFTY',
        'expiry_day': 1  # Tuesday
    },
    'BANKNIFTY': {
        'name': 'BANK NIFTY',
        'prefix': 'BANKNIFTY',
        'expiry_day': 2  # Wednesday
    },
    'FINNIFTY': {
        'name': 'FIN NIFTY',
        'prefix': 'FINNIFTY',
        'expiry_day': 1  # Tuesday
    },
    'MIDCPNIFTY': {
        'name': 'MIDCAP NIFTY',
        'prefix': 'MIDCPNIFTY',
        'expiry_day': 0  # Monday
    }
}

# ==================== FUTURES SYMBOL GENERATOR ====================
def get_futures_symbol(index_name: str) -> str:
    """
    Generate current futures symbol with CORRECT DATE FORMAT
    Format: NSE_FO|NIFTY25NOV25FUT (includes expiry date!)
    
    Based on Upstox actual format:
    - NSE_FO = Futures & Options segment
    - Symbol = {PREFIX}{DD}{MONTH}{YY}FUT
    - DD = Expiry day (25, 26, 27, 28)
    - MONTH = 3-letter month (NOV, DEC)
    - YY = 2-digit year (25)
    """
    now = datetime.now(IST)
    year = now.year
    month = now.month
    
    config = FUTURES_CONFIG[index_name]
    expiry_day_of_week = config['expiry_day']  # 0=Mon, 1=Tue, 2=Wed
    
    # Find last occurrence of expiry day in current month
    last_day = monthrange(year, month)[1]
    last_date = datetime(year, month, last_day, tzinfo=IST)
    days_back = (last_date.weekday() - expiry_day_of_week) % 7
    expiry_date = last_date - timedelta(days=days_back)
    
    # If past expiry, use next month
    if now.date() > expiry_date.date() or (
        now.date() == expiry_date.date() and now.time().hour >= 15
    ):
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        
        # Recalculate for next month
        last_day = monthrange(year, month)[1]
        last_date = datetime(year, month, last_day, tzinfo=IST)
        days_back = (last_date.weekday() - expiry_day_of_week) % 7
        expiry_date = last_date - timedelta(days=days_back)
    
    # Format components
    day = expiry_date.day  # Expiry day (25, 26, 27, etc.)
    month_name = expiry_date.strftime('%b').upper()  # NOV, DEC, etc.
    year_short = expiry_date.year % 100  # 25 for 2025
    prefix = config['prefix']
    
    # 🔥 CORRECT FORMAT: PREFIX + DD + MONTH + YY + FUT
    symbol = f"NSE_FO|{prefix}{day:02d}{month_name}{year_short:02d}FUT"
    
    logger.info(f"✅ {config['name']}: {symbol} (Expiry: {expiry_date.strftime('%d-%b-%Y')})")
    return symbol

# ==================== DATA FETCHER ====================
class FuturesDataFetcher:
    """Fetch historical candles for futures"""
    
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
    
    async def fetch_candles(self, symbol: str, index_name: str) -> dict:
        """
        Fetch last 10 candles (1-minute interval)
        
        Endpoint: /v2/historical-candle/{instrument}/1minute/{to_date}/{from_date}
        
        Returns:
        {
            "index": "NIFTY 50",
            "symbol": "NSE_FO|NIFTY25NOVFUT",
            "candles": [
                {
                    "timestamp": "2025-11-25T14:30:00+05:30",
                    "open": 24500.0,
                    "high": 24520.0,
                    "low": 24495.0,
                    "close": 24510.0,
                    "volume": 125000,
                    "oi": 5000000
                },
                ...
            ],
            "total_volume": 1250000
        }
        """
        async with aiohttp.ClientSession() as session:
            # Date range: last 2 days to ensure we get today's data
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            from_date = (datetime.now(IST) - timedelta(days=2)).strftime('%Y-%m-%d')
            
            # URL encode symbol
            enc_symbol = urllib.parse.quote(symbol)
            
            url = f"https://api.upstox.com/v2/historical-candle/{enc_symbol}/1minute/{to_date}/{from_date}"
            
            logger.info(f"🔍 Fetching: {index_name}")
            logger.info(f"   URL: {url}")
            
            try:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data.get('status') == 'success' and 'data' in data:
                            raw_candles = data['data'].get('candles', [])
                            
                            if not raw_candles:
                                logger.warning(f"⚠️ {index_name}: No candles received")
                                return None
                            
                            # Get last 10 candles (most recent first in response)
                            last_10 = raw_candles[:10]
                            
                            # Parse candles
                            parsed_candles = []
                            total_volume = 0
                            
                            for candle in last_10:
                                # Upstox format: [timestamp, open, high, low, close, volume, oi]
                                parsed = {
                                    "timestamp": candle[0],
                                    "open": float(candle[1]),
                                    "high": float(candle[2]),
                                    "low": float(candle[3]),
                                    "close": float(candle[4]),
                                    "volume": int(candle[5]),
                                    "oi": int(candle[6]) if len(candle) > 6 else 0
                                }
                                parsed_candles.append(parsed)
                                total_volume += parsed['volume']
                            
                            logger.info(f"✅ {index_name}: {len(parsed_candles)} candles | Vol: {total_volume:,}")
                            
                            return {
                                "index": FUTURES_CONFIG[index_name]['name'],
                                "symbol": symbol,
                                "candles": parsed_candles,
                                "total_volume": total_volume,
                                "timestamp": datetime.now(IST).isoformat()
                            }
                        else:
                            logger.error(f"❌ {index_name}: Invalid response - {data}")
                            return None
                    
                    elif resp.status == 429:
                        logger.warning(f"⏳ Rate limited")
                        return None
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ {index_name}: HTTP {resp.status} - {error_text}")
                        return None
            
            except Exception as e:
                logger.error(f"💥 {index_name}: {e}")
                return None
    
    async def fetch_all_indices(self) -> dict:
        """Fetch data for all 4 indices"""
        results = {
            "fetch_time": datetime.now(IST).strftime('%d-%b-%Y %I:%M:%S %p'),
            "indices": {}
        }
        
        for index_name in FUTURES_CONFIG.keys():
            symbol = get_futures_symbol(index_name)
            data = await self.fetch_candles(symbol, index_name)
            
            if data:
                results['indices'][index_name] = data
            else:
                results['indices'][index_name] = {
                    "error": "Failed to fetch data"
                }
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
        return results

# ==================== TELEGRAM SENDER ====================
class TelegramSender:
    """Send JSON data to Telegram"""
    
    def __init__(self):
        if not TELEGRAM_AVAILABLE:
            raise Exception("Telegram library not available")
        
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    async def send_json(self, data: dict):
        """Send data as formatted JSON message"""
        
        # Create summary message
        summary = f"""
🔥 FUTURES DATA TEST

⏰ {data['fetch_time']}

━━━━━━━━━━━━━━━━━━━━━━━━
📊 DATA SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for index_name, index_data in data['indices'].items():
            if 'error' not in index_data:
                config = FUTURES_CONFIG[index_name]
                candle_count = len(index_data.get('candles', []))
                total_vol = index_data.get('total_volume', 0)
                
                # Latest candle
                if index_data.get('candles'):
                    latest = index_data['candles'][0]
                    summary += f"""
📈 {config['name']}
   Symbol: {index_data['symbol']}
   Candles: {candle_count}
   Volume: {total_vol:,}
   Latest: {latest['close']:.2f}
   Time: {latest['timestamp'][-14:-9]}

"""
            else:
                summary += f"""
❌ {FUTURES_CONFIG[index_name]['name']}
   Error: Failed to fetch

"""
        
        summary += """━━━━━━━━━━━━━━━━━━━━━━━━

📎 Full JSON data below ⬇️
"""
        
        try:
            # Send summary
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=summary
            )
            
            # Send full JSON as document
            json_str = json.dumps(data, indent=2)
            json_bytes = json_str.encode('utf-8')
            
            from io import BytesIO
            json_file = BytesIO(json_bytes)
            json_file.name = f"futures_data_{datetime.now(IST).strftime('%H%M%S')}.json"
            
            await self.bot.send_document(
                chat_id=TELEGRAM_CHAT_ID,
                document=json_file,
                caption="📊 Full JSON Data"
            )
            
            logger.info("✅ Sent to Telegram")
        
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")

# ==================== MAIN ====================
async def main():
    """Main testing loop"""
    
    logger.info("=" * 80)
    logger.info("🚀 FUTURES DATA TESTING BOT")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📊 Testing Indices:")
    for name, config in FUTURES_CONFIG.items():
        logger.info(f"   - {config['name']}")
    logger.info("")
    logger.info("⏱️ Interval: Every 60 seconds")
    logger.info("📦 Data: Last 10 candles (1-minute)")
    logger.info("")
    logger.info("=" * 80)
    
    fetcher = FuturesDataFetcher()
    sender = TelegramSender()
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 Iteration #{iteration}")
            logger.info(f"{'='*80}")
            
            # Fetch data
            data = await fetcher.fetch_all_indices()
            
            # Send to Telegram
            await sender.send_json(data)
            
            # Wait 60 seconds
            logger.info("\n⏳ Waiting 60 seconds...\n")
            await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped by user")
            break
        
        except Exception as e:
            logger.error(f"💥 Error: {e}")
            logger.info("   Retrying in 60 seconds...")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Shutdown complete")
