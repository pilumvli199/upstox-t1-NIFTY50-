#!/usr/bin/env python3
"""
FUTURES DATA BOT - INTRADAY API VERSION
========================================
Uses Upstox INTRADAY API for live today's data
Fetches last 10 candles for 4 indices
Sends to Telegram every 60 seconds
"""

import os
import asyncio
import aiohttp
import urllib.parse
from datetime import datetime, timedelta
import pytz
import json
import logging
import gzip
from io import BytesIO

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
logger = logging.getLogger("FuturesBot")

# API Credentials
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Upstox Instruments JSON URL
INSTRUMENTS_JSON_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

# Index names to search
INDEX_NAMES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# ==================== INSTRUMENTS FETCHER ====================
class InstrumentsFetcher:
    """Download and parse Upstox instruments JSON"""
    
    def __init__(self):
        self.instruments = []
        self.futures_map = {}
    
    async def download_instruments(self):
        """Download Upstox instruments JSON file (gzipped)"""
        logger.info("📥 Downloading Upstox instruments...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(INSTRUMENTS_JSON_URL) as resp:
                    if resp.status == 200:
                        compressed = await resp.read()
                        decompressed = gzip.decompress(compressed)
                        self.instruments = json.loads(decompressed)
                        logger.info(f"✅ Loaded {len(self.instruments)} instruments")
                        return True
                    else:
                        logger.error(f"❌ HTTP {resp.status}")
                        return False
            except Exception as e:
                logger.error(f"💥 Download failed: {e}")
                return False
    
    def find_current_month_futures(self):
        """Find current month futures for our indices"""
        logger.info("🔍 Finding current month futures...")
        
        now = datetime.now(IST)
        
        for instrument in self.instruments:
            if instrument.get('segment') != 'NSE_FO':
                continue
            
            if instrument.get('instrument_type') != 'FUT':
                continue
            
            name = instrument.get('name', '')
            
            if name not in INDEX_NAMES:
                continue
            
            expiry_ms = instrument.get('expiry', 0)
            if not expiry_ms:
                continue
            
            expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=IST)
            
            if expiry_dt < now:
                continue
            
            if name not in self.futures_map:
                self.futures_map[name] = {
                    'instrument_key': instrument.get('instrument_key'),
                    'exchange_token': instrument.get('exchange_token'),
                    'trading_symbol': instrument.get('trading_symbol'),
                    'expiry': expiry_dt.strftime('%d-%b-%Y'),
                    'expiry_timestamp': expiry_ms
                }
            else:
                if expiry_ms < self.futures_map[name]['expiry_timestamp']:
                    self.futures_map[name] = {
                        'instrument_key': instrument.get('instrument_key'),
                        'exchange_token': instrument.get('exchange_token'),
                        'trading_symbol': instrument.get('trading_symbol'),
                        'expiry': expiry_dt.strftime('%d-%b-%Y'),
                        'expiry_timestamp': expiry_ms
                    }
        
        logger.info(f"✅ Found {len(self.futures_map)} futures:")
        for name, info in self.futures_map.items():
            logger.info(f"   {name}: {info['instrument_key']}")
            logger.info(f"      Symbol: {info['trading_symbol']}")
            logger.info(f"      Expiry: {info['expiry']}")
        
        return len(self.futures_map) > 0
    
    async def initialize(self):
        """Download and parse instruments"""
        success = await self.download_instruments()
        if not success:
            return False
        
        return self.find_current_month_futures()

# ==================== DATA FETCHER (INTRADAY API) ====================
class FuturesDataFetcher:
    """Fetch historical candles using INTRADAY API for live data"""
    
    def __init__(self, instruments_map):
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.instruments_map = instruments_map
    
    async def fetch_candles(self, index_name: str) -> dict:
        """
        Fetch last 10 candles using INTRADAY API
        
        🔥 KEY CHANGE: Using /v2/historical-candle/intraday/{key}/1minute
        This returns ONLY TODAY'S candles - perfect for live data!
        
        Response: [timestamp, open, high, low, close, volume, oi]
        """
        if index_name not in self.instruments_map:
            logger.error(f"❌ {index_name}: Not found in instruments")
            return None
        
        info = self.instruments_map[index_name]
        instrument_key = info['instrument_key']
        
        # Check market hours
        now = datetime.now(IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_market_hours = market_open <= now <= market_close
        is_weekday = now.weekday() < 5  # Mon-Fri
        
        async with aiohttp.ClientSession() as session:
            # URL encode instrument key
            enc_key = urllib.parse.quote(instrument_key)
            
            # 🔥 INTRADAY API - Returns ONLY today's candles!
            url = f"https://api.upstox.com/v2/historical-candle/intraday/{enc_key}/1minute"
            
            logger.info(f"🔍 {index_name}: {instrument_key}")
            
            # Market status
            if is_market_hours and is_weekday:
                logger.info(f"   📊 Market: OPEN (Live data)")
            else:
                logger.info(f"   ⏸️ Market: CLOSED")
            
            try:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data.get('status') == 'success' and 'data' in data:
                            raw_candles = data['data'].get('candles', [])
                            
                            if not raw_candles:
                                logger.warning(f"⚠️ {index_name}: No candles (market not started?)")
                                return None
                            
                            # Last 10 candles
                            last_10 = raw_candles[:10]
                            
                            # Parse
                            parsed = []
                            total_vol = 0
                            
                            today_str = now.strftime('%Y-%m-%d')
                            
                            for c in last_10:
                                candle = {
                                    "timestamp": c[0],
                                    "open": float(c[1]),
                                    "high": float(c[2]),
                                    "low": float(c[3]),
                                    "close": float(c[4]),
                                    "volume": int(c[5]),
                                    "oi": int(c[6]) if len(c) > 6 else 0
                                }
                                parsed.append(candle)
                                total_vol += candle['volume']
                            
                            # Check data freshness
                            latest_time = datetime.fromisoformat(parsed[0]['timestamp'])
                            data_age_minutes = (now - latest_time).total_seconds() / 60
                            
                            if data_age_minutes < 5:
                                logger.info(f"🟢 {index_name}: {len(parsed)} candles | Vol: {total_vol:,} | LIVE DATA ({data_age_minutes:.1f}m old)")
                            else:
                                logger.info(f"🟡 {index_name}: {len(parsed)} candles | Vol: {total_vol:,} | {data_age_minutes:.0f}m old")
                            
                            return {
                                "index": index_name,
                                "instrument_key": instrument_key,
                                "trading_symbol": info['trading_symbol'],
                                "expiry": info['expiry'],
                                "candles": parsed,
                                "total_volume": total_vol,
                                "data_age_minutes": round(data_age_minutes, 1),
                                "is_live": data_age_minutes < 5,
                                "market_status": "OPEN" if (is_market_hours and is_weekday) else "CLOSED",
                                "timestamp": datetime.now(IST).isoformat()
                            }
                        else:
                            logger.error(f"❌ {index_name}: Invalid response")
                            return None
                    
                    elif resp.status == 401:
                        logger.error(f"🔑 Invalid token!")
                        return None
                    
                    elif resp.status == 429:
                        logger.warning(f"⏳ Rate limit")
                        return None
                    
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ {index_name}: HTTP {resp.status}")
                        logger.error(f"   {error_text[:200]}")
                        return None
            
            except Exception as e:
                logger.error(f"💥 {index_name}: {e}")
                return None
    
    async def fetch_all_indices(self) -> dict:
        """Fetch all indices"""
        results = {
            "fetch_time": datetime.now(IST).strftime('%d-%b-%Y %I:%M:%S %p'),
            "indices": {}
        }
        
        for index_name in INDEX_NAMES:
            data = await self.fetch_candles(index_name)
            
            if data:
                results['indices'][index_name] = data
            else:
                results['indices'][index_name] = {
                    "error": "Failed to fetch"
                }
            
            # Delay
            await asyncio.sleep(0.5)
        
        return results

# ==================== TELEGRAM SENDER ====================
class TelegramSender:
    """Send to Telegram"""
    
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    async def send_data(self, data: dict):
        """Send summary + JSON file"""
        
        # Check if any index has live data
        has_live_data = any(
            idx.get('is_live', False) 
            for idx in data['indices'].values() 
            if 'error' not in idx
        )
        
        market_emoji = "🟢" if has_live_data else "🟡"
        status_text = "LIVE DATA" if has_live_data else "DELAYED DATA"
        
        summary = f"""
{market_emoji} FUTURES DATA - {status_text}

⏰ {data['fetch_time']}

━━━━━━━━━━━━━━━━━━━━━━━━
📊 DATA SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for idx_name, idx_data in data['indices'].items():
            if 'error' not in idx_data:
                candles = idx_data.get('candles', [])
                
                if candles:
                    latest = candles[0]
                    latest_time = datetime.fromisoformat(latest['timestamp'])
                    
                    # Data freshness indicator
                    is_live = idx_data.get('is_live', False)
                    data_age = idx_data.get('data_age_minutes', 0)
                    
                    if is_live:
                        freshness = f"🟢 Live ({data_age:.1f}m)"
                    elif data_age < 60:
                        freshness = f"🟡 {data_age:.0f}m ago"
                    else:
                        freshness = f"🔴 {data_age/60:.1f}h ago"
                    
                    summary += f"""
📈 {idx_name}
   Symbol: {idx_data['trading_symbol']}
   Expiry: {idx_data['expiry']}
   Status: {freshness}
   Market: {idx_data.get('market_status', 'UNKNOWN')}
   Candles: {len(candles)}
   Volume: {idx_data['total_volume']:,}
   Latest: ₹{latest['close']:.2f}
   Time: {latest_time.strftime('%I:%M %p')}

"""
            else:
                summary += f"""
❌ {idx_name}
   Status: Failed to fetch

"""
        
        summary += "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        if not has_live_data:
            summary += "ℹ️ Market hours: Mon-Fri, 9:15 AM - 3:30 PM IST\n\n"
        
        summary += "📎 JSON attached"
        
        try:
            # Send summary
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=summary
            )
            
            # Send JSON
            json_str = json.dumps(data, indent=2)
            json_file = BytesIO(json_str.encode('utf-8'))
            json_file.name = f"futures_{datetime.now(IST).strftime('%H%M%S')}.json"
            
            await self.bot.send_document(
                chat_id=TELEGRAM_CHAT_ID,
                document=json_file,
                caption="📊 Full Data"
            )
            
            logger.info("✅ Sent to Telegram")
        
        except Exception as e:
            logger.error(f"❌ Telegram: {e}")

# ==================== MAIN ====================
async def main():
    """Main loop"""
    
    logger.info("=" * 80)
    logger.info("🚀 FUTURES DATA BOT - INTRADAY API VERSION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("🔥 Using INTRADAY API for live today's data!")
    logger.info("")
    
    # Initialize instruments
    logger.info("📥 Loading instruments from Upstox...")
    instruments_fetcher = InstrumentsFetcher()
    
    success = await instruments_fetcher.initialize()
    if not success:
        logger.error("❌ Failed to load instruments!")
        return
    
    if len(instruments_fetcher.futures_map) == 0:
        logger.error("❌ No futures found!")
        return
    
    logger.info("")
    logger.info("✅ Instruments loaded successfully")
    logger.info("")
    logger.info("⏱️ Interval: 60 seconds")
    logger.info("📦 Data: Last 10 candles (1-min)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    # Create fetcher and sender
    fetcher = FuturesDataFetcher(instruments_fetcher.futures_map)
    sender = TelegramSender()
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Iteration #{iteration}")
            logger.info(f"{'='*60}\n")
            
            # Fetch data
            data = await fetcher.fetch_all_indices()
            
            # Send to Telegram
            await sender.send_data(data)
            
            # Wait
            logger.info("\n⏳ Waiting 60 seconds...\n")
            await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped")
            break
        
        except Exception as e:
            logger.error(f"💥 Error: {e}")
            logger.info("   Retrying in 60s...")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Bye")
