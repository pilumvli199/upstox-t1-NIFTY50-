#!/usr/bin/env python3
"""
NIFTY50 HYBRID DATA BOT - FIXED VERSION
========================================
✅ Monthly Futures: 500 candles from intraday API
✅ Weekly Options: Option chain API (5 strikes)
✅ Uses proper Upstox API endpoints
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
logger = logging.getLogger("NiftyBot")

# API Credentials
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Upstox Instruments JSON URL
INSTRUMENTS_JSON_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

# NIFTY50 Index key for option chain
NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"

# ==================== EXPIRY CALCULATOR ====================
def get_next_tuesday_expiry(from_date=None):
    """Calculate next Tuesday expiry for weekly options"""
    if from_date is None:
        from_date = datetime.now(IST)
    
    days_until_tuesday = (1 - from_date.weekday()) % 7
    
    if days_until_tuesday == 0:
        market_close = from_date.replace(hour=15, minute=30, second=0, microsecond=0)
        if from_date > market_close:
            next_tuesday = from_date + timedelta(days=7)
        else:
            next_tuesday = from_date
    else:
        next_tuesday = from_date + timedelta(days=days_until_tuesday)
    
    return next_tuesday

def get_monthly_expiry(from_date=None):
    """
    Calculate monthly expiry (last TUESDAY of month)
    Changed since September 2024
    """
    if from_date is None:
        from_date = datetime.now(IST)
    
    # Get next month first day
    if from_date.month == 12:
        next_month = from_date.replace(year=from_date.year + 1, month=1, day=1)
    else:
        next_month = from_date.replace(month=from_date.month + 1, day=1)
    
    # Last day of current month
    last_day = next_month - timedelta(days=1)
    
    # Find last Tuesday
    while last_day.weekday() != 1:  # Tuesday = 1
        last_day -= timedelta(days=1)
    
    # If already passed, get next month
    if last_day.date() < from_date.date():
        if next_month.month == 12:
            next_next_month = next_month.replace(year=next_month.year + 1, month=1, day=1)
        else:
            next_next_month = next_month.replace(month=next_month.month + 1, day=1)
        
        last_day = next_next_month - timedelta(days=1)
        while last_day.weekday() != 1:
            last_day -= timedelta(days=1)
    
    return last_day

# ==================== INSTRUMENTS FETCHER ====================
class InstrumentsFetcher:
    """Find NIFTY50 monthly futures instrument"""
    
    def __init__(self):
        self.instruments = []
        self.monthly_future = None
    
    async def download_instruments(self):
        """Download Upstox instruments JSON"""
        logger.info("📥 Downloading instruments...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(INSTRUMENTS_JSON_URL, timeout=30) as resp:
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
    
    def find_monthly_future(self):
        """Find NIFTY monthly futures"""
        logger.info("🔍 Finding NIFTY monthly futures...")
        
        target_expiry = get_monthly_expiry()
        target_date = target_expiry.date()
        
        logger.info(f"🎯 Target Monthly Expiry: {target_date.strftime('%d-%b-%Y')} (Last Tuesday)")
        
        for instrument in self.instruments:
            if instrument.get('segment') != 'NSE_FO':
                continue
            if instrument.get('instrument_type') != 'FUT':
                continue
            if instrument.get('name') != 'NIFTY':
                continue
            
            expiry_ms = instrument.get('expiry', 0)
            if not expiry_ms:
                continue
            
            expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=IST)
            expiry_date = expiry_dt.date()
            
            if expiry_date == target_date:
                self.monthly_future = {
                    'instrument_key': instrument.get('instrument_key'),
                    'trading_symbol': instrument.get('trading_symbol'),
                    'expiry': expiry_dt.strftime('%d-%b-%Y'),
                    'expiry_date': target_date.strftime('%Y-%m-%d')
                }
                logger.info(f"✅ Found: {self.monthly_future['trading_symbol']}")
                logger.info(f"   Key: {self.monthly_future['instrument_key']}")
                return True
        
        logger.error(f"❌ No monthly future found")
        return False
    
    async def initialize(self):
        """Initialize"""
        success = await self.download_instruments()
        if not success:
            return False
        
        return self.find_monthly_future()

# ==================== DATA FETCHER ====================
class DataFetcher:
    """Fetch futures candles + option chain"""
    
    def __init__(self, monthly_future):
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.monthly_future = monthly_future
    
    async def fetch_futures_candles(self) -> dict:
        """Fetch intraday candles for monthly futures"""
        instrument_key = self.monthly_future['instrument_key']
        
        async with aiohttp.ClientSession() as session:
            enc_key = urllib.parse.quote(instrument_key)
            url = f"https://api.upstox.com/v2/historical-candle/intraday/{enc_key}/1minute"
            
            logger.info(f"📊 Fetching futures candles...")
            logger.info(f"   URL: {url}")
            
            try:
                async with session.get(url, headers=self.headers, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data.get('status') == 'success':
                            raw_candles = data.get('data', {}).get('candles', [])
                            
                            if not raw_candles:
                                logger.warning("⚠️ No candles available")
                                return None
                            
                            # Take up to 500 candles
                            candles_to_take = min(500, len(raw_candles))
                            selected = raw_candles[:candles_to_take]
                            
                            parsed = []
                            total_vol = 0
                            
                            for c in selected:
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
                            
                            logger.info(f"✅ Fetched {len(parsed)} candles | Vol: {total_vol:,}")
                            
                            return {
                                "candles": parsed,
                                "total_volume": total_vol,
                                "candle_count": len(parsed)
                            }
                        else:
                            logger.error(f"❌ API error: {data}")
                            return None
                    
                    elif resp.status == 401:
                        logger.error("❌ Invalid access token!")
                        return None
                    
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ HTTP {resp.status}")
                        logger.error(f"   Response: {error_text[:500]}")
                        return None
            
            except asyncio.TimeoutError:
                logger.error("❌ Request timeout")
                return None
            except Exception as e:
                logger.error(f"💥 Error: {e}")
                return None
    
    async def fetch_spot_price(self) -> float:
        """Get NIFTY50 spot price"""
        async with aiohttp.ClientSession() as session:
            enc_key = urllib.parse.quote(NIFTY_INDEX_KEY)
            url = f"https://api.upstox.com/v2/market-quote/quotes?symbol={enc_key}"
            
            try:
                async with session.get(url, headers=self.headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('status') == 'success':
                            quote = data['data'][NIFTY_INDEX_KEY]
                            spot = quote['last_price']
                            logger.info(f"✅ NIFTY Spot: ₹{spot:.2f}")
                            return spot
                    
                    logger.warning("⚠️ Using fallback spot: 24000")
                    return 24000.0
            
            except Exception as e:
                logger.warning(f"⚠️ Spot fetch error: {e}, using 24000")
                return 24000.0
    
    async def fetch_option_chain(self) -> dict:
        """
        Fetch option chain using Upstox Option Chain API
        This returns complete option chain data for given expiry
        """
        logger.info("📈 Fetching option chain...")
        
        # Get spot price first
        spot_price = await self.fetch_spot_price()
        
        # Calculate ATM strike (nearest 50 multiple)
        atm_strike = round(spot_price / 50) * 50
        
        # Get next Tuesday expiry
        next_tuesday = get_next_tuesday_expiry()
        expiry_date = next_tuesday.strftime('%Y-%m-%d')
        
        logger.info(f"🎯 Target Weekly Expiry: {expiry_date} (Tuesday)")
        logger.info(f"🎯 ATM Strike: {atm_strike} (Spot: ₹{spot_price:.2f})")
        
        # Option chain API
        async with aiohttp.ClientSession() as session:
            enc_key = urllib.parse.quote(NIFTY_INDEX_KEY)
            url = f"https://api.upstox.com/v2/option/chain?instrument_key={enc_key}&expiry_date={expiry_date}"
            
            logger.info(f"   URL: {url}")
            
            try:
                async with session.get(url, headers=self.headers, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data.get('status') == 'success':
                            chain_data = data.get('data', [])
                            
                            if not chain_data:
                                logger.error("❌ No option chain data")
                                return None
                            
                            logger.info(f"✅ Got option chain with {len(chain_data)} strikes")
                            
                            # Filter 5 strikes around ATM
                            target_strikes = [
                                atm_strike - 100,
                                atm_strike - 50,
                                atm_strike,
                                atm_strike + 50,
                                atm_strike + 100
                            ]
                            
                            result = {
                                'atm_strike': atm_strike,
                                'spot_price': spot_price,
                                'expiry': expiry_date,
                                'strikes': {}
                            }
                            
                            for item in chain_data:
                                strike = item.get('strike_price')
                                
                                if strike not in target_strikes:
                                    continue
                                
                                result['strikes'][strike] = {
                                    'CE': self._parse_option_data(item.get('call_options', {})),
                                    'PE': self._parse_option_data(item.get('put_options', {}))
                                }
                            
                            found_strikes = len(result['strikes'])
                            logger.info(f"✅ Found data for {found_strikes} strikes")
                            
                            for strike in sorted(result['strikes'].keys()):
                                atm_mark = " ← ATM" if strike == atm_strike else ""
                                logger.info(f"   Strike {strike}{atm_mark}")
                            
                            return result
                        else:
                            logger.error(f"❌ API error: {data}")
                            return None
                    
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ HTTP {resp.status}")
                        logger.error(f"   Response: {error_text[:500]}")
                        return None
            
            except Exception as e:
                logger.error(f"💥 Error: {e}")
                return None
    
    def _parse_option_data(self, option_data: dict) -> dict:
        """Parse option data from chain"""
        if not option_data:
            return {}
        
        market_data = option_data.get('market_data', {})
        
        return {
            'instrument_key': option_data.get('instrument_key', ''),
            'ltp': market_data.get('ltp', 0),
            'volume': market_data.get('volume', 0),
            'oi': market_data.get('oi', 0),
            'bid': market_data.get('bid_price', 0),
            'ask': market_data.get('ask_price', 0),
            'change': market_data.get('net_change', 0),
            'prev_oi': market_data.get('prev_oi', 0),
            'oi_change': market_data.get('oi', 0) - market_data.get('prev_oi', 0)
        }
    
    async def fetch_all_data(self) -> dict:
        """Fetch both futures and options"""
        logger.info("\n" + "="*60)
        logger.info("📊 FETCHING ALL DATA")
        logger.info("="*60)
        
        # Fetch futures
        futures_data = await self.fetch_futures_candles()
        
        # Small delay
        await asyncio.sleep(1)
        
        # Fetch options
        options_data = await self.fetch_option_chain()
        
        return {
            "timestamp": datetime.now(IST).isoformat(),
            "fetch_time": datetime.now(IST).strftime('%d-%b-%Y %I:%M:%S %p'),
            "monthly_future": {
                "symbol": self.monthly_future['trading_symbol'],
                "expiry": self.monthly_future['expiry'],
                "data": futures_data
            },
            "weekly_options": {
                "data": options_data
            }
        }

# ==================== TELEGRAM SENDER ====================
class TelegramSender:
    """Send data to Telegram"""
    
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    async def send_data(self, data: dict):
        """Send summary + JSON"""
        
        futures = data['monthly_future']['data']
        options = data['weekly_options']['data']
        
        if not futures or not options:
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text="❌ Failed to fetch data"
            )
            return
        
        latest_candle = futures['candles'][0] if futures['candles'] else None
        
        summary = f"""
🚀 NIFTY50 DATA

⏰ {data['fetch_time']}

━━━━━━━━━━━━━━━━━━━━━━━━
📊 MONTHLY FUTURES
━━━━━━━━━━━━━━━━━━━━━━━━

📈 {data['monthly_future']['symbol']}
📅 Expiry: {data['monthly_future']['expiry']}
📦 Candles: {futures['candle_count']}
📊 Volume: {futures['total_volume']:,}
"""
        
        if latest_candle:
            latest_time = datetime.fromisoformat(latest_candle['timestamp'])
            summary += f"""
💰 Latest: ₹{latest_candle['close']:.2f}
📈 High: ₹{latest_candle['high']:.2f}
📉 Low: ₹{latest_candle['low']:.2f}
🕐 {latest_time.strftime('%I:%M %p')}
📊 OI: {latest_candle['oi']:,}
"""
        
        summary += f"""
━━━━━━━━━━━━━━━━━━━━━━━━
📈 WEEKLY OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━

📅 Expiry: {options['expiry']}
🎯 Spot: ₹{options['spot_price']:.2f}
🎯 ATM: {options['atm_strike']}

"""
        
        for strike in sorted(options['strikes'].keys()):
            strike_data = options['strikes'][strike]
            ce = strike_data.get('CE', {})
            pe = strike_data.get('PE', {})
            
            atm_mark = "← ATM" if strike == options['atm_strike'] else ""
            
            summary += f"""Strike {strike} {atm_mark}
  CE: ₹{ce.get('ltp', 0):.2f} | Vol: {ce.get('volume', 0):,} | OI: {ce.get('oi', 0):,}
  PE: ₹{pe.get('ltp', 0):.2f} | Vol: {pe.get('volume', 0):,} | OI: {pe.get('oi', 0):,}

"""
        
        summary += "━━━━━━━━━━━━━━━━━━━━━━━━\n📎 JSON attached"
        
        try:
            # Send summary
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=summary
            )
            
            # Send JSON
            json_str = json.dumps(data, indent=2)
            json_file = BytesIO(json_str.encode('utf-8'))
            json_file.name = f"nifty_{datetime.now(IST).strftime('%H%M%S')}.json"
            
            await self.bot.send_document(
                chat_id=TELEGRAM_CHAT_ID,
                document=json_file,
                caption="📊 Complete Data"
            )
            
            logger.info("✅ Sent to Telegram")
        
        except Exception as e:
            logger.error(f"❌ Telegram: {e}")

# ==================== MAIN ====================
async def main():
    """Main loop"""
    
    logger.info("=" * 80)
    logger.info("🚀 NIFTY50 HYBRID BOT")
    logger.info("=" * 80)
    logger.info("")
    logger.info("⚠️ NEW EXPIRY (Sept 2024+):")
    logger.info("   📊 Monthly: Last Tuesday")
    logger.info("   📈 Weekly: Every Tuesday")
    logger.info("")
    
    # Initialize
    logger.info("📥 Initializing...")
    fetcher_init = InstrumentsFetcher()
    
    success = await fetcher_init.initialize()
    if not success:
        logger.error("❌ Init failed!")
        return
    
    logger.info("")
    logger.info("✅ Ready!")
    logger.info("⏱️ Interval: 60 seconds")
    logger.info("")
    logger.info("=" * 80)
    
    # Create fetcher and sender
    data_fetcher = DataFetcher(fetcher_init.monthly_future)
    sender = TelegramSender()
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Iteration #{iteration}")
            logger.info(f"{'='*60}\n")
            
            # Fetch
            data = await data_fetcher.fetch_all_data()
            
            # Send
            await sender.send_data(data)
            
            # Wait
            logger.info("\n⏳ Waiting 60s...\n")
            await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped")
            break
        
        except Exception as e:
            logger.error(f"💥 Error: {e}")
            import traceback
            traceback.print_exc()
            logger.info("   Retrying in 60s...")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Bye")
