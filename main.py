#!/usr/bin/env python3
"""
NIFTY50 HYBRID DATA BOT
=======================
✅ Monthly Futures: 500 candles + Volume data
✅ Weekly Options: Option chain (5 strikes ATM ±2)
✅ Auto-detects nearest Tuesday expiry for options
✅ Sends to Telegram every 60 seconds
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
logger = logging.getLogger("NiftyHybridBot")

# API Credentials
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Upstox Instruments JSON URL
INSTRUMENTS_JSON_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

# ==================== EXPIRY CALCULATOR ====================
def get_next_tuesday_expiry(from_date=None):
    """Calculate next Tuesday expiry for NIFTY50 weekly options"""
    if from_date is None:
        from_date = datetime.now(IST)
    
    # Find next Tuesday
    days_until_tuesday = (1 - from_date.weekday()) % 7
    
    if days_until_tuesday == 0:
        market_close = from_date.replace(hour=15, minute=30, second=0, microsecond=0)
        if from_date > market_close:
            next_tuesday = from_date + timedelta(days=7)
        else:
            next_tuesday = from_date
    else:
        next_tuesday = from_date + timedelta(days=days_until_tuesday)
    
    expiry_datetime = next_tuesday.replace(hour=15, minute=30, second=0, microsecond=0)
    return expiry_datetime

def get_monthly_expiry(from_date=None):
    """
    Calculate monthly expiry (last TUESDAY of month)
    ⚠️ CHANGED SINCE SEPTEMBER 2024
    NIFTY monthly futures now expire on LAST TUESDAY (not Thursday)
    """
    if from_date is None:
        from_date = datetime.now(IST)
    
    # Go to next month
    if from_date.month == 12:
        next_month = from_date.replace(year=from_date.year + 1, month=1, day=1)
    else:
        next_month = from_date.replace(month=from_date.month + 1, day=1)
    
    # Last day of current month
    last_day = next_month - timedelta(days=1)
    
    # Find last TUESDAY (changed from Thursday)
    while last_day.weekday() != 1:  # Tuesday = 1
        last_day -= timedelta(days=1)
    
    # If already passed, get next month
    if last_day.date() < from_date.date():
        if next_month.month == 12:
            next_next_month = next_month.replace(year=next_month.year + 1, month=1, day=1)
        else:
            next_next_month = next_month.replace(month=next_month.month + 1, day=1)
        
        last_day = next_next_month - timedelta(days=1)
        while last_day.weekday() != 1:  # Tuesday = 1
            last_day -= timedelta(days=1)
    
    return last_day.replace(hour=15, minute=30, second=0, microsecond=0)

# ==================== INSTRUMENTS FETCHER ====================
class InstrumentsFetcher:
    """Download and find NIFTY50 monthly futures + weekly options"""
    
    def __init__(self):
        self.instruments = []
        self.monthly_future = None
        self.weekly_options = {
            'atm_strike': None,
            'strikes': {}  # Will store CE and PE for 5 strikes
        }
        self.spot_price = None
    
    async def download_instruments(self):
        """Download Upstox instruments JSON"""
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
    
    async def get_spot_price(self):
        """Get NIFTY50 spot price to calculate ATM strike"""
        logger.info("🔍 Fetching NIFTY50 spot price...")
        
        headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        
        # NIFTY50 index instrument key
        nifty_index_key = "NSE_INDEX|Nifty 50"
        enc_key = urllib.parse.quote(nifty_index_key)
        
        url = f"https://api.upstox.com/v2/market-quote/quotes?symbol={enc_key}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('status') == 'success':
                            quote = data['data'][nifty_index_key]
                            self.spot_price = quote['last_price']
                            logger.info(f"✅ NIFTY50 Spot: ₹{self.spot_price:.2f}")
                            return True
                    else:
                        logger.error(f"❌ Failed to fetch spot price: {resp.status}")
                        # Fallback: use 24000 as default
                        self.spot_price = 24000.0
                        logger.warning(f"⚠️ Using fallback spot price: ₹{self.spot_price}")
                        return True
            except Exception as e:
                logger.error(f"💥 Error fetching spot: {e}")
                self.spot_price = 24000.0
                return True
    
    def find_monthly_future(self):
        """Find NIFTY50 monthly futures (last Thursday expiry)"""
        logger.info("🔍 Finding NIFTY50 monthly futures...")
        
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
                    'expiry_timestamp': expiry_ms
                }
                logger.info(f"✅ Monthly Future: {self.monthly_future['trading_symbol']}")
                logger.info(f"   Expiry: {self.monthly_future['expiry']}")
                return True
        
        logger.error(f"❌ No monthly future found for {target_date}")
        return False
    
    def find_weekly_options(self):
        """Find NIFTY50 weekly options (5 strikes around ATM)"""
        logger.info("🔍 Finding NIFTY50 weekly options...")
        
        target_expiry = get_next_tuesday_expiry()
        target_date = target_expiry.date()
        
        logger.info(f"🎯 Target Weekly Expiry: {target_date.strftime('%d-%b-%Y')} (Tuesday)")
        
        # Calculate ATM strike (nearest 50 multiple)
        atm_strike = round(self.spot_price / 50) * 50
        self.weekly_options['atm_strike'] = atm_strike
        
        # 5 strikes: ATM-100, ATM-50, ATM, ATM+50, ATM+100
        target_strikes = [
            atm_strike - 100,
            atm_strike - 50,
            atm_strike,
            atm_strike + 50,
            atm_strike + 100
        ]
        
        logger.info(f"🎯 ATM Strike: {atm_strike} (Spot: ₹{self.spot_price:.2f})")
        logger.info(f"🎯 Target Strikes: {target_strikes}")
        
        # Search for options
        for instrument in self.instruments:
            if instrument.get('segment') != 'NSE_FO':
                continue
            if instrument.get('instrument_type') not in ['CE', 'PE']:
                continue
            if instrument.get('name') != 'NIFTY':
                continue
            
            expiry_ms = instrument.get('expiry', 0)
            if not expiry_ms:
                continue
            
            expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=IST)
            expiry_date = expiry_dt.date()
            
            if expiry_date != target_date:
                continue
            
            strike = instrument.get('strike')
            if strike not in target_strikes:
                continue
            
            option_type = instrument.get('instrument_type')  # CE or PE
            
            if strike not in self.weekly_options['strikes']:
                self.weekly_options['strikes'][strike] = {}
            
            self.weekly_options['strikes'][strike][option_type] = {
                'instrument_key': instrument.get('instrument_key'),
                'trading_symbol': instrument.get('trading_symbol'),
                'strike': strike,
                'option_type': option_type,
                'expiry': expiry_dt.strftime('%d-%b-%Y')
            }
        
        found_count = len(self.weekly_options['strikes'])
        logger.info(f"✅ Found options for {found_count} strikes:")
        
        for strike in sorted(self.weekly_options['strikes'].keys()):
            ce = self.weekly_options['strikes'][strike].get('CE', {}).get('trading_symbol', 'N/A')
            pe = self.weekly_options['strikes'][strike].get('PE', {}).get('trading_symbol', 'N/A')
            atm_marker = " ← ATM" if strike == atm_strike else ""
            logger.info(f"   {strike}: CE={ce}, PE={pe}{atm_marker}")
        
        return found_count == 5
    
    async def initialize(self):
        """Download and parse instruments"""
        success = await self.download_instruments()
        if not success:
            return False
        
        await self.get_spot_price()
        
        if not self.find_monthly_future():
            return False
        
        if not self.find_weekly_options():
            logger.warning("⚠️ Could not find all 5 option strikes")
        
        return True

# ==================== DATA FETCHER ====================
class DataFetcher:
    """Fetch futures candles + option chain data"""
    
    def __init__(self, instruments_info):
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.monthly_future = instruments_info.monthly_future
        self.weekly_options = instruments_info.weekly_options
        self.spot_price = instruments_info.spot_price
    
    async def fetch_futures_candles(self) -> dict:
        """Fetch 500 candles from monthly futures (INTRADAY API)"""
        instrument_key = self.monthly_future['instrument_key']
        
        async with aiohttp.ClientSession() as session:
            enc_key = urllib.parse.quote(instrument_key)
            url = f"https://api.upstox.com/v2/historical-candle/intraday/{enc_key}/1minute"
            
            logger.info(f"📊 Fetching futures candles: {instrument_key}")
            
            try:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data.get('status') == 'success' and 'data' in data:
                            raw_candles = data['data'].get('candles', [])
                            
                            if not raw_candles:
                                logger.warning("⚠️ No candles available")
                                return None
                            
                            # Take last 500 (or all if less)
                            last_500 = raw_candles[:500]
                            
                            parsed = []
                            total_vol = 0
                            
                            for c in last_500:
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
                            
                            logger.info(f"✅ Fetched {len(parsed)} candles | Total Vol: {total_vol:,}")
                            
                            return {
                                "candles": parsed,
                                "total_volume": total_vol,
                                "candle_count": len(parsed)
                            }
                    
                    logger.error(f"❌ HTTP {resp.status}")
                    return None
            
            except Exception as e:
                logger.error(f"💥 Error fetching candles: {e}")
                return None
    
    async def fetch_option_chain(self) -> dict:
        """Fetch option chain data for 5 strikes"""
        logger.info("📈 Fetching option chain data...")
        
        option_data = {
            'atm_strike': self.weekly_options['atm_strike'],
            'spot_price': self.spot_price,
            'strikes': {}
        }
        
        # Collect all instrument keys
        instrument_keys = []
        for strike, options in self.weekly_options['strikes'].items():
            for option_type in ['CE', 'PE']:
                if option_type in options:
                    instrument_keys.append(options[option_type]['instrument_key'])
        
        # Batch fetch quotes (Upstox allows multiple symbols)
        if not instrument_keys:
            logger.error("❌ No option instruments found")
            return option_data
        
        # Build query string
        symbols_param = ','.join([urllib.parse.quote(key) for key in instrument_keys])
        url = f"https://api.upstox.com/v2/market-quote/quotes?symbol={symbols_param}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data.get('status') == 'success':
                            quotes = data.get('data', {})
                            
                            # Parse quotes
                            for strike, options in self.weekly_options['strikes'].items():
                                option_data['strikes'][strike] = {}
                                
                                for option_type in ['CE', 'PE']:
                                    if option_type not in options:
                                        continue
                                    
                                    inst_key = options[option_type]['instrument_key']
                                    
                                    if inst_key in quotes:
                                        quote = quotes[inst_key]
                                        
                                        option_data['strikes'][strike][option_type] = {
                                            'symbol': options[option_type]['trading_symbol'],
                                            'ltp': quote.get('last_price', 0),
                                            'bid': quote.get('depth', {}).get('buy', [{}])[0].get('price', 0),
                                            'ask': quote.get('depth', {}).get('sell', [{}])[0].get('price', 0),
                                            'volume': quote.get('volume', 0),
                                            'oi': quote.get('oi', 0),
                                            'oi_change': quote.get('oi_day_high', 0) - quote.get('oi_day_low', 0),
                                            'change': quote.get('net_change', 0),
                                            'change_pct': quote.get('change_percent', 0)
                                        }
                            
                            logger.info(f"✅ Fetched option chain for {len(option_data['strikes'])} strikes")
                            return option_data
                    
                    logger.error(f"❌ HTTP {resp.status}")
                    return option_data
            
            except Exception as e:
                logger.error(f"💥 Error fetching options: {e}")
                return option_data
    
    async def fetch_all_data(self) -> dict:
        """Fetch both futures candles and option chain"""
        logger.info("\n" + "="*60)
        logger.info("📊 FETCHING ALL DATA")
        logger.info("="*60)
        
        # Fetch futures
        futures_data = await self.fetch_futures_candles()
        
        # Small delay
        await asyncio.sleep(0.5)
        
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
                "expiry": next(iter(self.weekly_options['strikes'].values())).get('CE', {}).get('expiry', 'N/A'),
                "data": options_data
            }
        }

# ==================== TELEGRAM SENDER ====================
class TelegramSender:
    """Send comprehensive data to Telegram"""
    
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    async def send_data(self, data: dict):
        """Send summary + detailed JSON"""
        
        futures = data['monthly_future']['data']
        options = data['weekly_options']['data']
        
        # Check if we have data
        if not futures or not options:
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text="❌ Failed to fetch complete data"
            )
            return
        
        # Build summary
        latest_candle = futures['candles'][0] if futures['candles'] else None
        
        summary = f"""
🚀 NIFTY50 HYBRID DATA

⏰ {data['fetch_time']}

━━━━━━━━━━━━━━━━━━━━━━━━
📊 MONTHLY FUTURES DATA
━━━━━━━━━━━━━━━━━━━━━━━━

📈 Symbol: {data['monthly_future']['symbol']}
📅 Expiry: {data['monthly_future']['expiry']}
📦 Candles: {futures['candle_count']}
📊 Total Volume: {futures['total_volume']:,}
"""
        
        if latest_candle:
            latest_time = datetime.fromisoformat(latest_candle['timestamp'])
            summary += f"""
💰 Latest: ₹{latest_candle['close']:.2f}
📈 High: ₹{latest_candle['high']:.2f}
📉 Low: ₹{latest_candle['low']:.2f}
🕐 Time: {latest_time.strftime('%I:%M %p')}
📊 OI: {latest_candle['oi']:,}
"""
        
        summary += f"""
━━━━━━━━━━━━━━━━━━━━━━━━
📈 WEEKLY OPTIONS CHAIN
━━━━━━━━━━━━━━━━━━━━━━━━

📅 Expiry: {data['weekly_options']['expiry']}
🎯 Spot: ₹{options['spot_price']:.2f}
🎯 ATM Strike: {options['atm_strike']}

"""
        
        # Option chain table
        for strike in sorted(options['strikes'].keys()):
            strike_data = options['strikes'][strike]
            
            ce = strike_data.get('CE', {})
            pe = strike_data.get('PE', {})
            
            atm_marker = "← ATM" if strike == options['atm_strike'] else ""
            
            summary += f"""
Strike: {strike} {atm_marker}
  CE: ₹{ce.get('ltp', 0):.2f} | Vol: {ce.get('volume', 0):,} | OI: {ce.get('oi', 0):,}
  PE: ₹{pe.get('ltp', 0):.2f} | Vol: {pe.get('volume', 0):,} | OI: {pe.get('oi', 0):,}

"""
        
        summary += "━━━━━━━━━━━━━━━━━━━━━━━━\n\n📎 Full JSON attached"
        
        try:
            # Send summary
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=summary
            )
            
            # Send JSON
            json_str = json.dumps(data, indent=2)
            json_file = BytesIO(json_str.encode('utf-8'))
            json_file.name = f"nifty_data_{datetime.now(IST).strftime('%H%M%S')}.json"
            
            await self.bot.send_document(
                chat_id=TELEGRAM_CHAT_ID,
                document=json_file,
                caption="📊 Complete NIFTY50 Data (500 Candles + Option Chain)"
            )
            
            logger.info("✅ Sent to Telegram")
        
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")

# ==================== MAIN ====================
async def main():
    """Main loop"""
    
    logger.info("=" * 80)
    logger.info("🚀 NIFTY50 HYBRID DATA BOT")
    logger.info("=" * 80)
    logger.info("")
    logger.info("⚠️ NEW EXPIRY SCHEDULE (Since Sept 2024):")
    logger.info("   📊 Monthly: Last Tuesday of month")
    logger.info("   📈 Weekly: Every Tuesday")
    logger.info("")
    logger.info("📊 Monthly Futures: 500 candles + volume")
    logger.info("📈 Weekly Options: 5 strikes (ATM ±2)")
    logger.info("")
    
    # Initialize
    logger.info("📥 Initializing instruments...")
    fetcher_init = InstrumentsFetcher()
    
    success = await fetcher_init.initialize()
    if not success:
        logger.error("❌ Initialization failed!")
        return
    
    logger.info("")
    logger.info("✅ Initialization complete")
    logger.info("")
    logger.info("⏱️ Interval: 60 seconds")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    # Create fetcher and sender
    data_fetcher = DataFetcher(fetcher_init)
    sender = TelegramSender()
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Iteration #{iteration}")
            logger.info(f"{'='*60}\n")
            
            # Fetch all data
            data = await data_fetcher.fetch_all_data()
            
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
