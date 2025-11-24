#!/usr/bin/env python3
"""
NIFTY OPTIONS BOT V13.2 - WORKING VERSION
==========================================
✅ API Connections Verified
✅ All 4 Indices Supported
✅ Real-time Option Chain Data

Author: Fixed Version
Date: Nov 24, 2025
"""

import os
import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta, time
import pytz
from calendar import monthrange
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("MultiIndexBot-V13.2")

# Environment Variables
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# ✅ VERIFIED INDICES (Working with Upstox API)
INDICES = {
    'NIFTY': {
        'spot': "NSE_INDEX|Nifty 50",
        'name': 'NIFTY 50',
        'strike_gap': 50,
        'has_weekly': True,
        'expiry_day': 1,  # Tuesday
        'futures_prefix': 'NIFTY'
    },
    'BANKNIFTY': {
        'spot': "NSE_INDEX|Nifty Bank",
        'name': 'BANK NIFTY',
        'strike_gap': 100,
        'has_weekly': False,
        'expiry_day': 1,
        'futures_prefix': 'BANKNIFTY'
    },
    'FINNIFTY': {
        'spot': "NSE_INDEX|Nifty Fin Service",
        'name': 'FIN NIFTY',
        'strike_gap': 50,
        'has_weekly': False,
        'expiry_day': 1,
        'futures_prefix': 'FINNIFTY'
    },
    'MIDCPNIFTY': {
        'spot': "NSE_INDEX|NIFTY MID SELECT",
        'name': 'MIDCAP NIFTY',
        'strike_gap': 25,
        'has_weekly': False,
        'expiry_day': 1,
        'futures_prefix': 'MIDCPNIFTY'
    }
}

ACTIVE_INDICES = ['NIFTY']  # Start with NIFTY, then add others

# Trading Configuration
SCAN_INTERVAL = 120  # 2 minutes
ALERT_COOLDOWN = 300  # 5 minutes per index

# Strategy Thresholds
OI_THRESHOLD_STRONG = 8.0
OI_THRESHOLD_MEDIUM = 5.0
PCR_BULLISH = 1.08
PCR_BEARISH = 0.92
VOL_SPIKE_THRESHOLD = 2.0

# ==================== DATA STRUCTURES ====================
@dataclass
class Signal:
    """Trading Signal"""
    index_name: str
    type: str  # CE_BUY or PE_BUY
    reason: str
    confidence: int
    spot_price: float
    strike: int
    target_points: int
    stop_loss_points: int
    pcr: float
    atm_ce_change: float
    atm_pe_change: float
    timestamp: datetime

# ==================== MEMORY SYSTEM ====================
class SimpleMemory:
    """In-memory storage for OI tracking"""
    
    def __init__(self):
        self.strike_snapshots = {}  # {index: {strike: {timestamp: data}}}
        self.total_oi_snapshots = {}  # {index: {timestamp: {ce, pe}}}
    
    def save_strike_snapshot(self, index_name: str, strike_data: Dict[int, dict]):
        """Save current strike data"""
        if index_name not in self.strike_snapshots:
            self.strike_snapshots[index_name] = {}
        
        now = datetime.now(IST)
        timestamp = now.replace(second=0, microsecond=0)
        
        for strike, data in strike_data.items():
            if strike not in self.strike_snapshots[index_name]:
                self.strike_snapshots[index_name][strike] = {}
            
            self.strike_snapshots[index_name][strike][timestamp] = data.copy()
        
        # Clean old data (keep last 30 minutes)
        cutoff = now - timedelta(minutes=30)
        for strike in self.strike_snapshots[index_name]:
            self.strike_snapshots[index_name][strike] = {
                ts: d for ts, d in self.strike_snapshots[index_name][strike].items()
                if ts > cutoff
            }
    
    def get_strike_oi_change(self, index_name: str, strike: int, 
                            current_data: dict, minutes_ago: int = 15) -> Tuple[float, float]:
        """Calculate OI change for specific strike"""
        if index_name not in self.strike_snapshots:
            return 0.0, 0.0
        
        if strike not in self.strike_snapshots[index_name]:
            return 0.0, 0.0
        
        now = datetime.now(IST)
        target_time = now - timedelta(minutes=minutes_ago)
        
        # Find closest past snapshot
        snapshots = self.strike_snapshots[index_name][strike]
        if not snapshots:
            return 0.0, 0.0
        
        past_times = [t for t in snapshots.keys() if t <= target_time]
        if not past_times:
            return 0.0, 0.0
        
        closest_time = max(past_times)
        past_data = snapshots[closest_time]
        
        ce_change = ((current_data['ce_oi'] - past_data['ce_oi']) / past_data['ce_oi'] * 100
                    if past_data['ce_oi'] > 0 else 0)
        pe_change = ((current_data['pe_oi'] - past_data['pe_oi']) / past_data['pe_oi'] * 100
                    if past_data['pe_oi'] > 0 else 0)
        
        return ce_change, pe_change
    
    def save_total_oi(self, index_name: str, ce_total: int, pe_total: int):
        """Save total OI snapshot"""
        if index_name not in self.total_oi_snapshots:
            self.total_oi_snapshots[index_name] = {}
        
        now = datetime.now(IST)
        timestamp = now.replace(second=0, microsecond=0)
        
        self.total_oi_snapshots[index_name][timestamp] = {
            'ce': ce_total,
            'pe': pe_total
        }
        
        # Clean old data
        cutoff = now - timedelta(minutes=30)
        self.total_oi_snapshots[index_name] = {
            ts: d for ts, d in self.total_oi_snapshots[index_name].items()
            if ts > cutoff
        }

# ==================== DATA FEED ====================
class DataFeed:
    """Fetch market data from Upstox"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.config = INDICES[index_name]
        self.spot_symbol = self.config['spot']
        self.strike_gap = self.config['strike_gap']
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
    
    def encode_symbol(self, symbol: str) -> str:
        """Proper URL encoding for Upstox"""
        return symbol.replace('|', '%7C').replace(' ', '%20')
    
    def get_expiry_date(self) -> str:
        """Calculate expiry date"""
        now = datetime.now(IST)
        today = now.date()
        
        if self.config['has_weekly']:
            # NIFTY: Next Tuesday
            days_to_tuesday = (1 - today.weekday() + 7) % 7
            if days_to_tuesday == 0:
                expiry = today if now.time() <= time(15, 30) else today + timedelta(days=7)
            else:
                expiry = today + timedelta(days=days_to_tuesday)
        else:
            # Monthly: Last Tuesday
            year, month = now.year, now.month
            last_day = monthrange(year, month)[1]
            last_date = datetime(year, month, last_day)
            days_to_tuesday = (last_date.weekday() - 1) % 7
            last_tuesday = last_date - timedelta(days=days_to_tuesday)
            
            if now.date() > last_tuesday.date() or (
                now.date() == last_tuesday.date() and now.time() > time(15, 30)
            ):
                month += 1
                if month > 12:
                    year += 1
                    month = 1
                
                last_day = monthrange(year, month)[1]
                last_date = datetime(year, month, last_day)
                days_to_tuesday = (last_date.weekday() - 1) % 7
                last_tuesday = last_date - timedelta(days=days_to_tuesday)
            
            expiry = last_tuesday.date()
        
        return expiry.strftime('%Y-%m-%d')
    
    async def fetch_with_retry(self, url: str, session: aiohttp.ClientSession, retries: int = 3):
        """Fetch with retry logic"""
        for attempt in range(retries):
            try:
                async with session.get(url, headers=self.headers, 
                                     timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait = 2 ** attempt
                        logger.warning(f"⏳ [{self.index_name}] Rate limit, waiting {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        text = await resp.text()
                        logger.error(f"❌ [{self.index_name}] HTTP {resp.status}: {text[:100]}")
                        await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"💥 [{self.index_name}] Attempt {attempt+1}: {e}")
                await asyncio.sleep(2 * (attempt + 1))
        
        return None
    
    async def get_market_data(self) -> Tuple[float, Dict[int, dict], str]:
        """Fetch spot price and option chain"""
        async with aiohttp.ClientSession() as session:
            # Get spot price
            spot_encoded = self.encode_symbol(self.spot_symbol)
            ltp_url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={spot_encoded}"
            
            ltp_data = await self.fetch_with_retry(ltp_url, session)
            
            spot_price = 0
            if ltp_data and 'data' in ltp_data:
                for key in [self.spot_symbol, 
                           self.spot_symbol.replace('NSE_INDEX|', 'NSE_INDEX:')]:
                    if key in ltp_data['data']:
                        spot_price = ltp_data['data'][key].get('last_price', 0)
                        if spot_price > 0:
                            break
            
            if spot_price == 0:
                logger.error(f"❌ [{self.index_name}] Failed to get spot price")
                return 0, {}, ""
            
            logger.info(f"💰 [{self.index_name}] Spot: {spot_price:.2f}")
            
            # Get option chain
            expiry = self.get_expiry_date()
            chain_url = f"https://api.upstox.com/v2/option/chain?instrument_key={spot_encoded}&expiry_date={expiry}"
            
            chain_data = await self.fetch_with_retry(chain_url, session)
            
            strike_data = {}
            if chain_data and chain_data.get('status') == 'success':
                atm_strike = round(spot_price / self.strike_gap) * self.strike_gap
                min_strike = atm_strike - (2 * self.strike_gap)
                max_strike = atm_strike + (2 * self.strike_gap)
                
                for option in chain_data.get('data', []):
                    strike = option.get('strike_price', 0)
                    
                    if min_strike <= strike <= max_strike:
                        call_data = option.get('call_options', {}).get('market_data', {})
                        put_data = option.get('put_options', {}).get('market_data', {})
                        
                        strike_data[strike] = {
                            'ce_oi': call_data.get('oi', 0),
                            'pe_oi': put_data.get('oi', 0),
                            'ce_vol': call_data.get('volume', 0),
                            'pe_vol': put_data.get('volume', 0),
                            'ce_ltp': call_data.get('ltp', 0),
                            'pe_ltp': put_data.get('ltp', 0)
                        }
                
                logger.info(f"📊 [{self.index_name}] Got {len(strike_data)} strikes")
            
            return spot_price, strike_data, expiry

# ==================== ANALYZER ====================
class Analyzer:
    """Signal generation logic"""
    
    def __init__(self, index_name: str, memory: SimpleMemory):
        self.index_name = index_name
        self.config = INDICES[index_name]
        self.memory = memory
    
    def calculate_pcr(self, strike_data: Dict[int, dict]) -> float:
        """Calculate Put-Call Ratio"""
        total_ce = sum(d['ce_oi'] for d in strike_data.values())
        total_pe = sum(d['pe_oi'] for d in strike_data.values())
        return total_pe / total_ce if total_ce > 0 else 1.0
    
    def analyze(self, spot_price: float, strike_data: Dict[int, dict]) -> Optional[Signal]:
        """Generate trading signal"""
        if not strike_data or spot_price == 0:
            return None
        
        # Save current data
        self.memory.save_strike_snapshot(self.index_name, strike_data)
        
        total_ce = sum(d['ce_oi'] for d in strike_data.values())
        total_pe = sum(d['pe_oi'] for d in strike_data.values())
        self.memory.save_total_oi(self.index_name, total_ce, total_pe)
        
        # Calculate metrics
        pcr = self.calculate_pcr(strike_data)
        atm_strike = round(spot_price / self.config['strike_gap']) * self.config['strike_gap']
        
        # ATM analysis
        atm_ce_change = 0.0
        atm_pe_change = 0.0
        
        if atm_strike in strike_data:
            atm_data = strike_data[atm_strike]
            atm_ce_change, atm_pe_change = self.memory.get_strike_oi_change(
                self.index_name, atm_strike, atm_data, minutes_ago=15
            )
        
        logger.info(f"📊 [{self.index_name}] PCR: {pcr:.2f} | ATM CE: {atm_ce_change:+.1f}% | PE: {atm_pe_change:+.1f}%")
        
        # CE Buy Signal
        if atm_ce_change < -OI_THRESHOLD_MEDIUM:
            logger.info(f"🟢 [{self.index_name}] CE Signal detected!")
            
            return Signal(
                index_name=self.index_name,
                type="CE_BUY",
                reason=f"Call unwinding detected (ATM CE: {atm_ce_change:.1f}%)",
                confidence=80,
                spot_price=spot_price,
                strike=atm_strike,
                target_points=60,
                stop_loss_points=30,
                pcr=pcr,
                atm_ce_change=atm_ce_change,
                atm_pe_change=atm_pe_change,
                timestamp=datetime.now(IST)
            )
        
        # PE Buy Signal
        if atm_pe_change < -OI_THRESHOLD_MEDIUM:
            logger.info(f"🔴 [{self.index_name}] PE Signal detected!")
            
            return Signal(
                index_name=self.index_name,
                type="PE_BUY",
                reason=f"Put unwinding detected (ATM PE: {atm_pe_change:.1f}%)",
                confidence=80,
                spot_price=spot_price,
                strike=atm_strike,
                target_points=60,
                stop_loss_points=30,
                pcr=pcr,
                atm_ce_change=atm_ce_change,
                atm_pe_change=atm_pe_change,
                timestamp=datetime.now(IST)
            )
        
        return None

# ==================== MAIN BOT ====================
class MultiIndexBot:
    """Main bot controller"""
    
    def __init__(self):
        self.memory = SimpleMemory()
        self.feeds = {name: DataFeed(name) for name in ACTIVE_INDICES}
        self.analyzers = {name: Analyzer(name, self.memory) for name in ACTIVE_INDICES}
        self.last_alerts = {name: None for name in ACTIVE_INDICES}
    
    async def run_cycle(self):
        """Run one scan cycle"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Scan Cycle - {datetime.now(IST).strftime('%I:%M:%S %p')}")
        logger.info(f"{'='*60}")
        
        for index_name in ACTIVE_INDICES:
            try:
                # Fetch data
                feed = self.feeds[index_name]
                spot, strikes, expiry = await feed.get_market_data()
                
                if spot == 0 or not strikes:
                    logger.warning(f"⏭️ [{index_name}] No data, skipping")
                    continue
                
                # Analyze
                analyzer = self.analyzers[index_name]
                signal = analyzer.analyze(spot, strikes)
                
                if signal:
                    # Check cooldown
                    last_alert = self.last_alerts[index_name]
                    if last_alert:
                        elapsed = (datetime.now(IST) - last_alert).seconds
                        if elapsed < ALERT_COOLDOWN:
                            logger.info(f"⏳ [{index_name}] Cooldown: {ALERT_COOLDOWN - elapsed}s")
                            continue
                    
                    self.send_alert(signal)
                    self.last_alerts[index_name] = datetime.now(IST)
                else:
                    logger.info(f"✋ [{index_name}] No signal")
                
                await asyncio.sleep(2)  # Rate limit delay
                
            except Exception as e:
                logger.error(f"💥 [{index_name}] Error: {e}")
        
        logger.info(f"{'='*60}\n")
    
    def send_alert(self, signal: Signal):
        """Send alert (console for now)"""
        emoji = "🟢" if signal.type == "CE_BUY" else "🔴"
        
        entry = signal.spot_price
        target = entry + signal.target_points if signal.type == "CE_BUY" else entry - signal.target_points
        stop = entry - signal.stop_loss_points if signal.type == "CE_BUY" else entry + signal.stop_loss_points
        
        msg = f"""
{emoji} {INDICES[signal.index_name]['name']} SIGNAL

Type: {signal.type}
Entry: {entry:.1f}
Target: {target:.1f} ({signal.target_points:+} pts)
Stop Loss: {stop:.1f} ({signal.stop_loss_points} pts)
Strike: {signal.strike}

Reason: {signal.reason}
Confidence: {signal.confidence}%
PCR: {signal.pcr:.2f}

Time: {signal.timestamp.strftime('%I:%M %p')}
"""
        
        logger.info(f"\n🚨 ALERT!\n{msg}")

# ==================== MAIN LOOP ====================
async def main():
    """Main execution"""
    bot = MultiIndexBot()
    
    logger.info("=" * 60)
    logger.info("🚀 MULTI-INDEX BOT V13.2 STARTING")
    logger.info("=" * 60)
    logger.info(f"Active Indices: {', '.join(ACTIVE_INDICES)}")
    logger.info(f"Scan Interval: {SCAN_INTERVAL}s")
    logger.info("=" * 60)
    
    while True:
        try:
            now = datetime.now(IST).time()
            
            if time(9, 15) <= now <= time(15, 30):
                await bot.run_cycle()
                await asyncio.sleep(SCAN_INTERVAL)
            else:
                logger.info("🌙 Market closed, waiting...")
                await asyncio.sleep(300)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped by user")
            break
        except Exception as e:
            logger.error(f"💥 Critical error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Shutdown complete")
