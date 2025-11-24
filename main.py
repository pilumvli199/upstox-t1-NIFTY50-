#!/usr/bin/env python3
"""
MULTI-INDEX DATA FETCHING TEST - V1.0
=====================================
✅ Test data fetching for all 4 indices
✅ Verify Upstox API responses
✅ Check data quality

TESTING:
- Spot prices
- Futures candles
- Option chain (5 strikes)

Author: Data Fetching Test Version
Date: November 24, 2025
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta, time
from calendar import monthrange
import pytz
from typing import Dict, Tuple, Optional
import pandas as pd

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')

# ⚠️ PUT YOUR UPSTOX TOKEN HERE
UPSTOX_ACCESS_TOKEN = 'YOUR_TOKEN_HERE'

# ✅ ALL 4 INDICES - VERIFIED INSTRUMENT KEYS
INDICES = {
    'NIFTY': {
        'spot': "NSE_INDEX|Nifty 50",
        'name': 'NIFTY 50',
        'strike_gap': 50,
        'has_weekly': True,
        'futures_prefix': 'NIFTY'
    },
    'BANKNIFTY': {
        'spot': "NSE_INDEX|Nifty Bank",
        'name': 'BANK NIFTY',
        'strike_gap': 100,
        'has_weekly': False,
        'futures_prefix': 'BANKNIFTY'
    },
    'FINNIFTY': {
        'spot': "NSE_INDEX|Nifty Fin Service",
        'name': 'FIN NIFTY',
        'strike_gap': 50,
        'has_weekly': False,
        'futures_prefix': 'FINNIFTY'
    },
    'MIDCPNIFTY': {
        'spot': "NSE_INDEX|NIFTY MID SELECT",
        'name': 'MIDCAP NIFTY',
        'strike_gap': 25,
        'has_weekly': False,
        'futures_prefix': 'MIDCPNIFTY'
    }
}

# Test only these indices
TEST_INDICES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# ==================== HELPER FUNCTIONS ====================
def get_current_futures_symbol(index_name: str) -> str:
    """
    Generate current month futures symbol
    Format: NSE_FO|NIFTY24DECFUT
    """
    now = datetime.now(IST)
    year = now.year
    month = now.month
    
    # Find last Tuesday of current month
    last_day = monthrange(year, month)[1]
    last_date = datetime(year, month, last_day, tzinfo=IST)
    days_to_tuesday = (last_date.weekday() - 1) % 7
    last_tuesday = last_date - timedelta(days=days_to_tuesday)
    
    # If already past expiry, move to next month
    if now.date() > last_tuesday.date() or (
        now.date() == last_tuesday.date() and now.time() > time(15, 30)
    ):
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    
    # Format: NIFTY24DEC
    year_short = year % 100
    month_name = datetime(year, month, 1).strftime('%b').upper()
    
    prefix = INDICES[index_name]['futures_prefix']
    symbol = f"NSE_FO|{prefix}{year_short:02d}{month_name}FUT"
    
    return symbol

def get_expiry_date(index_name: str) -> str:
    """
    Get expiry date in YYYY-MM-DD format
    """
    now = datetime.now(IST)
    today = now.date()
    config = INDICES[index_name]
    
    if config['has_weekly']:
        # NIFTY: Next Tuesday
        days_to_tuesday = (1 - today.weekday() + 7) % 7
        
        if days_to_tuesday == 0:
            if now.time() > time(15, 30):
                expiry = today + timedelta(days=7)
            else:
                expiry = today
        else:
            expiry = today + timedelta(days=days_to_tuesday)
    
    else:
        # Others: Last Tuesday of month
        year = now.year
        month = now.month
        
        last_day = monthrange(year, month)[1]
        last_date = datetime(year, month, last_day)
        
        days_to_tuesday = (last_date.weekday() - 1) % 7
        last_tuesday = last_date - timedelta(days=days_to_tuesday)
        
        if now.date() > last_tuesday.date() or (
            now.date() == last_tuesday.date() and now.time() > time(15, 30)
        ):
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            
            last_day = monthrange(year, month)[1]
            last_date = datetime(year, month, last_day)
            days_to_tuesday = (last_date.weekday() - 1) % 7
            last_tuesday = last_date - timedelta(days=days_to_tuesday)
        
        expiry = last_tuesday.date()
    
    return expiry.strftime('%Y-%m-%d')

# ==================== DATA FETCHER ====================
class DataFetcher:
    """Fetch and test data for one index"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.config = INDICES[index_name]
        self.spot_symbol = self.config['spot']
        self.strike_gap = self.config['strike_gap']
        
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        
        self.results = {
            'spot': {'status': 'pending', 'data': None, 'error': None},
            'futures': {'status': 'pending', 'data': None, 'error': None},
            'options': {'status': 'pending', 'data': None, 'error': None}
        }
    
    def encode_symbol(self, symbol: str) -> str:
        """Proper URL encoding for Upstox"""
        return symbol.replace('|', '%7C').replace(' ', '%20')
    
    async def fetch_url(self, url: str, session: aiohttp.ClientSession) -> Optional[dict]:
        """Fetch URL with error handling"""
        try:
            async with session.get(url, headers=self.headers, 
                                 timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    return {'error': f'HTTP {resp.status}', 'details': error_text}
        except Exception as e:
            return {'error': str(e)}
    
    async def test_spot_price(self, session: aiohttp.ClientSession):
        """Test 1: Fetch spot price"""
        print(f"\n📊 [{self.config['name']}] Testing Spot Price...")
        print(f"   Symbol: {self.spot_symbol}")
        
        spot_encoded = self.encode_symbol(self.spot_symbol)
        url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={spot_encoded}"
        
        print(f"   URL: {url}")
        
        result = await self.fetch_url(url, session)
        
        if result and 'data' in result:
            # Try different possible keys
            spot_price = 0
            found_key = None
            
            for key in [self.spot_symbol, 
                       self.spot_symbol.replace('NSE_INDEX|', 'NSE_INDEX:')]:
                if key in result['data']:
                    spot_price = result['data'][key].get('last_price', 0)
                    found_key = key
                    break
            
            if spot_price > 0:
                self.results['spot']['status'] = 'success'
                self.results['spot']['data'] = {
                    'price': spot_price,
                    'key_used': found_key
                }
                print(f"   ✅ SUCCESS: {spot_price}")
            else:
                self.results['spot']['status'] = 'failed'
                self.results['spot']['error'] = 'Price not found in response'
                print(f"   ❌ FAILED: Price not found")
                print(f"   Response keys: {list(result.get('data', {}).keys())}")
        else:
            self.results['spot']['status'] = 'failed'
            self.results['spot']['error'] = result.get('error', 'Unknown error')
            print(f"   ❌ FAILED: {result.get('error')}")
    
    async def test_futures_candles(self, session: aiohttp.ClientSession):
        """Test 2: Fetch futures candles"""
        print(f"\n📈 [{self.config['name']}] Testing Futures Candles...")
        
        futures_symbol = get_current_futures_symbol(self.index_name)
        print(f"   Symbol: {futures_symbol}")
        
        futures_encoded = self.encode_symbol(futures_symbol)
        to_date = datetime.now(IST).strftime('%Y-%m-%d')
        from_date = (datetime.now(IST) - timedelta(days=10)).strftime('%Y-%m-%d')
        
        url = f"https://api.upstox.com/v2/historical-candle/{futures_encoded}/1minute/{to_date}/{from_date}"
        
        print(f"   URL: {url}")
        
        result = await self.fetch_url(url, session)
        
        if result and result.get('status') == 'success':
            candles = result.get('data', {}).get('candles', [])
            
            if candles:
                # Convert to DataFrame
                df = pd.DataFrame(
                    candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(IST)
                
                # Filter today's data
                today = datetime.now(IST).date()
                df = df[df['timestamp'].dt.date == today]
                
                if not df.empty:
                    latest = df.iloc[-1]
                    self.results['futures']['status'] = 'success'
                    self.results['futures']['data'] = {
                        'total_candles': len(df),
                        'latest_price': latest['close'],
                        'latest_time': latest['timestamp'].strftime('%H:%M:%S'),
                        'sample_candle': latest.to_dict()
                    }
                    print(f"   ✅ SUCCESS: {len(df)} candles")
                    print(f"   Latest: {latest['close']} @ {latest['timestamp'].strftime('%H:%M:%S')}")
                else:
                    self.results['futures']['status'] = 'failed'
                    self.results['futures']['error'] = 'No candles for today'
                    print(f"   ⚠️ WARNING: No candles for today")
            else:
                self.results['futures']['status'] = 'failed'
                self.results['futures']['error'] = 'Empty candles array'
                print(f"   ❌ FAILED: Empty candles")
        else:
            self.results['futures']['status'] = 'failed'
            self.results['futures']['error'] = result.get('error', 'Unknown error')
            print(f"   ❌ FAILED: {result.get('error')}")
    
    async def test_option_chain(self, session: aiohttp.ClientSession):
        """Test 3: Fetch option chain"""
        print(f"\n⚔️ [{self.config['name']}] Testing Option Chain...")
        
        # Need spot price first
        if self.results['spot']['status'] != 'success':
            print(f"   ⚠️ SKIPPED: Need spot price first")
            return
        
        spot_price = self.results['spot']['data']['price']
        expiry = get_expiry_date(self.index_name)
        
        print(f"   Spot: {spot_price}")
        print(f"   Expiry: {expiry}")
        
        spot_encoded = self.encode_symbol(self.spot_symbol)
        url = f"https://api.upstox.com/v2/option/chain?instrument_key={spot_encoded}&expiry_date={expiry}"
        
        print(f"   URL: {url}")
        
        result = await self.fetch_url(url, session)
        
        if result and result.get('status') == 'success':
            all_options = result.get('data', [])
            
            # Filter 5 strikes around ATM
            atm_strike = round(spot_price / self.strike_gap) * self.strike_gap
            min_strike = atm_strike - (2 * self.strike_gap)
            max_strike = atm_strike + (2 * self.strike_gap)
            
            print(f"   ATM: {atm_strike}")
            print(f"   Range: {min_strike} to {max_strike}")
            
            strike_data = {}
            total_ce_oi = 0
            total_pe_oi = 0
            
            for option in all_options:
                strike = option.get('strike_price', 0)
                
                if min_strike <= strike <= max_strike:
                    call_data = option.get('call_options', {}).get('market_data', {})
                    put_data = option.get('put_options', {}).get('market_data', {})
                    
                    ce_oi = call_data.get('oi', 0)
                    pe_oi = put_data.get('oi', 0)
                    
                    strike_data[strike] = {
                        'ce_oi': ce_oi,
                        'pe_oi': pe_oi,
                        'ce_ltp': call_data.get('ltp', 0),
                        'pe_ltp': put_data.get('ltp', 0),
                        'ce_volume': call_data.get('volume', 0),
                        'pe_volume': put_data.get('volume', 0)
                    }
                    
                    total_ce_oi += ce_oi
                    total_pe_oi += pe_oi
            
            if strike_data:
                pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
                
                self.results['options']['status'] = 'success'
                self.results['options']['data'] = {
                    'strikes_found': len(strike_data),
                    'atm_strike': atm_strike,
                    'total_ce_oi': total_ce_oi,
                    'total_pe_oi': total_pe_oi,
                    'pcr': pcr,
                    'strikes': strike_data
                }
                
                print(f"   ✅ SUCCESS: {len(strike_data)} strikes")
                print(f"   PCR: {pcr:.2f}")
                print(f"   Total CE OI: {total_ce_oi:,}")
                print(f"   Total PE OI: {total_pe_oi:,}")
                
                # Show strike details
                print(f"\n   Strike Details:")
                for strike in sorted(strike_data.keys()):
                    data = strike_data[strike]
                    marker = "📍 ATM" if strike == atm_strike else ""
                    print(f"   {strike}: CE={data['ce_oi']:,} PE={data['pe_oi']:,} {marker}")
            else:
                self.results['options']['status'] = 'failed'
                self.results['options']['error'] = 'No strikes in range'
                print(f"   ❌ FAILED: No strikes found")
        else:
            self.results['options']['status'] = 'failed'
            self.results['options']['error'] = result.get('error', 'Unknown error')
            print(f"   ❌ FAILED: {result.get('error')}")
    
    async def run_all_tests(self):
        """Run all tests for this index"""
        async with aiohttp.ClientSession() as session:
            await self.test_spot_price(session)
            await asyncio.sleep(1)  # Rate limit protection
            
            await self.test_futures_candles(session)
            await asyncio.sleep(1)
            
            await self.test_option_chain(session)
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print(f"📋 SUMMARY: {self.config['name']}")
        print(f"{'='*60}")
        
        for test_name, result in self.results.items():
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_emoji} {test_name.upper()}: {result['status']}")
            if result['error']:
                print(f"   Error: {result['error']}")
        
        print(f"{'='*60}\n")

# ==================== MAIN TEST ====================
async def test_all_indices():
    """Test data fetching for all indices"""
    
    print("="*70)
    print("🧪 MULTI-INDEX DATA FETCHING TEST")
    print("="*70)
    print(f"⏰ Started: {datetime.now(IST).strftime('%d-%b %I:%M:%S %p')}")
    print(f"🎯 Testing {len(TEST_INDICES)} indices")
    print("="*70)
    
    # Check token
    if UPSTOX_ACCESS_TOKEN == 'YOUR_TOKEN_HERE':
        print("\n⚠️ ERROR: Please set UPSTOX_ACCESS_TOKEN")
        return
    
    all_results = {}
    
    for index_name in TEST_INDICES:
        print(f"\n\n{'#'*70}")
        print(f"🔍 TESTING: {INDICES[index_name]['name']}")
        print(f"{'#'*70}")
        
        fetcher = DataFetcher(index_name)
        await fetcher.run_all_tests()
        await asyncio.sleep(2)  # Delay between indices
        
        all_results[index_name] = fetcher
    
    # Final Summary
    print("\n\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    
    for index_name, fetcher in all_results.items():
        fetcher.print_summary()
    
    # Overall Status
    print("="*70)
    print("🎯 OVERALL STATUS")
    print("="*70)
    
    for index_name, fetcher in all_results.items():
        spot_ok = fetcher.results['spot']['status'] == 'success'
        futures_ok = fetcher.results['futures']['status'] == 'success'
        options_ok = fetcher.results['options']['status'] == 'success'
        
        all_ok = spot_ok and futures_ok and options_ok
        status = "✅ READY" if all_ok else "⚠️ PARTIAL" if spot_ok else "❌ FAILED"
        
        print(f"{status} - {INDICES[index_name]['name']}")
        if not all_ok:
            if not spot_ok:
                print(f"   ⚠️ Spot: {fetcher.results['spot']['error']}")
            if not futures_ok:
                print(f"   ⚠️ Futures: {fetcher.results['futures']['error']}")
            if not options_ok:
                print(f"   ⚠️ Options: {fetcher.results['options']['error']}")
    
    print("="*70)
    print(f"⏰ Completed: {datetime.now(IST).strftime('%d-%b %I:%M:%S %p')}")
    print("="*70)

# ==================== RUN ====================
if __name__ == "__main__":
    try:
        asyncio.run(test_all_indices())
    except KeyboardInterrupt:
        print("\n\n👋 Test stopped by user")
    except Exception as e:
        print(f"\n\n💥 Critical error: {e}")
        import traceback
        traceback.print_exc()
