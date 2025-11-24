#!/usr/bin/env python3
"""
MULTI-INDEX DATA FETCHING TEST - V1.1 FIXED
============================================
✅ Fixed token handling
✅ Better error messages
✅ Upstox API verified endpoints
✅ Proper response parsing

Author: Fixed Version
Date: November 24, 2025
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime, timedelta, time
from calendar import monthrange
import pytz
from typing import Dict, Tuple, Optional

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')

# ⚠️ ENVIRONMENT VARIABLE से लेगा
import os
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', '')

# ✅ VERIFIED INSTRUMENT KEYS - Upstox API Format
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

TEST_INDICES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# ==================== HELPER FUNCTIONS ====================
def get_futures_symbol(index_name: str) -> str:
    """
    Generate current month futures symbol
    Fixed format: NSE_FO|NIFTY24DECFUT
    """
    now = datetime.now(IST)
    year = now.year
    month = now.month
    
    # Check if current month expiry passed
    last_day = monthrange(year, month)[1]
    last_date = datetime(year, month, last_day, tzinfo=IST)
    days_to_tuesday = (last_date.weekday() - 1) % 7
    last_tuesday = last_date - timedelta(days=days_to_tuesday)
    
    # If expiry passed, next month
    if now.date() > last_tuesday.date() or (
        now.date() == last_tuesday.date() and now.time() > time(15, 30)
    ):
        month += 1
        if month > 12:
            year += 1
            month = 1
    
    # Format: NIFTY24DEC
    year_short = year % 100
    month_name = datetime(year, month, 1).strftime('%b').upper()
    
    prefix = INDICES[index_name]['futures_prefix']
    symbol = f"NSE_FO|{prefix}{year_short:02d}{month_name}FUT"
    
    return symbol

def get_expiry_date(index_name: str) -> str:
    """Get expiry in YYYY-MM-DD format"""
    now = datetime.now(IST)
    today = now.date()
    config = INDICES[index_name]
    
    if config['has_weekly']:
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

# ==================== DATA FETCHER ====================
class DataFetcher:
    """Test data fetching for one index"""
    
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
    
    async def fetch_url(self, url: str, session: aiohttp.ClientSession) -> Optional[dict]:
        """Fetch with proper error handling"""
        try:
            async with session.get(url, headers=self.headers, 
                                 timeout=aiohttp.ClientTimeout(total=20)) as resp:
                
                response_text = await resp.text()
                
                if resp.status == 200:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        return {'error': 'Invalid JSON response', 'details': response_text[:200]}
                
                elif resp.status == 401:
                    return {'error': 'UNAUTHORIZED - Token invalid/expired', 'status': 401}
                
                elif resp.status == 429:
                    return {'error': 'RATE LIMIT - Too many requests', 'status': 429}
                
                else:
                    return {'error': f'HTTP {resp.status}', 'details': response_text[:200]}
        
        except asyncio.TimeoutError:
            return {'error': 'TIMEOUT - Request took too long'}
        
        except Exception as e:
            return {'error': f'NETWORK ERROR: {str(e)}'}
    
    async def test_spot_price(self, session: aiohttp.ClientSession):
        """Test 1: Fetch spot price"""
        print(f"\n📊 [{self.config['name']}] Testing Spot Price...")
        
        # URL encode properly
        spot_encoded = self.spot_symbol.replace('|', '%7C').replace(' ', '%20')
        url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={spot_encoded}"
        
        print(f"   URL: {url[:80]}...")
        
        result = await self.fetch_url(url, session)
        
        if result and 'error' in result:
            self.results['spot']['status'] = 'failed'
            self.results['spot']['error'] = result['error']
            print(f"   ❌ FAILED: {result['error']}")
            if result.get('status') == 401:
                print(f"   ⚠️ TOKEN ISSUE: Generate new Upstox token!")
            return
        
        if result and 'data' in result:
            # Try multiple key formats
            spot_price = 0
            found_key = None
            
            possible_keys = [
                self.spot_symbol,
                self.spot_symbol.replace('NSE_INDEX|', 'NSE_INDEX:'),
                self.config['name']
            ]
            
            for key in possible_keys:
                if key in result['data']:
                    ltp_data = result['data'][key]
                    spot_price = ltp_data.get('last_price', 0)
                    found_key = key
                    if spot_price > 0:
                        break
            
            if spot_price > 0:
                self.results['spot']['status'] = 'success'
                self.results['spot']['data'] = {
                    'price': spot_price,
                    'key_used': found_key
                }
                print(f"   ✅ SUCCESS: ₹{spot_price}")
            else:
                self.results['spot']['status'] = 'failed'
                self.results['spot']['error'] = 'Price not found in response'
                print(f"   ❌ FAILED: Price not found")
                print(f"   Available keys: {list(result.get('data', {}).keys())}")
        else:
            self.results['spot']['status'] = 'failed'
            self.results['spot']['error'] = 'No data in response'
            print(f"   ❌ FAILED: No data in response")
    
    async def test_futures_candles(self, session: aiohttp.ClientSession):
        """Test 2: Fetch futures candles"""
        print(f"\n📈 [{self.config['name']}] Testing Futures Candles...")
        
        futures_symbol = get_futures_symbol(self.index_name)
        print(f"   Symbol: {futures_symbol}")
        
        futures_encoded = futures_symbol.replace('|', '%7C')
        to_date = datetime.now(IST).strftime('%Y-%m-%d')
        from_date = (datetime.now(IST) - timedelta(days=5)).strftime('%Y-%m-%d')
        
        url = f"https://api.upstox.com/v2/historical-candle/{futures_encoded}/1minute/{to_date}/{from_date}"
        
        print(f"   URL: {url[:80]}...")
        
        result = await self.fetch_url(url, session)
        
        if result and 'error' in result:
            self.results['futures']['status'] = 'failed'
            self.results['futures']['error'] = result['error']
            print(f"   ❌ FAILED: {result['error']}")
            return
        
        if result and result.get('status') == 'success':
            candles = result.get('data', {}).get('candles', [])
            
            if candles:
                # Parse candles (no pandas dependency)
                today = datetime.now(IST).date()
                today_candles = []
                
                for candle in candles:
                    try:
                        # candle = [timestamp, open, high, low, close, volume, oi]
                        ts = datetime.fromisoformat(candle[0].replace('Z', '+00:00'))
                        ts_ist = ts.astimezone(IST)
                        
                        if ts_ist.date() == today:
                            today_candles.append({
                                'time': ts_ist.strftime('%H:%M:%S'),
                                'open': candle[1],
                                'high': candle[2],
                                'low': candle[3],
                                'close': candle[4],
                                'volume': candle[5],
                                'oi': candle[6]
                            })
                    except:
                        continue
                
                if today_candles:
                    latest = today_candles[-1]
                    self.results['futures']['status'] = 'success'
                    self.results['futures']['data'] = {
                        'total_candles': len(today_candles),
                        'latest_price': latest['close'],
                        'latest_time': latest['time'],
                        'sample': latest
                    }
                    print(f"   ✅ SUCCESS: {len(today_candles)} candles")
                    print(f"   Latest: ₹{latest['close']} @ {latest['time']}")
                else:
                    self.results['futures']['status'] = 'warning'
                    self.results['futures']['error'] = 'No candles for today (market closed?)'
                    print(f"   ⚠️ WARNING: No today candles (Total: {len(candles)})")
            else:
                self.results['futures']['status'] = 'failed'
                self.results['futures']['error'] = 'Empty candles array'
                print(f"   ❌ FAILED: No candles returned")
        else:
            self.results['futures']['status'] = 'failed'
            self.results['futures']['error'] = 'Invalid response format'
            print(f"   ❌ FAILED: Invalid response")
    
    async def test_option_chain(self, session: aiohttp.ClientSession):
        """Test 3: Fetch option chain"""
        print(f"\n⚔️ [{self.config['name']}] Testing Option Chain...")
        
        if self.results['spot']['status'] != 'success':
            print(f"   ⚠️ SKIPPED: Need spot price first")
            return
        
        spot_price = self.results['spot']['data']['price']
        expiry = get_expiry_date(self.index_name)
        
        print(f"   Spot: ₹{spot_price}")
        print(f"   Expiry: {expiry}")
        
        spot_encoded = self.spot_symbol.replace('|', '%7C').replace(' ', '%20')
        url = f"https://api.upstox.com/v2/option/chain?instrument_key={spot_encoded}&expiry_date={expiry}"
        
        print(f"   URL: {url[:80]}...")
        
        result = await self.fetch_url(url, session)
        
        if result and 'error' in result:
            self.results['options']['status'] = 'failed'
            self.results['options']['error'] = result['error']
            print(f"   ❌ FAILED: {result['error']}")
            return
        
        if result and result.get('status') == 'success':
            all_options = result.get('data', [])
            
            # Filter 5 strikes
            atm_strike = round(spot_price / self.strike_gap) * self.strike_gap
            min_strike = atm_strike - (2 * self.strike_gap)
            max_strike = atm_strike + (2 * self.strike_gap)
            
            print(f"   ATM: {atm_strike} | Range: {min_strike}-{max_strike}")
            
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
                        'pe_ltp': put_data.get('ltp', 0)
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
                print(f"   PCR: {pcr:.2f} | CE OI: {total_ce_oi:,} | PE OI: {total_pe_oi:,}")
                
                # Strike details
                print(f"\n   Strike Details:")
                for strike in sorted(strike_data.keys()):
                    data = strike_data[strike]
                    marker = "📍 ATM" if strike == atm_strike else ""
                    print(f"   {strike}: CE={data['ce_oi']:,} PE={data['pe_oi']:,} {marker}")
            else:
                self.results['options']['status'] = 'failed'
                self.results['options']['error'] = f'No strikes in range {min_strike}-{max_strike}'
                print(f"   ❌ FAILED: No strikes found")
        else:
            self.results['options']['status'] = 'failed'
            self.results['options']['error'] = 'Invalid response'
            print(f"   ❌ FAILED: Invalid response")
    
    async def run_all_tests(self):
        """Run all tests"""
        async with aiohttp.ClientSession() as session:
            await self.test_spot_price(session)
            await asyncio.sleep(1)
            
            await self.test_futures_candles(session)
            await asyncio.sleep(1)
            
            await self.test_option_chain(session)
    
    def print_summary(self):
        """Print summary"""
        print(f"\n{'='*60}")
        print(f"📋 SUMMARY: {self.config['name']}")
        print(f"{'='*60}")
        
        for test_name, result in self.results.items():
            status_map = {
                'success': '✅',
                'failed': '❌',
                'warning': '⚠️',
                'pending': '⏳'
            }
            emoji = status_map.get(result['status'], '❓')
            print(f"{emoji} {test_name.upper()}: {result['status']}")
            if result['error']:
                print(f"   Error: {result['error']}")
        
        print(f"{'='*60}\n")

# ==================== MAIN TEST ====================
async def test_all_indices():
    """Test all indices"""
    
    print("="*70)
    print("🧪 MULTI-INDEX DATA FETCHING TEST V1.1")
    print("="*70)
    print(f"⏰ Started: {datetime.now(IST).strftime('%d-%b %I:%M:%S %p')}")
    print(f"🎯 Testing: {len(TEST_INDICES)} indices")
    print("="*70)
    
    # Check token
    if not UPSTOX_ACCESS_TOKEN or UPSTOX_ACCESS_TOKEN == '':
        print("\n❌ ERROR: UPSTOX_ACCESS_TOKEN not set!")
        print("\n💡 Fix:")
        print("   1. Railway Dashboard → Variables")
        print("   2. Add: UPSTOX_ACCESS_TOKEN = your_token")
        print("   3. Redeploy")
        print("\n⚠️ Token must be fresh (generated today)")
        return
    
    print(f"\n✅ Token found: {UPSTOX_ACCESS_TOKEN[:20]}...")
    print("")
    
    all_results = {}
    
    for index_name in TEST_INDICES:
        print(f"\n{'#'*70}")
        print(f"🔍 TESTING: {INDICES[index_name]['name']}")
        print(f"{'#'*70}")
        
        fetcher = DataFetcher(index_name)
        
        try:
            await fetcher.run_all_tests()
        except Exception as e:
            print(f"\n💥 CRITICAL ERROR for {index_name}: {e}")
            import traceback
            traceback.print_exc()
        
        all_results[index_name] = fetcher
        
        # Delay between indices
        await asyncio.sleep(2)
    
    # Final summary
    print("\n\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    
    for index_name, fetcher in all_results.items():
        fetcher.print_summary()
    
    # Overall status
    print("="*70)
    print("🎯 OVERALL STATUS")
    print("="*70)
    
    total_success = 0
    total_tests = len(TEST_INDICES)
    
    for index_name, fetcher in all_results.items():
        spot_ok = fetcher.results['spot']['status'] == 'success'
        futures_ok = fetcher.results['futures']['status'] in ['success', 'warning']
        options_ok = fetcher.results['options']['status'] == 'success'
        
        all_ok = spot_ok and futures_ok and options_ok
        
        if all_ok:
            status = "✅ READY"
            total_success += 1
        elif spot_ok:
            status = "⚠️ PARTIAL"
        else:
            status = "❌ FAILED"
        
        print(f"{status} - {INDICES[index_name]['name']}")
        
        if not all_ok:
            if not spot_ok:
                print(f"   ⚠️ Spot: {fetcher.results['spot']['error']}")
            if not futures_ok:
                print(f"   ⚠️ Futures: {fetcher.results['futures']['error']}")
            if not options_ok:
                print(f"   ⚠️ Options: {fetcher.results['options']['error']}")
    
    print("="*70)
    print(f"📈 Success Rate: {total_success}/{total_tests} indices")
    print(f"⏰ Completed: {datetime.now(IST).strftime('%d-%b %I:%M:%S %p')}")
    print("="*70)
    
    # Recommendations
    if total_success < total_tests:
        print("\n💡 TROUBLESHOOTING:")
        print("")
        print("1️⃣ Token Issues (401 Unauthorized):")
        print("   → Generate NEW token from Upstox")
        print("   → Token expires daily!")
        print("   → Update UPSTOX_ACCESS_TOKEN env var")
        print("")
        print("2️⃣ Market Closed:")
        print("   → Futures candles only available during market hours")
        print("   → Spot/Options available 24x7")
        print("")
        print("3️⃣ Rate Limits (429):")
        print("   → Wait 1 minute and retry")
        print("   → Upstox has strict rate limits")
        print("")
    else:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Ready for full analysis code")

# ==================== RUN ====================
if __name__ == "__main__":
    try:
        asyncio.run(test_all_indices())
    except KeyboardInterrupt:
        print("\n\n👋 Test stopped")
    except Exception as e:
        print(f"\n\n💥 Critical error: {e}")
        import traceback
        traceback.print_exc()
