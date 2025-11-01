#!/usr/bin/env python3
"""
HYBRID TRADING BOT v19.0 - HOLIDAY-AWARE EXPIRY
=========================================================
✅ Automatic holiday detection
✅ Smart expiry day selection (skips holidays)
✅ NSE/BSE trading calendar integration
✅ Fallback to API-based expiry detection
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
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import traceback
import re

# Redis import with fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - running without OI tracking")

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONFIG ====================
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

# Trading thresholds
CONFIDENCE_MIN = 75
SCORE_MIN = 90
SCAN_INTERVAL = 300  # 5 minutes
SIGNAL_COOLDOWN = 1800  # 30 minutes

# ==================== INDICES CONFIG ====================
INDICES = {
    'NSE_INDEX|Nifty 50': {
        'name': 'NIFTY50',
        'display_name': 'NIFTY 50',
        'preferred_expiry_day': 1,  # Tuesday (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri)
        'lot_size': 50,
        'tick_size': 0.05
    },
    'NSE_INDEX|Sensex': {
        'name': 'SENSEX',
        'display_name': 'SENSEX',
        'preferred_expiry_day': 3,  # Thursday
        'lot_size': 10,
        'tick_size': 0.05
    }
}

# ==================== NSE HOLIDAY CALENDAR 2025 ====================
# Source: https://www.nseindia.com/regulations/trading-holidays
NSE_HOLIDAYS_2025 = [
    '2025-01-26',  # Republic Day
    '2025-03-14',  # Holi
    '2025-03-31',  # Id-Ul-Fitr
    '2025-04-10',  # Mahavir Jayanti
    '2025-04-14',  # Dr. Ambedkar Jayanti
    '2025-04-18',  # Good Friday
    '2025-05-01',  # Maharashtra Day
    '2025-06-07',  # Id-Ul-Adha (Bakri Id)
    '2025-07-07',  # Muharram
    '2025-08-15',  # Independence Day
    '2025-08-27',  # Ganesh Chaturthi
    '2025-10-02',  # Gandhi Jayanti
    '2025-10-21',  # Dussehra
    '2025-11-01',  # Diwali (Laxmi Pujan)*
    '2025-11-03',  # Diwali - Balipratipada
    '2025-11-05',  # Gurunanak Jayanti
    '2025-12-25',  # Christmas
]

# ==================== DATA CLASSES ====================
@dataclass
class KeyOIZones:
    """OI zones with PCR and key strikes"""
    pcr: float = 0.0
    resistance_strike: int = 0
    support_strike: int = 0
    max_call_oi: int = 0
    max_put_oi: int = 0
    total_call_oi: int = 0
    total_put_oi: int = 0
    timestamp: str = ""

@dataclass
class MultiTimeframeData:
    """Candlestick data with spot price"""
    df_15m: pd.DataFrame
    spot_price: float

@dataclass
class NewsData:
    """Market news data"""
    headline: str
    summary: str
    sentiment: str
    impact_score: int
    url: str

@dataclass
class DeepAnalysis:
    """Complete trade analysis from AI"""
    opportunity: str
    confidence: int
    total_score: int
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: str
    recommended_strike: int
    market_structure: str
    support_levels: List[float]
    resistance_levels: List[float]
    chart_bias: str
    pattern_signal: str = "N/A"
    oi_flow_signal: str = "N/A"
    risk_factors: List[str] = field(default_factory=list)
    news_sentiment: str = "NEUTRAL"
    news_impact: int = 0

# ==================== HOLIDAY MANAGER ====================
class HolidayManager:
    """Smart holiday-aware expiry calculator"""

    def __init__(self):
        self.holidays = set(NSE_HOLIDAYS_2025)
        logger.info(f"📅 Loaded {len(self.holidays)} NSE holidays for 2025")

    def is_trading_day(self, date: datetime) -> bool:
        """Check if given date is a trading day"""
        # Weekend check
        if date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Holiday check
        date_str = date.strftime('%Y-%m-%d')
        if date_str in self.holidays:
            logger.info(f"  🚫 {date_str} is a holiday")
            return False

        return True

    def get_next_trading_day(self, start_date: datetime, target_weekday: int) -> datetime:
        """
        Get next trading day for target weekday, accounting for holidays
        
        Args:
            start_date: Starting date
            target_weekday: Target day (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri)
        
        Returns:
            Next valid trading day
        """
        current = start_date

        # Find next occurrence of target weekday
        days_ahead = target_weekday - current.weekday()
        if days_ahead <= 0:  # Target day already passed this week
            days_ahead += 7

        target_date = current + timedelta(days=days_ahead)

        # Check if it's a trading day
        max_attempts = 10  # Prevent infinite loop
        attempts = 0

        while not self.is_trading_day(target_date) and attempts < max_attempts:
            logger.info(f"  ⚠️ {target_date.strftime('%Y-%m-%d')} is not a trading day")

            # Try previous day first (prepone expiry)
            prev_day = target_date - timedelta(days=1)
            if self.is_trading_day(prev_day):
                logger.info(f"  ✅ Expiry preponed to {prev_day.strftime('%Y-%m-%d %A')}")
                return prev_day

            # Try next day (postpone expiry)
            next_day = target_date + timedelta(days=1)
            if self.is_trading_day(next_day):
                logger.info(f"  ✅ Expiry postponed to {next_day.strftime('%Y-%m-%d %A')}")
                return next_day

            # If both fail, try day before previous
            prev_prev = target_date - timedelta(days=2)
            if self.is_trading_day(prev_prev):
                logger.info(f"  ✅ Expiry preponed to {prev_prev.strftime('%Y-%m-%d %A')}")
                return prev_prev

            # Last resort: move to next week
            target_date = target_date + timedelta(days=7)
            attempts += 1

        if attempts >= max_attempts:
            logger.error("❌ Could not find valid trading day after 10 attempts!")

        return target_date

    def get_current_week_expiry(self, target_weekday: int) -> Optional[str]:
        """
        Get expiry for current week
        
        Args:
            target_weekday: Preferred expiry day (0=Mon, 1=Tue, etc.)
        
        Returns:
            Expiry date string in 'YYYY-MM-DD' format
        """
        now = datetime.now(IST)

        # If it's after 3:30 PM on target day, look for next week
        if now.weekday() == target_weekday and now.time() > time(15, 30):
            now = now + timedelta(days=7)

        expiry_date = self.get_next_trading_day(now, target_weekday)
        expiry_str = expiry_date.strftime('%Y-%m-%d')

        logger.info(f"  📅 Calculated expiry: {expiry_str} ({expiry_date.strftime('%A')})")
        return expiry_str

# ==================== FINNHUB NEWS API ====================
class FinnhubNewsAPI:
    def __init__(self):
        self.api_key = FINNHUB_API_KEY
        self.base_url = "https://finnhub.io/api/v1"
        self.connected = bool(self.api_key)
        if self.connected:
            logger.info("✅ Finnhub API connected")
        else:
            logger.warning("⚠️ Finnhub API key not found")

    def get_market_news(self, limit: int = 15) -> List[Dict]:
        """Fetch latest market news"""
        if not self.connected:
            return []

        try:
            url = f"{self.base_url}/news?category=general&token={self.api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                news_list = response.json()[:limit]
                logger.info(f"📰 Fetched {len(news_list)} news articles")
                return news_list
            else:
                logger.warning(f"News API returned status {response.status_code}")

        except Exception as e:
            logger.error(f"News fetch error: {e}")

        return []

# ==================== REDIS CACHE ====================
class RedisCache:
    def __init__(self):
        self.redis_client = None
        self.connected = False

        if not REDIS_AVAILABLE:
            logger.warning("⚠️ Redis library not available")
            return

        try:
            self.redis_client = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.connected = True
            logger.info("✅ Redis connected successfully!")

        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.connected = False

    def store_and_get_oi_snapshots(
        self,
        symbol: str,
        current_zones: KeyOIZones
    ) -> Tuple[KeyOIZones, Optional[KeyOIZones]]:
        """Store current OI and return previous for comparison"""

        if not self.connected:
            return current_zones, None

        try:
            now = datetime.now(IST)
            current_zones_json = json.dumps(current_zones.__dict__)

            # Daily opening snapshot key
            key_opening = f"oi_snapshot:opening:{symbol}:{now.date()}"
            # Rolling 30-min snapshot key
            key_30min = f"oi_snapshot:30min:{symbol}"

            # Store opening snapshot once per day
            if not self.redis_client.exists(key_opening):
                self.redis_client.setex(key_opening, 86400, current_zones_json)
                logger.info(f"💾 Stored opening OI snapshot for {symbol}")
                return current_zones, None

            # Determine which snapshot to compare against
            if now.time() < time(9, 45):
                # First 30 mins: compare with opening
                prev_json = self.redis_client.get(key_opening)
                comparison_label = "Market Opening"
                logger.info(f"  📊 Comparing with Opening OI (before 9:45)")
            else:
                # After 9:45: compare with 30-min old snapshot
                prev_json = self.redis_client.get(key_30min)
                comparison_label = "30-min ago"
                logger.info(f"  📊 Comparing with 30-min old OI")

            # Update the 30-min snapshot with current data
            self.redis_client.setex(key_30min, 1800, current_zones_json)

            # Parse previous snapshot
            previous_zones = None
            if prev_json:
                try:
                    previous_zones = KeyOIZones(**json.loads(prev_json))
                    logger.info(f"  ✅ Got previous OI data from {comparison_label}")
                except Exception as e:
                    logger.error(f"Error parsing previous OI: {e}")

            return current_zones, previous_zones

        except Exception as e:
            logger.error(f"Redis OI snapshot error: {e}")
            return current_zones, None

# ==================== UPSTOX DATA FETCHER ====================
class UpstoxDataFetcher:
    def __init__(self):
        self.connected = self.test_connection()
        self.holiday_manager = HolidayManager()

    def test_connection(self) -> bool:
        """Test Upstox API connection"""
        try:
            if not UPSTOX_ACCESS_TOKEN:
                logger.error("❌ Upstox Access Token not found")
                return False

            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
            }

            response = requests.get(
                f"{BASE_URL}/v2/user/profile",
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                user_data = response.json().get('data', {})
                logger.info(f"✅ Upstox connected! User: {user_data.get('user_name', 'Unknown')}")
                return True
            else:
                logger.error(f"❌ Upstox connection failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Upstox connection error: {e}")
            return False

    def get_expiries_from_api(self, instrument_key: str) -> List[str]:
        """Get all available expiries from Upstox API"""
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
        }

        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/option/contract?instrument_key={encoded_key}"

        try:
            resp = requests.get(url, headers=headers, timeout=10)

            if resp.status_code == 200:
                contracts = resp.json().get('data', [])
                expiries = sorted(list(set(
                    c['expiry'] for c in contracts if 'expiry' in c
                )))
                logger.info(f"  📋 Found {len(expiries)} expiries from API")
                return expiries
            else:
                logger.warning(f"Expiries fetch failed: {resp.status_code}")

        except Exception as e:
            logger.error(f"Error fetching expiries from API: {e}")

        return []

    def get_next_expiry(
        self,
        instrument_key: str,
        preferred_expiry_day: int,
        symbol_name: str
    ) -> str:
        """
        Smart expiry selection with holiday awareness
        
        Strategy:
        1. Calculate expiry using holiday calendar
        2. Verify against API expiries
        3. Fallback to API if calculation fails
        """
        logger.info(f"  🔍 Finding next expiry for {symbol_name}...")

        # Method 1: Calculate using holiday calendar
        calculated_expiry = self.holiday_manager.get_current_week_expiry(
            preferred_expiry_day
        )

        # Method 2: Get actual expiries from API
        api_expiries = self.get_expiries_from_api(instrument_key)

        if not api_expiries:
            # No API data - trust calculation
            logger.info(f"  ✅ Using calculated expiry: {calculated_expiry}")
            return calculated_expiry

        # Filter future expiries
        today = datetime.now(IST).date()
        now_time = datetime.now(IST).time()

        future_expiries = [
            exp for exp in api_expiries
            if datetime.strptime(exp, '%Y-%m-%d').date() > today
            or (datetime.strptime(exp, '%Y-%m-%d').date() == today
                and now_time < time(15, 30))
        ]

        if not future_expiries:
            logger.warning("  ⚠️ No future expiries found in API")
            return calculated_expiry

        # Check if calculated expiry matches API
        if calculated_expiry in future_expiries:
            logger.info(f"  ✅ Expiry verified: {calculated_expiry} (matches API)")
            return calculated_expiry

        # If mismatch, prefer nearest API expiry
        nearest_expiry = min(future_expiries)
        logger.warning(
            f"  ⚠️ Calculated expiry {calculated_expiry} not in API. "
            f"Using nearest: {nearest_expiry}"
        )

        return nearest_expiry

    @staticmethod
    def get_option_chain(instrument_key: str, expiry: str) -> List[Dict]:
        """Fetch complete option chain for given expiry"""
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
        }

        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"

        try:
            resp = requests.get(url, headers=headers, timeout=15)

            if resp.status_code == 200:
                chain = resp.json().get('data', [])
                sorted_chain = sorted(chain, key=lambda x: x.get('strike_price', 0))
                logger.info(f"  ✅ Fetched option chain: {len(sorted_chain)} strikes")
                return sorted_chain
            else:
                logger.error(f"Option chain fetch failed: {resp.status_code}")

        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")

        return []

    @staticmethod
    def get_candlestick_data(instrument_key: str) -> Optional[MultiTimeframeData]:
        """Fetch 15-min candlestick data"""
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
        }

        encoded_key = urllib.parse.quote(instrument_key, safe='')
        all_candles = []

        try:
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            from_date = (datetime.now(IST) - timedelta(days=7)).strftime('%Y-%m-%d')

            url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/15minute/{to_date}/{from_date}"

            resp = requests.get(url, headers=headers, timeout=20)

            if resp.status_code == 200 and resp.json().get('status') == 'success':
                candles = resp.json().get('data', {}).get('candles', [])
                all_candles.extend(candles)
                logger.info(f"  📊 Fetched {len(candles)} candles (15min)")
            else:
                logger.warning(f"Historical data fetch failed: {resp.text}")
                return None

        except Exception as e:
            logger.error(f"Candlestick data error: {e}")
            return None

        if not all_candles:
            logger.warning("❌ No candlestick data available")
            return None

        # Create DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(IST)
        df = df.set_index('timestamp').astype(float).sort_index(ascending=True)

        spot_price = df['close'].iloc[-1] if not df.empty else 0

        return MultiTimeframeData(df_15m=df, spot_price=spot_price)

# ==================== NEWS ANALYZER ====================
class NewsAnalyzer:
    @staticmethod
    def filter_and_analyze_news(
        symbol: str,
        news_list: List[Dict]
    ) -> Optional[NewsData]:
        """Filter and analyze relevant news"""

        if not news_list:
            return None

        try:
            relevant_news = []

            # Indian market keywords
            keywords = [
                "nifty", "sensex", "rbi", "rupee", "inflation", "gdp",
                "sebi", "indian market", "finance minister", "monetary policy",
                "repo rate", "bse", "nse"
            ]

            # Filter relevant news
            for news in news_list:
                combined_text = (
                    news.get('headline', '') + " " + news.get('summary', '')
                ).lower()

                if any(kw in combined_text for kw in keywords):
                    # Exclude US-centric news
                    us_keywords = [
                        "fed", "dollar", "nasdaq", "dow jones",
                        "s&p 500", "wall street"
                    ]
                    if not any(us_kw in combined_text for us_kw in us_keywords):
                        relevant_news.append(news)

            if not relevant_news:
                return None

            # Use top news
            top_news = relevant_news[0]

            return NewsData(
                headline=top_news.get('headline', '')[:100],
                summary=top_news.get('summary', '')[:200] + '...',
                sentiment="NEUTRAL",
                impact_score=70,
                url=top_news.get('url', '')
            )

        except Exception as e:
            logger.error(f"News analysis error: {e}")
            return None

# ==================== OI ANALYZER ====================
class OIAnalyzer:
    def get_key_oi_zones(
        self,
        strikes: List[Dict],
        spot_price: float
    ) -> KeyOIZones:
        """Extract key OI zones from option chain"""

        if not strikes:
            return KeyOIZones()

        total_pe_oi = 0
        total_ce_oi = 0
        max_pe_oi = 0
        max_ce_oi = 0
        support_strike = 0
        resistance_strike = 0

        # Consider strikes within +/- 5% of spot
        strike_range_min = spot_price * 0.95
        strike_range_max = spot_price * 1.05

        for s in strikes:
            sp = s.get('strike_price', 0)

            if not (strike_range_min <= sp <= strike_range_max):
                continue

            ce_oi = s.get('call_options', {}).get('market_data', {}).get('oi', 0)
            pe_oi = s.get('put_options', {}).get('market_data', {}).get('oi', 0)

            total_ce_oi += ce_oi
            total_pe_oi += pe_oi

            # Track max Call OI (resistance)
            if ce_oi > max_ce_oi:
                max_ce_oi = ce_oi
                resistance_strike = sp

            # Track max Put OI (support)
            if pe_oi > max_pe_oi:
                max_pe_oi = pe_oi
                support_strike = sp

        pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0

        return KeyOIZones(
            pcr=pcr,
            resistance_strike=int(resistance_strike),
            support_strike=int(support_strike),
            max_call_oi=int(max_ce_oi),
            max_put_oi=int(max_pe_oi),
            total_call_oi=int(total_ce_oi),
            total_put_oi=int(total_pe_oi),
            timestamp=datetime.now(IST).isoformat()
        )

# ==================== AI ANALYZER ====================
class AIAnalyzer:
    @staticmethod
    def extract_json(content: str) -> Optional[Dict]:
        """Extract JSON from AI response"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        logger.error("Failed to extract JSON from AI response")
        return None

    @staticmethod
    def validate_targets(
        opportunity: str,
        entry: float,
        sl: float,
        t1: float,
        t2: float
    ) -> bool:
        """Validate target prices and R:R ratio"""

        if opportunity == "WAIT":
            return True

        # Validate direction
        if opportunity == "CE_BUY":
            if not (t1 > entry > sl):
                logger.warning(
                    f"⚠️ Invalid CE_BUY levels: Entry={entry}, SL={sl}, T1={t1}"
                )
                return False
            risk = entry - sl
            reward = t1 - entry

        elif opportunity == "PE_BUY":
            if not (t1 < entry < sl):
                logger.warning(
                    f"⚠️ Invalid PE_BUY levels: Entry={entry}, SL={sl}, T1={t1}"
                )
                return False
            risk = sl - entry
            reward = entry - t1
        else:
            return False

        # Check minimum R:R ratio
        rr_ratio = reward / risk if risk > 0 else 0

        if rr_ratio < 1.5:
            logger.warning(
                f"⚠️ Poor R:R ratio: {rr_ratio:.2f} (minimum 1:1.5 required)"
            )
            return False

        logger.info(f"  ✅ Validation passed. R:R = {rr_ratio:.2f}")
        return True

    @staticmethod
    def deep_analysis(
        symbol: str,
        mtf_data: MultiTimeframeData,
        current_oi: KeyOIZones,
        prev_oi: Optional[KeyOIZones],
        news_data: Optional[NewsData]
    ) -> Optional[DeepAnalysis]:
        """Complete AI analysis with all data"""

        try:
            # Prepare candlestick data
            candle_df = mtf_data.df_15m.tail(500).reset_index()
            candle_df['timestamp'] = candle_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            candles_json = candle_df[
                ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            ].to_json(orient='records')

            # Build OI comparison section
            now = datetime.now(IST)
            comparison_period = "Market Opening" if now.time() < time(9, 45) else "30 Min Ago"

            current_oi_text = f"""**CURRENT OI DATA:**
- Resistance (Max Call OI): {current_oi.resistance_strike} (OI: {current_oi.max_call_oi:,})
- Support (Max Put OI): {current_oi.support_strike} (OI: {current_oi.max_put_oi:,})
- PCR: {current_oi.pcr}
- Total Call OI: {current_oi.total_call_oi:,}
- Total Put OI: {current_oi.total_put_oi:,}
"""

            if prev_oi:
                resistance_change = current_oi.resistance_strike - prev_oi.resistance_strike
                support_change = current_oi.support_strike - prev_oi.support_strike
                pcr_change = current_oi.pcr - prev_oi.pcr

                resistance_arrow = "↑" if resistance_change > 0 else "↓" if resistance_change < 0 else "→"
                support_arrow = "↑" if support_change > 0 else "↓" if support_change < 0 else "→"

                prev_oi_text = f"""**PREVIOUS OI DATA ({comparison_period}):**
- Resistance: {prev_oi.resistance_strike} → {current_oi.resistance_strike} {resistance_arrow} (Change: {resistance_change})
- Support: {prev_oi.support_strike} → {current_oi.support_strike} {support_arrow} (Change: {support_change})
- PCR: {prev_oi.pcr} → {current_oi.pcr} (Change: {pcr_change:+.2f})

**OI FLOW INTERPRETATION:**
- Support moving UP + Price holding = Bullish strength
- Resistance moving DOWN + Price weak = Bearish pressure
- PCR > 1.2 = Bullish bias | PCR < 0.8 = Bearish bias
"""
            else:
                prev_oi_text = f"**PREVIOUS OI DATA:** Not available yet (first scan)\n"

            # News section
            news_text = ""
            if news_data:
                news_text = f"""**NEWS CONTEXT:**
- Headline: {news_data.headline}
- Summary: {news_data.summary}
"""

            # Construct prompt
            prompt = f"""You are an elite F&O trader analyzing {symbol} for intraday options trading on NSE/BSE.

📊 **CANDLESTICK DATA (15min, Last 7 days):**
{candles_json}

💹 **CURRENT MARKET STATUS:**
- Spot Price: ₹{mtf_data.spot_price:.2f}
- Time: {datetime.now(IST).strftime('%d-%b %H:%M')}

🔢 **OPTION INTEREST ANALYSIS:**

{current_oi_text}

{prev_oi_text}

{news_text}

🎯 **ANALYSIS REQUIREMENTS:**
1. Identify chart patterns (Support/Resistance, trendlines, candlestick patterns)
2. Analyze volume confirmation
3. Interpret OI flow changes comprehensively
4. Calculate precise entry/exit levels
5. Consider news sentiment if applicable

⚠️ **CRITICAL TRADING RULES:**
- **CE_BUY:** target_1 > entry_price > stop_loss (upside trade)
- **PE_BUY:** target_1 < entry_price < stop_loss (downside trade)
- **Minimum R:R:** 1:1.5 (reward must be 1.5x the risk)
- **Strike selection:** ATM or 1-2 strikes OTM for best liquidity
- **confidence:** Be honest - only high confidence (75+) if setup is clear
- **total_score:** Rate overall setup quality (0-100)

📋 **OUTPUT FORMAT (JSON ONLY, NO OTHER TEXT):**
{{
  "chart_bias": "Bullish/Bearish/Neutral",
  "market_structure": "Clear description of current price structure",
  "pattern_signal": "Any chart patterns identified (e.g., 'Higher highs forming')",
  "oi_flow_signal": "Detailed OI flow interpretation based on changes",
  "opportunity": "CE_BUY/PE_BUY/WAIT",
  "confidence": 85,
  "total_score": 90,
  "entry_price": {mtf_data.spot_price:.2f},
  "stop_loss": 0.0,
  "target_1": 0.0,
  "target_2": 0.0,
  "risk_reward": "1:2.5",
  "recommended_strike": {round(mtf_data.spot_price/50)*50},
  "support_levels": [0.0],
  "resistance_levels": [0.0],
  "risk_factors": ["List specific risks"]
}}

Respond with ONLY the JSON object, nothing else."""

            # Call DeepSeek API
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert F&O trader. Respond in JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }

            logger.info("🧠 Sending data to AI for analysis...")
            response = requests.post(url, json=payload, headers=headers, timeout=60)

            if response.status_code != 200:
                logger.error(f"AI API error: {response.status_code} - {response.text}")
                return None

            # Extract analysis
            ai_content = response.json()['choices'][0]['message']['content']
            analysis_dict = AIAnalyzer.extract_json(ai_content)

            if not analysis_dict:
                logger.error("Failed to parse AI response")
                return None

            logger.info(
                f"  ✅ AI Response: {analysis_dict.get('opportunity')} | "
                f"Confidence: {analysis_dict.get('confidence')}% | "
                f"Score: {analysis_dict.get('total_score')}"
            )

            # Validate targets
            opportunity = analysis_dict.get('opportunity', 'WAIT')
            if opportunity != "WAIT":
                if not AIAnalyzer.validate_targets(
                    opportunity,
                    analysis_dict.get('entry_price', 0),
                    analysis_dict.get('stop_loss', 0),
                    analysis_dict.get('target_1', 0),
                    analysis_dict.get('target_2', 0)
                ):
                    logger.warning("  ❌ Validation failed, rejecting signal")
                    return None

            # Create analysis object
            return DeepAnalysis(
                **analysis_dict,
                news_sentiment=news_data.sentiment if news_data else "NEUTRAL",
                news_impact=news_data.impact_score if news_data else 0
            )

        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            traceback.print_exc()
            return None

# ==================== CHART GENERATOR ====================
class ChartGenerator:
    @staticmethod
    def create_chart(
        mtf_data: MultiTimeframeData,
        symbol: str,
        analysis: DeepAnalysis,
        current_oi: KeyOIZones,
        prev_oi: Optional[KeyOIZones]
    ) -> Optional[io.BytesIO]:
        """Generate trading chart with analysis"""

        try:
            df_plot = mtf_data.df_15m.tail(120).copy()

            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(18, 11),
                gridspec_kw={'height_ratios': [3, 1]},
                facecolor='white'
            )

            # Plot candlesticks
            for i, row in enumerate(df_plot.itertuples()):
                color = '#26a69a' if row.close >= row.open else '#ef5350'

                # Wick
                ax1.plot([i, i], [row.low, row.high], color=color, linewidth=1.2)

                # Body
                ax1.add_patch(Rectangle(
                    (i - 0.35, min(row.open, row.close)),
                    0.7,
                    abs(row.close - row.open),
                    facecolor=color,
                    edgecolor=color
                ))

            # OI levels
            ax1.axhline(
                y=current_oi.support_strike,
                color='#2196F3',
                linestyle='-.',
                linewidth=2,
                alpha=0.8,
                label=f'OI Support: {current_oi.support_strike}'
            )

            ax1.axhline(
                y=current_oi.resistance_strike,
                color='#9C27B0',
                linestyle='-.',
                linewidth=2,
                alpha=0.8,
                label=f'OI Resistance: {current_oi.resistance_strike}'
            )

            # Current price line
            ax1.axhline(
                y=mtf_data.spot_price,
                color='#FF6B00',
                linestyle='-',
                linewidth=2.5,
                label=f'CMP: {mtf_data.spot_price:.1f}'
            )

            # Entry, SL, Targets
            if analysis.opportunity != "WAIT":
                ax1.axhline(
                    y=analysis.entry_price,
                    color='#4CAF50',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    label=f'Entry: {analysis.entry_price:.1f}'
                )

                ax1.axhline(
                    y=analysis.stop_loss,
                    color='#F44336',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    label=f'SL: {analysis.stop_loss:.1f}'
                )

                ax1.axhline(
                    y=analysis.target_1,
                    color='#00E676',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    label=f'T1: {analysis.target_1:.1f}'
                )

            # OI change indicator
            if prev_oi:
                oi_change_text = (
                    f"OI Flow: Support {prev_oi.support_strike}→{current_oi.support_strike} | "
                    f"Resistance {prev_oi.resistance_strike}→{current_oi.resistance_strike} | "
                    f"PCR {prev_oi.pcr}→{current_oi.pcr}"
                )

                ax1.text(
                    0.02, 0.98,
                    oi_change_text,
                    transform=ax1.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
                )

            # Chart title
            title = (
                f'{symbol} | 15min | {analysis.chart_bias} | '
                f'Confidence: {analysis.confidence}% | Score: {analysis.total_score}/100'
            )
            ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)

            ax1.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
            ax1.tick_params(axis='x', bottom=False, labelbottom=False)

            # Volume bars
            vol_colors = [
                '#26a69a' if row.close >= row.open else '#ef5350'
                for row in df_plot.itertuples()
            ]

            ax2.bar(range(len(df_plot)), df_plot['volume'], color=vol_colors, alpha=0.7)
            ax2.set_ylabel('Volume', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Time', fontsize=11)
            ax2.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)

            plt.tight_layout(pad=2.0)

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)

            return buf

        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            traceback.print_exc()
            return None

# ==================== TELEGRAM NOTIFIER ====================
class TelegramNotifier:
    def __init__(self, api_status: Dict):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.api_status = api_status

    async def send_startup_message(self):
        """Send bot startup notification"""
        try:
            status_emojis = {
                'upstox': '🟢' if self.api_status.get('upstox') else '🔴',
                'redis': '🟢' if self.api_status.get('redis') else '🟡',
                'finnhub': '🟢' if self.api_status.get('finnhub') else '🟡',
                'deepseek': '🟢' if self.api_status.get('deepseek') else '🔴'
            }

            msg = f"""🚀 **HYBRID TRADING BOT v19.0 - HOLIDAY AWARE** 🚀

**📡 API STATUS:**
{status_emojis['upstox']} Upstox API
{status_emojis['redis']} Redis Cache (OI Tracking)
{status_emojis['finnhub']} Finnhub News
{status_emojis['deepseek']} DeepSeek AI

**📊 TRACKING:**
- NIFTY 50 (Tuesday Weekly) - Auto-adjusts for holidays
- SENSEX (Thursday Weekly) - Auto-adjusts for holidays

**⚙️ CONFIGURATION:**
- Scan Interval: {SCAN_INTERVAL//60} minutes
- Min Confidence: {CONFIDENCE_MIN}%
- Min Score: {SCORE_MIN}/100
- Signal Cooldown: {SIGNAL_COOLDOWN//60} mins

**🧠 ANALYSIS ENGINE:**
✅ Raw candlestick analysis (15min)
✅ Smart OI zones with flow comparison
✅ 30-min OI snapshot tracking
✅ Market news integration
✅ AI-powered signal generation

**🎯 NEW IN v19.0:**
✅ Holiday-aware expiry calculation
✅ Automatic expiry day adjustment
✅ NSE trading calendar integration
✅ Prepone/postpone logic for holidays

**📝 MODE:** Paper Trading Ready

Bot is now scanning... 🔍"""

            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )

            logger.info("✅ Startup message sent to Telegram")

        except Exception as e:
            logger.error(f"Telegram startup message error: {e}")

    async def send_alert(
        self,
        symbol: str,
        analysis: DeepAnalysis,
        mtf: MultiTimeframeData,
        current_oi: KeyOIZones,
        prev_oi: Optional[KeyOIZones],
        expiry: str,
        index_info: Dict
    ):
        """Send trading alert with chart"""
        try:
            # Signal emoji
            signal_map = {
                "CE_BUY": "🟢 CALL BUY",
                "PE_BUY": "🔴 PUT BUY",
                "WAIT": "⚪ NO TRADE"
            }
            signal = signal_map.get(analysis.opportunity, "⚪ WAIT")

            # Parse expiry date to show day
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
            expiry_display = expiry_date.strftime('%d-%b-%Y (%A)')

            # Build alert message
            alert = f"""🎯 **{index_info['display_name']} SIGNAL** 🎯

**{signal}**

━━━━━━━━━━━━━━━━━━━━
**📊 ANALYSIS:**
- **Market Bias:** {analysis.chart_bias}
- **Confidence:** {analysis.confidence}%
- **Quality Score:** {analysis.total_score}/100
- **Structure:** {analysis.market_structure}

**🔢 OI INSIGHTS:**
- **Current PCR:** {current_oi.pcr}
- **Support Level:** {current_oi.support_strike} (OI: {current_oi.max_put_oi:,})
- **Resistance Level:** {current_oi.resistance_strike} (OI: {current_oi.max_call_oi:,})
- **OI Flow:** {analysis.oi_flow_signal}

━━━━━━━━━━━━━━━━━━━━
**💰 TRADE SETUP:**

📍 **Entry:** `{analysis.entry_price:.2f}`
🛑 **Stop Loss:** `{analysis.stop_loss:.2f}`
🎯 **Target 1:** `{analysis.target_1:.2f}`
🎯 **Target 2:** `{analysis.target_2:.2f}`
📊 **Risk:Reward:** {analysis.risk_reward}

**📝 RECOMMENDED OPTION:**
- **Strike:** {analysis.recommended_strike}
- **Expiry:** {expiry_display}
- **Lot Size:** {index_info['lot_size']}

━━━━━━━━━━━━━━━━━━━━
**⚠️ RISK FACTORS:**
"""

            for risk in analysis.risk_factors:
                alert += f"• {risk}\n"

            if analysis.news_sentiment != "NEUTRAL":
                alert += f"\n**📰 News Sentiment:** {analysis.news_sentiment}"

            alert += f"""

━━━━━━━━━━━━━━━━━━━━
⏰ *Signal Time: {datetime.now(IST).strftime('%d %b %Y, %H:%M:%S IST')}*
📌 *CMP: {mtf.spot_price:.2f}*

_This is for paper trading. Trade at your own risk._"""

            # Generate chart
            chart = ChartGenerator.create_chart(
                mtf, symbol, analysis, current_oi, prev_oi
            )

            # Send alert
            if chart:
                await self.bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=chart,
                    caption=alert,
                    parse_mode='Markdown'
                )
            else:
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=alert,
                    parse_mode='Markdown'
                )

            logger.info(f"✅ Alert sent for {symbol}: {analysis.opportunity}")

        except Exception as e:
            logger.error(f"Telegram alert error: {e}")
            traceback.print_exc()

    async def send_error_notification(self, error_msg: str):
        """Send error notification"""
        try:
            msg = f"⚠️ **BOT ERROR**\n\n`{error_msg}`"
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error notification failed: {e}")

# ==================== MAIN BOT CLASS ====================
class HybridBot:
    def __init__(self):
        logger.info("="*60)
        logger.info("Initializing Hybrid Trading Bot v19.0 (Holiday-Aware)...")
        logger.info("="*60)

        # Initialize components
        self.redis = RedisCache()
        self.fetcher = UpstoxDataFetcher()
        self.finnhub = FinnhubNewsAPI()
        self.oi_analyzer = OIAnalyzer()
        self.ai_analyzer = AIAnalyzer()

        # API status
        api_status = {
            'upstox': self.fetcher.connected,
            'redis': self.redis.connected,
            'finnhub': self.finnhub.connected,
            'deepseek': bool(DEEPSEEK_API_KEY)
        }

        self.notifier = TelegramNotifier(api_status)

        # Signal tracking
        self.last_signals = {}  # Track last signal time per symbol

        logger.info("✅ Bot initialized successfully!")
        logger.info("="*60)

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now(IST)

        # Weekend check
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Holiday check
        date_str = now.strftime('%Y-%m-%d')
        if date_str in NSE_HOLIDAYS_2025:
            logger.info(f"🚫 Today ({date_str}) is a market holiday")
            return False

        # Market hours: 9:15 AM to 3:30 PM IST
        market_start = time(9, 15)
        market_end = time(15, 30)

        return market_start <= now.time() <= market_end

    async def scan_indices(self):
        """Main scanning function for all indices"""

        logger.info("\n" + "="*60)
        logger.info(f"SCAN CYCLE STARTED @ {datetime.now(IST).strftime('%d-%b-%Y %H:%M:%S')}")
        logger.info("="*60)

        # Fetch market news once per cycle
        market_news = self.finnhub.get_market_news()

        # Scan each index
        for instrument_key, index_info in INDICES.items():
            try:
                symbol = index_info['name']
                display_name = index_info['display_name']

                logger.info(f"\n🔍 Analyzing {display_name}...")
                logger.info("-" * 40)

                # Step 1: Fetch candlestick data
                logger.info("  📊 Fetching candlestick data...")
                mtf = self.fetcher.get_candlestick_data(instrument_key)

                if not mtf or mtf.df_15m.empty or mtf.spot_price == 0:
                    logger.warning(f"  ❌ No market data available for {symbol}")
                    continue

                logger.info(f"  ✅ Spot Price: ₹{mtf.spot_price:.2f}")

                # Step 2: Get smart expiry (holiday-aware)
                logger.info("  📅 Calculating expiry (holiday-aware)...")
                expiry = self.fetcher.get_next_expiry(
                    instrument_key,
                    index_info['preferred_expiry_day'],
                    symbol
                )

                expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
                logger.info(f"  ✅ Using expiry: {expiry} ({expiry_date.strftime('%A')})")

                # Step 3: Fetch option chain
                logger.info("  🔢 Fetching option chain...")
                strikes = self.fetcher.get_option_chain(instrument_key, expiry)

                if not strikes:
                    logger.warning(f"  ❌ No option chain data for {symbol}")
                    continue

                # Analyze OI
                logger.info("  🔢 Analyzing Open Interest...")
                current_oi = self.oi_analyzer.get_key_oi_zones(strikes, mtf.spot_price)

                logger.info(f"  ✅ OI Data: PCR={current_oi.pcr}, "
                            f"Support={current_oi.support_strike}, "
                            f"Resistance={current_oi.resistance_strike}")

                # Get previous OI for comparison
                current_oi, prev_oi = self.redis.store_and_get_oi_snapshots(
                    symbol, current_oi
                )

                # Step 4: Filter relevant news
                news_data = NewsAnalyzer.filter_and_analyze_news(symbol, market_news)

                if news_data:
                    logger.info(f"  📰 Relevant news found: {news_data.headline[:50]}...")

                # Step 5: AI Deep Analysis
                logger.info("  🧠 Requesting AI analysis...")
                deep = self.ai_analyzer.deep_analysis(
                    symbol, mtf, current_oi, prev_oi, news_data
                )

                if not deep:
                    logger.info("  ⚪ No actionable signal from AI")
                    continue

                logger.info(
                    f"  ✅ AI Analysis Complete:\n"
                    f"      • Signal: {deep.opportunity}\n"
                    f"      • Bias: {deep.chart_bias}\n"
                    f"      • Confidence: {deep.confidence}%\n"
                    f"      • Score: {deep.total_score}/100"
                )

                # Step 6: Check thresholds and cooldown
                if deep.opportunity == "WAIT":
                    logger.info("  ⚪ AI suggests WAIT - No trade setup")
                    continue

                if deep.confidence < CONFIDENCE_MIN:
                    logger.info(
                        f"  ⚠️ Confidence too low ({deep.confidence}% < {CONFIDENCE_MIN}%)"
                    )
                    continue

                if deep.total_score < SCORE_MIN:
                    logger.info(
                        f"  ⚠️ Score too low ({deep.total_score} < {SCORE_MIN})"
                    )
                    continue

                # Check signal cooldown
                now = datetime.now(IST)
                last_signal_time = self.last_signals.get(
                    symbol,
                    datetime.min.replace(tzinfo=IST)
                )

                time_since_last = (now - last_signal_time).total_seconds()

                if time_since_last < SIGNAL_COOLDOWN:
                    remaining = int((SIGNAL_COOLDOWN - time_since_last) / 60)
                    logger.info(
                        f"  ⏳ Signal cooldown active. "
                        f"Wait {remaining} more minutes."
                    )
                    continue

                # Step 7: Send alert
                logger.info(f"  🚀 HIGH CONFIDENCE SIGNAL DETECTED!")
                logger.info(f"  📤 Sending alert to Telegram...")

                await self.notifier.send_alert(
                    symbol, deep, mtf, current_oi, prev_oi, expiry, index_info
                )

                # Update last signal time
                self.last_signals[symbol] = now

                logger.info(f"  ✅ Alert sent successfully!")

                # Small delay between alerts
                await asyncio.sleep(3)

            except Exception as e:
                logger.error(f"❌ Error scanning {instrument_key}: {e}")
                traceback.print_exc()

                # Send error notification
                await self.notifier.send_error_notification(
                    f"Error scanning {instrument_key}: {str(e)[:100]}"
                )

            # Delay between indices
            await asyncio.sleep(2)

        logger.info("\n" + "="*60)
        logger.info("SCAN CYCLE COMPLETED")
        logger.info("="*60)

    async def run(self):
        """Main bot loop"""

        # Send startup message
        await self.notifier.send_startup_message()

        logger.info("\n🟢 Bot is now running...\n")

        while True:
            try:
                if self.is_market_open():
                    # Market is open - scan indices
                    await self.scan_indices()

                    # Wait for next scan
                    logger.info(
                        f"\n⏳ Waiting {SCAN_INTERVAL//60} minutes "
                        f"until next scan...\n"
                    )
                    await asyncio.sleep(SCAN_INTERVAL)

                else:
                    # Market is closed
                    now = datetime.now(IST)
                    logger.info(
                        f"💤 Market closed. Current time: {now.strftime('%H:%M:%S')} | "
                        f"Checking again in 1 minute..."
                    )
                    await asyncio.sleep(60)

            except KeyboardInterrupt:
                logger.info("\n🛑 Bot stopped by user (Ctrl+C)")
                break

            except Exception as e:
                logger.error(f"❌ Main loop error: {e}")
                traceback.print_exc()

                # Send error notification
                await self.notifier.send_error_notification(
                    f"Main loop error: {str(e)[:100]}"
                )

                # Wait before retry
                logger.info("⏳ Waiting 60 seconds before retry...")
                await asyncio.sleep(60)

# ==================== ENTRY POINT ====================
async def main():
    """Main entry point"""

    # Validate environment variables
    required_vars = [
        ('UPSTOX_ACCESS_TOKEN', UPSTOX_ACCESS_TOKEN),
        ('TELEGRAM_BOT_TOKEN', TELEGRAM_BOT_TOKEN),
        ('TELEGRAM_CHAT_ID', TELEGRAM_CHAT_ID),
        ('DEEPSEEK_API_KEY', DEEPSEEK_API_KEY),
        ('FINNHUB_API_KEY', FINNHUB_API_KEY)
    ]

    missing_vars = [name for name, value in required_vars if not value]

    if missing_vars:
        logger.critical(
            f"❌ CRITICAL ERROR: Missing environment variables: "
            f"{', '.join(missing_vars)}"
        )
        logger.critical("Please set all required environment variables and restart.")
        return

    # Start bot
    logger.info("🚀 Starting Hybrid Trading Bot v19.0 (Holiday-Aware)...")
    bot = HybridBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Bot shutdown complete. Goodbye!")
    except Exception as e:
        logger.critical(f"❌ Fatal error: {e}")
        traceback.print_exc()
