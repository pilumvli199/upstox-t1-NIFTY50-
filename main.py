"""
🚀 NIFTY 50 OPTIONS BOT v6.3 PRO
==================================
Platform  : NSE via Upstox API v2
Asset     : NIFTY 50 Weekly Options
Lot Size  : 75 (updated Apr 2024)
Updated   : Feb 2026

✅ v6.3 CHANGES:
- POLLING: 5-min → 3-min (NSE OI updates every ~3 min)
- OI WINDOW: 30-min removed → 15-min only (cache[-5] = 15-min at 3-min polling)
- ADAPTIVE CACHE: cache[-5] preferred, fallback [-3] → [-1] if cache small
- STRONG OI DIRECT AI: PUT/CALL OI >20% + Vol >25% → lगech AI call, no Phase wait
- CANDLE RESAMPLE: 1min → 3min (was 5min, matches polling)
- OPTION CHAIN VWAP: total CE+PE volume from snapshots (fixes ₹0 bug for index)
- ANALYSIS CYCLE: every 5th cycle = 15-min (was 30-min)
- STRIKE-WISE COMPARE: ATM shift safe (compares per-strike, not ATM-to-ATM)
- START: 9:25 IST compatible (adaptive cache handles small cache gracefully)

✅ v6.2 FIXES KEPT:
- Phase 1+2 same-cycle bug fixed (MIN_P1_P2_GAP = 4 min)
- VWAP volume=0 fallback (now replaced with option chain volume)
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import logging
import os
import pytz
import time as time_module

# ============================================================
#  CONFIGURATION
# ============================================================
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_TOKEN")
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN",  "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID",     "YOUR_CHAT_ID")
DEEPSEEK_API_KEY    = os.getenv("DEEPSEEK_API_KEY",     "YOUR_DEEPSEEK_KEY")

UPSTOX_BASE    = "https://api.upstox.com/v2"
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

LOT_SIZE        = 75
STRIKE_INTERVAL = 50
ATM_RANGE       = 4          # ±4 strikes = ±200 pts (9 strikes total)

# v6.3: 3-min polling
SNAPSHOT_INTERVAL = 3 * 60
CANDLE_COUNT      = 20

# Cache: 3-min snapshots, 72 = 3.6 hrs
CACHE_SIZE = 72

# v6.3: OI comparison windows (in cache indices at 3-min polling)
# cache[-1] = 3-min ago
# cache[-3] = 9-min ago
# cache[-5] = 15-min ago  ← primary
WINDOW_SHORT  = 1   # 3-min ago
WINDOW_MEDIUM = 5   # 15-min ago  ← main window

# v6.3: Analysis every 5th cycle = 15-min
ANALYSIS_EVERY_N = 5

# OI thresholds
MIN_OI_CHANGE    = 8.0
STRONG_OI_CHANGE = 15.0
MIN_VOLUME_CHG   = 15.0
PCR_BULL         = 1.2
PCR_BEAR         = 0.8

# v6.3: Strong OI → direct AI trigger thresholds
STRONG_OI_DIRECT_PCT = 20.0   # % OI change in 15-min
STRONG_VOL_PCT       = 25.0   # % volume change in 15-min
STRONG_OI_COOLDOWN   = 10 * 60  # 10-min cooldown between strong OI AI calls

MIN_CONFIDENCE = 7

# Phase detection
PHASE1_OI_BUILD_PCT   = 6.0
PHASE1_VOL_MAX_PCT    = 12.0
PHASE2_VOL_SPIKE_PCT  = 15.0
PHASE2_OI_MIN_PCT     = 3.0
PHASE3_PRICE_MOVE_PCT = 0.25

# Alert thresholds
OI_ALERT_PCT  = 12.0
VOL_SPIKE_PCT = 25.0
PCR_ALERT_PCT = 10.0
ATM_PROX_PTS  = 50

# Weights
ATM_WEIGHT      = 3.0
NEAR_ATM_WEIGHT = 2.0
FAR_WEIGHT      = 1.0
MTF_OVERALL_THRESHOLD = 8

MAX_RETRIES      = 3
API_DELAY        = 0.3
DEEPSEEK_TIMEOUT = 45

IST = pytz.timezone("Asia/Kolkata")
MARKET_START = (9, 15)
MARKET_END   = (15, 30)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
#  DATA STRUCTURES
# ============================================================

@dataclass
class OISnapshot:
    strike:    int
    ce_oi:     float
    pe_oi:     float
    ce_volume: float
    pe_volume: float
    ce_ltp:    float
    pe_ltp:    float
    ce_iv:     float
    pe_iv:     float
    pcr:       float
    timestamp: datetime


@dataclass
class MarketSnapshot:
    timestamp:    datetime
    spot_price:   float
    atm_strike:   int
    expiry:       str
    strikes_oi:   Dict[int, OISnapshot]
    overall_pcr:  float
    total_ce_oi:  float
    total_pe_oi:  float
    total_ce_vol: float
    total_pe_vol: float


@dataclass
class PhaseSignal:
    phase:            int
    dominant_side:    str
    direction:        str
    oi_change_pct:    float
    vol_change_pct:   float
    price_change_pct: float
    atm_strike:       int
    spot_price:       float
    confidence:       float
    message:          str


@dataclass
class TrendInfo:
    day_trend:      str
    intraday_trend: str
    trend_3min:     str
    vwap:           float
    spot_vs_vwap:   str
    all_agree:      bool
    summary:        str


@dataclass
class PriceActionInsight:
    price_change_3m:   float
    price_change_15m:  float
    price_momentum:    str
    vol_rolling_avg:   float
    vol_spike_ratio:   float
    oi_vol_corr:       float
    support_levels:    List[float]
    resistance_levels: List[float]
    trend_strength:    float
    triple_confirmed:  bool
    trend:             TrendInfo


@dataclass
class StrikeAnalysis:
    strike:         int
    is_atm:         bool
    distance_atm:   int
    weight:         float
    ce_oi:          float
    pe_oi:          float
    ce_volume:      float
    pe_volume:      float
    ce_ltp:         float
    pe_ltp:         float
    ce_iv:          float
    pe_iv:          float
    # 3-min changes
    ce_oi_3:        float
    pe_oi_3:        float
    ce_vol_3:       float
    pe_vol_3:       float
    # 15-min changes
    ce_oi_15:       float
    pe_oi_15:       float
    ce_vol_15:      float
    pe_vol_15:      float
    pcr_ch_15:      float
    pcr:            float
    ce_action:      str
    pe_action:      str
    tf3_signal:     str
    tf15_signal:    str
    mtf_confirmed:  bool
    vol_confirms:   bool
    vol_strength:   str
    is_support:     bool
    is_resistance:  bool
    bull_strength:  float
    bear_strength:  float
    recommendation: str
    confidence:     float


@dataclass
class SupportResistance:
    support_strike:     int
    support_put_oi:     float
    resistance_strike:  int
    resistance_call_oi: float
    near_support:       bool
    near_resistance:    bool


# ============================================================
#  SINGLE CACHE (3-min snapshots)
# ============================================================

class SnapshotCache:
    """
    v6.3: Single cache at 3-min polling.
    cache[-1] = 3-min ago
    cache[-5] = 15-min ago (primary comparison window)
    Adaptive: uses best available window if cache is small (early market).
    """
    def __init__(self):
        self._cache = deque(maxlen=CACHE_SIZE)
        self._lock  = asyncio.Lock()

    async def add(self, snap: MarketSnapshot):
        async with self._lock:
            self._cache.append(snap)
        logger.info(
            f"📦 Cache: {len(self._cache)}/{CACHE_SIZE} | "
            f"PCR:{snap.overall_pcr:.2f} | Spot:{snap.spot_price:,.0f}"
        )

    async def get_ago(self, n: int) -> Optional[MarketSnapshot]:
        """Get snapshot n indices ago. cache[-1] = most recent."""
        async with self._lock:
            lst = list(self._cache)
            # lst[-1] = latest, lst[-(n+1)] = n ago
            idx = len(lst) - 1 - n
            return lst[idx] if idx >= 0 else None

    async def get_best_prev(self) -> Tuple[Optional[MarketSnapshot], int]:
        """
        Adaptive cache: returns best available previous snapshot.
        Preferred: 15-min ago (cache[-5])
        Fallback:  9-min ago (cache[-3])
        Minimum:   3-min ago (cache[-1])
        Returns (snapshot, actual_minutes_ago)
        """
        async with self._lock:
            lst = list(self._cache)
            n   = len(lst)
            if n > WINDOW_MEDIUM:    # >= 6 snapshots → 15-min
                return lst[-(WINDOW_MEDIUM+1)], 15
            elif n > 3:              # 4-5 snapshots → 9-min
                return lst[-4], 9
            elif n > 1:              # 2-3 snapshots → 3-6 min
                return lst[-2], 3
            return None, 0

    async def get_recent(self, n: int) -> List[MarketSnapshot]:
        async with self._lock:
            lst = list(self._cache)
            return lst[-n:] if len(lst) >= n else lst

    async def get_short(self) -> Optional[MarketSnapshot]:
        """3-min ago snapshot"""
        return await self.get_ago(WINDOW_SHORT)

    def size(self) -> int:
        return len(self._cache)

    def has_data(self) -> bool:
        return len(self._cache) >= 2


# ============================================================
#  UPSTOX CLIENT
# ============================================================

class UpstoxClient:

    def __init__(self, token: str):
        self.token   = token
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept":        "application/json"
        }

    async def init(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )

    async def close(self):
        if self.session:
            await self.session.close()

    async def _get(self, url: str, params: Dict = None) -> Optional[Dict]:
        for attempt in range(MAX_RETRIES):
            try:
                async with self.session.get(url, params=params) as r:
                    if r.status == 200:
                        return await r.json()
                    if r.status == 429:
                        await asyncio.sleep((attempt + 1) * 5)
                        continue
                    txt = await r.text()
                    logger.warning(f"⚠️ {r.status}: {txt[:80]}")
                    return None
            except aiohttp.ClientConnectorError:
                logger.error(f"❌ Network ({attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"❌ Request: {e}")
                await asyncio.sleep(2)
        return None

    async def get_nearest_expiry(self) -> Optional[str]:
        data = await self._get(
            f"{UPSTOX_BASE}/option/contract",
            params={"instrument_key": INSTRUMENT_KEY}
        )
        if not data or data.get("status") != "success":
            return None

        now_ist = datetime.now(IST)
        cutoff  = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        today   = now_ist.date()

        expiries = sorted(set(
            c.get("expiry") for c in data.get("data", []) if c.get("expiry")
        ))
        for exp in expiries:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            if exp_date < today: continue
            if exp_date == today and now_ist >= cutoff: continue
            logger.info(f"📅 Expiry: {exp}")
            return exp
        return None

    async def _fetch_raw_candles(self, resolution: str) -> pd.DataFrame:
        """
        Upstox V2 supports: 1minute, 30minute only.
        """
        url  = f"{UPSTOX_BASE}/historical-candle/intraday/{INSTRUMENT_KEY}/{resolution}"
        data = await self._get(url)
        if not data or data.get("status") != "success":
            return pd.DataFrame()

        raw = data.get("data", {}).get("candles", [])
        if not raw:
            return pd.DataFrame()

        rows = []
        for c in raw:
            try:
                rows.append({
                    "timestamp": pd.to_datetime(c[0]),
                    "open":      float(c[1]),
                    "high":      float(c[2]),
                    "low":       float(c[3]),
                    "close":     float(c[4]),
                    "volume":    int(c[5]) if len(c) > 5 else 0
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).set_index("timestamp").sort_index()

    async def get_candles(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        2 API calls (Upstox V2):
        - 1min raw → resample to 3min (matches polling)
        - 30min direct → day context, S/R
        Returns: (df_1m, df_3m, df_30m)
        """
        df_1m_raw = await self._fetch_raw_candles("1minute")
        await asyncio.sleep(API_DELAY)
        df_30m_raw = await self._fetch_raw_candles("30minute")

        df_1m = df_1m_raw.tail(CANDLE_COUNT) if not df_1m_raw.empty else pd.DataFrame()

        # Resample 1min → 3min
        if not df_1m_raw.empty:
            df_3m = df_1m_raw.resample("3min").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum"
            }).dropna().tail(CANDLE_COUNT)
            logger.info(f"📊 1min→3min: {len(df_3m)} candles")
        else:
            df_3m = pd.DataFrame()

        df_30m = df_30m_raw.tail(CANDLE_COUNT) if not df_30m_raw.empty else pd.DataFrame()
        logger.info(f"📊 30min: {len(df_30m)} candles")

        return df_1m, df_3m, df_30m

    async def fetch_snapshot(self, expiry: str) -> Optional[MarketSnapshot]:
        url  = f"{UPSTOX_BASE}/option/chain"
        data = await self._get(url, params={
            "instrument_key": INSTRUMENT_KEY,
            "expiry_date":    expiry
        })
        if not data or data.get("status") != "success":
            return None

        chain = data.get("data", [])
        if not chain:
            return None

        spot = 0.0
        for item in chain:
            spot = float(item.get("underlying_spot_price", 0))
            if spot > 0:
                break
        if spot <= 0:
            return None

        atm   = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
        min_s = atm - ATM_RANGE * STRIKE_INTERVAL
        max_s = atm + ATM_RANGE * STRIKE_INTERVAL

        strikes_oi: Dict[int, OISnapshot] = {}
        t_ce_oi = t_pe_oi = t_ce_vol = t_pe_vol = 0.0

        for item in chain:
            strike = int(item.get("strike_price", 0))
            if not (min_s <= strike <= max_s):
                continue

            ce = item.get("call_options", {}).get("market_data", {})
            pe = item.get("put_options",  {}).get("market_data", {})
            cg = item.get("call_options", {}).get("option_greeks", {})
            pg = item.get("put_options",  {}).get("option_greeks", {})

            ce_oi  = float(ce.get("oi",     0) or 0)
            pe_oi  = float(pe.get("oi",     0) or 0)
            ce_vol = float(ce.get("volume", 0) or 0)
            pe_vol = float(pe.get("volume", 0) or 0)
            ce_ltp = float(ce.get("ltp",    0) or 0)
            pe_ltp = float(pe.get("ltp",    0) or 0)
            ce_iv  = float(cg.get("iv",     0) or 0) * 100
            pe_iv  = float(pg.get("iv",     0) or 0) * 100
            pcr    = (pe_oi / ce_oi) if ce_oi > 0 else 0.0

            t_ce_oi  += ce_oi;  t_pe_oi  += pe_oi
            t_ce_vol += ce_vol; t_pe_vol += pe_vol

            strikes_oi[strike] = OISnapshot(
                strike=strike, ce_oi=ce_oi, pe_oi=pe_oi,
                ce_volume=ce_vol, pe_volume=pe_vol,
                ce_ltp=ce_ltp, pe_ltp=pe_ltp,
                ce_iv=ce_iv, pe_iv=pe_iv, pcr=pcr,
                timestamp=datetime.now(IST)
            )

        if not strikes_oi:
            return None

        overall_pcr = (t_pe_oi / t_ce_oi) if t_ce_oi > 0 else 0.0
        logger.info(
            f"💰 NIFTY:{spot:,.2f} | ATM:{atm} | "
            f"Strikes:{len(strikes_oi)} | PCR:{overall_pcr:.2f}"
        )
        return MarketSnapshot(
            timestamp=datetime.now(IST),
            spot_price=spot, atm_strike=atm, expiry=expiry,
            strikes_oi=strikes_oi, overall_pcr=overall_pcr,
            total_ce_oi=t_ce_oi, total_pe_oi=t_pe_oi,
            total_ce_vol=t_ce_vol, total_pe_vol=t_pe_vol
        )


# ============================================================
#  TREND CALCULATOR
# ============================================================

class TrendCalculator:

    @staticmethod
    def vwap_from_snapshots(snaps: List[MarketSnapshot]) -> float:
        """
        v6.3: Option chain volume VWAP.
        Fixes ₹0 bug — NSE index candles have volume=0.
        Uses total CE+PE volume from each snapshot as weight.
        """
        if not snaps:
            return 0.0
        total_vol = sum(s.total_ce_vol + s.total_pe_vol for s in snaps)
        if total_vol <= 0:
            # Fallback: simple price average
            return float(np.mean([s.spot_price for s in snaps]))
        weighted = sum(
            s.spot_price * (s.total_ce_vol + s.total_pe_vol)
            for s in snaps
        )
        return weighted / total_vol

    @staticmethod
    def trend_from_snaps(snaps: List[MarketSnapshot]) -> str:
        if len(snaps) < 3:
            return "SIDEWAYS"
        prices = [s.spot_price for s in snaps]
        if prices[-1] > prices[-3] * 1.001:  return "UPTREND"
        if prices[-1] < prices[-3] * 0.999:  return "DOWNTREND"
        return "SIDEWAYS"

    @staticmethod
    def trend_from_df(df: pd.DataFrame) -> str:
        if df.empty or len(df) < 3:
            return "SIDEWAYS"
        c = df["close"].values
        if c[-1] > c[-3] * 1.001:  return "UPTREND"
        if c[-1] < c[-3] * 0.999:  return "DOWNTREND"
        return "SIDEWAYS"

    @staticmethod
    def calculate(snaps: List[MarketSnapshot],
                  df_3m: pd.DataFrame, df_30m: pd.DataFrame) -> TrendInfo:
        spot      = snaps[-1].spot_price if snaps else 0.0
        vwap_val  = TrendCalculator.vwap_from_snapshots(snaps)
        day_t     = TrendCalculator.trend_from_df(df_30m)
        intra_t   = TrendCalculator.trend_from_df(df_3m)
        t3        = TrendCalculator.trend_from_snaps(snaps[-5:] if len(snaps) >= 5 else snaps)
        spot_vwap = "ABOVE" if vwap_val > 0 and spot > vwap_val else "BELOW"
        tfs       = [day_t, intra_t, t3]
        all_agree = len(set(tfs)) == 1 and tfs[0] != "SIDEWAYS"

        if all(t == "UPTREND"   for t in tfs):
            summary = f"📈 ALL UPTREND | Spot {spot_vwap} VWAP ₹{vwap_val:.0f}"
        elif all(t == "DOWNTREND" for t in tfs):
            summary = f"📉 ALL DOWNTREND | Spot {spot_vwap} VWAP ₹{vwap_val:.0f}"
        else:
            summary = f"Mixed | Spot {spot_vwap} VWAP ₹{vwap_val:.0f}"

        return TrendInfo(day_t, intra_t, t3, vwap_val, spot_vwap, all_agree, summary)


# ============================================================
#  PRICE ACTION CALCULATOR
# ============================================================

class PriceActionCalculator:

    @staticmethod
    def calculate(snaps: List[MarketSnapshot],
                  df_3m: pd.DataFrame, df_30m: pd.DataFrame) -> PriceActionInsight:
        trend = TrendCalculator.calculate(snaps, df_3m, df_30m)

        if len(snaps) < 2:
            return PriceActionCalculator._empty(trend)

        prices = np.array([s.spot_price for s in snaps])
        curr   = prices[-1]

        def pct(ago: int) -> float:
            return ((curr - prices[-(ago+1)]) / prices[-(ago+1)] * 100
                    if len(prices) > ago else 0.0)

        p3m  = pct(1)   # 3-min price change
        p15m = pct(5)   # 15-min price change
        momentum = "BULLISH" if p3m > 0.15 else "BEARISH" if p3m < -0.15 else "NEUTRAL"

        # Volume spike from option chain (CE+PE vol)
        vols        = np.array([s.total_ce_vol + s.total_pe_vol for s in snaps])
        vol_rolling = float(np.mean(vols[:-1])) if len(vols) > 1 else float(vols[-1])
        vol_spike   = float(vols[-1] / vol_rolling) if vol_rolling > 0 else 1.0

        # OI-Vol correlation
        ce_ois   = np.array([s.total_ce_oi for s in snaps])
        pe_ois   = np.array([s.total_pe_oi for s in snaps])
        oi_total = ce_ois + pe_ois
        oi_vol_c = float(np.corrcoef(oi_total, vols)[0, 1]) if (
            len(oi_total) > 2 and np.std(oi_total) > 0 and np.std(vols) > 0
        ) else 0.0

        # S/R from 30min candles
        supports, resistances = [], []
        if not df_30m.empty and len(df_30m) >= 5:
            lws = df_30m["low"].values
            hws = df_30m["high"].values
            for i in range(1, len(lws) - 1):
                if lws[i] < lws[i-1] and lws[i] < lws[i+1]:
                    supports.append(float(lws[i]))
                if hws[i] > hws[i-1] and hws[i] > hws[i+1]:
                    resistances.append(float(hws[i]))
            supports    = sorted(supports,    key=lambda x: abs(x - curr))[:3]
            resistances = sorted(resistances, key=lambda x: abs(x - curr))[:3]

        # Trend strength score
        ts = 0.0
        if abs(p3m) >= 0.3:   ts += 3.0
        elif abs(p3m) >= 0.15: ts += 1.5
        if vol_spike >= 1.5:   ts += 3.0
        elif vol_spike >= 1.2: ts += 1.5
        oi_ch = ((oi_total[-1] - oi_total[0]) / oi_total[0] * 100) if oi_total[0] > 0 else 0
        if abs(oi_ch) >= 10:  ts += 4.0
        elif abs(oi_ch) >= 5: ts += 2.0

        price_bull = p3m > 0.15
        price_bear = p3m < -0.15
        oi_bull    = len(pe_ois) > 1 and pe_ois[-1] > pe_ois[0]
        oi_bear    = len(ce_ois) > 1 and ce_ois[-1] > ce_ois[0]
        vol_ok     = vol_spike >= 1.2
        triple     = ((price_bull and oi_bull and vol_ok) or
                      (price_bear and oi_bear and vol_ok))

        return PriceActionInsight(
            price_change_3m=round(p3m, 3),
            price_change_15m=round(p15m, 3),
            price_momentum=momentum,
            vol_rolling_avg=round(vol_rolling, 0),
            vol_spike_ratio=round(vol_spike, 2),
            oi_vol_corr=round(oi_vol_c, 2),
            support_levels=supports,
            resistance_levels=resistances,
            trend_strength=round(min(10.0, ts), 1),
            triple_confirmed=triple,
            trend=trend
        )

    @staticmethod
    def _empty(trend: TrendInfo) -> PriceActionInsight:
        return PriceActionInsight(0, 0, "NEUTRAL", 0, 1.0, 0, [], [], 0, False, trend)


# ============================================================
#  MTF OI ANALYZER (3-min + 15-min)
# ============================================================

class MTFAnalyzer:
    """
    v6.3: 2 timeframes only (30-min removed):
    - 3-min  (cache short window)
    - 15-min (cache medium window, adaptive)
    Strike-wise comparison — safe against ATM shifts.
    """

    def __init__(self, cache: SnapshotCache):
        self.cache = cache

    @staticmethod
    def _pct(c: float, p: float) -> float:
        return ((c - p) / p * 100) if p > 0 else 0.0

    @staticmethod
    def _action(ch: float) -> str:
        if ch >= 8:  return "BUILDING"
        if ch <= -8: return "UNWINDING"
        return "NEUTRAL"

    @staticmethod
    def _tf_signal(ce: float, pe: float, cv: float, pv: float) -> str:
        if pe >= MIN_OI_CHANGE and pv >= MIN_VOLUME_CHG: return "BULLISH"
        if ce >= MIN_OI_CHANGE and cv >= MIN_VOLUME_CHG: return "BEARISH"
        if pe <= -MIN_OI_CHANGE: return "BEARISH"
        if ce <= -MIN_OI_CHANGE: return "BULLISH"
        return "NEUTRAL"

    @staticmethod
    def _vol_confirm(oi_ch: float, vol_ch: float) -> Tuple[bool, str]:
        if oi_ch > 8  and vol_ch > MIN_VOLUME_CHG: return True,  "STRONG"
        if oi_ch > 4  and vol_ch > 10:             return True,  "MODERATE"
        if abs(oi_ch) < 4 and abs(vol_ch) < 4:    return True,  "WEAK"
        return False, "WEAK"

    def _strength(self, ce15: float, pe15: float, cv15: float, pv15: float,
                  weight: float, mtf: bool) -> Tuple[float, float]:
        bull = bear = 0.0
        boost = 1.5 if mtf else 0.8
        if   pe15 >= STRONG_OI_CHANGE: bull = 9.0
        elif pe15 >= MIN_OI_CHANGE:    bull = 7.0
        elif pe15 >= 5:                bull = 4.0
        if   ce15 >= STRONG_OI_CHANGE: bear = 9.0
        elif ce15 >= MIN_OI_CHANGE:    bear = 7.0
        elif ce15 >= 5:                bear = 4.0
        if pe15 <= -STRONG_OI_CHANGE:  bear = max(bear, 8.0)
        elif pe15 <= -MIN_OI_CHANGE:   bear = max(bear, 6.0)
        if ce15 <= -STRONG_OI_CHANGE:  bull = max(bull, 8.0)
        elif ce15 <= -MIN_OI_CHANGE:   bull = max(bull, 6.0)
        return min(10.0, bull * weight * boost), min(10.0, bear * weight * boost)

    async def analyze(self, current: MarketSnapshot) -> Dict:
        s3          = await self.cache.get_short()                  # 3-min ago
        s15, mins15 = await self.cache.get_best_prev()              # 15-min ago (adaptive)

        if not s3:
            return {"available": False, "reason": "Building cache..."}

        analyses: List[StrikeAnalysis] = []

        for strike in sorted(current.strikes_oi.keys()):
            c   = current.strikes_oi[strike]
            p3  = s3.strikes_oi.get(strike)   # 3-min prev, same strike
            p15 = s15.strikes_oi.get(strike) if s15 else None  # 15-min prev

            # 3-min change (per strike — ATM shift safe)
            ce3  = self._pct(c.ce_oi,     p3.ce_oi     if p3 else 0)
            pe3  = self._pct(c.pe_oi,     p3.pe_oi     if p3 else 0)
            cv3  = self._pct(c.ce_volume, p3.ce_volume  if p3 else 0)
            pv3  = self._pct(c.pe_volume, p3.pe_volume  if p3 else 0)

            # 15-min change (adaptive window)
            ce15 = self._pct(c.ce_oi,     p15.ce_oi     if p15 else 0)
            pe15 = self._pct(c.pe_oi,     p15.pe_oi     if p15 else 0)
            cv15 = self._pct(c.ce_volume, p15.ce_volume  if p15 else 0)
            pv15 = self._pct(c.pe_volume, p15.pe_volume  if p15 else 0)
            pc15 = self._pct(c.pcr,       p15.pcr        if p15 else 0)

            is_atm = (strike == current.atm_strike)
            dist   = abs(strike - current.atm_strike)
            weight = (ATM_WEIGHT if is_atm else
                      NEAR_ATM_WEIGHT if dist <= STRIKE_INTERVAL else FAR_WEIGHT)

            tf3  = self._tf_signal(ce3,  pe3,  cv3,  pv3)
            tf15 = self._tf_signal(ce15, pe15, cv15, pv15)
            mtf  = (tf3 == tf15 and tf3 != "NEUTRAL")

            vc, vs = self._vol_confirm((ce15 + pe15) / 2, (cv15 + pv15) / 2)
            bull, bear = self._strength(ce15, pe15, cv15, pv15, weight, mtf)
            if mtf:
                bull = min(10.0, bull * 1.3)
                bear = min(10.0, bear * 1.3)

            if   bull >= 7 and bull > bear: rec, conf = "STRONG_CALL", bull
            elif bear >= 7 and bear > bull: rec, conf = "STRONG_PUT",  bear
            else:                           rec, conf = "WAIT", max(bull, bear)

            analyses.append(StrikeAnalysis(
                strike=strike, is_atm=is_atm, distance_atm=dist, weight=weight,
                ce_oi=c.ce_oi, pe_oi=c.pe_oi,
                ce_volume=c.ce_volume, pe_volume=c.pe_volume,
                ce_ltp=c.ce_ltp, pe_ltp=c.pe_ltp,
                ce_iv=c.ce_iv, pe_iv=c.pe_iv,
                ce_oi_3=ce3, pe_oi_3=pe3, ce_vol_3=cv3, pe_vol_3=pv3,
                ce_oi_15=ce15, pe_oi_15=pe15, ce_vol_15=cv15, pe_vol_15=pv15,
                pcr_ch_15=pc15,
                pcr=c.pcr, ce_action=self._action(ce15), pe_action=self._action(pe15),
                tf3_signal=tf3, tf15_signal=tf15,
                mtf_confirmed=mtf, vol_confirms=vc, vol_strength=vs,
                is_support=False, is_resistance=False,
                bull_strength=bull, bear_strength=bear,
                recommendation=rec, confidence=conf
            ))

        sr = self._find_sr(current, analyses)
        for sa in analyses:
            sa.is_support    = (sa.strike == sr.support_strike)
            sa.is_resistance = (sa.strike == sr.resistance_strike)

        prev_pcr  = s15.overall_pcr if s15 else current.overall_pcr
        pcr_trend = "BULLISH" if current.overall_pcr > prev_pcr else "BEARISH"
        pcr_ch    = self._pct(current.overall_pcr, prev_pcr)
        tb        = sum(sa.bull_strength for sa in analyses)
        tr_b      = sum(sa.bear_strength for sa in analyses)
        overall   = ("BULLISH" if tb > tr_b and tb >= MTF_OVERALL_THRESHOLD
                     else "BEARISH" if tr_b > tb and tr_b >= MTF_OVERALL_THRESHOLD
                     else "NEUTRAL")

        return {
            "available":       True,
            "strike_analyses": analyses,
            "sr":              sr,
            "overall":         overall,
            "total_bull":      tb,
            "total_bear":      tr_b,
            "overall_pcr":     current.overall_pcr,
            "pcr_trend":       pcr_trend,
            "pcr_ch_pct":      pcr_ch,
            "window_mins":     mins15,
            "has_strong":      any(
                sa.mtf_confirmed and sa.confidence >= MIN_CONFIDENCE
                for sa in analyses
            )
        }

    def _find_sr(self, current: MarketSnapshot,
                 analyses: List[StrikeAnalysis]) -> SupportResistance:
        mp = max(analyses, key=lambda x: x.pe_oi, default=None)
        mc = max(analyses, key=lambda x: x.ce_oi, default=None)
        sup = mp.strike if mp else current.atm_strike
        res = mc.strike if mc else current.atm_strike
        return SupportResistance(
            support_strike=sup, support_put_oi=mp.pe_oi if mp else 0,
            resistance_strike=res, resistance_call_oi=mc.ce_oi if mc else 0,
            near_support=abs(current.spot_price - sup) <= ATM_PROX_PTS,
            near_resistance=abs(current.spot_price - res) <= ATM_PROX_PTS
        )


# ============================================================
#  STRONG OI CHECKER — v6.3 NEW
# ============================================================

class StrongOIChecker:
    """
    v6.3 NEW: Direct AI trigger on strong OI buildup.
    Does NOT wait for Phase 1→2→3.
    Fires when: PUT or CALL OI change (15-min) > STRONG_OI_DIRECT_PCT
                AND volume confirms (>STRONG_VOL_PCT)
                AND both agree on direction

    This fixes the core miss problem:
    Old: Phase system missed 10:14 alert (PUT OI +61.7%) because price was down
    New: Strong OI → immediate AI call regardless of price direction
    """
    def __init__(self):
        self._last: Dict[str, float] = {}

    def _can(self, k: str) -> bool:
        return (time_module.time() - self._last.get(k, 0)) >= STRONG_OI_COOLDOWN

    def _mark(self, k: str):
        self._last[k] = time_module.time()

    @staticmethod
    def _pct(c: float, p: float) -> float:
        return ((c - p) / p * 100) if p > 0 else 0.0

    async def check(self, current: MarketSnapshot,
                    prev: Optional[MarketSnapshot],
                    pa: PriceActionInsight) -> Optional[Dict]:
        """
        Returns signal dict if strong OI detected, else None.
        """
        if not prev:
            return None

        ac = current.strikes_oi.get(current.atm_strike)
        ap = prev.strikes_oi.get(current.atm_strike)
        if not ac or not ap:
            return None

        # Per-strike percentage change
        ce_oi_ch  = self._pct(ac.ce_oi,    ap.ce_oi)
        pe_oi_ch  = self._pct(ac.pe_oi,    ap.pe_oi)
        ce_vol_ch = self._pct(ac.ce_volume, ap.ce_volume)
        pe_vol_ch = self._pct(ac.pe_volume, ap.pe_volume)

        # Also check total OI (all strikes combined)
        total_ce_ch = self._pct(current.total_ce_oi,  prev.total_ce_oi)
        total_pe_ch = self._pct(current.total_pe_oi,  prev.total_pe_oi)
        total_cv_ch = self._pct(current.total_ce_vol, prev.total_ce_vol)
        total_pv_ch = self._pct(current.total_pe_vol, prev.total_pe_vol)

        # Detect strong OI event
        # Use weighted: ATM strike + total OI both must confirm
        strong_put  = (pe_oi_ch  >= STRONG_OI_DIRECT_PCT and
                       pe_vol_ch >= STRONG_VOL_PCT and
                       total_pe_ch >= 10.0)   # total pn building
        strong_call = (ce_oi_ch  >= STRONG_OI_DIRECT_PCT and
                       ce_vol_ch >= STRONG_VOL_PCT and
                       total_ce_ch >= 10.0)

        if not (strong_put or strong_call):
            return None

        dom   = "PUT"  if (strong_put  and pe_oi_ch >= ce_oi_ch) else "CALL"
        dirn  = "BULLISH" if dom == "PUT" else "BEARISH"
        key   = f"STRONG_{dom}"

        if not self._can(key):
            return None

        self._mark(key)

        oi_ch  = pe_oi_ch  if dom == "PUT" else ce_oi_ch
        vol_ch = pe_vol_ch if dom == "PUT" else ce_vol_ch

        logger.info(
            f"🔥 STRONG OI DETECTED: {dom} OI={oi_ch:+.1f}% "
            f"Vol={vol_ch:+.1f}% → {dirn} → AI call!"
        )

        return {
            "dominant":   dom,
            "direction":  dirn,
            "oi_ch":      oi_ch,
            "vol_ch":     vol_ch,
            "total_pe":   total_pe_ch,
            "total_ce":   total_ce_ch,
            "atm_strike": current.atm_strike,
            "spot":       current.spot_price
        }


# ============================================================
#  PHASE DETECTOR (kept, runs in parallel)
# ============================================================

class PhaseDetector:

    COOLDOWN_P1 = 15 * 60
    COOLDOWN_P2 = 10 * 60
    COOLDOWN_P3 =  5 * 60
    MIN_P1_P2_GAP = 4 * 60   # v6.2 fix: same-cycle bug

    def __init__(self):
        self._last:   Dict[str, float] = {}
        self._p1_at   = 0.0
        self._p2_at   = 0.0
        self._p1_side = ""

    def _can(self, k: str, cd: int) -> bool:
        return (time_module.time() - self._last.get(k, 0)) >= cd

    def _mark(self, k: str):
        self._last[k] = time_module.time()

    @staticmethod
    def _pct(c: float, p: float) -> float:
        return ((c - p) / p * 100) if p > 0 else 0.0

    async def detect(self, curr: MarketSnapshot,
                     prev: Optional[MarketSnapshot],
                     pa: PriceActionInsight) -> List[PhaseSignal]:
        signals = []
        if not prev:
            return signals

        ac = curr.strikes_oi.get(curr.atm_strike)
        ap = prev.strikes_oi.get(curr.atm_strike)
        if not ac or not ap:
            return signals

        ce_oi_ch  = self._pct(ac.ce_oi,    ap.ce_oi)
        pe_oi_ch  = self._pct(ac.pe_oi,    ap.pe_oi)
        ce_vol_ch = self._pct(ac.ce_volume, ap.ce_volume)
        pe_vol_ch = self._pct(ac.pe_volume, ap.pe_volume)

        call_bld = ce_oi_ch >= PHASE1_OI_BUILD_PCT
        put_bld  = pe_oi_ch >= PHASE1_OI_BUILD_PCT
        if not (call_bld or put_bld):
            return signals

        dom   = "PUT" if (put_bld and pe_oi_ch >= ce_oi_ch) else "CALL"
        oi_ch = pe_oi_ch if dom == "PUT" else ce_oi_ch
        v_ch  = pe_vol_ch if dom == "PUT" else ce_vol_ch
        dirn  = "BULLISH" if dom == "PUT" else "BEARISH"
        now   = time_module.time()

        # Phase 1
        if (oi_ch >= PHASE1_OI_BUILD_PCT and abs(v_ch) < PHASE1_VOL_MAX_PCT
                and self._can("P1", self.COOLDOWN_P1)):
            self._p1_at = now; self._p1_side = dom; self._mark("P1")
            signals.append(PhaseSignal(
                phase=1, dominant_side=dom, direction=dirn,
                oi_change_pct=oi_ch, vol_change_pct=v_ch,
                price_change_pct=pa.price_change_3m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, oi_ch / 1.5),
                message=(
                    f"⚡ PHASE 1 - SMART MONEY\n\n"
                    f"NIFTY: {curr.spot_price:,.2f} | ATM: {curr.atm_strike}\n"
                    f"{dom} OI: {oi_ch:+.1f}% | Vol: {v_ch:+.1f}% (quiet)\n"
                    f"Signal: {dirn}\n{pa.trend.summary}\n\n"
                    f"⏳ Wait for Phase 2!\n"
                    f"{datetime.now(IST).strftime('%H:%M IST')}"
                )
            ))

        # Phase 2 — v6.2 fix: min gap
        p1_gap_ok = (now - self._p1_at) >= self.MIN_P1_P2_GAP
        p1_recent = self._p1_at > 0 and p1_gap_ok and (now - self._p1_at) < (25 * 60)
        if (pa.vol_spike_ratio >= (1 + PHASE2_VOL_SPIKE_PCT / 100)
                and oi_ch >= PHASE2_OI_MIN_PCT
                and p1_recent and self._p1_side == dom
                and self._can("P2", self.COOLDOWN_P2)):
            self._p2_at = now; self._mark("P2")
            signals.append(PhaseSignal(
                phase=2, dominant_side=dom, direction=dirn,
                oi_change_pct=oi_ch, vol_change_pct=v_ch,
                price_change_pct=pa.price_change_3m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, pa.vol_spike_ratio * 3),
                message=(
                    f"🔥 PHASE 2 - VOL SPIKE! MOVE IMMINENT\n\n"
                    f"NIFTY: {curr.spot_price:,.2f} | ATM: {curr.atm_strike}\n"
                    f"Volume: {pa.vol_spike_ratio:.1f}x avg | OI: {oi_ch:+.1f}%\n"
                    f"Signal: {dirn}\n{pa.trend.summary}\n\n"
                    f"👆 {'BUY CALL' if dirn=='BULLISH' else 'BUY PUT'} near {curr.atm_strike}\n"
                    f"{datetime.now(IST).strftime('%H:%M IST')}"
                )
            ))

        # Phase 3
        p2_recent = (now - self._p2_at) < (15 * 60)
        price_ok  = ((dirn == "BULLISH" and pa.price_change_3m >= PHASE3_PRICE_MOVE_PCT) or
                     (dirn == "BEARISH" and pa.price_change_3m <= -PHASE3_PRICE_MOVE_PCT))
        if (p2_recent and price_ok and pa.triple_confirmed
                and self._can("P3", self.COOLDOWN_P3)):
            self._mark("P3")
            signals.append(PhaseSignal(
                phase=3, dominant_side=dom, direction=dirn,
                oi_change_pct=oi_ch, vol_change_pct=v_ch,
                price_change_pct=pa.price_change_3m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, 7 + pa.trend_strength / 3),
                message=(
                    f"🚀 PHASE 3 - CONFIRMED! EXECUTE!\n\n"
                    f"NIFTY: {curr.spot_price:,.2f} ({pa.price_change_3m:+.2f}%/3m)\n"
                    f"ATM: {curr.atm_strike}\n"
                    f"Signal: {'BUY_CALL' if dirn=='BULLISH' else 'BUY_PUT'}\n"
                    f"✅ Triple: OI+Vol+Price confirmed!\n\n"
                    f"VWAP: ₹{pa.trend.vwap:.0f} | Spot {pa.trend.spot_vs_vwap}\n"
                    f"OI:{oi_ch:+.1f}% | Vol:{pa.vol_spike_ratio:.1f}x | "
                    f"Price:{pa.price_change_3m:+.2f}%\n"
                    f"{datetime.now(IST).strftime('%H:%M IST')}\n"
                    f"⏳ AI confirming..."
                )
            ))

        return signals


# ============================================================
#  STANDALONE ALERT CHECKER
# ============================================================

class AlertChecker:
    COOLDOWN = 15 * 60   # v6.3: 15-min cooldown (was 30-min)

    def __init__(self, cache: SnapshotCache, alerter):
        self.cache   = cache
        self.alerter = alerter
        self._last: Dict[str, float] = {}

    def _can(self, k: str) -> bool:
        return (time_module.time() - self._last.get(k, 0)) >= self.COOLDOWN

    def _mark(self, k: str):
        self._last[k] = time_module.time()

    @staticmethod
    def _pct(c: float, p: float) -> float:
        return ((c - p) / p * 100) if p > 0 else 0.0

    async def check_all(self, curr: MarketSnapshot):
        prev, _ = await self.cache.get_best_prev()
        if not prev:
            return
        await self._oi(curr, prev)
        await self._vol(curr, prev)
        await self._pcr(curr, prev)
        await self._prox(curr)

    async def _oi(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("OI"): return
        ac = curr.strikes_oi.get(curr.atm_strike)
        ap = prev.strikes_oi.get(curr.atm_strike)
        if not ac or not ap or ap.ce_oi == 0 or ap.pe_oi == 0: return
        ce = self._pct(ac.ce_oi, ap.ce_oi)
        pe = self._pct(ac.pe_oi, ap.pe_oi)
        if abs(ce) < OI_ALERT_PCT and abs(pe) < OI_ALERT_PCT: return
        txt = (
            f"📊 OI CHANGE ALERT\n\n"
            f"NIFTY: {curr.spot_price:,.2f} | ATM: {curr.atm_strike}\n\n"
            f"CALL OI: {ce:+.1f}% {'BUILDING 🔴' if ce > 0 else 'UNWINDING'}\n"
            f"PUT  OI: {pe:+.1f}% {'BUILDING 🟢' if pe > 0 else 'UNWINDING'}\n\n"
            f"PCR: {curr.overall_pcr:.2f}\n"
            f"{datetime.now(IST).strftime('%H:%M IST')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("OI")

    async def _vol(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("VOL"): return
        ac = curr.strikes_oi.get(curr.atm_strike)
        ap = prev.strikes_oi.get(curr.atm_strike)
        if not ac or not ap or ap.ce_volume == 0 or ap.pe_volume == 0: return
        cv = self._pct(ac.ce_volume, ap.ce_volume)
        pv = self._pct(ac.pe_volume, ap.pe_volume)
        if max(cv, pv) < VOL_SPIKE_PCT: return
        dom = "CALL" if cv >= pv else "PUT"
        txt = (
            f"⚡ VOLUME SPIKE ALERT\n\n"
            f"NIFTY: {curr.spot_price:,.2f} | ATM: {curr.atm_strike}\n\n"
            f"CALL Vol: {cv:+.1f}% | PUT Vol: {pv:+.1f}%\n"
            f"Dominant: {dom} → {'BEARISH 🔴' if dom=='CALL' else 'BULLISH 🟢'}\n"
            f"{datetime.now(IST).strftime('%H:%M IST')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("VOL")

    async def _pcr(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("PCR") or prev.overall_pcr <= 0: return
        ch = self._pct(curr.overall_pcr, prev.overall_pcr)
        if abs(ch) < PCR_ALERT_PCT: return
        txt = (
            f"📈 PCR CHANGE ALERT\n\n"
            f"NIFTY: {curr.spot_price:,.2f}\n\n"
            f"PCR: {prev.overall_pcr:.2f} → {curr.overall_pcr:.2f} ({ch:+.1f}%)\n"
            f"{'Bulls gaining 🟢' if ch > 0 else 'Bears gaining 🔴'}\n"
            f"{datetime.now(IST).strftime('%H:%M IST')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("PCR")

    async def _prox(self, curr: MarketSnapshot):
        if not self._can("PROX"): return
        max_pe = max(curr.strikes_oi.items(), key=lambda x: x[1].pe_oi, default=None)
        max_ce = max(curr.strikes_oi.items(), key=lambda x: x[1].ce_oi, default=None)
        for level, item, kind in [("SUPPORT", max_pe, "PUT"), ("RESISTANCE", max_ce, "CALL")]:
            if not item: continue
            strike, oi_s = item
            dist = abs(curr.spot_price - strike)
            if dist > ATM_PROX_PTS: continue
            oi_v = oi_s.pe_oi if kind == "PUT" else oi_s.ce_oi
            txt  = (
                f"🎯 PRICE NEAR {level}\n\n"
                f"NIFTY: {curr.spot_price:,.2f}\n"
                f"{level}: {strike} (OI: {oi_v:,.0f})\n"
                f"Distance: {dist:.0f} pts\n"
                f"{datetime.now(IST).strftime('%H:%M IST')}"
            )
            await self.alerter.send_raw(txt)
            self._mark("PROX")
            break


# ============================================================
#  CANDLESTICK PATTERN DETECTOR
# ============================================================

class PatternDetector:

    @staticmethod
    def detect(df: pd.DataFrame) -> List[Dict]:
        pats = []
        if df.empty or len(df) < 2: return pats
        for i in range(1, len(df)):
            c, p = df.iloc[i], df.iloc[i-1]
            bc  = abs(c.close - c.open)
            bp  = abs(p.close - p.open)
            rng = c.high - c.low
            if rng == 0: continue
            if (c.close > c.open and p.close < p.open
                    and c.open <= p.close and c.close >= p.open and bc > bp * 1.2):
                pats.append({"time": c.name, "pattern": "BULLISH_ENGULFING",
                             "type": "BULLISH", "strength": 8, "price": c.close})
            elif (c.close < c.open and p.close > p.open
                    and c.open >= p.close and c.close <= p.open and bc > bp * 1.2):
                pats.append({"time": c.name, "pattern": "BEARISH_ENGULFING",
                             "type": "BEARISH", "strength": 8, "price": c.close})
            else:
                lw = min(c.open, c.close) - c.low
                hw = c.high - max(c.open, c.close)
                if lw > bc * 2 and hw < bc * 0.3 and bc < rng * 0.35:
                    pats.append({"time": c.name, "pattern": "HAMMER",
                                 "type": "BULLISH", "strength": 7, "price": c.close})
                elif hw > bc * 2 and lw < bc * 0.3 and bc < rng * 0.35:
                    pats.append({"time": c.name, "pattern": "SHOOTING_STAR",
                                 "type": "BEARISH", "strength": 7, "price": c.close})
                elif bc < rng * 0.1:
                    pats.append({"time": c.name, "pattern": "DOJI",
                                 "type": "NEUTRAL", "strength": 4, "price": c.close})
        return pats[-5:]

    @staticmethod
    def sr(df: pd.DataFrame) -> Tuple[float, float]:
        if df.empty or len(df) < 5: return 0.0, 0.0
        d = df.tail(20)
        return float(d.low.min()), float(d.high.max())


# ============================================================
#  DEEPSEEK PROMPT BUILDER
# ============================================================

class PromptBuilder:

    @staticmethod
    def _candles(df: pd.DataFrame, label: str) -> str:
        if df.empty: return f"{label}: no data\n"
        out = f"\n{label} (TIME|O|H|L|C|DIR):\n"
        for ts, row in df.tail(CANDLE_COUNT).iterrows():
            t = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)[:5]
            d = "▲" if row.close > row.open else "▼" if row.close < row.open else "-"
            out += f"{t}|{row.open:.0f}|{row.high:.0f}|{row.low:.0f}|{row.close:.0f}|{d}\n"
        return out

    @staticmethod
    def build(snap: MarketSnapshot, oi: Dict, pa: PriceActionInsight,
              df_3m: pd.DataFrame, df_30m: pd.DataFrame,
              patterns: List[Dict], p_sup: float, p_res: float,
              trigger: str = "MTF_ANALYSIS",
              strong_oi: Optional[Dict] = None) -> str:

        now = datetime.now(IST).strftime("%H:%M IST")
        sr  = oi["sr"]
        pcr = oi["overall_pcr"]
        t   = pa.trend

        p  = "You are an expert NIFTY 50 options trader. Analyze everything and give precise signal.\n\n"
        p += f"=== MARKET | {now} | Expiry:{snap.expiry} | Lot:{LOT_SIZE} | Trigger:{trigger} ===\n"
        p += f"NIFTY: ₹{snap.spot_price:,.2f} | ATM:{snap.atm_strike} | PCR:{pcr:.2f}({oi['pcr_trend']} Δ{oi['pcr_ch_pct']:+.1f}%)\n"
        p += f"OI Support:{sr.support_strike} | OI Resistance:{sr.resistance_strike}\n"
        if sr.near_support:    p += "⚠️ NEAR OI SUPPORT!\n"
        if sr.near_resistance: p += "⚠️ NEAR OI RESISTANCE!\n"

        if strong_oi:
            p += f"\n=== 🔥 STRONG OI EVENT (Direct Trigger) ===\n"
            p += f"Dominant:{strong_oi['dominant']} OI:{strong_oi['oi_ch']:+.1f}% Vol:{strong_oi['vol_ch']:+.1f}%\n"
            p += f"Total PE:{strong_oi['total_pe']:+.1f}% | Total CE:{strong_oi['total_ce']:+.1f}%\n"
            p += f"Direction:{strong_oi['direction']}\n"

        p += f"\n=== TREND ===\n"
        p += f"Day(30m):{t.day_trend} | Intraday(3m):{t.intraday_trend} | Short:{t.trend_3min}\n"
        p += f"VWAP(OI-weighted):₹{t.vwap:.0f} | Spot {t.spot_vs_vwap} VWAP | All agree:{'✅' if t.all_agree else '❌'}\n"
        p += f"{t.summary}\n"

        p += f"\n=== PRICE ACTION ===\n"
        p += f"Price: 3m={pa.price_change_3m:+.2f}% | 15m={pa.price_change_15m:+.2f}%\n"
        p += f"Momentum:{pa.price_momentum} | Vol spike:{pa.vol_spike_ratio:.2f}x\n"
        p += f"Trend strength:{pa.trend_strength:.1f}/10 | Triple confirmed:{'✅' if pa.triple_confirmed else '❌'}\n"
        if pa.support_levels:    p += f"Price support: {', '.join(f'₹{s:.0f}' for s in pa.support_levels)}\n"
        if pa.resistance_levels: p += f"Price resistance: {', '.join(f'₹{r:.0f}' for r in pa.resistance_levels)}\n"

        p += f"\n=== OI ANALYSIS (3-min + {oi['window_mins']}-min window) ===\n"
        p += "STRIKE | W | CE_OI(3%/15%) | CE_VOL15% | CE_ACT | PE_OI(3%/15%) | PE_VOL15% | PE_ACT | PCR | TF3/TF15 | MTF | IV(CE/PE) | Bull | Bear\n"
        for sa in oi["strike_analyses"]:
            tag = "ATM" if sa.is_atm else ("SUP" if sa.is_support else ("RES" if sa.is_resistance else "   "))
            p += (
                f"{sa.strike}({tag}) W{sa.weight:.0f} | "
                f"CE:{sa.ce_oi_3:+.0f}%/{sa.ce_oi_15:+.0f}% V:{sa.ce_vol_15:+.0f}%({sa.ce_action}) | "
                f"PE:{sa.pe_oi_3:+.0f}%/{sa.pe_oi_15:+.0f}% V:{sa.pe_vol_15:+.0f}%({sa.pe_action}) | "
                f"PCR:{sa.pcr:.2f} | {sa.tf3_signal[:3]}/{sa.tf15_signal[:3]} | "
                f"MTF:{'✅' if sa.mtf_confirmed else '❌'} | "
                f"IV:{sa.ce_iv:.1f}%/{sa.pe_iv:.1f}% | "
                f"B:{sa.bull_strength:.0f} Br:{sa.bear_strength:.0f}\n"
            )

        p += PromptBuilder._candles(df_3m,  "3MIN CANDLES")
        p += PromptBuilder._candles(df_30m, "30MIN CANDLES")

        if patterns:
            p += "\n=== PATTERNS ===\n"
            for pat in patterns:
                ts = pat["time"].strftime("%H:%M") if hasattr(pat["time"], "strftime") else str(pat["time"])[:5]
                p += f"{ts}|{pat['pattern']}|{pat['type']}|{pat['strength']}/10|@₹{pat['price']:.0f}\n"

        if p_sup or p_res: p += f"\nCandle S/R: Sup=₹{p_sup:.0f} | Res=₹{p_res:.0f}\n"

        p += f"""
=== NIFTY RULES ===
OI: CALL OI↑+Vol↑=Resistance=BEARISH→BUY PUT | PUT OI↑+Vol↑=Support=BULLISH→BUY CALL
OI↑ Vol flat = TRAP! | MTF(3+15 agree)=HIGH confidence
STRONG OI (>20% in 15min): High probability directional move — DO NOT ignore
TREND: Trade with day trend | All TFs+VWAP agree = best entry
VWAP: Spot above=bullish bias | Spot below=bearish bias
PCR>{PCR_BULL}=bullish | PCR<{PCR_BEAR}=bearish
IV>20%=expensive, ATM only | IV<12%=cheap, ATM or 1-ITM
Lot={LOT_SIZE} | Strike=₹{STRIKE_INTERVAL} | SL=1 strike | Tgt=2 strikes
Entry: MTF+Volume+Trend all confirm

RESPOND ONLY JSON:
{{
  "signal": "BUY_CALL"|"BUY_PUT"|"WAIT",
  "primary_strike": {snap.atm_strike},
  "confidence": 0-10,
  "stop_loss_strike": 0,
  "target_strike": 0,
  "trend_analysis": {{"day": "", "intraday": "", "all_agree": true, "vwap_confirms": true, "note": ""}},
  "mtf": {{"tf3": "", "tf15": "", "confirmed": true}},
  "price_action": {{"momentum": "", "triple_confirmed": true, "confirms_signal": true}},
  "candle_pattern": {{"pattern": "", "type": "", "confirms_signal": true, "near_sr": true}},
  "atm": {{"ce_action": "", "pe_action": "", "vol_confirms": true, "strength": ""}},
  "iv_note": {{"ce_iv": 0, "pe_iv": 0, "note": ""}},
  "pcr": {{"value": {pcr:.2f}, "trend": "{oi['pcr_trend']}", "supports": true}},
  "volume": {{"ok": true, "spike_ratio": {pa.vol_spike_ratio:.2f}, "trap_warning": ""}},
  "entry": {{"now": true, "reason": "", "wait_for": ""}},
  "rr": {{"sl_pts": 0, "tgt_pts": 0, "ratio": 0}},
  "levels": {{"oi_support": {sr.support_strike}, "oi_resistance": {sr.resistance_strike}, "candle_sup": {p_sup:.0f}, "candle_res": {p_res:.0f}}}
}}"""
        return p


# ============================================================
#  DEEPSEEK CLIENT
# ============================================================

class DeepSeekClient:
    URL   = "https://api.deepseek.com/v1/chat/completions"
    MODEL = "deepseek-chat"

    def __init__(self, key: str):
        self.key = key

    async def analyze(self, prompt: str) -> Optional[Dict]:
        hdrs = {"Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json"}
        payload = {"model": self.MODEL,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.2, "max_tokens": 1500}
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=DEEPSEEK_TIMEOUT)
            ) as sess:
                async with sess.post(self.URL, headers=hdrs, json=payload) as r:
                    if r.status != 200:
                        logger.error(f"DeepSeek {r.status}")
                        return None
                    data    = await r.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    for f in ("```json", "```"):
                        content = content.replace(f, "")
                    return json.loads(content.strip())
        except asyncio.TimeoutError:
            logger.error("❌ DeepSeek timeout")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ DeepSeek: {e}")
            return None


# ============================================================
#  TELEGRAM ALERTER
# ============================================================

class TelegramAlerter:

    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self.session = None

    async def _sess(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    async def send_raw(self, text: str):
        await self._sess()
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            async with self.session.post(url, json={
                "chat_id": self.chat_id, "text": text, "parse_mode": "HTML"
            }) as r:
                if r.status != 200:
                    logger.error(f"Telegram {r.status}: {await r.text()[:80]}")
        except Exception as e:
            logger.error(f"❌ Telegram: {e}")

    async def send_signal(self, sig: Dict, snap: MarketSnapshot,
                          oi: Dict, pa: PriceActionInsight,
                          trigger: str = "MTF"):
        mtf  = sig.get("mtf",            {})
        atma = sig.get("atm",            {})
        pcra = sig.get("pcr",            {})
        vol  = sig.get("volume",         {})
        ent  = sig.get("entry",          {})
        rr   = sig.get("rr",             {})
        cndl = sig.get("candle_pattern", {})
        trnd = sig.get("trend_analysis", {})
        iv   = sig.get("iv_note",        {})
        st   = sig.get("signal", "WAIT")
        opt  = "CE" if "CALL" in st else "PE" if "PUT" in st else ""
        t    = pa.trend
        conf = sig.get("confidence", 0)
        bar  = "🟢" * min(conf, 10) + "⚪" * (10 - min(conf, 10))

        msg = (
            f"🚀 NIFTY OPTIONS v6.3 | {trigger}\n"
            f"📅 {datetime.now(IST).strftime('%d-%b %H:%M IST')}\n\n"
            f"💰 NIFTY: ₹{snap.spot_price:,.2f}\n"
            f"📊 Signal: <b>{st}</b>\n"
            f"⭐ Confidence: {conf}/10 {bar}\n"
            f"📅 Expiry: {snap.expiry} | Lot:{LOT_SIZE}\n\n"
            f"━━━ TRADE SETUP ━━━\n"
            f"Entry: <b>{sig.get('primary_strike',0)} {opt}</b>\n"
            f"SL: {sig.get('stop_loss_strike',0)} {opt} | "
            f"Target: {sig.get('target_strike',0)} {opt}\n"
            f"RR: {rr.get('ratio','N/A')} "
            f"(SL:{rr.get('sl_pts',0)}pt→Tgt:{rr.get('tgt_pts',0)}pt)\n\n"
            f"━━━ TREND ━━━\n"
            f"Day:{t.day_trend} | 3min:{t.intraday_trend}\n"
            f"VWAP:₹{t.vwap:.0f} | Spot {t.spot_vs_vwap}\n"
            f"All agree:{'✅' if trnd.get('all_agree') else '❌'} | "
            f"VWAP confirms:{'✅' if trnd.get('vwap_confirms') else '❌'}\n\n"
            f"━━━ OI (3m+15m) ━━━\n"
            f"ATM CE:{atma.get('ce_action','N/A')} | ATM PE:{atma.get('pe_action','N/A')}\n"
            f"TF3:{mtf.get('tf3','N/A')} | TF15:{mtf.get('tf15','N/A')}\n"
            f"MTF:{'✅ HIGH CONF' if mtf.get('confirmed') else '❌ Weak'} | "
            f"Vol:{'✅' if atma.get('vol_confirms') else '❌ TRAP?'}\n\n"
            f"━━━ PRICE ACTION ━━━\n"
            f"3m:{pa.price_change_3m:+.2f}% | 15m:{pa.price_change_15m:+.2f}%\n"
            f"Triple:{'✅' if pa.triple_confirmed else '❌'} | "
            f"Vol:{pa.vol_spike_ratio:.1f}x\n\n"
            f"━━━ PATTERN & PCR ━━━\n"
            f"Pattern:{cndl.get('pattern','None')} | "
            f"Confirms:{'✅' if cndl.get('confirms_signal') else '❌'}\n"
            f"PCR:{pcra.get('value','N/A')} ({pcra.get('trend','N/A')}) | "
            f"Supports:{'✅' if pcra.get('supports') else '❌'}\n"
            f"IV: CE={iv.get('ce_iv',0):.1f}% PE={iv.get('pe_iv',0):.1f}%\n\n"
            f"━━━ ENTRY ━━━\n"
            f"{'✅ ENTER NOW' if ent.get('now') else '⏳ WAIT'}\n"
            f"{ent.get('reason','')}\n\n"
            f"DeepSeek V3 | NIFTY v6.3 | 3-min polling"
        )
        if vol.get("trap_warning"):
            msg += f"\n\n⚠️ {vol['trap_warning']}"
        await self.send_raw(msg.strip())


# ============================================================
#  MAIN BOT
# ============================================================

class NiftyOptionsBot:

    def __init__(self):
        self.upstox  = UpstoxClient(UPSTOX_ACCESS_TOKEN)
        self.cache   = SnapshotCache()
        self.mtf     = MTFAnalyzer(self.cache)
        self.ai      = DeepSeekClient(DEEPSEEK_API_KEY)
        self.alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.checker = AlertChecker(self.cache, self.alerter)
        self.phase   = PhaseDetector()
        self.strong  = StrongOIChecker()
        self._cycle  = 0
        self._expiry: Optional[str] = None
        self._expiry_date: Optional[str] = None

    def is_market_open(self) -> bool:
        now = datetime.now(IST)
        if now.weekday() >= 5: return False
        s = now.replace(hour=MARKET_START[0], minute=MARKET_START[1], second=0)
        e = now.replace(hour=MARKET_END[0],   minute=MARKET_END[1],   second=0)
        return s <= now <= e

    async def _refresh_expiry(self):
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if self._expiry_date == today and self._expiry:
            return
        logger.info("🔄 Fetching expiry...")
        self._expiry      = await self.upstox.get_nearest_expiry()
        self._expiry_date = today

    async def run(self):
        logger.info("=" * 60)
        logger.info("NIFTY OPTIONS BOT v6.3 PRO — Koyeb")
        logger.info(f"Polling: {SNAPSHOT_INTERVAL}s | ATM±{ATM_RANGE} | Lot:{LOT_SIZE}")
        logger.info(f"Strong OI threshold: >{STRONG_OI_DIRECT_PCT}% OI + >{STRONG_VOL_PCT}% Vol")
        logger.info(f"Phase1≥{PHASE1_OI_BUILD_PCT}% | Phase2≥{PHASE2_VOL_SPIKE_PCT}%")
        logger.info("=" * 60)

        await self.upstox.init()
        await self._refresh_expiry()

        try:
            while True:
                if not self.is_market_open():
                    logger.info("Market closed — 60s")
                    await asyncio.sleep(60)
                    continue
                try:
                    await self._cycle_run()
                except aiohttp.ClientConnectorError:
                    logger.error("❌ Network — retry 60s")
                    await asyncio.sleep(60)
                except Exception as e:
                    logger.error(f"❌ Cycle: {e}")
                    logger.exception("Traceback:")
                    await asyncio.sleep(30)

                logger.info(
                    f"⏱ Next in {SNAPSHOT_INTERVAL}s | "
                    f"Cache:{self.cache.size()}/{CACHE_SIZE}"
                )
                await asyncio.sleep(SNAPSHOT_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Stopped")
        finally:
            await self.upstox.close()
            await self.alerter.close()

    async def _cycle_run(self):
        self._cycle += 1
        is_analysis = (self._cycle % ANALYSIS_EVERY_N == 0)  # every 15-min

        await self._refresh_expiry()
        if not self._expiry:
            logger.warning("No expiry")
            return

        logger.info(
            f"{'📊 ANALYSIS' if is_analysis else '📦 SNAP'} "
            f"#{self._cycle} | {datetime.now(IST).strftime('%H:%M:%S IST')}"
        )

        # 1. Option chain snapshot
        snap = await self.upstox.fetch_snapshot(self._expiry)
        if not snap:
            return

        await self.cache.add(snap)

        # 2. Standalone alerts
        await self.checker.check_all(snap)

        # 3. Candle data (2 API calls)
        df_1m, df_3m, df_30m = await self.upstox.get_candles()

        # 4. Price action (uses option chain volume for VWAP)
        recent = await self.cache.get_recent(20)
        pa     = PriceActionCalculator.calculate(recent, df_3m, df_30m)
        logger.info(
            f"Price: 3m={pa.price_change_3m:+.2f}% 15m={pa.price_change_15m:+.2f}% | "
            f"Vol:{pa.vol_spike_ratio:.2f}x | VWAP:₹{pa.trend.vwap:.0f} | "
            f"Spot {pa.trend.spot_vs_vwap}"
        )

        # 5. Strong OI → direct AI (most important, no phase wait)
        prev_15, _ = await self.cache.get_best_prev()
        strong_ev  = await self.strong.check(snap, prev_15, pa)
        if strong_ev:
            await self._strong_oi_ai_call(snap, pa, strong_ev, df_3m, df_30m)

        # 6. Phase detection (parallel system)
        prev_3 = await self.cache.get_short()
        phases = await self.phase.detect(snap, prev_3, pa)
        for ps in phases:
            logger.info(f"⚡ Phase {ps.phase}: {ps.direction}")
            await self.alerter.send_raw(ps.message)
            if ps.phase == 3:
                await self._phase3_ai_call(snap, pa, ps, df_3m, df_30m)

        # 7. Periodic MTF analysis (every 15-min)
        if is_analysis:
            await self._full_analysis(snap, pa, df_3m, df_30m)

    async def _strong_oi_ai_call(self, snap: MarketSnapshot, pa: PriceActionInsight,
                                  strong_ev: Dict, df_3m: pd.DataFrame, df_30m: pd.DataFrame):
        """v6.3 NEW: Direct AI call on strong OI — no phase wait."""
        logger.info("🔥 Strong OI → DeepSeek V3...")
        oi = await self.mtf.analyze(snap)
        if not oi["available"]:
            oi = self._fallback_oi(snap, strong_ev["direction"])

        patterns     = PatternDetector.detect(df_3m)
        p_sup, p_res = PatternDetector.sr(df_30m)
        prompt       = PromptBuilder.build(
            snap, oi, pa, df_3m, df_30m, patterns, p_sup, p_res,
            trigger="STRONG_OI", strong_oi=strong_ev
        )
        ai_sig = await self.ai.analyze(prompt)
        if ai_sig and ai_sig.get("confidence", 0) >= MIN_CONFIDENCE:
            await self.alerter.send_signal(ai_sig, snap, oi, pa, trigger="🔥 STRONG_OI")

    async def _phase3_ai_call(self, snap: MarketSnapshot, pa: PriceActionInsight,
                               ps: PhaseSignal, df_3m: pd.DataFrame, df_30m: pd.DataFrame):
        logger.info("🚀 Phase 3 → DeepSeek V3...")
        oi = await self.mtf.analyze(snap)
        if not oi["available"]:
            oi = self._fallback_oi(snap, ps.direction)

        patterns     = PatternDetector.detect(df_3m)
        p_sup, p_res = PatternDetector.sr(df_30m)
        prompt       = PromptBuilder.build(
            snap, oi, pa, df_3m, df_30m, patterns, p_sup, p_res,
            trigger="PHASE3"
        )
        ai_sig = await self.ai.analyze(prompt)
        if ai_sig and ai_sig.get("confidence", 0) >= MIN_CONFIDENCE:
            await self.alerter.send_signal(ai_sig, snap, oi, pa, trigger="🚀 PHASE3")

    async def _full_analysis(self, snap: MarketSnapshot, pa: PriceActionInsight,
                              df_3m: pd.DataFrame, df_30m: pd.DataFrame):
        logger.info("📊 15-min MTF analysis...")
        oi = await self.mtf.analyze(snap)
        if not oi["available"]:
            logger.info(oi["reason"])
            return
        if not oi["has_strong"]:
            logger.info("No strong MTF signal — skip AI")
            return

        patterns     = PatternDetector.detect(df_3m)
        p_sup, p_res = PatternDetector.sr(df_30m)
        prompt       = PromptBuilder.build(
            snap, oi, pa, df_3m, df_30m, patterns, p_sup, p_res,
            trigger="MTF_15MIN"
        )
        logger.info("🤖 DeepSeek V3...")
        ai_sig = await self.ai.analyze(prompt)

        if not ai_sig:
            sa = next((s for s in oi["strike_analyses"] if s.is_atm), None)
            ai_sig = self._fallback_signal(snap, pa, oi, sa)

        conf = ai_sig.get("confidence", 0)
        logger.info(f"Signal:{ai_sig.get('signal','WAIT')} | Conf:{conf}/10")
        if conf >= MIN_CONFIDENCE:
            await self.alerter.send_signal(ai_sig, snap, oi, pa, trigger="📊 MTF_15MIN")

    def _fallback_oi(self, snap: MarketSnapshot, direction: str) -> Dict:
        return {
            "available": True, "strike_analyses": [],
            "sr": SupportResistance(snap.atm_strike, 0, snap.atm_strike, 0, False, False),
            "overall": direction, "total_bull": 0, "total_bear": 0,
            "overall_pcr": snap.overall_pcr, "pcr_trend": "N/A",
            "pcr_ch_pct": 0, "window_mins": 0, "has_strong": True
        }

    def _fallback_signal(self, snap, pa, oi, sa) -> Dict:
        fb = ("BUY_CALL" if sa and sa.bull_strength > sa.bear_strength
              else "BUY_PUT" if sa else "WAIT")
        fc = min(10, max(sa.bull_strength, sa.bear_strength)) if sa else 3
        return {
            "signal": fb, "confidence": fc,
            "primary_strike": snap.atm_strike,
            "mtf":           {"tf3":"N/A","tf15":"N/A","confirmed":False},
            "entry":         {"now":False,"reason":"AI timeout—MTF fallback","wait_for":""},
            "price_action":  {"momentum":pa.price_momentum,
                              "triple_confirmed":pa.triple_confirmed,"confirms_signal":False},
            "candle_pattern":{"pattern":"N/A","type":"","confirms_signal":False,"near_sr":False},
            "trend_analysis":{"day":pa.trend.day_trend,"intraday":pa.trend.intraday_trend,
                              "all_agree":pa.trend.all_agree,"vwap_confirms":False,"note":""},
            "iv_note":       {"ce_iv":0,"pe_iv":0,"note":""},
            "volume":        {"ok":False,"spike_ratio":pa.vol_spike_ratio,"trap_warning":""},
            "rr":{},"atm":{},"pcr":{},"levels":{}
        }


# ============================================================
#  HTTP SERVER — Koyeb
# ============================================================

bot_instance: Optional[NiftyOptionsBot] = None

async def health(request):
    if bot_instance:
        sz  = bot_instance.cache.size()
        mkt = "OPEN" if bot_instance.is_market_open() else "CLOSED"
        exp = bot_instance._expiry or "fetching..."
        txt = (
            f"NIFTY Bot v6.3 | ALIVE ✅\n"
            f"{datetime.now(IST).strftime('%d-%b %H:%M IST')} | Market:{mkt}\n"
            f"Expiry:{exp} | Cache:{sz}/{CACHE_SIZE} (3-min)\n"
            f"ATM±{ATM_RANGE} | StrongOI>{STRONG_OI_DIRECT_PCT}% | Phase1≥{PHASE1_OI_BUILD_PCT}%"
        )
    else:
        txt = "NIFTY Bot v6.3 | Starting..."
    return aiohttp.web.Response(
        text=txt,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache", "Expires": "0"
        }
    )

async def start_bot(app):
    global bot_instance
    bot_instance    = NiftyOptionsBot()
    app["bot_task"] = asyncio.create_task(bot_instance.run())

async def stop_bot(app):
    if "bot_task" in app:
        app["bot_task"].cancel()
        try:
            await app["bot_task"]
        except asyncio.CancelledError:
            pass
    if bot_instance:
        await bot_instance.upstox.close()
        await bot_instance.alerter.close()


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    from aiohttp import web

    app = web.Application()
    app.router.add_get("/",       health)
    app.router.add_get("/health", health)
    app.on_startup.append(start_bot)
    app.on_cleanup.append(stop_bot)

    port = int(os.getenv("PORT", 8000))

    print(f"""
╔══════════════════════════════════════════╗
║   NIFTY 50 OPTIONS BOT v6.3 PRO          ║
║   Platform: Koyeb | Upstox V2            ║
╠══════════════════════════════════════════╣
║  Polling  : Every 3 min (NSE OI refresh) ║
║  ATM Range: ±{ATM_RANGE} strikes (±₹{ATM_RANGE*STRIKE_INTERVAL})       ║
║  Lot Size : {LOT_SIZE} | Strike: ₹{STRIKE_INTERVAL}           ║
╠══════════════════════════════════════════╣
║  v6.3 KEY CHANGES:                       ║
║  ✅ 3-min polling                        ║
║  ✅ Strong OI (>20%+Vol>25%) → AI direct ║
║  ✅ 15-min OI window (was 30-min)        ║
║  ✅ 3-min candle resample                ║
║  ✅ Option chain VWAP (no ₹0 bug)       ║
║  ✅ Adaptive cache (9:25 compatible)     ║
║  ✅ Strike-wise compare (ATM shift safe) ║
║  ✅ Analysis every 15-min (was 30-min)  ║
╠══════════════════════════════════════════╣
║  ENV VARS:                               ║
║  UPSTOX_ACCESS_TOKEN                     ║
║  TELEGRAM_BOT_TOKEN + CHAT_ID            ║
║  DEEPSEEK_API_KEY                        ║
║  PORT (default: 8000)                    ║
╚══════════════════════════════════════════╝
Starting on port {port}...
""")

    web.run_app(app, host="0.0.0.0", port=port)
