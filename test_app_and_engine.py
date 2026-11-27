import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import stock_engine  # noqa: E402


def make_synthetic_ohlcv(
    bars: int = 320,
    start_price: float = 1000.0,
    trend_mode: str = "breakout",
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic OHLCV test data for different market regimes."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-06-01", periods=bars, freq="B")

    closes = np.zeros(bars, dtype=float)
    opens = np.zeros(bars, dtype=float)
    highs = np.zeros(bars, dtype=float)
    lows = np.zeros(bars, dtype=float)
    volumes = np.zeros(bars, dtype=float)

    price = start_price
    for i in range(bars):
        if trend_mode == "breakout":
            # Oscillation with upward drift and a clean volume breakout at the end
            if i < bars - 15:
                drift = 0.0012 + 0.008 * np.sin(i / 8.0)
                vol = 1_000_000 * (1.0 + 0.2 * rng.random())
            elif i < bars - 2:
                # Tight consolidation coil below resistance
                drift = 0.0005 * np.sin(i)
                vol = 750_000
            else:
                # Strong breakout surge
                drift = 0.032
                vol = 2_400_000
        elif trend_mode == "resistance_capped":
            # Repeatedly rejected at 1200, currently at 1140
            cycle = 1100.0 + 95.0 * np.sin(i / 6.0)
            price = cycle
            drift = 0.0
            vol = 900_000
        else:
            # Mixed alternating waves so historical setups produce wins & stops
            drift = 0.002 * np.sin(i / 5.0) + rng.normal(0.0005, 0.012)
            vol = 1_100_000 * (0.8 + 0.5 * rng.random())

        if trend_mode != "resistance_capped":
            price = max(50.0, price * (1.0 + drift + rng.normal(0.0, 0.006)))
        o = price * (1.0 + rng.normal(0.0, 0.003))
        spread = price * (0.012 + 0.006 * rng.random())
        h = max(o, price) + spread * 0.5
        l = min(o, price) - spread * 0.5

        if trend_mode == "breakout" and i >= bars - 2:
            o = price * 0.975
            l = o * 0.996
            h = price * 1.003  # Close near high

        opens[i] = round(o, 2)
        highs[i] = round(h, 2)
        lows[i] = round(l, 2)
        closes[i] = round(price, 2)
        volumes[i] = float(vol)

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )


class TestStockEngineQuantLogic(unittest.TestCase):
    def test_breakout_signal_and_targets_on_surge(self):
        raw_df = make_synthetic_ohlcv(bars=300, trend_mode="breakout")
        df = stock_engine.enrich_indicators(raw_df)
        report = stock_engine.generate_full_stock_report(df, risk_reward_ratio=2.0, timeframe="1d")

        analysis = report["analysis"]
        breakout = report["breakout"]
        feasibility = report["feasibility"]
        holding = report["holding_plan"]
        backtest = report["backtest"]
        past_perf = report["past_performance"]

        self.assertEqual(analysis["bias"], "Bullish")
        self.assertLess(analysis["sl"], analysis["cmp"])
        self.assertLess(analysis["cmp"], analysis["target_1"])
        self.assertLess(analysis["target_1"], analysis["target_2"])

        self.assertIn("BREAKOUT", breakout["status"])
        self.assertGreaterEqual(breakout["vol_ratio"], 1.5)
        self.assertGreaterEqual(breakout["score"], 60)

        self.assertGreaterEqual(feasibility["t1"]["touch_probability"], 10)
        self.assertLessEqual(feasibility["t1"]["touch_probability"], 92)
        self.assertGreaterEqual(feasibility["t2"]["touch_probability"], 10)
        self.assertLessEqual(feasibility["t2"]["touch_probability"], 92)

        self.assertTrue(len(holding["steps"]) >= 3)
        self.assertGreater(holding["time_stop_bars"], 0)

        self.assertEqual(len(backtest["sensitivity"]), 5)
        self.assertGreaterEqual(backtest["summary"]["t1_hit_rate"], backtest["summary"]["t2_hit_rate"])
        self.assertLessEqual(past_perf["max_drawdown_pct"], 0.0)
        self.assertGreaterEqual(past_perf["annualized_vol_pct"], 0.0)

    def test_high_rr_reduces_or_flags_stretched_target(self):
        raw_df = make_synthetic_ohlcv(bars=320, trend_mode="resistance_capped")
        df = stock_engine.enrich_indicators(raw_df)

        report_low = stock_engine.generate_full_stock_report(df, risk_reward_ratio=1.5, timeframe="1d")
        report_high = stock_engine.generate_full_stock_report(df, risk_reward_ratio=3.5, timeframe="1d")

        t2_low = report_low["feasibility"]["t2"]["touch_probability"]
        t2_high = report_high["feasibility"]["t2"]["touch_probability"]

        self.assertGreaterEqual(t2_low, t2_high)
        self.assertGreater(
            report_high["feasibility"]["t2"]["atr_multiple"],
            report_low["feasibility"]["t2"]["atr_multiple"],
        )

    def test_backtest_invariant_t1_ge_t2_and_non_overlapping(self):
        raw_df = make_synthetic_ohlcv(bars=350, trend_mode="mixed")
        df = stock_engine.enrich_indicators(raw_df)
        bt = stock_engine.backtest_stock_strategy(df, timeframe="1d", selected_rr=2.5)

        for row in bt["sensitivity"]:
            self.assertGreaterEqual(row["t1_hit_rate"], row["t2_hit_rate"])
            self.assertGreaterEqual(row["t1_hit_rate"], 0.0)
            self.assertLessEqual(row["t1_hit_rate"], 100.0)

        self.assertIn(bt["optimal_rr"], stock_engine.RR_GRID)

    def test_screener_handles_single_stock_flat_download(self):
        sample_df = make_synthetic_ohlcv(bars=120, trend_mode="breakout")
        universe = [stock_engine.INDIAN_STOCKS_METADATA[14]]  # TATAPOWER.NS

        with patch("stock_engine.yf.download", return_value=sample_df):
            results, health = stock_engine.fetch_screener_universe(universe, timeframe="1d")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["symbol"], "TATAPOWER.NS")
        self.assertIn("breakout_status", results[0])
        self.assertIn("breakout_score", results[0])
        self.assertIn("breakouts", health)


class TestStreamlitAppBehavior(unittest.TestCase):
    APP_PATH = os.path.join(PROJECT_ROOT, "app.py")

    @staticmethod
    def _mock_fetch_stock_data(symbol: str, timeframe: str = "1d", period=None):
        clean = symbol.strip().upper()
        if not clean.endswith((".NS", ".BO")) and not clean.startswith("^"):
            clean = f"{clean}.NS"
        df = stock_engine.enrich_indicators(make_synthetic_ohlcv(bars=260, trend_mode="breakout"))
        return {
            "df": df,
            "symbol": clean,
            "name": clean.replace(".NS", ""),
            "sector": "Power & Energy",
            "mcap_cr": 135000,
            "currency": "₹",
        }

    @staticmethod
    def _mock_fetch_news(company_name: str, symbol: str, limit: int = 4):
        return [
            {
                "title": f"{company_name} wins major grid expansion order",
                "link": "https://example.com/news1",
                "published": "2026-11-27",
                "source": "Market Wire",
                "sentiment": "Positive",
                "sentiment_score": 0.6,
            }
        ]

    def test_initial_load_does_not_default_to_reliance(self):
        at = AppTest.from_file(self.APP_PATH, default_timeout=10).run()
        self.assertFalse(at.exception)
        self.assertIsNone(at.selectbox(key="stock").value)
        self.assertTrue(len(at.info) >= 1)
        self.assertIn("No default stock is forced", at.info[0].value)

    def test_selecting_stock_and_adjusting_rr_updates_touch_logic(self):
        with (
            patch("stock_engine.fetch_stock_data", side_effect=self._mock_fetch_stock_data),
            patch("stock_engine.fetch_stock_news", side_effect=self._mock_fetch_news),
        ):
            at = AppTest.from_file(self.APP_PATH, default_timeout=10).run()
            at.selectbox(key="stock").select("TATAPOWER.NS").run()
            self.assertFalse(at.exception)
            self.assertEqual(at.selectbox(key="stock").value, "TATAPOWER.NS")
            self.assertTrue(any("Target touch logic" in m.value for m in at.markdown))
            self.assertTrue(any("Logical holding plan" in m.value for m in at.markdown))
            self.assertTrue(any("Breakout signal" in m.value for m in at.markdown))
            self.assertTrue(any("Past performances" in m.value for m in at.markdown))

            t2_before = at.metric[3].value
            at.slider(key="rr").set_value(3.5).run()
            self.assertFalse(at.exception)
            self.assertEqual(at.slider(key="rr").value, 3.5)
            t2_after = at.metric[3].value
            self.assertNotEqual(t2_before, t2_after)

    def test_quick_pick_selects_stock_and_radar_returns_home(self):
        with (
            patch("stock_engine.fetch_stock_data", side_effect=self._mock_fetch_stock_data),
            patch("stock_engine.fetch_stock_news", side_effect=self._mock_fetch_news),
        ):
            at = AppTest.from_file(self.APP_PATH, default_timeout=10).run()
            at.button(key="quick_pick_HAL").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.selectbox(key="stock").value, "HAL.NS")
            self.assertTrue(any("HAL.NS" in s.value for s in at.subheader))

    def test_refresh_persists_custom_ticker_via_query_params(self):
        with (
            patch("stock_engine.fetch_stock_data", side_effect=self._mock_fetch_stock_data),
            patch("stock_engine.fetch_stock_news", side_effect=self._mock_fetch_news),
        ):
            # Simulate a full browser refresh on a custom ticker URL (?stock=IREDA&tf=1h&rr=2.5)
            at = AppTest.from_file(self.APP_PATH, default_timeout=10)
            at.query_params["stock"] = "IREDA"
            at.query_params["tf"] = "1h"
            at.query_params["rr"] = "2.5"
            at.run()

            self.assertFalse(at.exception)
            self.assertEqual(at.selectbox(key="stock").value, "IREDA")
            self.assertEqual(at.get("segmented_control")[0].value, "1h")
            self.assertEqual(at.slider(key="rr").value, 2.5)
            self.assertTrue(any("IREDA.NS" in s.value for s in at.subheader))

    def test_formatted_url_stock_does_not_duplicate_dropdown_options(self):
        formatted_label = "TATAPOWER.NS — Tata Power (Power & Energy)"
        with (
            patch("stock_engine.fetch_stock_data", side_effect=self._mock_fetch_stock_data),
            patch("stock_engine.fetch_stock_news", side_effect=self._mock_fetch_news),
        ):
            at = AppTest.from_file(self.APP_PATH, default_timeout=10)
            at.query_params["stock"] = formatted_label
            at.run()

            self.assertFalse(at.exception)
            self.assertEqual(at.selectbox(key="stock").value, "TATAPOWER.NS")
            self.assertEqual(at.selectbox(key="stock").options.count(formatted_label), 1)
            self.assertEqual(len(at.selectbox(key="stock").options), len(stock_engine.INDIAN_STOCKS_METADATA))


if __name__ == "__main__":
    unittest.main()
