"""Unit tests for compute_export_quality_volume in tier.py."""
import pandas as pd
import numpy as np
import pytest

from tier import compute_export_quality_volume

THRESHOLDS = {"Top": 2500, "Low": 900}
SAL_VAR = "BANKSEC_MAX14DAY_"
VOL_VAR = "C_CAA003_"

MI_NAMES = ["A", "B", "C", "D", "E", "F", "Units"]


def _make_col(part_b, units="UMHOS/CM"):
    """Build a single 7-level MultiIndex column tuple."""
    return ("CALSIM", part_b, "2020D09E", "1MON", "L2020A", "DV", units)


def _build_df(columns_tuples, n_months, start="1921-10-31", value=1.0):
    """Return a DataFrame with MultiIndex columns and monthly DatetimeIndex."""
    idx = pd.date_range(start, periods=n_months, freq="ME")
    mi = pd.MultiIndex.from_tuples(columns_tuples, names=MI_NAMES)
    data = np.full((n_months, len(columns_tuples)), value)
    return pd.DataFrame(data, index=idx, columns=mi)


# -- helpers to build paired salinity / volume frames --

def _pair(sid, n_months=12, sal_value=900.0, vol_value=100.0, start="1921-10-31"):
    sal_col = _make_col(f"{SAL_VAR}{sid}", units="UMHOS/CM")
    vol_col = _make_col(f"{VOL_VAR}{sid}", units="CFS")
    sal_df = _build_df([sal_col], n_months, start=start, value=sal_value)
    vol_df = _build_df([vol_col], n_months, start=start, value=vol_value)
    return sal_df, vol_df


# -- Tests --

class TestWeightCalculation:
    """Verify weight = clip((ht - salinity) / (ht - lt), 0, 1)."""

    def test_weight_at_low_threshold(self):
        """salinity == Low (900) => weight = 1.0 => qv = volume * 1.0."""
        sal_df, vol_df = _pair("s0020", sal_value=900.0, vol_value=100.0)
        result = compute_export_quality_volume(
            sal_df, vol_df, SAL_VAR, VOL_VAR, THRESHOLDS, ["s0020"]
        )
        expected = 100.0 * 1.0 * 12  # 12 months, weight 1.0
        assert result["s0020"].iloc[0] == pytest.approx(expected)

    def test_weight_at_high_threshold(self):
        """salinity == Top (2500) => weight = 0.0 => qv = 0."""
        sal_df, vol_df = _pair("s0020", sal_value=2500.0, vol_value=100.0)
        result = compute_export_quality_volume(
            sal_df, vol_df, SAL_VAR, VOL_VAR, THRESHOLDS, ["s0020"]
        )
        assert result["s0020"].iloc[0] == pytest.approx(0.0)

    def test_weight_at_midpoint(self):
        """salinity == 1700 (midpoint) => weight = 0.5 => qv = volume * 0.5."""
        sal_df, vol_df = _pair("s0020", sal_value=1700.0, vol_value=100.0)
        result = compute_export_quality_volume(
            sal_df, vol_df, SAL_VAR, VOL_VAR, THRESHOLDS, ["s0020"]
        )
        expected = 100.0 * 0.5 * 12
        assert result["s0020"].iloc[0] == pytest.approx(expected)

    def test_weight_clipping_below_low(self):
        """salinity = 500 (below Low) => weight clips to 1.0, not >1."""
        sal_df, vol_df = _pair("s0020", sal_value=500.0, vol_value=100.0)
        result = compute_export_quality_volume(
            sal_df, vol_df, SAL_VAR, VOL_VAR, THRESHOLDS, ["s0020"]
        )
        expected = 100.0 * 1.0 * 12
        assert result["s0020"].iloc[0] == pytest.approx(expected)

    def test_weight_clipping_above_high(self):
        """salinity = 3000 (above Top) => weight clips to 0.0, not negative."""
        sal_df, vol_df = _pair("s0020", sal_value=3000.0, vol_value=100.0)
        result = compute_export_quality_volume(
            sal_df, vol_df, SAL_VAR, VOL_VAR, THRESHOLDS, ["s0020"]
        )
        assert result["s0020"].iloc[0] == pytest.approx(0.0)


class TestAnnualResampling:
    """Verify water-year resampling (Oct-Sep)."""

    def test_annual_water_year_sum(self):
        """12 months Oct-Sep should produce exactly 1 annual sum."""
        sal_df, vol_df = _pair("s0020", n_months=12, sal_value=900.0, vol_value=50.0)
        result = compute_export_quality_volume(
            sal_df, vol_df, SAL_VAR, VOL_VAR, THRESHOLDS, ["s0020"]
        )
        assert len(result) == 1
        assert result["s0020"].iloc[0] == pytest.approx(50.0 * 12)


class TestMultipleScenarios:
    """Verify multi-scenario handling."""

    def test_multiple_scenarios(self):
        """2 scenario IDs with matching columns => 2 result columns."""
        sal1, vol1 = _pair("s0020", sal_value=900.0, vol_value=100.0)
        sal2, vol2 = _pair("s0021", sal_value=2500.0, vol_value=100.0)
        sal_df = pd.concat([sal1, sal2], axis=1)
        vol_df = pd.concat([vol1, vol2], axis=1)

        result = compute_export_quality_volume(
            sal_df, vol_df, SAL_VAR, VOL_VAR, THRESHOLDS, ["s0020", "s0021"]
        )
        assert set(result.columns) == {"s0020", "s0021"}
        assert result["s0020"].iloc[0] == pytest.approx(100.0 * 12)
        assert result["s0021"].iloc[0] == pytest.approx(0.0)


class TestMissingScenarios:
    """Verify graceful handling of missing columns."""

    def test_missing_scenario_skipped(self):
        """Scenario with no matching columns should be silently skipped."""
        sal_df, vol_df = _pair("s0020", sal_value=900.0, vol_value=100.0)
        result = compute_export_quality_volume(
            sal_df, vol_df, SAL_VAR, VOL_VAR, THRESHOLDS, ["s0020", "s9999"]
        )
        assert "s0020" in result.columns
        assert "s9999" not in result.columns

    def test_empty_result(self):
        """All scenario IDs missing => returns empty DataFrame."""
        sal_df, vol_df = _pair("s0020", sal_value=900.0, vol_value=100.0)
        result = compute_export_quality_volume(
            sal_df, vol_df, SAL_VAR, VOL_VAR, THRESHOLDS, ["s9998", "s9999"]
        )
        assert result.empty
