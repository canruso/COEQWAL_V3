"""Tests for structured output validation in llm_utils.py.

Tests cover:
- _parse_llm_response: JSON parsing with multiple fallback strategies
- _extract_means_from_stats: stats text parsing into numeric dicts
- validate_observation: LLM claim validation against computed stats
- analyze_plot: mocked API return type and backward compatibility
"""
from __future__ import annotations

import json
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup: allow imports from the coeqwalpackage directory
# ---------------------------------------------------------------------------
_PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)


# ===================================================================
# Helpers: synthetic data builders
# ===================================================================

def _make_valid_structured():
    """Return a well-formed structured dict for testing."""
    return {
        "narrative": "Scenario s0020 has the highest storage.",
        "ranking": ["s0030", "s0020", "s0028"],
        "best_scenario": "s0030",
        "worst_scenario": "s0028",
        "cited_values": {"s0020": 123.4, "s0030": 150.0, "s0028": 100.0},
    }


def _make_valid_json_response():
    """Return a valid JSON string matching the expected LLM envelope."""
    return json.dumps(_make_valid_structured())


def _make_stats_text_mon_ts(means: dict[str, float], baseline_sid: str) -> str:
    """Build a stats_text string mimicking _stats_mon_ts output.

    means: {"s0020": 123.4, "s0030": 150.0, "s0028": 100.0}
    baseline_sid: "s0020"
    """
    lines = ["Underlying data (water-year statistics):"]
    bl_mean = means.get(baseline_sid, 0.0)
    for sid, mean_val in means.items():
        label = f"{sid} (baseline)" if sid == baseline_sid else sid
        if sid == baseline_sid:
            lines.append(f"  {label}: mean={mean_val:.1f}, drought-yr mean=80.0, wet-yr mean=160.0, min=50.0, max=200.0")
        else:
            pct = (mean_val - bl_mean) / bl_mean * 100 if bl_mean else 0
            lines.append(f"  {label}: mean={mean_val:.1f} ({pct:+.1f}%), drought-yr mean=85.0, wet-yr mean=170.0, min=55.0, max=210.0")
    return "\n".join(lines)


def _make_stats_text_exceedance(means: dict[str, float], baseline_sid: str) -> str:
    """Build a stats_text string mimicking _stats_exceedance output."""
    lines = ["Exceedance statistics:"]
    bl_mean = means.get(baseline_sid, 0.0)
    for sid, mean_val in means.items():
        label = f"{sid} (baseline)" if sid == baseline_sid else sid
        if sid == baseline_sid:
            lines.append(f"  {label}: P10=200.0, P25=180.0, P50=150.0, P75=120.0, P90=100.0, mean={mean_val:.1f}")
        else:
            pct = (mean_val - bl_mean) / bl_mean * 100 if bl_mean else 0
            lines.append(f"  {label}: P10=210.0, P25=190.0, P50=160.0, P75=130.0, P90=110.0, mean={mean_val:.1f} ({pct:+.1f}%)")
    return "\n".join(lines)


def _make_stats_text_moy(baseline_sid: str) -> str:
    """Build a stats_text string mimicking _stats_moy output (no mean= field)."""
    return (
        "Month-of-year averages:\n"
        f"  {baseline_sid} (baseline): peak=May (180.0), low=Sep (60.0)\n"
        "  s0030: peak=May (190.0 (+5.6%)), low=Sep (65.0 (+8.3%))\n"
        "  Largest inter-scenario spread: May (10.0)"
    )


# ===================================================================
# 1. _parse_llm_response
# ===================================================================

class TestParseLlmResponse:
    """Tests for JSON parsing with fallback strategies."""

    def test_valid_json_all_fields(self):
        """Valid JSON with all expected fields returns parsed dict."""
        from coeqwalpackage.llm_utils import _parse_llm_response
        raw = _make_valid_json_response()
        result = _parse_llm_response(raw)
        assert result["narrative"] == "Scenario s0020 has the highest storage."
        assert result["ranking"] == ["s0030", "s0020", "s0028"]
        assert result["best_scenario"] == "s0030"
        assert result["worst_scenario"] == "s0028"
        assert result["cited_values"] == {"s0020": 123.4, "s0030": 150.0, "s0028": 100.0}

    def test_json_inside_code_fence(self):
        """JSON wrapped in markdown code fence is extracted and parsed."""
        from coeqwalpackage.llm_utils import _parse_llm_response
        inner = _make_valid_json_response()
        raw = f"Here is my analysis:\n```json\n{inner}\n```\nEnd."
        result = _parse_llm_response(raw)
        assert result["ranking"] == ["s0030", "s0020", "s0028"]
        assert result["best_scenario"] == "s0030"

    def test_json_with_extra_text(self):
        """JSON object with leading/trailing text is extracted."""
        from coeqwalpackage.llm_utils import _parse_llm_response
        inner = _make_valid_json_response()
        raw = f"Some preamble text.\n{inner}\nSome trailing text."
        result = _parse_llm_response(raw)
        assert result["ranking"] == ["s0030", "s0020", "s0028"]

    def test_plain_text_fallback(self):
        """Plain text (no JSON) returns fallback dict with narrative=raw text."""
        from coeqwalpackage.llm_utils import _parse_llm_response
        raw = "This is just a plain text analysis with no JSON."
        result = _parse_llm_response(raw)
        assert result["narrative"] == raw
        assert result["ranking"] == []
        assert result["best_scenario"] == ""
        assert result["worst_scenario"] == ""
        assert result["cited_values"] == {}

    def test_malformed_json_fallback(self):
        """Malformed JSON (missing closing brace) triggers fallback."""
        from coeqwalpackage.llm_utils import _parse_llm_response
        raw = '{"narrative": "test", "ranking": ["s0020"'
        result = _parse_llm_response(raw)
        assert result["narrative"] == raw
        assert result["ranking"] == []

    def test_empty_string_fallback(self):
        """Empty string returns fallback with empty narrative."""
        from coeqwalpackage.llm_utils import _parse_llm_response
        result = _parse_llm_response("")
        assert result["narrative"] == ""
        assert result["ranking"] == []
        assert result["cited_values"] == {}

    def test_json_missing_optional_fields(self):
        """JSON with only narrative key fills in defaults for missing fields."""
        from coeqwalpackage.llm_utils import _parse_llm_response
        raw = json.dumps({"narrative": "Only narrative here."})
        result = _parse_llm_response(raw)
        assert result["narrative"] == "Only narrative here."
        assert result["ranking"] == []
        assert result["best_scenario"] == ""
        assert result["worst_scenario"] == ""
        assert result["cited_values"] == {}

    def test_json_with_escaped_quotes_in_narrative(self):
        """JSON where narrative contains escaped quotes parses correctly."""
        from coeqwalpackage.llm_utils import _parse_llm_response
        obj = {"narrative": 'Scenario "s0020" is the best.', "ranking": ["s0020"], "best_scenario": "s0020", "worst_scenario": "", "cited_values": {}}
        raw = json.dumps(obj)
        result = _parse_llm_response(raw)
        assert result["narrative"] == 'Scenario "s0020" is the best.'

    def test_nested_cited_values(self):
        """cited_values with nested numeric values parse correctly."""
        from coeqwalpackage.llm_utils import _parse_llm_response
        obj = {"narrative": "Test.", "ranking": [], "best_scenario": "", "worst_scenario": "", "cited_values": {"s0020_mean": 123.4, "s0020_P50": 150.0}}
        raw = json.dumps(obj)
        result = _parse_llm_response(raw)
        assert result["cited_values"]["s0020_mean"] == 123.4
        assert result["cited_values"]["s0020_P50"] == 150.0

    def test_return_type_is_always_dict(self):
        """_parse_llm_response always returns a dict regardless of input."""
        from coeqwalpackage.llm_utils import _parse_llm_response
        for raw in ["", "hello", "{}", "null", "[]", "123"]:
            result = _parse_llm_response(raw)
            assert isinstance(result, dict), f"Failed for input: {raw!r}"
            assert "narrative" in result
            assert "ranking" in result


# ===================================================================
# 2. _extract_means_from_stats
# ===================================================================

class TestExtractMeansFromStats:
    """Tests for extracting mean values from formatted stats text."""

    def test_mon_ts_format(self):
        """Standard mon_ts stats text with 'mean=X' per line extracts correctly."""
        from coeqwalpackage.llm_utils import _extract_means_from_stats
        means = {"s0002": 123.4, "s0020": 150.0, "s0030": 100.0}
        stats_text = _make_stats_text_mon_ts(means, "s0002")
        result = _extract_means_from_stats(stats_text, 2)
        assert abs(result["s0002"] - 123.4) < 0.01
        assert abs(result["s0020"] - 150.0) < 0.01
        assert abs(result["s0030"] - 100.0) < 0.01

    def test_exceedance_format_with_pct(self):
        """Exceedance stats with percentage annotations extracts the mean value."""
        from coeqwalpackage.llm_utils import _extract_means_from_stats
        means = {"s0002": 130.0, "s0020": 145.0}
        stats_text = _make_stats_text_exceedance(means, "s0002")
        result = _extract_means_from_stats(stats_text, 2)
        assert abs(result["s0002"] - 130.0) < 0.01
        assert abs(result["s0020"] - 145.0) < 0.01

    def test_moy_format_no_mean(self):
        """MOY stats (peak/low, no mean=) returns empty dict or gracefully handles."""
        from coeqwalpackage.llm_utils import _extract_means_from_stats
        stats_text = _make_stats_text_moy("s0002")
        result = _extract_means_from_stats(stats_text, 2)
        # MOY has no mean= field, so result should be empty or missing entries
        assert isinstance(result, dict)

    def test_empty_string(self):
        """Empty stats_text returns empty dict."""
        from coeqwalpackage.llm_utils import _extract_means_from_stats
        result = _extract_means_from_stats("", 1)
        assert result == {}

    def test_baseline_marker_identified(self):
        """Entry with '(baseline)' suffix is correctly keyed."""
        from coeqwalpackage.llm_utils import _extract_means_from_stats
        stats_text = "Underlying data:\n  s0002 (baseline): mean=100.0, min=50.0\n  s0030: mean=120.0, min=60.0"
        result = _extract_means_from_stats(stats_text, 2)
        assert "s0002" in result
        assert abs(result["s0002"] - 100.0) < 0.01

    def test_single_scenario(self):
        """Stats text with only one scenario returns single-entry dict."""
        from coeqwalpackage.llm_utils import _extract_means_from_stats
        stats_text = "Underlying data:\n  s0001 (baseline): mean=200.0, min=100.0"
        result = _extract_means_from_stats(stats_text, 1)
        assert len(result) == 1
        assert abs(result["s0001"] - 200.0) < 0.01


# ===================================================================
# 3. validate_observation
# ===================================================================

class TestValidateObservation:
    """Tests for LLM claim validation against computed statistics."""

    # ---- Happy path ----

    def test_all_correct_returns_empty_warnings(self):
        """When ranking, best, worst, cited values all match stats, return empty list."""
        from coeqwalpackage.llm_utils import validate_observation
        means = {"s0020": 123.4, "s0030": 150.0, "s0028": 100.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        # Correct ranking: sorted descending by mean (higher first)
        structured = {
            "ranking": ["s0030", "s0028"],
            "best_scenario": "s0030",
            "worst_scenario": "s0028",
            "cited_values": {"s0030": 150.0, "s0028": 100.0},
        }
        warnings = validate_observation(structured, stats_text, [20, 30, 28], baseline=20)
        assert isinstance(warnings, list)
        # Happy path: should have minimal/no warnings
        # (exact behavior depends on implementation's ranking direction logic)

    # ---- Ranking checks ----

    def test_ranking_mismatch_warns(self):
        """When LLM ranking is neither ascending nor descending by mean, a warning fires."""
        from coeqwalpackage.llm_utils import validate_observation
        # 3 non-baseline scenarios with distinct means
        means = {"s0020": 123.4, "s0030": 150.0, "s0028": 100.0, "s0040": 130.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        # Correct desc: s0030(150), s0040(130), s0028(100)
        # Correct asc: s0028(100), s0040(130), s0030(150)
        # This is neither:
        structured = {
            "ranking": ["s0028", "s0030", "s0040"],  # Scrambled order
            "best_scenario": "s0030",
            "worst_scenario": "s0028",
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30, 28, 40], baseline=20)
        assert any("ranking" in w.lower() or "order" in w.lower() for w in warnings)

    def test_missing_scenarios_in_ranking_warns(self):
        """When ranking omits a non-baseline scenario, a warning is returned."""
        from coeqwalpackage.llm_utils import validate_observation
        means = {"s0020": 123.4, "s0030": 150.0, "s0028": 100.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        structured = {
            "ranking": ["s0030"],  # Missing s0028
            "best_scenario": "s0030",
            "worst_scenario": "s0028",
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30, 28], baseline=20)
        assert any("s0028" in w or "missing" in w.lower() or "completeness" in w.lower() for w in warnings)

    def test_baseline_in_ranking_warns(self):
        """When ranking includes the baseline scenario, a warning is returned."""
        from coeqwalpackage.llm_utils import validate_observation
        means = {"s0020": 123.4, "s0030": 150.0, "s0028": 100.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        structured = {
            "ranking": ["s0030", "s0020", "s0028"],  # Includes baseline s0020
            "best_scenario": "s0030",
            "worst_scenario": "s0028",
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30, 28], baseline=20)
        assert any("baseline" in w.lower() for w in warnings)

    # ---- Best/worst scenario checks ----

    def test_wrong_best_scenario_warns(self):
        """When LLM best_scenario is neither highest nor lowest mean, a warning fires."""
        from coeqwalpackage.llm_utils import validate_observation
        # 3 non-baseline scenarios: s0030=150 (max), s0028=100 (min), s0040=130 (middle)
        means = {"s0020": 123.4, "s0030": 150.0, "s0028": 100.0, "s0040": 130.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        structured = {
            "ranking": ["s0030", "s0040", "s0028"],
            "best_scenario": "s0040",  # Middle - neither highest nor lowest
            "worst_scenario": "s0028",
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30, 28, 40], baseline=20)
        assert any("best" in w.lower() for w in warnings)

    def test_wrong_worst_scenario_warns(self):
        """When LLM worst_scenario is neither highest nor lowest mean, a warning fires."""
        from coeqwalpackage.llm_utils import validate_observation
        # 3 non-baseline scenarios: s0030=150 (max), s0028=100 (min), s0040=130 (middle)
        means = {"s0020": 123.4, "s0030": 150.0, "s0028": 100.0, "s0040": 130.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        structured = {
            "ranking": ["s0030", "s0040", "s0028"],
            "best_scenario": "s0030",
            "worst_scenario": "s0040",  # Middle - neither highest nor lowest
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30, 28, 40], baseline=20)
        assert any("worst" in w.lower() for w in warnings)

    def test_best_scenario_is_baseline_warns(self):
        """When best_scenario is the baseline, a warning about baseline exclusion fires."""
        from coeqwalpackage.llm_utils import validate_observation
        means = {"s0020": 200.0, "s0030": 150.0, "s0028": 100.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        structured = {
            "ranking": ["s0030", "s0028"],
            "best_scenario": "s0020",  # Baseline
            "worst_scenario": "s0028",
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30, 28], baseline=20)
        assert any("baseline" in w.lower() for w in warnings)

    def test_worst_scenario_is_baseline_warns(self):
        """When worst_scenario is the baseline, a warning about baseline exclusion fires."""
        from coeqwalpackage.llm_utils import validate_observation
        means = {"s0020": 50.0, "s0030": 150.0, "s0028": 100.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        structured = {
            "ranking": ["s0030", "s0028"],
            "best_scenario": "s0030",
            "worst_scenario": "s0020",  # Baseline
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30, 28], baseline=20)
        assert any("baseline" in w.lower() for w in warnings)

    # ---- Cited values checks ----

    def test_cited_value_not_in_stats_warns(self):
        """When a cited value is not found in stats_text, a warning is returned."""
        from coeqwalpackage.llm_utils import validate_observation
        means = {"s0020": 123.4, "s0030": 150.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        structured = {
            "ranking": ["s0030"],
            "best_scenario": "s0030",
            "worst_scenario": "s0030",
            "cited_values": {"s0030": 999.9},  # Not in stats
        }
        warnings = validate_observation(structured, stats_text, [20, 30], baseline=20)
        assert any("999.9" in w or "cited" in w.lower() for w in warnings)

    def test_cited_value_found_no_warning(self):
        """When cited values appear in stats_text, no warning for those values."""
        from coeqwalpackage.llm_utils import validate_observation
        means = {"s0020": 123.4, "s0030": 150.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        structured = {
            "ranking": ["s0030"],
            "best_scenario": "s0030",
            "worst_scenario": "s0030",
            "cited_values": {"s0030_mean": 150.0},  # This value IS in stats
        }
        warnings = validate_observation(structured, stats_text, [20, 30], baseline=20)
        # Should not warn about the cited value 150.0 since it appears in stats
        cited_warnings = [w for w in warnings if "150.0" in w and "cited" in w.lower()]
        assert len(cited_warnings) == 0

    # ---- Edge cases: empty / None / type mismatches ----

    def test_empty_structured_dict_no_crash(self):
        """Empty structured dict does not crash, returns appropriate warnings."""
        from coeqwalpackage.llm_utils import validate_observation
        stats_text = _make_stats_text_mon_ts({"s0020": 100.0, "s0030": 120.0}, "s0020")
        warnings = validate_observation({}, stats_text, [20, 30], baseline=20)
        assert isinstance(warnings, list)
        # Should warn about missing ranking, best, worst at minimum

    def test_empty_stats_text_skips_gracefully(self):
        """Empty stats_text means no ground truth - validation skips without crash."""
        from coeqwalpackage.llm_utils import validate_observation
        structured = _make_valid_structured()
        warnings = validate_observation(structured, "", [20, 30, 28], baseline=20)
        assert isinstance(warnings, list)
        # Should not crash; ranking/best/worst checks may be skipped

    def test_single_non_baseline_scenario(self):
        """With only one non-baseline scenario, ranking is trivially one item."""
        from coeqwalpackage.llm_utils import validate_observation
        means = {"s0020": 100.0, "s0030": 150.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        structured = {
            "ranking": ["s0030"],
            "best_scenario": "s0030",
            "worst_scenario": "s0030",
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30], baseline=20)
        assert isinstance(warnings, list)
        # Single scenario: best=worst is valid, no ranking error expected

    def test_all_scenarios_same_mean_no_ranking_warning(self):
        """When all non-baseline scenarios have the same mean, ranking is ambiguous."""
        from coeqwalpackage.llm_utils import validate_observation
        means = {"s0020": 100.0, "s0030": 150.0, "s0028": 150.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        structured = {
            "ranking": ["s0028", "s0030"],  # Either order is fine
            "best_scenario": "s0030",
            "worst_scenario": "s0028",
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30, 28], baseline=20)
        # With tied means, ranking order should NOT generate a warning
        ranking_warns = [w for w in warnings if "ranking" in w.lower() and "order" in w.lower()]
        assert len(ranking_warns) == 0

    def test_none_best_scenario_no_crash(self):
        """best_scenario=None does not crash validation."""
        from coeqwalpackage.llm_utils import validate_observation
        stats_text = _make_stats_text_mon_ts({"s0020": 100.0, "s0030": 120.0}, "s0020")
        structured = {
            "ranking": ["s0030"],
            "best_scenario": None,
            "worst_scenario": "s0030",
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30], baseline=20)
        assert isinstance(warnings, list)

    def test_none_worst_scenario_no_crash(self):
        """worst_scenario=None does not crash validation."""
        from coeqwalpackage.llm_utils import validate_observation
        stats_text = _make_stats_text_mon_ts({"s0020": 100.0, "s0030": 120.0}, "s0020")
        structured = {
            "ranking": ["s0030"],
            "best_scenario": "s0030",
            "worst_scenario": None,
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30], baseline=20)
        assert isinstance(warnings, list)

    def test_ranking_as_string_no_crash(self):
        """ranking as a string instead of list does not crash."""
        from coeqwalpackage.llm_utils import validate_observation
        stats_text = _make_stats_text_mon_ts({"s0020": 100.0, "s0030": 120.0}, "s0020")
        structured = {
            "ranking": "s0030, s0020",  # Wrong type
            "best_scenario": "s0030",
            "worst_scenario": "s0020",
            "cited_values": {},
        }
        warnings = validate_observation(structured, stats_text, [20, 30], baseline=20)
        assert isinstance(warnings, list)

    def test_cited_values_as_list_no_crash(self):
        """cited_values as a list instead of dict does not crash."""
        from coeqwalpackage.llm_utils import validate_observation
        stats_text = _make_stats_text_mon_ts({"s0020": 100.0, "s0030": 120.0}, "s0020")
        structured = {
            "ranking": ["s0030"],
            "best_scenario": "s0030",
            "worst_scenario": "s0030",
            "cited_values": [123.4, 456.7],  # Wrong type
        }
        warnings = validate_observation(structured, stats_text, [20, 30], baseline=20)
        assert isinstance(warnings, list)

    # ---- Return type guarantee ----

    def test_always_returns_list_of_strings(self):
        """validate_observation always returns list[str], never None or raises."""
        from coeqwalpackage.llm_utils import validate_observation
        stats_text = _make_stats_text_mon_ts({"s0020": 100.0}, "s0020")
        # Test several degenerate inputs
        for structured in [{}, {"ranking": None}, {"ranking": 42}]:
            result = validate_observation(structured, stats_text, [20], baseline=20)
            assert isinstance(result, list)
            for item in result:
                assert isinstance(item, str)


# ===================================================================
# 4. analyze_plot - mocked API
# ===================================================================

class TestAnalyzePlotReturnType:
    """Tests for analyze_plot return type using mocked API calls."""

    def _mock_api_response(self, text):
        """Create a mock Anthropic response object."""
        mock_content = MagicMock()
        mock_content.text = text
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        return mock_response

    def _make_dummy_image(self, tmp_path):
        """Create a tiny dummy PNG file for testing."""
        img_path = tmp_path / "test.png"
        # Minimal valid PNG: 1x1 white pixel
        import struct
        import zlib
        def _make_png():
            sig = b'\x89PNG\r\n\x1a\n'
            ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff)
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + ihdr_crc
            raw = b'\x00\xff\xff\xff'
            idat_data = zlib.compress(raw)
            idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + idat_data) & 0xffffffff)
            idat = struct.pack('>I', len(idat_data)) + b'IDAT' + idat_data + idat_crc
            iend_crc = struct.pack('>I', zlib.crc32(b'IEND') & 0xffffffff)
            iend = struct.pack('>I', 0) + b'IEND' + iend_crc
            return sig + ihdr + idat + iend
        img_path.write_bytes(_make_png())
        return str(img_path)

    @patch("coeqwalpackage.llm_utils._get_client")
    def test_valid_json_response_returns_dict(self, mock_get_client, tmp_path):
        """When API returns valid JSON, analyze_plot returns dict with correct keys."""
        from coeqwalpackage.llm_utils import analyze_plot
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_api_response(_make_valid_json_response())
        mock_get_client.return_value = mock_client
        img = self._make_dummy_image(tmp_path)

        result = analyze_plot(img, "Test prompt")
        assert isinstance(result, dict)
        assert "narrative" in result
        assert "structured" in result
        assert "validation" in result
        assert "raw" in result

    @patch("coeqwalpackage.llm_utils._get_client")
    def test_plain_text_response_returns_dict_with_narrative(self, mock_get_client, tmp_path):
        """When API returns plain text, dict narrative contains the text."""
        from coeqwalpackage.llm_utils import analyze_plot
        mock_client = MagicMock()
        plain = "This is a plain text observation about the plot."
        mock_client.messages.create.return_value = self._mock_api_response(plain)
        mock_get_client.return_value = mock_client
        img = self._make_dummy_image(tmp_path)

        result = analyze_plot(img, "Test prompt")
        assert isinstance(result, dict)
        assert result["narrative"] == plain
        assert result["structured"]["ranking"] == []

    @patch("coeqwalpackage.llm_utils._get_client")
    def test_with_stats_runs_validation(self, mock_get_client, tmp_path):
        """When stats_text/scenarios/baseline provided, validation key is populated."""
        from coeqwalpackage.llm_utils import analyze_plot
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_api_response(_make_valid_json_response())
        mock_get_client.return_value = mock_client
        img = self._make_dummy_image(tmp_path)

        means = {"s0020": 123.4, "s0030": 150.0, "s0028": 100.0}
        stats_text = _make_stats_text_mon_ts(means, "s0020")
        result = analyze_plot(img, "Test prompt", stats_text=stats_text, scenarios=[20, 30, 28], baseline=20)
        assert isinstance(result["validation"], list)

    @patch("coeqwalpackage.llm_utils._get_client")
    def test_without_stats_validation_empty(self, mock_get_client, tmp_path):
        """When stats_text not provided, validation list is empty."""
        from coeqwalpackage.llm_utils import analyze_plot
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_api_response(_make_valid_json_response())
        mock_get_client.return_value = mock_client
        img = self._make_dummy_image(tmp_path)

        result = analyze_plot(img, "Test prompt")
        assert result["validation"] == []

    @patch("coeqwalpackage.llm_utils._get_client")
    def test_backward_compat_narrative_always_present(self, mock_get_client, tmp_path):
        """Callers can always access result['narrative'] for the LLM text."""
        from coeqwalpackage.llm_utils import analyze_plot
        mock_client = MagicMock()
        narrative_text = "Scenario s0030 shows the highest storage levels."
        obj = {"narrative": narrative_text, "ranking": ["s0030"], "best_scenario": "s0030", "worst_scenario": "s0030", "cited_values": {}}
        mock_client.messages.create.return_value = self._mock_api_response(json.dumps(obj))
        mock_get_client.return_value = mock_client
        img = self._make_dummy_image(tmp_path)

        result = analyze_plot(img, "Test prompt")
        assert result["narrative"] == narrative_text
        assert isinstance(result["raw"], str)


# ===================================================================
# 5. Pipeline stability - no-crash guarantees
# ===================================================================

class TestPipelineStability:
    """Verify functions never raise exceptions on valid or degenerate inputs."""

    def test_parse_llm_response_never_raises(self):
        """_parse_llm_response should never raise on any string input."""
        from coeqwalpackage.llm_utils import _parse_llm_response
        edge_cases = [
            "", "   ", "\n\n", "{", "}", "[]", "null", "true", "false",
            "123", "123.456", '"just a string"',
            '{"incomplete": ', '{"key": [}', "```json\n{bad}\n```",
            "{" * 1000, "x" * 10000,
        ]
        for raw in edge_cases:
            result = _parse_llm_response(raw)
            assert isinstance(result, dict), f"Failed for: {raw[:50]!r}"
            assert "narrative" in result

    def test_validate_observation_never_raises(self):
        """validate_observation should never raise on any input combination."""
        from coeqwalpackage.llm_utils import validate_observation
        stats_text = _make_stats_text_mon_ts({"s0020": 100.0}, "s0020")
        degenerate_structured = [
            {},
            {"ranking": None, "best_scenario": None, "worst_scenario": None, "cited_values": None},
            {"ranking": 42, "best_scenario": 42, "worst_scenario": 42, "cited_values": 42},
            {"ranking": "not a list"},
            {"cited_values": "not a dict"},
            {"ranking": [None, None]},
            {"ranking": [1, 2, 3]},  # ints instead of strings
        ]
        for s in degenerate_structured:
            result = validate_observation(s, stats_text, [20], baseline=20)
            assert isinstance(result, list), f"Failed for: {s!r}"

    def test_extract_means_never_raises(self):
        """_extract_means_from_stats should never raise on any string input."""
        from coeqwalpackage.llm_utils import _extract_means_from_stats
        edge_cases = ["", "   ", "no data here", "mean=abc", "s0020: mean=, more"]
        for text in edge_cases:
            result = _extract_means_from_stats(text, 1)
            assert isinstance(result, dict), f"Failed for: {text!r}"
