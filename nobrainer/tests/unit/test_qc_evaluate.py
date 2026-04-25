"""Unit tests for nobrainer.qc.evaluate."""

from __future__ import annotations


class TestParseQcResponse:
    def test_structured_format(self):
        from nobrainer.qc.evaluate import parse_qc_response

        text = "SCORE: 4\nREASON: Good quality scan with minimal noise."
        result = parse_qc_response(text)
        assert result["score"] == 4
        assert result["parse_success"] is True
        assert "Good quality" in result["reason"]

    def test_fallback_digit_extraction(self):
        from nobrainer.qc.evaluate import parse_qc_response

        text = "This scan is rated 3 out of 5 due to moderate motion."
        result = parse_qc_response(text)
        assert result["score"] == 3
        assert result["parse_success"] is False

    def test_no_score_found(self):
        from nobrainer.qc.evaluate import parse_qc_response

        text = "Unable to determine quality."
        result = parse_qc_response(text)
        assert result["score"] is None
        assert result["parse_success"] is False

    def test_out_of_range_score_ignored(self):
        from nobrainer.qc.evaluate import parse_qc_response

        text = "SCORE: 9\nREASON: Invalid score."
        result = parse_qc_response(text)
        # 9 is out of 1-5 range, should not be accepted as structured
        assert result["score"] is None or result["parse_success"] is False

    def test_reason_without_score_prefix(self):
        from nobrainer.qc.evaluate import parse_qc_response

        text = "The image has severe motion artifacts and is unusable."
        result = parse_qc_response(text)
        assert result["reason"] == text.strip()

    def test_multiline_reason(self):
        from nobrainer.qc.evaluate import parse_qc_response

        text = (
            "SCORE: 2\n"
            "REASON: Severe motion artifacts visible.\n"
            "Additional ghosting in the posterior region."
        )
        result = parse_qc_response(text)
        assert result["score"] == 2
        assert result["parse_success"] is True
        assert "ghosting" in result["reason"]

    def test_qc_prompt_is_string(self):
        from nobrainer.qc.evaluate import QC_PROMPT

        assert isinstance(QC_PROMPT, str)
        assert "SCORE" in QC_PROMPT
        assert "REASON" in QC_PROMPT


class TestParseDualQcResponse:
    def test_structured_format(self):
        from nobrainer.qc.evaluate import parse_dual_qc_response

        result = parse_dual_qc_response("Quality: 4 Thickness: 3")
        assert result["quality"] == 4
        assert result["thickness"] == 3
        assert result["parse_success"] is True

    def test_case_insensitive(self):
        from nobrainer.qc.evaluate import parse_dual_qc_response

        result = parse_dual_qc_response("quality: 5 thickness: 2")
        assert result["quality"] == 5
        assert result["thickness"] == 2
        assert result["parse_success"] is True

    def test_decimal_values_truncated(self):
        from nobrainer.qc.evaluate import parse_dual_qc_response

        result = parse_dual_qc_response("Quality: 4.0 Thickness: 3.5")
        assert result["quality"] == 4
        assert result["thickness"] == 3
        assert result["parse_success"] is True

    def test_multiline_format(self):
        from nobrainer.qc.evaluate import parse_dual_qc_response

        result = parse_dual_qc_response("Quality: 2\nThickness: 4")
        assert result["quality"] == 2
        assert result["thickness"] == 4
        assert result["parse_success"] is True

    def test_quality_out_of_range_rejected(self):
        from nobrainer.qc.evaluate import parse_dual_qc_response

        result = parse_dual_qc_response("Quality: 9 Thickness: 3")
        assert result["quality"] is None
        assert result["thickness"] == 3
        assert result["parse_success"] is False

    def test_thickness_out_of_range_rejected(self):
        from nobrainer.qc.evaluate import parse_dual_qc_response

        result = parse_dual_qc_response("Quality: 4 Thickness: 0")
        assert result["quality"] == 4
        assert result["thickness"] is None
        assert result["parse_success"] is False

    def test_only_quality_present(self):
        from nobrainer.qc.evaluate import parse_dual_qc_response

        result = parse_dual_qc_response("Quality: 4")
        assert result["quality"] == 4
        assert result["thickness"] is None
        assert result["parse_success"] is False

    def test_no_match(self):
        from nobrainer.qc.evaluate import parse_dual_qc_response

        result = parse_dual_qc_response("Unable to assess this scan.")
        assert result["quality"] is None
        assert result["thickness"] is None
        assert result["parse_success"] is False

    def test_qc_dual_prompt_constant(self):
        from nobrainer.qc.evaluate import QC_DUAL_PROMPT

        assert isinstance(QC_DUAL_PROMPT, str)
        assert "Quality:" in QC_DUAL_PROMPT
        assert "Thickness:" in QC_DUAL_PROMPT
