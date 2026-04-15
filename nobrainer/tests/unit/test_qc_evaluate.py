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
