"""Unit tests for nobrainer.qc.gate."""

from __future__ import annotations

import pytest


class TestQCGate:
    def test_accept(self):
        from nobrainer.qc.gate import QCGate

        gate = QCGate(accept_threshold=3.5, reject_threshold=2.0)
        decision = gate.decide(4.0)
        assert decision.action == "accept"

    def test_reject(self):
        from nobrainer.qc.gate import QCGate

        gate = QCGate(accept_threshold=3.5, reject_threshold=2.0)
        decision = gate.decide(1.5)
        assert decision.action == "reject"

    def test_review(self):
        from nobrainer.qc.gate import QCGate

        gate = QCGate(accept_threshold=3.5, reject_threshold=2.0)
        decision = gate.decide(2.5)
        assert decision.action == "review"

    def test_nan_triggers_review(self):
        from nobrainer.qc.gate import QCGate

        gate = QCGate()
        decision = gate.decide(float("nan"))
        assert decision.action == "review"

    def test_boundary_accept(self):
        from nobrainer.qc.gate import QCGate

        gate = QCGate(accept_threshold=3.5, reject_threshold=2.0)
        decision = gate.decide(3.5)
        assert decision.action == "accept"

    def test_boundary_reject(self):
        from nobrainer.qc.gate import QCGate

        gate = QCGate(accept_threshold=3.5, reject_threshold=2.0)
        decision = gate.decide(2.0)
        assert decision.action == "reject"

    def test_invalid_thresholds(self):
        from nobrainer.qc.gate import QCGate

        with pytest.raises(ValueError, match="must be <"):
            QCGate(accept_threshold=2.0, reject_threshold=3.0)

    def test_equal_thresholds_invalid(self):
        from nobrainer.qc.gate import QCGate

        with pytest.raises(ValueError, match="must be <"):
            QCGate(accept_threshold=3.0, reject_threshold=3.0)

    def test_decision_has_reason(self):
        from nobrainer.qc.gate import QCGate

        gate = QCGate()
        decision = gate.decide(4.0)
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0
