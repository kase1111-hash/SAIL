"""
Unit tests for the Intervention Framework (Phase 7).

Tests cover:
- Base types and enumerations
- Risk assessment engine
- Mode handlers (Ambient, Advisory, Guardian, Crisis)
- Intervention engine orchestration
"""

import asyncio
from datetime import datetime, timedelta

import pytest
from src.intervention.base import (
    Intervention,
    InterventionConfig,
    InterventionEventType,
    InterventionMode,
    InterventionPriority,
    InterventionResponse,
    InterventionType,
    RiskAssessment,
    RiskCategory,
    RiskFactor,
    RiskLevel,
)

# ============================================================================
# Base Types Tests
# ============================================================================


class TestInterventionMode:
    """Tests for InterventionMode enum."""

    def test_all_modes_defined(self):
        """Test all intervention modes are defined."""
        modes = list(InterventionMode)
        assert len(modes) == 4
        assert InterventionMode.AMBIENT in modes
        assert InterventionMode.ADVISORY in modes
        assert InterventionMode.GUARDIAN in modes
        assert InterventionMode.CRISIS in modes

    def test_mode_priority(self):
        """Test mode priority ordering."""
        assert InterventionMode.AMBIENT.get_priority() < InterventionMode.ADVISORY.get_priority()
        assert InterventionMode.ADVISORY.get_priority() < InterventionMode.GUARDIAN.get_priority()
        assert InterventionMode.GUARDIAN.get_priority() < InterventionMode.CRISIS.get_priority()

    def test_mode_values(self):
        """Test mode string values."""
        assert InterventionMode.AMBIENT.value == "ambient"
        assert InterventionMode.CRISIS.value == "crisis"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_all_levels_defined(self):
        """Test all risk levels are defined."""
        levels = list(RiskLevel)
        assert len(levels) == 5

    def test_level_scores(self):
        """Test risk level scoring."""
        assert RiskLevel.NONE.get_score() == 0
        assert RiskLevel.LOW.get_score() == 1
        assert RiskLevel.CRITICAL.get_score() == 4

    def test_from_score(self):
        """Test creating risk level from score."""
        assert RiskLevel.from_score(0) == RiskLevel.NONE
        assert RiskLevel.from_score(2) == RiskLevel.MODERATE
        assert RiskLevel.from_score(10) == RiskLevel.CRITICAL


class TestRiskFactor:
    """Tests for RiskFactor data class."""

    def test_factor_creation(self):
        """Test creating a risk factor."""
        factor = RiskFactor(
            category=RiskCategory.LOCATION,
            name="unfamiliar_area",
            description="User is in unfamiliar area",
            weight=0.7,
            score=0.8,
        )

        assert factor.category == RiskCategory.LOCATION
        assert factor.name == "unfamiliar_area"
        assert factor.weight == 0.7
        assert factor.score == 0.8

    def test_weighted_score(self):
        """Test weighted score calculation."""
        factor = RiskFactor(
            category=RiskCategory.LOCATION,
            name="test",
            description="Test",
            weight=0.5,
            score=0.8,
            confidence=1.0,
        )

        assert factor.weighted_score == 0.4  # 0.5 * 0.8 * 1.0

    def test_weighted_score_with_confidence(self):
        """Test weighted score with reduced confidence."""
        factor = RiskFactor(
            category=RiskCategory.LOCATION,
            name="test",
            description="Test",
            weight=0.5,
            score=0.8,
            confidence=0.5,
        )

        assert factor.weighted_score == 0.2  # 0.5 * 0.8 * 0.5

    def test_factor_to_dict(self):
        """Test converting factor to dictionary."""
        factor = RiskFactor(
            category=RiskCategory.TEMPORAL,
            name="late_night",
            description="Late night hours",
            weight=0.6,
            score=0.5,
        )

        data = factor.to_dict()
        assert data["category"] == "temporal"
        assert data["name"] == "late_night"
        assert "weighted_score" in data


class TestIntervention:
    """Tests for Intervention data class."""

    def test_intervention_creation(self):
        """Test creating an intervention."""
        intervention = Intervention(
            id="test_001",
            type=InterventionType.SUGGESTION,
            priority=InterventionPriority.NORMAL,
            mode=InterventionMode.ADVISORY,
            message="Test message",
        )

        assert intervention.id == "test_001"
        assert intervention.type == InterventionType.SUGGESTION
        assert intervention.mode == InterventionMode.ADVISORY

    def test_intervention_not_expired(self):
        """Test intervention expiration check - not expired."""
        intervention = Intervention(
            id="test",
            type=InterventionType.INFO,
            priority=InterventionPriority.LOW,
            mode=InterventionMode.AMBIENT,
            message="Test",
            expires_at=datetime.now() + timedelta(hours=1),
        )

        assert not intervention.is_expired()

    def test_intervention_expired(self):
        """Test intervention expiration check - expired."""
        intervention = Intervention(
            id="test",
            type=InterventionType.INFO,
            priority=InterventionPriority.LOW,
            mode=InterventionMode.AMBIENT,
            message="Test",
            expires_at=datetime.now() - timedelta(hours=1),
        )

        assert intervention.is_expired()

    def test_intervention_to_dict(self):
        """Test converting intervention to dictionary."""
        intervention = Intervention(
            id="test",
            type=InterventionType.WARNING,
            priority=InterventionPriority.HIGH,
            mode=InterventionMode.GUARDIAN,
            message="Warning message",
            details="Details here",
        )

        data = intervention.to_dict()
        assert data["id"] == "test"
        assert data["type"] == "warning"
        assert data["priority"] == "high"


class TestInterventionConfig:
    """Tests for InterventionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = InterventionConfig()

        assert config.ambient_threshold == 0.0
        assert config.advisory_threshold == 0.25
        assert config.guardian_threshold == 0.5
        assert config.crisis_threshold == 0.75

    def test_custom_config(self):
        """Test custom configuration."""
        config = InterventionConfig(
            advisory_threshold=0.3,
            guardian_threshold=0.6,
        )

        assert config.advisory_threshold == 0.3
        assert config.guardian_threshold == 0.6


# ============================================================================
# Risk Assessment Tests
# ============================================================================


class TestRiskAssessmentEngine:
    """Tests for RiskAssessmentEngine."""

    def test_engine_creation(self):
        """Test creating risk assessment engine."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        assert engine is not None

    def test_assess_empty_context(self):
        """Test assessment with empty context."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        assessment = engine.assess({})

        assert assessment.level == RiskLevel.NONE
        assert assessment.score == 0.0
        assert assessment.recommended_mode == InterventionMode.AMBIENT

    def test_assess_low_risk(self):
        """Test assessment with low risk context."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        assessment = engine.assess({
            "location_context": "familiar",
            "is_late_night": False,
        })

        assert assessment.level in (RiskLevel.NONE, RiskLevel.LOW)

    def test_assess_moderate_risk(self):
        """Test assessment with moderate risk context."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        assessment = engine.assess({
            "location_context": "unfamiliar",
            "is_late_night": True,
        })

        assert assessment.level in (RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH)
        assert len(assessment.factors) > 0

    def test_assess_high_risk(self):
        """Test assessment with high risk context."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        assessment = engine.assess({
            "location_context": "unfamiliar",
            "is_late_night": True,
            "stress_level": "high",
            "threat_cues": ["raised_voices"],
        })

        assert assessment.level in (RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert assessment.recommended_mode != InterventionMode.AMBIENT

    def test_high_speed_driving_is_detected(self):
        """speed_mps was read from context and then never used."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        assessment = engine.assess({
            "motion_state": "driving",
            "speed_mps": 40.0,  # 144 km/h
        })

        names = [f.name for f in assessment.factors]
        assert "high_speed_driving" in names

    def test_normal_speed_driving_is_not_flagged(self):
        """Ordinary highway speed must not register as a risk factor."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        assessment = engine.assess({
            "motion_state": "driving",
            "speed_mps": 25.0,  # 90 km/h
        })

        names = [f.name for f in assessment.factors]
        assert "high_speed_driving" not in names

    def test_high_speed_ignored_when_not_driving(self):
        """The factor is about driving, so it needs the motion state to agree."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        assessment = engine.assess({"motion_state": "stationary", "speed_mps": 40.0})

        names = [f.name for f in assessment.factors]
        assert "high_speed_driving" not in names

    def test_repeated_pattern_is_counted_once(self):
        """Late night + unfamiliar + driving raised late_night_travel twice.

        The score is a weighted average, so the repeat doubled that pattern's
        weight against every other factor instead of adding information.
        """
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        assessment = engine.assess({
            "location_context": "unfamiliar",
            "is_late_night": True,
            "motion_state": "driving",
        })

        names = [f.name for f in assessment.factors]
        assert names.count("late_night_travel") == 1
        assert len(names) == len(set(names))

    def test_repeated_pattern_keeps_highest_score(self):
        """Collapsing repeats must not discard the more severe reading."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        assessment = engine.assess({
            "location_context": "unfamiliar",
            "is_late_night": True,
            "motion_state": "driving",
        })

        travel = next(
            f for f in assessment.factors if f.name == "late_night_travel"
        )
        assert travel.score == 0.8

    def test_risk_speed_threshold_matches_fusion(self):
        """The two risk paths score the same signal and must not drift apart."""
        from src.intervention.risk import HIGH_SPEED_KMH
        from src.sensors.fusion import HIGH_SPEED_KMH as FUSION_HIGH_SPEED_KMH

        assert HIGH_SPEED_KMH == FUSION_HIGH_SPEED_KMH

    def test_assess_with_factors(self):
        """Test assessment with pre-detected factors."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        factors = [
            RiskFactor(
                category=RiskCategory.LOCATION,
                name="test_factor",
                description="Test",
                weight=0.8,
                score=0.9,
            )
        ]

        assessment = engine.assess({}, detected_factors=factors)
        assert len(assessment.factors) == 1

    def test_dismissal_tracking(self):
        """Test false positive dismissal tracking."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()
        config = InterventionConfig(false_positive_threshold=2)
        engine.config = config

        # Dismiss pattern multiple times
        engine.record_dismissal("test_pattern")
        engine.record_dismissal("test_pattern")

        # Pattern should be suppressed
        assert "test_pattern" in engine._suppressed_patterns

    def test_risk_trend(self):
        """Test risk trend analysis."""
        from src.intervention.risk import RiskAssessmentEngine

        engine = RiskAssessmentEngine()

        # Generate some assessments
        for _ in range(5):
            engine.assess({"is_late_night": False})

        trend = engine.get_risk_trend()
        assert trend in ("increasing", "stable", "decreasing")


class TestRiskPatternDetector:
    """Tests for RiskPatternDetector."""

    def test_detector_creation(self):
        """Test creating pattern detector."""
        from src.intervention.risk import RiskPatternDetector

        detector = RiskPatternDetector()
        assert detector is not None

    def test_detect_pressure_tactics(self):
        """Test detecting pressure tactics."""
        from src.intervention.risk import RiskPatternDetector

        detector = RiskPatternDetector()

        # Text with pressure tactics
        text = "You need to act now! This is urgent, don't tell anyone."
        factor = detector.detect_pressure_tactics(text)

        assert factor is not None
        assert factor.category == RiskCategory.SOCIAL
        assert "pressure" in factor.name

    def test_detect_financial_risk(self):
        """Test detecting financial risk."""
        from src.intervention.risk import RiskPatternDetector

        detector = RiskPatternDetector()

        text = "Please send a wire transfer immediately and include your bank account number."
        factor = detector.detect_financial_risk(text)

        assert factor is not None
        assert factor.category == RiskCategory.FINANCIAL

    def test_no_false_positive(self):
        """Test no false positive on normal text."""
        from src.intervention.risk import RiskPatternDetector

        detector = RiskPatternDetector()

        text = "Let's meet for coffee tomorrow at 3pm."
        pressure = detector.detect_pressure_tactics(text)
        financial = detector.detect_financial_risk(text)

        assert pressure is None
        assert financial is None


# ============================================================================
# Mode Handler Tests
# ============================================================================


class TestAmbientModeHandler:
    """Tests for AmbientModeHandler."""

    @pytest.mark.asyncio
    async def test_handler_creation(self):
        """Test creating ambient handler."""
        from src.intervention.modes import AmbientModeHandler

        handler = AmbientModeHandler()
        assert handler.mode == InterventionMode.AMBIENT

    @pytest.mark.asyncio
    async def test_no_intervention_when_not_addressed(self):
        """Test no intervention when user hasn't addressed system."""
        from src.intervention.modes import AmbientModeHandler

        handler = AmbientModeHandler()
        handler.activate()

        assessment = RiskAssessment(
            level=RiskLevel.MODERATE,
            score=0.4,
            factors=[],
            recommended_mode=InterventionMode.ADVISORY,
            summary="Test",
        )

        intervention = await handler.evaluate({}, assessment)
        assert intervention is None

    @pytest.mark.asyncio
    async def test_intervention_when_addressed(self):
        """Test intervention when user addresses system."""
        from src.intervention.modes import AmbientModeHandler

        handler = AmbientModeHandler()
        handler.activate()
        handler.set_addressed(True)

        assessment = RiskAssessment(
            level=RiskLevel.NONE,
            score=0.0,
            factors=[],
            recommended_mode=InterventionMode.AMBIENT,
            summary="Test",
        )

        intervention = await handler.evaluate({}, assessment)
        assert intervention is not None
        assert intervention.type == InterventionType.INFO


class TestAdvisoryModeHandler:
    """Tests for AdvisoryModeHandler."""

    @pytest.mark.asyncio
    async def test_handler_creation(self):
        """Test creating advisory handler."""
        from src.intervention.modes import AdvisoryModeHandler

        handler = AdvisoryModeHandler()
        assert handler.mode == InterventionMode.ADVISORY

    @pytest.mark.asyncio
    async def test_suggestion_for_low_risk(self):
        """Test suggestion generation for low risk."""
        from src.intervention.modes import AdvisoryModeHandler

        handler = AdvisoryModeHandler()
        handler.activate()

        factor = RiskFactor(
            category=RiskCategory.LOCATION,
            name="unfamiliar_area",
            description="In unfamiliar area",
            weight=0.6,
            score=0.5,
        )

        assessment = RiskAssessment(
            level=RiskLevel.LOW,
            score=0.2,
            factors=[factor],
            recommended_mode=InterventionMode.ADVISORY,
            summary="Test",
        )

        intervention = await handler.evaluate({}, assessment)
        # May or may not generate intervention based on cooldown
        if intervention:
            assert intervention.type == InterventionType.SUGGESTION
            assert intervention.priority == InterventionPriority.LOW


class TestGuardianModeHandler:
    """Tests for GuardianModeHandler."""

    @pytest.mark.asyncio
    async def test_handler_creation(self):
        """Test creating guardian handler."""
        from src.intervention.modes import GuardianModeHandler

        handler = GuardianModeHandler()
        assert handler.mode == InterventionMode.GUARDIAN

    @pytest.mark.asyncio
    async def test_alert_for_high_risk(self):
        """Test alert generation for high risk."""
        from src.intervention.modes import GuardianModeHandler

        handler = GuardianModeHandler()
        handler.activate()

        factor = RiskFactor(
            category=RiskCategory.ENVIRONMENTAL,
            name="threat_sounds",
            description="Threat sounds detected",
            weight=0.9,
            score=0.8,
        )

        assessment = RiskAssessment(
            level=RiskLevel.HIGH,
            score=0.6,
            factors=[factor],
            recommended_mode=InterventionMode.GUARDIAN,
            summary="Test",
        )

        intervention = await handler.evaluate({}, assessment)
        if intervention:
            assert intervention.priority == InterventionPriority.HIGH
            assert intervention.type in (InterventionType.WARNING, InterventionType.VERIFICATION)


class TestCrisisModeHandler:
    """Tests for CrisisModeHandler."""

    @pytest.mark.asyncio
    async def test_handler_creation(self):
        """Test creating crisis handler."""
        from src.intervention.modes import CrisisModeHandler

        handler = CrisisModeHandler()
        assert handler.mode == InterventionMode.CRISIS

    @pytest.mark.asyncio
    async def test_guidance_for_critical_risk(self):
        """Test guidance generation for critical risk."""
        from src.intervention.modes import CrisisModeHandler

        handler = CrisisModeHandler()
        handler.activate()

        factor = RiskFactor(
            category=RiskCategory.ENVIRONMENTAL,
            name="threat_sounds",
            description="Threat detected",
            weight=0.9,
            score=0.95,
        )

        assessment = RiskAssessment(
            level=RiskLevel.CRITICAL,
            score=0.9,
            factors=[factor],
            recommended_mode=InterventionMode.CRISIS,
            summary="Critical situation",
        )

        intervention = await handler.evaluate({}, assessment)
        assert intervention is not None
        assert intervention.priority == InterventionPriority.CRITICAL
        assert intervention.type == InterventionType.GUIDANCE

    @pytest.mark.asyncio
    async def test_emergency_response_handling(self):
        """Test handling emergency keyword in response."""
        from src.intervention.modes import CrisisModeHandler

        handler = CrisisModeHandler()
        handler.activate()

        intervention = Intervention(
            id="test",
            type=InterventionType.GUIDANCE,
            priority=InterventionPriority.CRITICAL,
            mode=InterventionMode.CRISIS,
            message="Test",
        )

        response = InterventionResponse(
            intervention_id="test",
            acknowledged=True,
            response_type="custom",
            response_text="I need help please call 911",
        )

        follow_up = await handler.handle_response(intervention, response)
        assert follow_up is not None
        assert follow_up.type == InterventionType.EMERGENCY

    @pytest.mark.asyncio
    async def test_red_safety_word(self):
        """Test 'red' safety word triggers silent emergency."""
        from src.intervention.modes import CrisisModeHandler

        handler = CrisisModeHandler()
        handler.activate()

        intervention = Intervention(
            id="test",
            type=InterventionType.GUIDANCE,
            priority=InterventionPriority.CRITICAL,
            mode=InterventionMode.CRISIS,
            message="Test",
        )

        response = InterventionResponse(
            intervention_id="test",
            acknowledged=True,
            response_type="custom",
            response_text="red",
        )

        follow_up = await handler.handle_response(intervention, response)
        assert follow_up is not None
        assert follow_up.metadata.get("silent_emergency") is True


# ============================================================================
# Intervention Engine Tests
# ============================================================================


class TestInterventionQueue:
    """Tests for InterventionQueue."""

    def test_queue_creation(self):
        """Test creating intervention queue."""
        from src.intervention.engine import InterventionQueue

        queue = InterventionQueue()
        assert len(queue) == 0

    def test_add_and_get(self):
        """Test adding and getting interventions."""
        from src.intervention.engine import InterventionQueue

        queue = InterventionQueue()

        intervention = Intervention(
            id="test",
            type=InterventionType.INFO,
            priority=InterventionPriority.NORMAL,
            mode=InterventionMode.AMBIENT,
            message="Test",
        )

        queue.add(intervention)
        assert len(queue) == 1

        retrieved = queue.get_next()
        assert retrieved is not None
        assert retrieved.id == "test"
        assert len(queue) == 0

    def test_priority_ordering(self):
        """Test interventions are ordered by priority."""
        from src.intervention.engine import InterventionQueue

        queue = InterventionQueue()

        low = Intervention(
            id="low",
            type=InterventionType.INFO,
            priority=InterventionPriority.LOW,
            mode=InterventionMode.AMBIENT,
            message="Low",
        )

        high = Intervention(
            id="high",
            type=InterventionType.ALERT,
            priority=InterventionPriority.HIGH,
            mode=InterventionMode.GUARDIAN,
            message="High",
        )

        queue.add(low)
        queue.add(high)

        # High priority should come first
        first = queue.get_next()
        assert first.id == "high"

    def test_expired_removal(self):
        """Test expired interventions are removed."""
        from src.intervention.engine import InterventionQueue

        queue = InterventionQueue()

        expired = Intervention(
            id="expired",
            type=InterventionType.INFO,
            priority=InterventionPriority.NORMAL,
            mode=InterventionMode.AMBIENT,
            message="Expired",
            expires_at=datetime.now() - timedelta(hours=1),
        )

        queue.add(expired)

        # Should be removed when getting next
        result = queue.get_next()
        assert result is None


class TestInterventionEngine:
    """Tests for InterventionEngine."""

    @pytest.mark.asyncio
    async def test_engine_creation(self):
        """Test creating intervention engine."""
        from src.intervention.engine import InterventionEngine

        engine = InterventionEngine()
        assert engine.current_mode == InterventionMode.AMBIENT

    @pytest.mark.asyncio
    async def test_evaluate_low_risk(self):
        """Test evaluation with low risk context."""
        from src.intervention.engine import InterventionEngine

        engine = InterventionEngine()
        await engine.start()

        intervention = await engine.evaluate({
            "location_context": "home",
            "is_late_night": False,
        })

        # Ambient mode doesn't proactively intervene
        assert intervention is None
        assert engine.current_mode == InterventionMode.AMBIENT

    @pytest.mark.asyncio
    async def test_mode_escalation(self):
        """Test mode escalation on high risk."""
        from src.intervention.engine import InterventionEngine

        config = InterventionConfig(
            mode_escalation_delay=0.0,  # Immediate escalation for test
        )
        engine = InterventionEngine(config)
        await engine.start()

        # High risk context should escalate
        await engine.evaluate({
            "location_context": "unfamiliar",
            "is_late_night": True,
            "stress_level": "high",
            "threat_cues": ["raised_voices", "screaming"],
        })

        # Mode should escalate
        assert engine.current_mode != InterventionMode.AMBIENT

    @pytest.mark.asyncio
    async def test_force_mode(self):
        """Test forcing a specific mode."""
        from src.intervention.engine import InterventionEngine

        engine = InterventionEngine()
        await engine.start()

        engine.force_mode(InterventionMode.GUARDIAN, "Test override")

        # Allow async transition to complete
        await asyncio.sleep(0.1)

        assert engine.current_mode == InterventionMode.GUARDIAN

    @pytest.mark.asyncio
    async def test_response_handling(self):
        """Test handling user response."""
        from src.intervention.engine import InterventionEngine
        from src.intervention.modes import generate_intervention_id

        engine = InterventionEngine()
        await engine.start()

        # Create and track an intervention
        intervention = Intervention(
            id=generate_intervention_id(),
            type=InterventionType.WARNING,
            priority=InterventionPriority.HIGH,
            mode=InterventionMode.GUARDIAN,
            message="Test warning",
            metadata={"factor": "test_factor"},
        )
        engine._delivered_interventions[intervention.id] = intervention

        response = InterventionResponse(
            intervention_id=intervention.id,
            acknowledged=False,  # Dismissed, not acknowledged
            response_type="dismissed",
        )

        follow_up = await engine.handle_response(intervention.id, response)
        # Dismissal tracking should work
        assert "test_factor" in engine._risk_engine._dismissed_patterns

    @pytest.mark.asyncio
    async def test_set_addressed(self):
        """Test setting addressed state for ambient mode."""
        from src.intervention.engine import InterventionEngine

        engine = InterventionEngine()
        await engine.start()

        engine.set_addressed(True)

        # Now ambient mode should respond
        intervention = await engine.evaluate({})
        assert intervention is not None

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting engine statistics."""
        from src.intervention.engine import InterventionEngine

        engine = InterventionEngine()
        await engine.start()

        await engine.evaluate({})

        stats = engine.get_stats()

        assert "current_mode" in stats
        assert "pending_interventions" in stats
        assert "risk_engine_stats" in stats

    @pytest.mark.asyncio
    async def test_event_callback(self):
        """Test event callback notification."""
        from src.intervention.engine import InterventionEngine

        engine = InterventionEngine()

        events = []

        def on_event(event):
            events.append(event)

        engine.register_event_callback(on_event)
        await engine.start()

        engine.force_mode(InterventionMode.ADVISORY, "Test")
        await asyncio.sleep(0.1)

        # Should have mode change event
        mode_events = [e for e in events if e.event_type == InterventionEventType.MODE_CHANGED]
        assert len(mode_events) >= 1


class TestCreateInterventionEngine:
    """Tests for create_intervention_engine factory."""

    @pytest.mark.asyncio
    async def test_factory_default(self):
        """Test factory with default config."""
        from src.intervention.engine import create_intervention_engine

        engine = create_intervention_engine()
        assert engine is not None
        assert engine.current_mode == InterventionMode.AMBIENT

    @pytest.mark.asyncio
    async def test_factory_custom_config(self):
        """Test factory with custom config."""
        from src.intervention.engine import create_intervention_engine

        config = InterventionConfig(advisory_threshold=0.1)
        engine = create_intervention_engine(config)

        assert engine.config.advisory_threshold == 0.1


# ============================================================================
# Integration Tests
# ============================================================================


class TestInterventionIntegration:
    """Integration tests for intervention framework."""

    @pytest.mark.asyncio
    async def test_full_intervention_cycle(self):
        """Test complete intervention cycle from evaluation to response."""
        from src.intervention.engine import create_intervention_engine

        engine = create_intervention_engine(InterventionConfig(
            mode_escalation_delay=0.0,
        ))
        await engine.start()

        # Escalate to advisory mode with risk context
        await engine.evaluate({
            "location_context": "unfamiliar",
            "is_late_night": True,
        })

        # Force to guardian for testing
        engine.force_mode(InterventionMode.GUARDIAN, "Test")
        await asyncio.sleep(0.1)

        # Evaluate again to generate intervention
        intervention = await engine.evaluate({
            "threat_cues": ["raised_voices"],
            "stress_level": "high",
        })

        # May have generated an intervention
        if intervention:
            # Respond to it
            response = InterventionResponse(
                intervention_id=intervention.id,
                acknowledged=True,
                response_type="accepted",
            )

            await engine.handle_response(intervention.id, response)

        await engine.stop()

    @pytest.mark.asyncio
    async def test_crisis_escalation_flow(self):
        """Test escalation to crisis mode."""
        from src.intervention.engine import create_intervention_engine

        config = InterventionConfig(
            mode_escalation_delay=0.0,
            crisis_threshold=0.7,
        )
        engine = create_intervention_engine(config)
        await engine.start()

        # Critical context should trigger crisis
        await engine.evaluate({
            "location_context": "unfamiliar",
            "is_late_night": True,
            "stress_level": "high",
            "threat_cues": ["raised_voices", "screaming"],
        })

        # Should be in higher mode
        assert engine.current_mode.get_priority() > InterventionMode.AMBIENT.get_priority()

        await engine.stop()
