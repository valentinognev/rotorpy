"""Proposed vehicle-envelope limits and PN-lead design constant live in homing_config,
named and documented (plan Phase 7)."""

from rotorpy.guidance.command_limits import VehicleEnvelope
from rotorpy.guidance.homing_config import (
    PROPOSED_ENABLE_ACCELERATION_FEEDFORWARD,
    PROPOSED_PN_LEAD_TIME_S,
    PROPOSED_VEHICLE_MAX_ACCEL_M_S2,
    PROPOSED_VEHICLE_MAX_JERK_M_S3,
    PROPOSED_VEHICLE_MAX_SPEED_M_S,
    PROPOSED_VEHICLE_MAX_VERTICAL_SPEED_M_S,
    PROPOSED_VEHICLE_MAX_YAW_RATE_RAD_S,
    PROPOSED_VISIBILITY_MARGIN_FRAC,
    PROPOSED_VISIBILITY_MIN_SPEED_SCALE,
    vehicle_envelope,
)


def test_vehicle_envelope_thresholds_are_positive_and_named():
    thresholds = vehicle_envelope()
    assert thresholds == {
        "max_speed_m_s": PROPOSED_VEHICLE_MAX_SPEED_M_S,
        "max_accel_m_s2": PROPOSED_VEHICLE_MAX_ACCEL_M_S2,
        "max_jerk_m_s3": PROPOSED_VEHICLE_MAX_JERK_M_S3,
        "max_yaw_rate_rad_s": PROPOSED_VEHICLE_MAX_YAW_RATE_RAD_S,
        "max_vertical_speed_m_s": PROPOSED_VEHICLE_MAX_VERTICAL_SPEED_M_S,
        "visibility_margin_frac": PROPOSED_VISIBILITY_MARGIN_FRAC,
        "visibility_min_speed_scale": PROPOSED_VISIBILITY_MIN_SPEED_SCALE,
    }
    assert all(value > 0.0 for value in thresholds.values())


def test_vehicle_envelope_builds_a_working_envelope():
    envelope = VehicleEnvelope(**vehicle_envelope())
    assert envelope.max_speed_m_s == PROPOSED_VEHICLE_MAX_SPEED_M_S


def test_pn_lead_time_and_feedforward_flag_are_named_and_documented():
    assert PROPOSED_PN_LEAD_TIME_S > 0.0
    assert isinstance(PROPOSED_ENABLE_ACCELERATION_FEEDFORWARD, bool)
    assert PROPOSED_ENABLE_ACCELERATION_FEEDFORWARD is False
