"""Compliant follow + finger-plate close, once the palms are close
(proposal section 4.4)."""

# Finger hinge angle when fully closed (rad); see models/dap_hand.sdf.
HINGE_CLOSED = -1.6


class ClaspController:
    """Tracks the partner palm with a soft Cartesian impedance while driving
    the finger hinge from 0 to HINGE_CLOSED."""

    # TODO: impedance gains, close duration, success check (palms within
    # tolerance + plate in contact, no other collisions).
