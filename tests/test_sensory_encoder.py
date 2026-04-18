"""Unit tests for the SensoryEncoder."""

import numpy as np
import pytest
from connectome.loader import load_connectome
from simulation.lif_network import CElegansLIF
from simulation.sensory_encoder import SensoryEncoder


@pytest.fixture
def net_and_encoder():
    syn, gap, names = load_connectome()
    net     = CElegansLIF(syn, gap, names)
    encoder = SensoryEncoder(net, grid_size=20)
    return net, encoder


def test_clear_before_encode(net_and_encoder):
    net, encoder = net_and_encoder
    net.inject("ASEL", 5.0)   # pre-inject
    encoder.encode((10, 10), (15, 10), [(10, 10)], "RIGHT")
    # After encode, ASEL external_I reflects only the new injection (not stale 5 nA)
    idx = net.name_to_idx.get("ASEL", -1)
    if idx >= 0:
        assert net.external_I[idx] < 5e-9


def test_food_right_activates_food_right_neurons(net_and_encoder):
    net, encoder = net_and_encoder
    # Heading UP from (5, 10), food at (15, 10) is to the relative right
    encoder.encode((5, 10), (15, 10), [(5, 10)], "UP")
    idx = net.name_to_idx.get("ASER", -1)
    if idx >= 0:
        assert net.external_I[idx] > 0.0, "ASER should be activated when food is to the right"


def test_danger_front_activates_front_neurons(net_and_encoder):
    net, encoder = net_and_encoder
    # Wall directly in front (heading RIGHT, right edge at x=19)
    encoder.encode((19, 10), (5, 5), [(19, 10)], "RIGHT")
    idx = net.name_to_idx.get("AVM", -1)
    if idx >= 0:
        assert net.external_I[idx] > 0.0, "AVM should fire when wall is straight ahead"


def test_encode_no_crash_on_edge(net_and_encoder):
    """encode() at corners should not raise any exceptions."""
    net, encoder = net_and_encoder
    encoder.encode((0, 0), (19, 19), [(0, 0)], "UP")
    encoder.encode((19, 19), (0, 0), [(19, 19)], "DOWN")
