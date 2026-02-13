"""Unit tests for the CElegansLIF simulation."""

import numpy as np
import pytest
from simulation.lif_network import CElegansLIF, V_REST, V_THRESH, V_RESET, DT


@pytest.fixture
def tiny_net():
    """Minimal 5-neuron network for fast unit testing."""
    N = 5
    names = ["A", "B", "C", "D", "E"]
    syn = np.zeros((N, N), dtype=np.float32)
    gap = np.zeros((N, N), dtype=np.float32)
    # A → B excitatory
    syn[1, 0] = 2.0
    return CElegansLIF(syn, gap, names)


def test_resting_potential(tiny_net):
    """Network should stay at rest potential with no input."""
    for _ in range(100):
        tiny_net.step()
    assert np.allclose(tiny_net.V, V_REST, atol=1e-5), (
        f"Expected resting potential {V_REST}, got {tiny_net.V}"
    )


def test_spike_on_threshold(tiny_net):
    """Neuron A should spike when voltage exceeds threshold.

    We set V slightly above V_THRESH (+ 1 mV) so that the Euler leak term
    does not pull it back below threshold within the same step. This mirrors
    the biological reality: a spike occurs when V strictly exceeds threshold.
    """
    tiny_net.V[0] = V_THRESH + 1e-3   # 1 mV above threshold
    spikes = tiny_net.step()
    assert spikes[0], "Neuron A should have fired above threshold"
    assert tiny_net.V[0] == pytest.approx(V_RESET, abs=1e-6), (
        f"Post-spike reset failed: V={tiny_net.V[0]}"
    )


def test_refractory_prevents_double_spike(tiny_net):
    """A neuron in refractory period should not spike again immediately."""
    tiny_net.V[0] = V_THRESH + 1e-3   # above threshold — fires
    tiny_net.step()
    tiny_net.V[0] = V_THRESH + 1e-3   # force above threshold again
    spikes2 = tiny_net.step()          # should NOT fire (refractory)
    assert not spikes2[0], "Neuron should be in refractory period"


def test_external_injection_causes_spike(tiny_net):
    """Sustained injection should eventually cause a spike."""
    tiny_net.inject("A", 10.0)   # strong injection
    spiked = False
    for _ in range(500):
        spikes = tiny_net.step()
        if spikes[0]:
            spiked = True
            break
    assert spiked, "Sustained injection should cause neuron A to spike"


def test_inject_unknown_neuron_is_silent(tiny_net):
    """Injecting into a non-existent neuron should not raise."""
    tiny_net.inject("DOES_NOT_EXIST", 100.0)
    tiny_net.step()   # should not raise


def test_clear_external(tiny_net):
    """clear_external should zero all injected currents."""
    tiny_net.inject("A", 5.0)
    tiny_net.clear_external()
    assert np.allclose(tiny_net.external_I, 0.0)


def test_spike_count_accumulation(tiny_net):
    """spike_counts should accumulate correctly and reset."""
    tiny_net.V[0] = V_THRESH + 1e-3
    tiny_net.step()   # fires
    assert tiny_net.spike_counts[0] == 1
    tiny_net.reset_counts()
    assert tiny_net.spike_counts[0] == 0
