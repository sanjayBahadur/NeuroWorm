"""Unit tests for the biological SynapticCircuit and neuroevolution weights."""

import numpy as np
import pytest
from connectome.loader import build_network
from simulation.synaptic_circuit import (
    SynapticCircuit, SYNAPSE_DEFINITIONS, GENE_COUNT, DEFAULT_SYNAPTIC_GENOME
)


def test_synapse_definitions_count():
    """Verify exactly 36 biological pathways are defined."""
    assert len(SYNAPSE_DEFINITIONS) == 36
    assert GENE_COUNT == 36
    assert len(DEFAULT_SYNAPTIC_GENOME) == 36


def test_synaptic_circuit_apply_genome():
    """Verify that applying genome directly updates the connectome W_syn matrix."""
    net = build_network()
    circuit = SynapticCircuit(net)

    # Custom mutant genome with elevated weights
    custom_genome = DEFAULT_SYNAPTIC_GENOME.copy()
    custom_genome[0] = 17.5  # ASEL -> AIYL
    custom_genome[10] = 24.0 # AVM -> AVAL

    circuit.apply_genome(custom_genome)

    asel_idx = net.name_to_idx["ASEL"]
    aiyl_idx = net.name_to_idx["AIYL"]
    avm_idx  = net.name_to_idx["AVM"]
    aval_idx = net.name_to_idx["AVAL"]

    assert net.W_syn[aiyl_idx, asel_idx] == pytest.approx(17.5, abs=1e-5)
    assert net.W_syn[aval_idx, avm_idx] == pytest.approx(24.0, abs=1e-5)


def test_synaptic_circuit_summary():
    """Verify get_summary generates high-school friendly trait cards."""
    net = build_network()
    circuit = SynapticCircuit(net)
    summary = circuit.get_summary(DEFAULT_SYNAPTIC_GENOME)

    assert len(summary) == 36
    for card in summary:
        assert "label" in card
        assert "pathway" in card
        assert "category" in card
        assert "value" in card
        assert "strength" in card
        assert "pct_change" in card
