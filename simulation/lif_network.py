"""
Leaky Integrate-and-Fire simulation of the C. elegans connectome.

All 302 neurons are simulated in parallel using vectorized numpy operations.
The model uses a standard LIF neuron with exponentially decaying synaptic
conductances and instantaneous gap junction coupling.

References:
  - Leaky Integrate-and-Fire: Gerstner & Kistler (2002)
  - C. elegans parameters: Wicks et al. (1996), Kunert et al. (2014)
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# BIOPHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DT          = 0.5e-3   # timestep: 0.5 ms
TAU_M       = 10e-3    # membrane time constant: 10 ms
V_REST      = -65e-3   # resting potential: -65 mV
V_THRESH    = -55e-3   # spike threshold: -55 mV
V_RESET     = -70e-3   # post-spike reset: -70 mV (afterhyperpolarization)
TAU_SYN     = 5e-3     # synaptic conductance decay: 5 ms
R_M         = 1e7      # membrane resistance: 10 MΩ
REFRACTORY  = 2e-3     # refractory period: 2 ms
I_EXT_SCALE = 1e-10    # scale factor for all current inputs (converts to Amps)


class CElegansLIF:
    """
    Vectorized LIF simulation of the full C. elegans 302-neuron connectome.

    Usage:
        net = CElegansLIF(syn_weights, gap_weights, neuron_names)
        net.inject("ASEL", 2.0)   # inject 2 nA into ASEL
        spikes = net.step()        # advance 0.5 ms
        rate   = net.firing_rate("AVBL")  # Hz
    """

    def __init__(
        self,
        syn_weights: np.ndarray,
        gap_weights: np.ndarray,
        neuron_names: list[str],
    ):
        """
        Args:
            syn_weights:  (N, N) float array. syn_weights[i, j] = signed
                          synaptic strength from neuron j to neuron i.
                          Positive = excitatory, negative = inhibitory.
            gap_weights:  (N, N) float array. Symmetric. gap_weights[i, j]
                          = gap junction conductance between neurons i and j.
            neuron_names: list of N neuron name strings.
        """
        self.N = len(neuron_names)
        self.names = list(neuron_names)
        self.name_to_idx: dict[str, int] = {
            n: i for i, n in enumerate(self.names)
        }

        # Connectivity — stored as float32 for vectorized efficiency
        self.W_syn = syn_weights.astype(np.float32)  # (N, N)
        self.W_gap = gap_weights.astype(np.float32)  # (N, N)

        # Precompute row sums for gap junction current (constant topology)
        self._gap_row_sum = self.W_gap.sum(axis=1)   # (N,)

        # ── State variables ───────────────────────────────────────────────
        self.V = np.full(self.N, V_REST, dtype=np.float32)
        """Membrane voltages (Volts). Shape: (N,)"""

        self.g_syn = np.zeros(self.N, dtype=np.float32)
        """Synaptic conductance arriving at each neuron. Shape: (N,)"""

        self.refractory_timer = np.zeros(self.N, dtype=np.float32)
        """Countdown to end of refractory period. 0 = can spike. Shape: (N,)"""

        self.spikes = np.zeros(self.N, dtype=bool)
        """Boolean spike state from last timestep. Shape: (N,)"""

        self.spike_counts = np.zeros(self.N, dtype=np.int32)
        """Cumulative spike count since last reset_counts(). Shape: (N,)"""

        self.external_I = np.zeros(self.N, dtype=np.float32)
        """External injected current from game (Amps). Shape: (N,)"""

        self._decay = float(np.exp(-DT / TAU_SYN))
        """Precomputed synaptic decay factor per timestep."""

    # ─────────────────────────────────────────────────────────────────────
    # CORE SIMULATION STEP
    # ─────────────────────────────────────────────────────────────────────
    def step(self) -> np.ndarray:
        """
        Advance the simulation by one DT timestep (0.5 ms).

        Implements:
          1. Synaptic conductance update (exponential decay + spike input)
          2. Gap junction current (instantaneous, ohmic)
          3. LIF voltage integration (Euler method)
          4. Spike detection and reset
          5. Refractory timer management

        Returns:
            spikes: (N,) boolean array — True if neuron fired this step.
        """
        # ── 1. Synaptic conductance update ─────────────────────────────
        # Decay existing conductance, add new contributions from spiking neurons.
        # W_syn[i, j] * spikes[j] = conductance injected into i by j's spike.
        self.g_syn = (
            self.g_syn * self._decay
            + self.W_syn @ self.spikes  # (N,N) @ (N,) = (N,)
        )

        # ── 2. Gap junction current ─────────────────────────────────────
        # I_gap[i] = Σ_j W_gap[i,j] * (V[j] - V[i])
        #          = (W_gap @ V)[i] - (Σ_j W_gap[i,j]) * V[i]
        I_gap = self.W_gap @ self.V - self._gap_row_sum * self.V

        # ── 3. Total input current ──────────────────────────────────────
        I_total = (
            self.g_syn * I_EXT_SCALE   # synaptic
            + I_gap    * I_EXT_SCALE   # electrical coupling
            + self.external_I          # game stimulation
        )

        # ── 4. LIF voltage integration (forward Euler) ─────────────────
        # τ_m * dV/dt = -(V - V_rest) + R_m * I
        dV = (DT / TAU_M) * (
            -(self.V - V_REST)
            + R_M * I_total
        )
        # Neurons in refractory period cannot integrate input
        dV[self.refractory_timer > 0] = 0.0
        self.V += dV

        # ── 5. Spike detection ──────────────────────────────────────────
        self.spikes = self.V >= V_THRESH

        # ── 6. Spike reset ──────────────────────────────────────────────
        self.V[self.spikes] = V_RESET
        self.refractory_timer[self.spikes] = REFRACTORY
        self.refractory_timer = np.maximum(self.refractory_timer - DT, 0.0)

        # ── 7. Accumulate counts ────────────────────────────────────────
        self.spike_counts += self.spikes.astype(np.int32)

        return self.spikes

    # ─────────────────────────────────────────────────────────────────────
    # EXTERNAL INPUT
    # ─────────────────────────────────────────────────────────────────────
    def inject(self, neuron_name: str, current_nA: float) -> None:
        """
        Inject a sustained current into a named neuron.

        Call before each batch of step() calls. Positive current is
        depolarizing (promotes spiking). Negative is hyperpolarizing.

        Args:
            neuron_name: Exact name as in neuron_names list (e.g. "ASEL").
            current_nA:  Current in nanoamps.
        """
        if neuron_name not in self.name_to_idx:
            return  # silently ignore unknown neurons
        idx = self.name_to_idx[neuron_name]
        self.external_I[idx] += current_nA * 1e-9   # nA → A (additive)

    def clear_external(self) -> None:
        """Zero all external injected currents. Call once per game frame."""
        self.external_I[:] = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # READOUT
    # ─────────────────────────────────────────────────────────────────────
    def firing_rate(self, neuron_name: str, window_steps: int = 200) -> float:
        """
        Estimate firing rate (Hz) from spike_counts accumulated since
        the last reset_counts() call.

        Args:
            neuron_name:   Name of the neuron.
            window_steps:  Number of timesteps in the measurement window.
                           Default 200 steps = 100 ms.
        Returns:
            Firing rate in Hz.
        """
        if neuron_name not in self.name_to_idx:
            return 0.0
        idx = self.name_to_idx[neuron_name]
        window_s = window_steps * DT
        return float(self.spike_counts[idx]) / window_s

    def group_firing_rate(self, neuron_names: list[str], window_steps: int = 200) -> float:
        """Average firing rate across a group of named neurons."""
        rates = [
            self.firing_rate(n, window_steps)
            for n in neuron_names
            if n in self.name_to_idx
        ]
        return float(np.mean(rates)) if rates else 0.0

    def reset_counts(self) -> None:
        """Reset cumulative spike count accumulators. Call once per frame."""
        self.spike_counts[:] = 0

    def get_voltage_array(self) -> np.ndarray:
        """Return a copy of the current voltage state. Useful for visualization."""
        return self.V.copy()

    def get_spike_array(self) -> np.ndarray:
        """Return a copy of the last spike state."""
        return self.spikes.copy()
