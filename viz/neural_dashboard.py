"""
Real-time neural activity dashboard — rendered alongside the Snake game.

Shows a heatmap of firing rates for key neuron groups, updated every frame.
Uses matplotlib in interactive mode for a live-updating side panel.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # off-screen render for embedding in pygame
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from simulation.lif_network import CElegansLIF
from connectome.neuron_groups import GROUP_LABELS


class NeuralDashboard:
    """
    Renders a heatmap of neural group firing rates.
    Call update() each frame; get_surface() returns a numpy RGB array
    ready to blit into pygame.
    """

    PANEL_W = 256
    PANEL_H = 512
    DPI     = 72
    HISTORY = 100   # frames of firing rate history to show

    def __init__(self, network: CElegansLIF):
        self.net    = network
        self.groups = list(GROUP_LABELS.keys())
        self.n_groups = len(self.groups)

        # Rolling history: shape (n_groups, HISTORY)
        self.history = np.zeros((self.n_groups, self.HISTORY), dtype=np.float32)

        # Matplotlib figure (off-screen)
        fig_w = self.PANEL_W / self.DPI
        fig_h = self.PANEL_H / self.DPI
        self.fig, self.ax = plt.subplots(
            figsize=(fig_w, fig_h), dpi=self.DPI
        )
        self.fig.patch.set_facecolor("#1a0f0a")
        self.ax.set_facecolor("#1a0f0a")
        self._setup_axes()

        # Colormap: black → green (neuroscience convention)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "worm", ["#1a0f0a", "#00ff88", "#ffcc00"]
        )
        self.img = self.ax.imshow(
            self.history,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=80,   # Hz ceiling for display
            interpolation="nearest",
        )
        self.fig.tight_layout(pad=0.5)

    def _setup_axes(self) -> None:
        self.ax.set_yticks(range(self.n_groups))
        self.ax.set_yticklabels(
            self.groups, fontsize=6, color="#c8b87e"
        )
        self.ax.set_xticks([])
        self.ax.set_title(
            "NERVOUS SYSTEM", fontsize=7, color="#c8b87e",
            fontfamily="monospace", pad=4
        )
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#5a3a1a")

    def update(self) -> None:
        """Read firing rates from network and update heatmap history."""
        rates = np.array([
            self.net.group_firing_rate(GROUP_LABELS[g])
            for g in self.groups
        ], dtype=np.float32)

        # Shift history left by one column and append new column
        self.history = np.roll(self.history, -1, axis=1)
        self.history[:, -1] = rates

        self.img.set_data(self.history)
        self.fig.canvas.draw()

    def get_rgb_array(self) -> np.ndarray:
        """
        Returns (H, W, 3) uint8 RGB array suitable for pygame blitting.
        """
        buf = self.fig.canvas.buffer_rgba()
        rgb = np.frombuffer(buf, dtype=np.uint8).reshape(
            self.PANEL_H, self.PANEL_W, 4
        )
        return rgb[:, :, :3]   # drop alpha channel
