"""Unit tests for the SensoryRadar visualization."""

import pygame as pg
import pytest
from viz.sensory_radar import SensoryRadar, RADAR_W, RADAR_H


@pytest.fixture(scope="module")
def pygame_init():
    pg.init()
    yield
    pg.quit()


def test_sensory_radar_render(pygame_init):
    radar = SensoryRadar(grid_size=20)
    obs = ((10, 10), (15, 10), [(10, 10), (9, 10), (8, 10)], "RIGHT")
    telemetry = {
        "front_dist": 9,
        "left_dist": 10,
        "right_dist": 10,
        "front_hit": (19, 10),
        "left_hit": (10, 0),
        "right_hit": (10, 19),
        "c_left": 12.0,
        "c_right": 14.0,
        "c_center": 13.0,
        "delta_c": -2.0,
        "dc_dt": 0.5,
    }
    surf = radar.render(obs, telemetry, "RIGHT")
    assert surf.get_width() == RADAR_W
    assert surf.get_height() == RADAR_H
