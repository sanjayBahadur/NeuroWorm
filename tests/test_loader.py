"""Unit tests for the connectome loader."""

import pytest
from pathlib import Path
from connectome import loader


def test_load_connectome_success():
    """Verify loading real bundled connectome returns 302 neurons and valid weight matrices."""
    syn, gap, names = loader.load_connectome()
    assert len(names) == 302
    assert syn.shape == (302, 302)
    assert gap.shape == (302, 302)
    assert "ASEL" in names
    assert "AVAL" in names
    # Gap junctions are symmetric
    assert (gap == gap.T).all()


def test_load_connectome_missing_file_raises_filenotfound(monkeypatch, tmp_path):
    """Verify that a missing data directory or file raises FileNotFoundError."""
    monkeypatch.setattr(loader, "DATA_DIR", tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError, match="Connectome CSV data files missing"):
        loader.load_connectome()
