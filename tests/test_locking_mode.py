"""The `file_io_locking` mode must stay off the cluster's back and on for macOS mounts.

The measurement behind it is in CLAUDE.md: on smbfs, tensorstore's default `os` locking
fails 3/3 on close with EIO where `lockfile` succeeds 3/3. The regression that matters in
both directions -- turning it on for Linux (unmeasured on `/nrs`) or losing it for a mount
(the run breaks) -- so pin both.
"""

import pytest

from spotlight import stores


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("SPOTLIGHT_IO_LOCKING", raising=False)


def test_linux_keeps_tensorstores_default(monkeypatch, tmp_path):
    monkeypatch.setattr(stores.sys, "platform", "linux")
    monkeypatch.chdir(tmp_path)
    assert stores._locking_mode() is None
    assert "file_io_locking" not in stores._context_spec()


def test_a_macos_mount_switches_to_lockfile(monkeypatch):
    monkeypatch.setattr(stores.sys, "platform", "darwin")
    monkeypatch.setattr(stores.Path, "cwd", staticmethod(lambda: stores.Path("/Volumes/x/exp")))
    assert stores._locking_mode() == "lockfile"
    assert stores._context_spec()["file_io_locking"] == {"mode": "lockfile"}


def test_macos_off_a_mount_keeps_the_default(monkeypatch, tmp_path):
    monkeypatch.setattr(stores.sys, "platform", "darwin")
    monkeypatch.chdir(tmp_path)          # a real local path, not /Volumes
    assert stores._locking_mode() is None


def test_the_env_var_wins_on_either_platform(monkeypatch, tmp_path):
    monkeypatch.setenv("SPOTLIGHT_IO_LOCKING", "none")
    monkeypatch.chdir(tmp_path)
    for platform in ("linux", "darwin"):
        monkeypatch.setattr(stores.sys, "platform", platform)
        assert stores._locking_mode() == "none"
        assert stores._context_spec()["file_io_locking"] == {"mode": "none"}


def test_the_mode_is_one_tensorstore_accepts():
    ts = pytest.importorskip("tensorstore")
    for mode in ("os", "lockfile", "none", "non_atomic"):
        ts.Context({"file_io_locking": {"mode": mode}})
    with pytest.raises(ValueError):
        ts.Context({"file_io_locking": {"mode": "lockf"}})
