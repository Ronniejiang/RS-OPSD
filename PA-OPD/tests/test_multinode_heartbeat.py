"""Fault injection for shared-filesystem ESTALE during multi-node training."""
import errno
from pathlib import Path

import pytest

from scripts import run_multinode_ray as runner


@pytest.mark.parametrize('failure', [errno.ESTALE, errno.ENOENT, errno.EIO, errno.ETIMEDOUT])
def test_read_reopens_after_transient_error(tmp_path, monkeypatch, failure):
    path = tmp_path/'status.json'
    runner.write_status(path, 'training')
    original = Path.read_text
    attempts = []
    def flaky_read(self, *args, **kwargs):
        attempts.append(self)
        if len(attempts) < 3:
            raise OSError(failure, 'injected transient read failure')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', flaky_read)
    monkeypatch.setattr(runner.time, 'sleep', lambda delay: None)
    assert runner.read_status(path)['state'] == 'training'
    assert len(attempts) == 3


def test_exhausted_reads_are_unavailable_not_immediate_worker_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.time, 'sleep', lambda delay: None)
    def unavailable(*args, **kwargs):
        raise OSError(errno.ESTALE, 'stale handle')
    monkeypatch.setattr(Path, 'read_text', unavailable)
    clock = [10.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    watchdog = runner.HeartbeatWatchdog(timeout=300)
    for tick in (10., 100., 309.):
        clock[0] = tick
        watchdog.observe(runner.read_status(tmp_path/'status.json'))
    clock[0] = 311.
    with pytest.raises(TimeoutError):
        watchdog.observe({})


def test_read_permission_errors_are_not_silenced(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise PermissionError(errno.EACCES, 'access denied')
    monkeypatch.setattr(Path, 'read_text', forbidden)
    with pytest.raises(PermissionError):
        runner.read_status(tmp_path/'status.json')


def test_invalid_json_and_status_are_retryable(tmp_path, monkeypatch):
    values = iter(['{', '{}', '{"state":"finished","sequence":1}',
                   '{"state":"finished","sequence":2,"returncode":0}'])
    monkeypatch.setattr(Path, 'read_text', lambda *args, **kwargs: next(values))
    monkeypatch.setattr(runner.time, 'sleep', lambda delay: None)
    assert runner.read_status(tmp_path/'status.json')['returncode'] == 0


def test_write_retries_rename_with_fresh_temporary_and_preserves_status(tmp_path, monkeypatch):
    path = tmp_path/'status.json'
    runner.write_status(path, 'starting')
    original = Path.replace
    temporaries = []
    def flaky_replace(self, target):
        temporaries.append(self)
        if len(temporaries) < 3:
            raise OSError(errno.ESTALE, 'stale rename')
        return original(self, target)
    monkeypatch.setattr(Path, 'replace', flaky_replace)
    monkeypatch.setattr(runner.time, 'sleep', lambda delay: None)
    assert runner.write_status(path, 'training')
    assert len(set(temporaries)) == 3
    assert runner.read_status(path)['state'] == 'training'
    assert not list(tmp_path.glob('*.tmp'))


def test_watchdog_requires_progress_and_ignores_cross_node_clock_skew(monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    watch = runner.HeartbeatWatchdog(timeout=300)
    watch.observe({'sequence':100, 'heartbeat':-100000000})
    clock[0] += 290
    watch.observe({})  # temporary missing/stale file is not heartbeat epoch zero
    watch.observe({'sequence':101, 'heartbeat':-100000000})
    clock[0] += 290
    watch.observe({'sequence':101})  # re-reading cached bytes isn't progress
    watch.observe({'sequence':100})  # nor is an older cached version
    clock[0] += 11
    with pytest.raises(TimeoutError):
        watch.observe({'sequence':101})


def test_head_tolerates_short_write_outages_but_not_permanent_failure(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    successful = [False]
    monkeypatch.setattr(runner, 'write_status', lambda *args, **kwargs: successful[0])
    writer = runner.HeartbeatWriter(Path('/unused'), timeout=300)
    clock[0] = 290
    assert not writer.publish('training')
    successful[0] = True
    assert writer.publish('training')
    successful[0] = False
    clock[0] = 589
    assert not writer.publish('training')
    clock[0] = 591
    with pytest.raises(TimeoutError):
        writer.publish('training')
