"""Thread and OS locks shared by chat and approval workers."""

import hashlib
import threading
from contextlib import contextmanager
from weakref import WeakValueDictionary

from filelock import FileLock

from src.config import settings

_LOCKS = WeakValueDictionary()
_GUARD = threading.Lock()


@contextmanager
def session_guard(session_key: str):
    with _GUARD:
        lock = _LOCKS.get(session_key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[session_key] = lock
    # File locks are released by the OS after a process crash, without a lease.
    name = hashlib.sha256(session_key.encode()).hexdigest() + ".lock"
    settings.workflow_lock_dir.mkdir(parents=True, exist_ok=True)
    with lock, FileLock(str(settings.workflow_lock_dir / name)):
        yield
