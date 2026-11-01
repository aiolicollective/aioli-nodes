"""
Journal des nodes. Les nodes ecrivent dans la console ComfyUI (print) ; le
panneau Resolution appelle les memes fonctions pour son apercu et les fait
taire avec quiet(). Le silence est propre au thread courant : il ne touche ni
sys.stdout ni les autres threads (runs, autres apercus).
"""

import threading
from contextlib import contextmanager

_state = threading.local()


def log(*args, **kwargs):
    if not getattr(_state, "quiet", False):
        print(*args, **kwargs)


@contextmanager
def quiet():
    prev = getattr(_state, "quiet", False)
    _state.quiet = True
    try:
        yield
    finally:
        _state.quiet = prev
