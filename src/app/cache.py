import time
from typing import Callable, Any, Dict, Tuple

def ttl_cache(seconds: int):
    store: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}
    def deco(fn: Callable):
        async def wrapped(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in store:
                ts, val = store[key]
                if now - ts < seconds:
                    return val
            val = await fn(*args, **kwargs)
            store[key] = (now, val)
            return val
        return wrapped
    return deco
