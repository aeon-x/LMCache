# SPDX-License-Identifier: Apache-2.0
"""Store and load chunks through an installed lmcache wheel's NIXL connector.

    python test_nixl_wheel.py <cu12|cu13>

Loads libnixl from the nixl-<variant> wheel, then moves data through
``lmcache.lmcache_nixl`` with the POSIX backend, so one lmcache wheel can be
checked against each NIXL CUDA line.
"""

# Standard
from pathlib import Path
from typing import Any
import ctypes
import mmap
import select
import site
import sys
import sysconfig
import tempfile
import time

_CHUNK = 4096
_COUNT = 4


def _wait(client: Any, future_id: int) -> tuple[bool, str, list[bool] | None]:
    deadline = time.monotonic() + 10
    while (remaining := deadline - time.monotonic()) > 0:
        if not select.select([client.event_fd()], [], [], remaining)[0]:
            break
        for completed_id, ok, error, results in client.drain_completions():
            if completed_id == future_id:
                return ok, error, results
    raise TimeoutError("no NIXL completion within 10s")


def main(variant: str) -> None:
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        print("skipped: nixl publishes no free-threaded wheels")
        return
    libs = f".nixl_{variant}.mesonpy.libs/libnixl.so"
    libnixl = next(p for d in site.getsitepackages() if (p := Path(d, libs)).exists())
    # Preload libnixl globally instead of `import nixl`, which picks the CUDA
    # line from torch's build (always cu13 here), so it couldn't test nixl-cu12.
    # The adapter's path (`import nixl`, then lmcache.lmcache_nixl reusing that
    # libnixl by SONAME) is checked by the image builds.
    ctypes.CDLL(str(libnixl), mode=ctypes.RTLD_GLOBAL)

    # First Party
    from lmcache.lmcache_nixl import LMCacheNixlClient

    arena = mmap.mmap(-1, 2 * _CHUNK * _COUNT)
    address = ctypes.addressof(ctypes.c_char.from_buffer(arena))
    view = memoryview(arena)
    source = [view[i * _CHUNK : (i + 1) * _CHUNK] for i in range(_COUNT)]
    target = [
        view[(_COUNT + i) * _CHUNK : (_COUNT + i + 1) * _CHUNK] for i in range(_COUNT)
    ]
    for i, chunk in enumerate(source):
        chunk[:] = bytes([i + 1]) * _CHUNK

    client = LMCacheNixlClient(
        backend="POSIX",
        backend_params={"file_path": tempfile.mkdtemp(), "use_direct_io": "false"},
        num_workers=1,
        l1_base=address,
        l1_size=len(arena),
        l1_alignment=_CHUNK,
    )
    keys = [f"model@00000000@0@{i:08x}" for i in range(_COUNT)]
    ok, error, _ = _wait(client, client.submit_batch_set(keys, source))
    assert ok, f"store failed: {error}"
    ok, error, results = _wait(client, client.submit_batch_get(keys, target))
    assert ok, f"load failed: {error}"
    assert results == [True] * _COUNT, results
    assert all(bytes(t) == bytes(s) for s, t in zip(source, target, strict=True))
    print(f"lmcache.lmcache_nixl round trip OK with nixl-{variant} ({libnixl})")


if __name__ == "__main__":
    main(sys.argv[1])
