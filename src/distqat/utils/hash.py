import hashlib
from typing import Union

PID = Union[int, str, bytes]

def _to_bytes(x: PID) -> bytes:
    if isinstance(x, bytes): return x
    if isinstance(x, int):   return x.to_bytes(8, "big", signed=False)
    return str(x).encode()

def hash64(b: bytes) -> int:
    # 64-bit stable hash (little-endian slice of BLAKE2b)
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")
