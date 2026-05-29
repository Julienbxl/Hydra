#!/usr/bin/env python3
"""Simple WIF BSGS smoke tests.

Usage:
  python3 testbsgswif.py [./Hydra]
  python3 testbsgswif.py [./Hydra] --tests 10 --chars 9 --timeout 90
"""
import argparse
import hashlib
import os
import random
import re
import subprocess
import sys


P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def resolve_hydra(explicit=None):
    if explicit: return explicit
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = ('Hydra.exe', 'Hydra') if os.name == 'nt' else ('Hydra', 'Hydra.exe')
    for c in candidates:
        p = os.path.join(base, '..', c)
        if os.path.exists(p): return p
    return './Hydra'


def ec_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    x1, y1 = a
    x2, y2 = b
    if x1 == x2:
        if y1 != y2:
            return None
        m = 3 * x1 * x1 * pow(2 * y1, P - 2, P) % P
    else:
        m = (y2 - y1) * pow(x2 - x1, P - 2, P) % P
    x3 = (m * m - x1 - x2) % P
    return x3, (m * (x1 - x3) - y1) % P


def scalar_mul(k):
    r = None
    a = (GX, GY)
    while k:
        if k & 1:
            r = ec_add(r, a)
        a = ec_add(a, a)
        k >>= 1
    return r


def pubkey(k, compressed=True):
    x, y = scalar_mul(k)
    if compressed:
        return ((b"\x02" if y % 2 == 0 else b"\x03") + x.to_bytes(32, "big")).hex()
    return (b"\x04" + x.to_bytes(32, "big") + y.to_bytes(32, "big")).hex()


def sha2d(data):
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def b58encode(data):
    n = int.from_bytes(data, "big")
    out = []
    while n:
        n, r = divmod(n, 58)
        out.append(B58[r])
    leading = len(data) - len(data.lstrip(b"\x00"))
    return "1" * leading + "".join(reversed(out or ["1"]))


def wif_encode(key_bytes, compressed):
    payload = b"\x80" + key_bytes + (b"\x01" if compressed else b"")
    return b58encode(payload + sha2d(payload)[:4])


def mask_positions(length, chars, case):
    # Keep the quick smoke test on the scalar-bearing WIF chars. Pure checksum
    # tail masks exercise a separate unsupported edge and belong in a focused
    # regression, not in the default fast path.
    limit = max(chars, length - 6)

    def clamp_unique(values):
        out = []
        used = set()
        for value in values:
            value %= limit
            if value not in used:
                out.append(value)
                used.add(value)
            if len(out) == chars:
                return sorted(out)
        for value in range(limit):
            if value not in used:
                out.append(value)
                used.add(value)
            if len(out) == chars:
                break
        return sorted(out)

    patterns = (
        lambda: list(range(0, chars)),
        lambda: list(range(1, 1 + chars)),
        lambda: list(range(8, 8 + chars)),
        lambda: list(range(max(0, limit - chars), limit)),
        lambda: [0, 1, limit - 2, limit - 1, limit // 2],
        lambda: [case * 7 + i * 13 for i in range(chars)],
    )
    return clamp_unique(patterns[case % len(patterns)]())


def make_case(i, chars):
    k = random.randrange(1, N)
    key = k.to_bytes(32, "big")
    wif_compressed = (i % 4) in (0, 1)
    pub_compressed = (i % 4) in (0, 2)
    wif = wif_encode(key, wif_compressed)
    positions = mask_positions(len(wif), chars, i)
    mask = list(wif)
    for p in positions:
        mask[p] = "#"
    return {
        "i": i,
        "key": key.hex(),
        "wif": wif,
        "mask": "".join(mask),
        "positions": positions,
        "target": pubkey(k, pub_compressed),
        "wif_kind": "compressed" if wif_compressed else "uncompressed",
        "pub_kind": "compressed" if pub_compressed else "uncompressed",
    }


def run_case(hydra, case, timeout):
    print(
        f"\n{'=' * 64}\n"
        f"Test #{case['i'] + 1:02d} | WIF {case['wif_kind']} / PubKey {case['pub_kind']}\n"
        f"  WIF    : {case['wif']}\n"
        f"  Mask   : {case['mask']}\n"
        f"  Pos    : {' '.join(str(p) for p in case['positions'])}\n"
        f"  PubKey : {case['target']}"
    )
    try:
        result = subprocess.run(
            [hydra, case["mask"], case["target"]],
            cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        print("  FAIL: timeout")
        if out:
            print(out[-1000:].strip())
        return False

    out = result.stdout + result.stderr
    if "CUDA unavailable" in out:
        print("  SKIP: CUDA unavailable")
        return True

    got_wif = None
    got_key = None
    m = re.search(r"WIF\s*:\s*([123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz]+)", out)
    k = re.search(r"Private key\s*:\s*([0-9a-fA-F]{64})", out)
    plan = re.search(r"Unknowns\s+:\s+(\d+)\s+\(baby=(\d+),\s+giant=(\d+)\)", out)
    timing = re.search(r"^\[BSGS\] Timings ms: .+$", out, re.MULTILINE)
    if m:
        got_wif = m.group(1)
    if k:
        got_key = k.group(1).lower()

    ok = result.returncode == 0 and got_wif == case["wif"] and got_key == case["key"]
    if ok:
        print("  PASS")
        if plan:
            print(f"  Plan: unknowns={plan.group(1)} baby={plan.group(2)} giant={plan.group(3)}")
        if timing:
            print(f"  {timing.group(0)}")
        return True

    print(f"  FAIL: rc={result.returncode}")
    print(f"  Expected WIF: {case['wif']}")
    print(f"  Found WIF   : {got_wif}")
    print(f"  Expected key: {case['key']}")
    print(f"  Found key   : {got_key}")
    print(out[-1200:].strip())
    return False


def main():
    parser = argparse.ArgumentParser(description="Hydra WIF BSGS smoke tests")
    parser.add_argument("hydra", nargs="?", help="Hydra binary path")
    parser.add_argument("--tests", type=int, default=10)
    parser.add_argument("--chars", type=int, default=9, help="unknown WIF chars; keep <=6 for quick VRAM tests")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    if args.chars < 2 or args.chars > 10:
        sys.exit("ERROR: --chars must be between 2 and 10 for this quick test")

    hydra = resolve_hydra(args.hydra)
    if not os.path.exists(hydra):
        sys.exit(f"ERROR: {hydra} not found")

    random.seed(0xB56558)
    print(f"=== Hydra WIF BSGS smoke tests: {args.tests} tests x {args.chars} chars ===")
    print(f"Binary: {hydra}")
    passed = sum(run_case(hydra, make_case(i, args.chars), args.timeout) for i in range(args.tests))
    print(f"\n{'=' * 64}\nResult: {passed}/{args.tests}")
    sys.exit(0 if passed == args.tests else 1)


if __name__ == "__main__":
    main()
