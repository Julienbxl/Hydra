#!/usr/bin/env python3
"""Simple HEX BSGS smoke tests.

Usage:
  python3 testbsgshex.py [./Hydra]
  python3 testbsgshex.py [./Hydra] --tests 10 --nibbles 9 --timeout 90
"""
import argparse
import os
import random
import re
import subprocess
import sys


P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


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


def positions_for_case(i, nibbles):
    patterns = (
        lambda: list(range(0, nibbles)),
        lambda: list(range(1, 1 + nibbles)),
        lambda: list(range(8, 8 + nibbles)),
        lambda: list(range(64 - nibbles, 64)),
        lambda: sorted({0, 1, 62, 63, 31, 32, 17, 48, 9})[:nibbles],
        lambda: sorted(((i * 11 + j * 17) % 64) for j in range(nibbles)),
    )
    pos = patterns[i % len(patterns)]()
    if len(pos) < nibbles:
        used = set(pos)
        for p in range(64):
            if p not in used:
                pos.append(p)
                used.add(p)
                if len(pos) == nibbles:
                    break
    return sorted(pos[:nibbles])


def make_case(i, nibbles):
    k = random.randrange(1, N)
    key = k.to_bytes(32, "big").hex()
    positions = positions_for_case(i, nibbles)
    mask = list(key)
    unknown = []
    for p in positions:
        unknown.append(mask[p])
        mask[p] = "#"
    return {
        "i": i,
        "key": key,
        "mask": "".join(mask),
        "positions": positions,
        "unknown": "".join(unknown),
        "target": pubkey(k, compressed=(i % 2 == 0)),
    }


def run_case(hydra, case, timeout):
    print(
        f"\n{'=' * 64}\n"
        f"Test #{case['i'] + 1:02d}\n"
        f"  Key     : {case['key']}\n"
        f"  Mask    : {case['mask']}\n"
        f"  Pos     : {' '.join(str(p) for p in case['positions'])}\n"
        f"  Unknown : {case['unknown']}\n"
        f"  PubKey  : {case['target']}"
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

    km = re.search(r"Private key\s*:\s*([0-9a-fA-F]{64})", out)
    plan = re.search(r"Unknowns\s+:\s+(\d+)\s+\(baby=(\d+),\s+giant=(\d+)\)", out)
    timing = re.search(r"^\[BSGS\] Timings ms: .+$", out, re.MULTILINE)
    got_key = km.group(1).lower() if km else None
    ok = result.returncode == 0 and got_key == case["key"]

    if ok:
        print("  PASS")
        if plan:
            print(f"  Plan: unknowns={plan.group(1)} baby={plan.group(2)} giant={plan.group(3)}")
        if timing:
            print(f"  {timing.group(0)}")
        return True

    print(f"  FAIL: rc={result.returncode}")
    print(f"  Expected key: {case['key']}")
    print(f"  Found key   : {got_key}")
    print(out[-1200:].strip())
    return False


def main():
    parser = argparse.ArgumentParser(description="Hydra HEX BSGS smoke tests")
    parser.add_argument("hydra", nargs="?", help="Hydra binary path")
    parser.add_argument("--tests", type=int, default=10)
    parser.add_argument("--nibbles", type=int, default=9, help="unknown HEX nibbles")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    if args.nibbles < 2 or args.nibbles > 12:
        sys.exit("ERROR: --nibbles must be between 2 and 12 for this quick test")

    hydra = resolve_hydra(args.hydra)
    if not os.path.exists(hydra):
        sys.exit(f"ERROR: {hydra} not found")

    random.seed(0xB565)
    print(f"=== Hydra HEX BSGS smoke tests: {args.tests} tests x {args.nibbles} nibbles ===")
    print(f"Binary: {hydra}")
    passed = sum(run_case(hydra, make_case(i, args.nibbles), args.timeout) for i in range(args.tests))
    print(f"\n{'=' * 64}\nResult: {passed}/{args.tests}")
    sys.exit(0 if passed == args.tests else 1)


if __name__ == "__main__":
    main()
