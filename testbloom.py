#!/usr/bin/env python3
"""testbloom.py — tests Bloom mode for Hydra (Linux + Windows)

Creates a temporary bloom.bin with a few known addresses, runs Hydra against it,
then restores the original bloom.bin.

Usage:
  python3 testbloom.py [Hydra|Hydra.exe]
"""

import hashlib
import os
import random
import struct
import subprocess
import sys

import testseed as seed_helpers

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BC = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

BLOOM_K = 16
BLOOM_M_BITS = 1 << 24   # 2 MiB filter, enough for deterministic test cases
BLOOM_BYTES = BLOOM_M_BITS // 8
SEED1 = 0x9747B28C


def resolve_hydra(argv_index=1):
    if len(sys.argv) > argv_index:
        return sys.argv[argv_index]
    candidates = ("Hydra.exe", "./Hydra") if os.name == "nt" else ("./Hydra", "Hydra.exe")
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return "./Hydra"


def _add(a, b):
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


def smul(k):
    r = None
    a = (Gx, Gy)
    while k:
        if k & 1:
            r = _add(r, a)
        a = _add(a, a)
        k >>= 1
    return r


def pub(k):
    px, py = smul(k)
    return (b"\x02" if py % 2 == 0 else b"\x03") + px.to_bytes(32, "big"), px, py


def sha2(data):
    return hashlib.sha256(data).digest()


def sha2d(data):
    return sha2(sha2(data))


def h160(data):
    r = hashlib.new("ripemd160")
    r.update(sha2(data))
    return r.digest()


def keccak256(data):
    rc = [
        0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
        0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
        0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
    ]
    rot = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]
    mask = 0xFFFFFFFFFFFFFFFF
    rol = lambda x, n: ((x << n) | (x >> (64 - n))) & mask

    msg = bytearray(data) + b"\x01"
    while len(msg) % 136:
        msg += b"\x00"
    msg[-1] |= 0x80
    st = [0] * 25
    for bs in range(0, len(msg), 136):
        for i in range(17):
            st[i] ^= int.from_bytes(msg[bs + i * 8:bs + i * 8 + 8], "little")
        for rnd in range(24):
            c = [st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20] for x in range(5)]
            d = [c[(x - 1) % 5] ^ rol(c[(x + 1) % 5], 1) for x in range(5)]
            st = [st[i] ^ d[i % 5] for i in range(25)]
            b = [0] * 25
            for x in range(5):
                for y in range(5):
                    b[y + 5 * ((2 * x + 3 * y) % 5)] = rol(st[x + 5 * y], rot[x][y])
            st = [b[x + 5 * y] ^ ((~b[(x + 1) % 5 + 5 * y]) & mask & b[(x + 2) % 5 + 5 * y])
                  for y in range(5) for x in range(5)]
            st[0] ^= rc[rnd]
    return b"".join(st[i].to_bytes(8, "little") for i in range(4))


def b58e(data):
    n = int.from_bytes(data, "big")
    out = []
    while n:
        out.append(B58[n % 58])
        n //= 58
    out += ["1"] * next((i for i, b in enumerate(data) if b), 0)
    return "".join(reversed(out))


def _pm(values):
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for x in values:
        b = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ x
        for i in range(5):
            chk ^= gen[i] if ((b >> i) & 1) else 0
    return chk


def _cv(data, from_bits, to_bits):
    acc = 0
    bits = 0
    out = []
    mask = (1 << to_bits) - 1
    for value in data:
        acc = (acc << from_bits) | value
        bits += from_bits
        while bits >= to_bits:
            bits -= to_bits
            out.append((acc >> bits) & mask)
    return out


def btc_legacy(k):
    p, _, _ = pub(k)
    payload = b"\x00" + h160(p)
    return b58e(payload + sha2d(payload)[:4])


def btc_segwit(k):
    p, _, _ = pub(k)
    h = h160(p)
    data = [0] + _cv(h, 8, 5)
    hrp_expand = [ord(x) >> 5 for x in "bc"] + [0] + [ord(x) & 31 for x in "bc"]
    polymod = _pm(hrp_expand + data + [0] * 6) ^ 1
    cs = [(polymod >> (5 * (5 - i))) & 31 for i in range(6)]
    return "bc1" + "".join(BC[x] for x in data + cs)


def eth_addr(k):
    _, px, py = pub(k)
    return "0x" + keccak256(px.to_bytes(32, "big") + py.to_bytes(32, "big"))[12:].hex()


def wif_enc(key_bytes):
    payload = b"\x80" + key_bytes + b"\x01"
    return b58e(payload + sha2d(payload)[:4])


def _rotl32(x, r):
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


def murmur3_20(data20, seed):
    c1, c2 = 0xCC9E2D51, 0x1B873593
    h = seed & 0xFFFFFFFF
    for i in range(5):
        k = struct.unpack_from("<I", data20, i * 4)[0]
        k = (k * c1) & 0xFFFFFFFF
        k = _rotl32(k, 15)
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = _rotl32(h, 13)
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    h ^= 20
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def bloom_add(arr, data20):
    h1 = murmur3_20(data20, SEED1)
    h2 = murmur3_20(data20, h1) or 1
    mask = BLOOM_M_BITS - 1
    for i in range(BLOOM_K):
        bit = (h1 + i * h2) & mask
        arr[bit >> 3] |= (1 << (bit & 7))


def build_bloom_blob(hash20_values):
    arr = bytearray(BLOOM_BYTES)
    for item in hash20_values:
        bloom_add(arr, item)
    return bytes(arr)


def resolve_hash20_from_address(addr):
    if addr.startswith("0x"):
        return bytes.fromhex(addr[2:])
    if addr.startswith("bc1"):
        raise ValueError("SegWit addresses are not used directly in this bloom test")
    raise ValueError(f"Unsupported address format: {addr}")


def make_hex_case(label, coin):
    key = random.randint(1, N - 1)
    kh = key.to_bytes(32, "big").hex()
    start = random.randint(8, 64 - 8)
    mask = kh[:start] + "#" * 8 + kh[start + 8:]
    if coin == "btc":
        addr = btc_legacy(key)
        target = "bloombtc"
        hash20 = h160(pub(key)[0])
    else:
        addr = eth_addr(key)
        target = "bloometh"
        hash20 = bytes.fromhex(addr[2:])
    return {
        "label": label,
        "kind": "hex",
        "mask": mask,
        "target": target,
        "address": addr,
        "hash20": hash20,
        "expect": kh,
    }


def make_wif_case():
    key = random.randint(1, N - 1)
    kb = key.to_bytes(32, "big")
    wif = wif_enc(kb)
    start = random.randint(1, 52 - 5)
    mask = wif[:start] + "#" * 5 + wif[start + 5:]
    addr = btc_legacy(key)
    return {
        "label": "wif-btc",
        "kind": "wif",
        "mask": mask,
        "target": "bloombtc",
        "address": addr,
        "hash20": h160(pub(key)[0]),
        "expect": wif,
    }


def make_seed_case(label, coin):
    wanted_at = "btc_legacy" if coin == "btc" else "eth"
    seed_test_idx = 0 if coin == "btc" else 2
    case = seed_helpers.gen(seed_test_idx, 12)
    while case["at"] != wanted_at:
        seed_test_idx += 1
        case = seed_helpers.gen(seed_test_idx, 12)

    privkey = seed_helpers.derive_bip44(case["mnemonic"], 0 if coin == "btc" else 60)
    if coin == "btc":
        target = "bloombtc"
        hash20 = h160(pub(int.from_bytes(privkey, "big"))[0])
    else:
        target = "bloometh"
        hash20 = bytes.fromhex(case["addr"][2:])

    return {
        "label": label,
        "kind": "seed",
        "mask": case["mask"],
        "target": target,
        "address": case["addr"],
        "hash20": hash20,
        "expect": case["mnemonic"],
    }


def run_case(hydra, case):
    print(f"\n{'=' * 62}")
    print(f"Test | {case['label']}")
    print(f"  Target   : {case['target']}")
    print(f"  Address  : {case['address']}")
    print(f"  Input    : {case['mask']}")
    try:
        result = subprocess.run(
            [hydra, case["mask"], case["target"]],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("  ❌ TIMEOUT")
        return False

    out = result.stdout + result.stderr
    # Bloom mode in Hydra only ends in final success if the live API reports a non-zero balance.
    # For synthetic test vectors we expect a bloom hit followed by a false-positive continuation.
    if case["kind"] in ("hex", "seed"):
        if "!!! BLOOM HIT !!!" not in out:
            print(f"  ❌ FAIL (no bloom hit, rc={result.returncode})")
            print(out[-700:].strip())
            return False
        if case["expect"] not in out:
            what = "private key mismatch" if case["kind"] == "hex" else "mnemonic mismatch"
            print(f"  ❌ FAIL ({what})")
            print(out[-700:].strip())
            return False
        if case["kind"] == "hex" and case["address"] not in out:
            print("  ❌ FAIL (expected address not shown)")
            print(out[-700:].strip())
            return False
        if "Zero balance -- false positive, continuing." not in out:
            print("  ❌ FAIL (expected zero-balance continuation not seen)")
            print(out[-700:].strip())
            return False
    else:
        if "!!! BLOOM HIT !!!" not in out:
            print(f"  ❌ FAIL (no bloom hit, rc={result.returncode})")
            print(out[-700:].strip())
            return False
        if case["address"] not in out:
            print("  ❌ FAIL (expected address not shown)")
            print(out[-700:].strip())
            return False
        if "Zero balance -- false positive, continuing." not in out:
            print("  ❌ FAIL (expected zero-balance continuation not seen)")
            print(out[-700:].strip())
            return False

    print("  ✅ PASS")
    for line in out.splitlines():
        if any(token in line for token in ("BLOOM HIT", "VICTORY", "Private key", "WIF", "BTC legacy", "ETH", "Zero balance")):
            print(f"     {line.strip()}")
    return True


def main():
    hydra = resolve_hydra()
    if not os.path.exists(hydra):
        sys.exit(f"ERROR: {hydra} not found")

    backup_path = None
    temp_path = "bloom.test.tmp"
    bloom_path = "bloom.bin"

    cases = [
        make_hex_case("hex-btc", "btc"),
        make_hex_case("hex-eth", "eth"),
        make_seed_case("seed-btc", "btc"),
        make_seed_case("seed-eth", "eth"),
        make_wif_case(),
    ]
    bloom_blob = build_bloom_blob([case["hash20"] for case in cases])

    print(f"=== Hydra BLOOM test suite — {len(cases)} tests ===")
    print(f"Binary : {hydra}")
    print(f"Bloom  : {BLOOM_BYTES // 1024} KiB temporary filter")

    with open(temp_path, "wb") as f:
        f.write(bloom_blob)

    try:
        if os.path.exists(bloom_path):
            backup_path = bloom_path + ".backup"
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.replace(bloom_path, backup_path)
        os.replace(temp_path, bloom_path)

        passed = sum(run_case(hydra, case) for case in cases)
        print(f"\n{'=' * 62}")
        print(f"Result : {passed}/{len(cases)}")
        sys.exit(0 if passed == len(cases) else 1)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(bloom_path):
            os.remove(bloom_path)
        if backup_path and os.path.exists(backup_path):
            os.replace(backup_path, bloom_path)


if __name__ == "__main__":
    main()
