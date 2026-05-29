#!/usr/bin/env python3
"""testelectrumv1.py -- tests old Electrum V1 mode

Usage:
  python3 testelectrumv1.py [./Hydra]
"""
import hashlib
import os
import re
import subprocess
import sys

PHRASE = "like just love know never want time out there make look eye"
WORDS = None
WORD_INDEX = None

P  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

def resolve_hydra(explicit=None):
    if explicit: return explicit
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = ('Hydra.exe', 'Hydra') if os.name == 'nt' else ('Hydra', 'Hydra.exe')
    for c in candidates:
        p = os.path.join(base, '..', c)
        if os.path.exists(p): return p
    return './Hydra'

def load_words():
    global WORDS, WORD_INDEX
    if WORDS is not None:
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(script_dir, '..'))
    paths = [
        os.path.join(repo_dir, "include", "ElectrumV1Words.h"),
        os.path.join(script_dir, "..", "include", "ElectrumV1Words.h"),
        os.path.join(os.getcwd(), "include", "ElectrumV1Words.h"),
        os.path.join(os.getcwd(), "ElectrumV1Words.h"),
        "/tmp/electrum_v1_words.txt",
        "/tmp/python_hdwallet_ref/hdwallet/mnemonics/electrum/v1/wordlist/english.txt",
    ]
    for path in paths:
        if os.path.exists(path):
            if path.endswith(".h"):
                text = open(path, encoding="utf-8").read()
                def array_values(name):
                    m = re.search(rf"{name}\[[^\]]+\]\s*=\s*\{{(.*?)\}};", text, re.S)
                    if not m:
                        raise RuntimeError(f"Cannot parse {name} from {path}")
                    return [int(x) for x in re.findall(r"\d+", m.group(1))]
                blob = bytes(array_values("h_ELECTRUM_V1_BLOB"))
                offs = array_values("h_ELECTRUM_V1_OFFS")
                lens = array_values("h_ELECTRUM_V1_LENS")
                WORDS = [blob[o:o+l].decode("ascii") for o, l in zip(offs, lens)]
            else:
                with open(path, encoding="utf-8") as f:
                    WORDS = [line.strip() for line in f if line.strip()]
            break
    if WORDS is None:
        raise RuntimeError("Cannot load Electrum V1 wordlist")
    if len(WORDS) != 1626:
        raise RuntimeError(f"Electrum V1 wordlist must contain 1626 words, got {len(WORDS)}")
    WORD_INDEX = {w: i for i, w in enumerate(WORDS)}

def add(A, B):
    if A is None:
        return B
    if B is None:
        return A
    x1, y1 = A
    x2, y2 = B
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
            r = add(r, a)
        a = add(a, a)
        k >>= 1
    return r

def sha256d(data):
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def hash160(data):
    h = hashlib.new("ripemd160")
    h.update(hashlib.sha256(data).digest())
    return h.digest()

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
def b58enc(data):
    n = int.from_bytes(data, "big")
    out = []
    while n:
        out.append(B58[n % 58])
        n //= 58
    return "1" * (len(data) - len(data.lstrip(b"\0"))) + "".join(reversed(out))

def electrumv1_seed_ascii(phrase):
    load_words()
    words = phrase.split()
    chunks = []
    n = 1626
    for g in range(4):
        w1 = WORD_INDEX[words[g * 3 + 0]]
        w2 = WORD_INDEX[words[g * 3 + 1]]
        w3 = WORD_INDEX[words[g * 3 + 2]]
        chunk = w1 + n * ((w2 - w1) % n) + n * n * ((w3 - w2) % n)
        chunks.append(chunk.to_bytes(4, "big").hex())
    return "".join(chunks).encode()

def electrumv1_master_priv(phrase):
    oldseed = electrumv1_seed_ascii(phrase)
    x = oldseed
    for _ in range(100000):
        x = hashlib.sha256(x + oldseed).digest()
    return x

def electrumv1_child_priv(phrase, change=0, address_index=0):
    master = electrumv1_master_priv(phrase)
    mx, my = smul(int.from_bytes(master, "big"))
    mpk = mx.to_bytes(32, "big") + my.to_bytes(32, "big")
    seq = sha256d(f"{address_index}:{change}:".encode() + mpk)
    child = (int.from_bytes(master, "big") + int.from_bytes(seq, "big")) % N
    return child.to_bytes(32, "big")

def p2pkh_uncompressed(priv):
    x, y = smul(int.from_bytes(priv, "big"))
    pub = b"\x04" + x.to_bytes(32, "big") + y.to_bytes(32, "big")
    payload = b"\x00" + hash160(pub)
    return b58enc(payload + sha256d(payload)[:4])

def run_case(hydra, phrase, change, address_index, mask_phrase=None):
    priv = electrumv1_child_priv(phrase, change, address_index)
    addr = p2pkh_uncompressed(priv)
    mask_phrase = mask_phrase or phrase
    print(f"  Path   : m/{change}/{address_index}")
    print(f"  Address: {addr}")
    print(f"  Mask   : {mask_phrase}")
    result = subprocess.run([hydra, mask_phrase, addr, "--electrumV1"], cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), capture_output=True, timeout=300)
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    expected_path = f"m/{change}/{address_index}"
    ok = "VICTORY" in output and phrase in output and expected_path in output
    if ok:
        print("  PASS")
        return True
    print("  FAIL")
    print(output[-1200:])
    return False

def main():
    hydra = resolve_hydra(next((a for a in sys.argv[1:] if "Hydra" in a or "hydra" in a), None))
    load_words()
    print("=== ELECTRUM V1 mode test ===")
    print(f"  Phrase : {PHRASE}")

    cases = [
        (0, 0, "like just # know never want time out there make look eye"),
        (0, 7, "like just love know # want time out there make look eye"),
        (1, 3, "like just love know never want time # there make look eye"),
        (0, 0, "# just love know never want time out there make look eye"),
    ]
    passed = 0
    for change, address_index, mask_phrase in cases:
        if run_case(hydra, PHRASE, change, address_index, mask_phrase):
            passed += 1
    print(f"Results: {passed}/{len(cases)} passed")
    if passed == len(cases):
        return 0
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
