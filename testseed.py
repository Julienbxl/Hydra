#!/usr/bin/env python3
"""testseed.py — tests mode SEED BIP39 (12 et 24 mots)
Usage :
  python3 testseed.py [./Hydra] [--words 12|24|both] [--n N]

Défaut : 5 tests × 12 mots + 5 tests × 24 mots, 2 mots inconnus par test
"""
import hashlib, hmac, os, random, subprocess, sys

def resolve_hydra(explicit=None):
    if explicit:
        return explicit
    candidates = ('Hydra.exe', './Hydra') if os.name == 'nt' else ('./Hydra', 'Hydra.exe')
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return './Hydra'

# ── BIP39 wordlist ─────────────────────────────────────────────────────────
def load_wordlist():
    for path in ["english.txt", "bip39_english.txt"]:
        if os.path.exists(path):
            with open(path) as f:
                w = [l.strip() for l in f if l.strip()]
            if len(w) == 2048: return w
    try:
        from mnemonic import Mnemonic
        return Mnemonic("english").wordlist
    except ImportError:
        pass
    sys.exit("ERROR: cannot load BIP39 wordlist.\n"
             "Fix: pip install mnemonic   OR   place english.txt in current dir.")

WORDS = load_wordlist()
assert len(WORDS) == 2048

# ── secp256k1 ──────────────────────────────────────────────────────────────
P  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
N  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

def _add(A, B):
    if A is None: return B
    if B is None: return A
    x1,y1=A; x2,y2=B
    if x1==x2:
        if y1!=y2: return None
        m=3*x1*x1*pow(2*y1,P-2,P)%P
    else: m=(y2-y1)*pow(x2-x1,P-2,P)%P
    x3=(m*m-x1-x2)%P; return x3,(m*(x1-x3)-y1)%P

def smul(k):
    r=None; a=(Gx,Gy)
    while k:
        if k&1: r=_add(r,a)
        a=_add(a,a); k>>=1
    return r

def sha256(d):  return hashlib.sha256(d).digest()
def sha256d(d): return sha256(sha256(d))
def h160(d):
    r=hashlib.new('ripemd160'); r.update(sha256(d)); return r.digest()

B58='123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
def b58e(d):
    n=int.from_bytes(d,'big'); r=[]
    while n: r.append(B58[n%58]); n//=58
    r+=['1']*next((i for i,b in enumerate(d) if b),0)
    return ''.join(reversed(r))

_BC='qpzry9x8gf2tvdw0s3jn54khce6mua7l'
def _pm(v):
    G=[0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3]; c=1
    for x in v:
        b=c>>25; c=((c&0x1ffffff)<<5)^x
        for i in range(5): c^=G[i] if(b>>i)&1 else 0
    return c
def _cv(d,f,t):
    a=0;b=0;r=[];m=(1<<t)-1
    for v in d: a=(a<<f)|v; b+=f
    while b>=t: b-=t; r.append((a>>b)&m)
    return r

def btc_legacy(k):
    px,py=smul(int.from_bytes(k,'big'))
    pub=(b'\x02'if py%2==0 else b'\x03')+px.to_bytes(32,'big')
    pl=b'\x00'+h160(pub); return b58e(pl+sha256d(pl)[:4])

def btc_segwit(k):
    px,py=smul(int.from_bytes(k,'big'))
    pub=(b'\x02'if py%2==0 else b'\x03')+px.to_bytes(32,'big')
    hh=h160(pub); d=[0]+_cv(hh,8,5)
    he=[ord(x)>>5 for x in'bc']+[0]+[ord(x)&31 for x in'bc']
    p2=_pm(he+d+[0]*6)^1; cs=[(p2>>5*(5-i))&31 for i in range(6)]
    return 'bc1'+''.join(_BC[x] for x in d+cs)

def keccak256(data):
    RC=[0x0000000000000001,0x0000000000008082,0x800000000000808A,0x8000000080008000,
        0x000000000000808B,0x0000000080000001,0x8000000080008081,0x8000000000008009,
        0x000000000000008A,0x0000000000000088,0x0000000080008009,0x000000008000000A,
        0x000000008000808B,0x800000000000008B,0x8000000000008089,0x8000000000008003,
        0x8000000000008002,0x8000000000000080,0x000000000000800A,0x800000008000000A,
        0x8000000080008081,0x8000000000008080,0x0000000080000001,0x8000000080008008]
    ROT=[[0,36,3,41,18],[1,44,10,45,2],[62,6,43,15,61],[28,55,25,21,56],[27,20,39,8,14]]
    M=0xFFFFFFFFFFFFFFFF; r64=lambda x,n:((x<<n)|(x>>(64-n)))&M
    msg=bytearray(data)+b'\x01'
    while len(msg)%136: msg+=b'\x00'
    msg[-1]|=0x80; st=[0]*25
    for bs in range(0,len(msg),136):
        for i in range(17): st[i]^=int.from_bytes(msg[bs+i*8:bs+i*8+8],'little')
        for rnd in range(24):
            C=[st[x]^st[x+5]^st[x+10]^st[x+15]^st[x+20] for x in range(5)]
            D=[C[(x-1)%5]^r64(C[(x+1)%5],1) for x in range(5)]
            st=[st[i]^D[i%5] for i in range(25)]
            B=[0]*25
            for x in range(5):
                for y in range(5): B[y+5*((2*x+3*y)%5)]=r64(st[x+5*y],ROT[x][y])
            st=[B[x+5*y]^((~B[(x+1)%5+5*y])&M&B[(x+2)%5+5*y]) for y in range(5) for x in range(5)]
            st[0]^=RC[rnd]
    return b''.join(st[i].to_bytes(8,'little') for i in range(4))

def eth_addr(k):
    px,py=smul(int.from_bytes(k,'big'))
    return '0x'+keccak256(px.to_bytes(32,'big')+py.to_bytes(32,'big'))[12:].hex()

# ── BIP39 : entropy → mnemonic ─────────────────────────────────────────────
def entropy_to_mnemonic(entropy: bytes) -> str:
    ent_bits = len(entropy) * 8          # 128 pour 12 mots, 256 pour 24 mots
    cs_bits  = ent_bits // 32
    checksum = (int(sha256(entropy).hex(),16) >> (256 - cs_bits)) & ((1<<cs_bits)-1)
    bits = (int.from_bytes(entropy,'big') << cs_bits) | checksum
    total_bits = ent_bits + cs_bits
    num_words  = total_bits // 11
    w = []
    for _ in range(num_words):
        w.append(WORDS[bits & 0x7FF]); bits >>= 11
    return ' '.join(reversed(w))

# ── BIP32 / BIP44 ──────────────────────────────────────────────────────────
def hmac512(key, data): return hmac.new(key, data, hashlib.sha512).digest()

def bip32_master(seed):
    I = hmac512(b'Bitcoin seed', seed); return I[:32], I[32:]

def child_hard(k, c, idx):
    I = hmac512(c, b'\x00'+k+(idx|0x80000000).to_bytes(4,'big'))
    ki = (int.from_bytes(I[:32],'big') + int.from_bytes(k,'big')) % N
    return ki.to_bytes(32,'big'), I[32:]

def child_norm(k, c, idx):
    px,py = smul(int.from_bytes(k,'big'))
    pub = (b'\x02'if py%2==0 else b'\x03')+px.to_bytes(32,'big')
    I = hmac512(c, pub+idx.to_bytes(4,'big'))
    ki = (int.from_bytes(I[:32],'big') + int.from_bytes(k,'big')) % N
    return ki.to_bytes(32,'big'), I[32:]

def derive_bip44(mnemonic, coin=0):
    seed = hashlib.pbkdf2_hmac('sha512', mnemonic.encode(),
                                b'mnemonic', 2048)
    k,c = bip32_master(seed)
    for idx in [44, coin, 0]: k,c = child_hard(k,c,idx)
    k,c = child_norm(k,c,0)
    k,c = child_norm(k,c,0)
    return k

# ── Génération d'un cas de test ────────────────────────────────────────────
ATYPES    = ['btc_legacy','btc_segwit','eth']
NUM_UNKN  = 2   # mots inconnus par test (≤ SEED_MAX_X=6)

def gen(test_idx, num_words):
    ent_bytes = {12: 16, 24: 32}[num_words]
    entropy   = os.urandom(ent_bytes)
    mnemonic  = entropy_to_mnemonic(entropy)
    words     = mnemonic.split()
    assert len(words) == num_words

    at    = ATYPES[test_idx % 3]
    coin  = 60 if at == 'eth' else 0
    privk = derive_bip44(mnemonic, coin)
    addr  = (btc_legacy(privk) if at=='btc_legacy'
             else btc_segwit(privk) if at=='btc_segwit'
             else eth_addr(privk))

    # Positions inconnues — éviter le dernier mot (checksum)
    unknown_pos = sorted(random.sample(range(num_words - 1), NUM_UNKN))
    masked = words.copy()
    for p in unknown_pos: masked[p] = '#'

    return dict(idx=test_idx, num_words=num_words, mnemonic=mnemonic,
                mask=' '.join(masked), addr=addr, at=at, unknown_pos=unknown_pos)

# ── Exécution d'un test ────────────────────────────────────────────────────
def run(t, hydra):
    print(f"\n{'='*65}")
    print(f"Test #{t['idx']+1:02d} | {t['num_words']} words | {t['at']}")
    print(f"  Mnemonic : {t['mnemonic']}")
    print(f"  Mask     : {t['mask']}")
    print(f"  Unknown  : positions {t['unknown_pos']}")
    print(f"  Address  : {t['addr']}")
    try:
        r = subprocess.run([hydra, t['mask'], t['addr']],
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  ❌ TIMEOUT"); return False
    out = r.stdout + r.stderr
    if 'VICTORY' in out:
        print("  ✅ PASS")
        for l in out.splitlines():
            if any(k in l for k in ('VICTORY','key','Key','seed','Seed','mnemonic')):
                print(f"     {l.strip()}")
        return True
    print(f"  ❌ FAIL (rc={r.returncode})\n  {out[-400:].strip()}")
    return False

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('hydra',  nargs='?', default=None)
    ap.add_argument('--words', choices=['12','24','both'], default='both',
                    help='Phrase length to test (default: both)')
    ap.add_argument('--n', type=int, default=5,
                    help='Number of tests per length (default: 5)')
    args = ap.parse_args()

    args.hydra = resolve_hydra(args.hydra)
    if not os.path.exists(args.hydra):
        sys.exit(f"ERROR: {args.hydra} not found")

    lengths = {'12':[12], '24':[24], 'both':[12,24]}[args.words]

    print(f"=== Hydra SEED test suite ===")
    print(f"Binary  : {args.hydra}")
    print(f"Lengths : {lengths}  |  {NUM_UNKN} unknown words per test")

    passed = 0; total = 0
    for nw in lengths:
        print(f"\n{'─'*65}  {nw}-WORD PHRASES")
        for i in range(args.n):
            t = gen(total, nw)
            if run(t, args.hydra): passed += 1
            total += 1

    print(f"\n{'='*65}")
    print(f"Result : {passed}/{total}")
    sys.exit(0 if passed == total else 1)
