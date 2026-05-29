#!/usr/bin/env python3
"""testelectrumv2.py — tests mode ELECTRUM V2
Usage :
  python3 testelectrumv2.py [./Hydra] [--n N]

Génère une phrase mnémonique Electrum v2 valide (préfixe HMAC "01"),
masque 2 mots, et vérifie que Hydra --electrumV2 retrouve la bonne adresse.
"""
import hashlib, hmac as hmac_mod, os, random, subprocess, sys, struct

def resolve_hydra(explicit=None):
    if explicit: return explicit
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = ('Hydra.exe', 'Hydra') if os.name == 'nt' else ('Hydra', 'Hydra.exe')
    for c in candidates:
        p = os.path.join(base, '..', c)
        if os.path.exists(p): return p
    return './Hydra'

# ── BIP39 wordlist ──────────────────────────────────────────────────────────
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
    sys.exit("ERROR: cannot load BIP39 wordlist. Fix: pip install mnemonic")

WORDS = load_wordlist()

# ── secp256k1 ───────────────────────────────────────────────────────────────
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

def sha256d(d): return hashlib.sha256(hashlib.sha256(d).digest()).digest()
def ripemd160(d):
    h=hashlib.new('ripemd160'); h.update(hashlib.sha256(d).digest()); return h.digest()

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
def b58enc(b):
    n=int.from_bytes(b,'big'); r=[]
    while n: r.append(B58[n%58]); n//=58
    return '1'*sum(1 for x in b if x==0)+''.join(reversed(r))

def privkey_to_p2pkh(priv_bytes):
    x,y=smul(int.from_bytes(priv_bytes,'big'))
    pub=bytes([0x02 if y%2==0 else 0x03])+x.to_bytes(32,'big')
    h160=ripemd160(pub)
    payload=b'\x00'+h160
    return b58enc(payload+sha256d(payload)[:4])

# ── BIP32 ───────────────────────────────────────────────────────────────────
def hmac_sha512(key, data):
    return hmac_mod.new(key, data, hashlib.sha512).digest()

def bip32_master(seed):
    out = hmac_sha512(b"Bitcoin seed", seed)
    return out[:32], out[32:]  # priv, chain

def bip32_ckd_normal(priv, chain, idx):
    x,y=smul(int.from_bytes(priv,'big'))
    pub=bytes([0x02 if y%2==0 else 0x03])+x.to_bytes(32,'big')
    data = pub + struct.pack('>I', idx)
    out = hmac_sha512(chain, data)
    tweak = int.from_bytes(out[:32],'big')
    child = (tweak + int.from_bytes(priv,'big')) % N
    return child.to_bytes(32,'big'), out[32:]

# ── Electrum v2 checksum ────────────────────────────────────────────────────
SEED_VERSION_KEY = b"Seed version"

def electrum_is_standard(phrase):
    h = hmac_mod.new(SEED_VERSION_KEY, phrase.encode('utf-8'), hashlib.sha512).digest()
    return h[0] == 0x01

def electrum_seed_to_privkey(phrase):
    """Dérive la clé privée Electrum v2 : PBKDF2('electrum') + m/0/0"""
    seed = hashlib.pbkdf2_hmac('sha512', phrase.encode('utf-8'), b'electrum', 2048)
    mpriv, mchain = bip32_master(seed)
    k0, c0 = bip32_ckd_normal(mpriv, mchain, 0)   # m/0
    k1, _  = bip32_ckd_normal(k0, c0, 0)           # m/0/0
    return k1

# ── Génère une phrase Electrum v2 valide ────────────────────────────────────
def gen_electrum_phrase(n_words=12, max_tries=50000):
    """Tire des phrases aléatoires jusqu'à trouver un préfixe HMAC '01'."""
    for attempt in range(max_tries):
        words = [random.choice(WORDS) for _ in range(n_words)]
        phrase = ' '.join(words)
        if electrum_is_standard(phrase):
            return phrase
    raise RuntimeError(f"Impossible de trouver une phrase valide en {max_tries} tentatives")

# ── Test principal ──────────────────────────────────────────────────────────
def run_test(hydra_bin, n_words=12, n_unknown=2, verbose=False):
    # 1. Génère phrase Electrum valide
    phrase = gen_electrum_phrase(n_words)
    words = phrase.split()

    # 2. Calcule l'adresse cible
    priv = electrum_seed_to_privkey(phrase)
    addr = privkey_to_p2pkh(priv)

    # 3. Masque n_unknown mots
    unk_pos = sorted(random.sample(range(n_words), n_unknown))
    masked = words[:]
    for p in unk_pos: masked[p] = '#'
    mask_phrase = ' '.join(masked)

    if verbose:
        print(f"  Phrase   : {phrase}")
        print(f"  Address  : {addr}")
        print(f"  Mask     : {mask_phrase}")

    # 4. Exécute Hydra
    cmd = [hydra_bin, mask_phrase, addr, '--electrumV2']
    result = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    if 'VICTORY' in output and phrase in output:
        return True, f"OK  ({n_words} mots, pos masquées: {unk_pos})"
    else:
        return False, f"FAIL\n  Expected: {phrase}\n  Output:\n{output[-500:]}"

def main():
    args = sys.argv[1:]
    hydra = resolve_hydra(next((a for a in args if 'Hydra' in a or 'hydra' in a), None))
    n = 3
    for a in args:
        if a.startswith('--n='): n = int(a[4:])
        elif a.startswith('--n') and a[2:].isdigit(): n = int(a[2:])

    print(f"=== ELECTRUM V2 mode test ({n} run(s), binary: {hydra}) ===")
    passed = failed = 0
    for i in range(n):
        ok, msg = run_test(hydra, verbose=True)
        status = "PASS" if ok else "FAIL"
        print(f"[{i+1}/{n}] {status} — {msg}")
        if ok: passed += 1
        else:  failed += 1

    print(f"\nResults: {passed}/{n} passed")
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()
