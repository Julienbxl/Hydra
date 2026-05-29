#!/usr/bin/env python3
"""testbrainwallet.py — tests brainwallet mode
Generates 10 test passphrases, derives targets for Direct, Inverse, and Rule-based paths,
and verifies Hydra --brainwallet finds all of them.
"""
import hashlib, os, random, string, subprocess, sys, tempfile

def resolve_hydra(explicit=None):
    if explicit: return explicit
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = ('Hydra.exe', 'Hydra') if os.name == 'nt' else ('Hydra', 'Hydra.exe')
    for c in candidates:
        p = os.path.join(base, '..', c)
        if os.path.exists(p): return p
    return './Hydra'

# ── secp256k1 ────────────────────────────────────────────────────────────────
P  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

def _add(A, B):
    if A is None: return B
    if B is None: return A
    x1, y1 = A; x2, y2 = B
    if x1 == x2:
        if y1 != y2: return None
        m = 3*x1*x1*pow(2*y1, P-2, P) % P
    else:
        m = (y2-y1)*pow(x2-x1, P-2, P) % P
    x3 = (m*m - x1 - x2) % P
    return x3, (m*(x1-x3) - y1) % P

def smul(k):
    r = None; a = (Gx, Gy)
    while k:
        if k & 1: r = _add(r, a)
        a = _add(a, a); k >>= 1
    return r

def sha256d(d): return hashlib.sha256(hashlib.sha256(d).digest()).digest()
def ripemd160_sha256(d):
    h = hashlib.new('ripemd160'); h.update(hashlib.sha256(d).digest()); return h.digest()

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
def b58enc(b):
    n = int.from_bytes(b, 'big'); r = []
    while n: r.append(B58[n%58]); n //= 58
    return '1'*sum(1 for x in b if x == 0) + ''.join(reversed(r))

def passphrase_to_priv(passphrase: str, is_inverse: bool = False) -> bytes:
    priv = hashlib.sha256(passphrase.encode('utf-8')).digest()
    if is_inverse:
        # Inverse: ~k (bitwise NOT)
        # Note: ~k = (2^256 - 1) - k
        priv = bytes(~b & 0xFF for b in priv)
    return priv

def priv_to_btc_address(priv: bytes) -> str:
    x, y = smul(int.from_bytes(priv, 'big'))
    pub = bytes([0x02 if y % 2 == 0 else 0x03]) + x.to_bytes(32, 'big')
    h160 = ripemd160_sha256(pub)
    payload = b'\x00' + h160
    return b58enc(payload + sha256d(payload)[:4])

def gen_passphrase(min_len=4, max_len=15) -> str:
    charset = string.ascii_letters + string.digits
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(charset, k=length))

def run_test(hydra_bin: str, targets: list) -> bool:
    print(f"--- Running {len(targets)} tests ---")
    
    # 1. Backup existing files
    bw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources', 'brainwallet.txt')
    rule_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources', 'rule.txt')
    backup_bw = None
    backup_rule = None
    
    if os.path.exists(bw_path):
        fd, backup_bw = tempfile.mkstemp(suffix='.txt', dir='.', prefix='bw_bak_')
        os.close(fd); os.unlink(backup_bw)
        os.replace(bw_path, backup_bw)
    
    if os.path.exists(rule_path):
        fd, backup_rule = tempfile.mkstemp(suffix='.txt', dir='.', prefix='rule_bak_')
        os.close(fd); os.unlink(backup_rule)
        os.replace(rule_path, backup_rule)
        
    try:
        # 2. Write test files
        with open(bw_path, 'w') as f:
            for t in targets:
                f.write(t['base'] + '\n')
                
        with open(rule_path, 'w') as f:
            f.write(":\n")         # Rule 0: Nothing (Direct)
            f.write("u\n")         # Rule 1: Uppercase
            f.write("c\n")         # Rule 2: Capitalize
            f.write("$!\n")        # Rule 3: Append '!'
        
        # 3. Run Hydra for each target
        passed = 0
        for i, t in enumerate(targets):
            print(f"[{i+1}/{len(targets)}] Testing '{t['type']}' -> Expected pass: '{t['expected']}'")
            cmd = [hydra_bin, 'brainwallet', t['addr']]
            result = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), capture_output=True, text=True, timeout=600)
            output = result.stdout + result.stderr
            
            if 'VICTORY' in output and t['expected'] in output:
                print(f"  -> PASS")
                passed += 1
            else:
                print(f"  -> FAIL\n  Target Addr: {t['addr']}\n  Expected: {t['expected']}\n  Output tail:\n{output[-600:]}")
                
        return passed == len(targets)
    finally:
        # 4. Restore files
        if os.path.exists(bw_path): os.unlink(bw_path)
        if os.path.exists(rule_path): os.unlink(rule_path)
        if backup_bw and os.path.exists(backup_bw): os.replace(backup_bw, bw_path)
        if backup_rule and os.path.exists(backup_rule): os.replace(backup_rule, rule_path)

def main():
    args = sys.argv[1:]
    hydra = resolve_hydra(next((a for a in args if 'Hydra' in a or 'hydra' in a), None))
    
    # Generate 10 targets
    targets = []
    
    # T1: Direct (No rule, No inverse)
    base1 = gen_passphrase()
    targets.append({
        'base': base1, 'expected': base1,
        'addr': priv_to_btc_address(passphrase_to_priv(base1, False)),
        'type': 'Direct'
    })
    
    # T2: Direct Inverse (No rule, Inverse)
    base2 = gen_passphrase()
    targets.append({
        'base': base2, 'expected': base2 + " (INVERSE)",
        'addr': priv_to_btc_address(passphrase_to_priv(base2, True)),
        'type': 'Direct + Inverse'
    })
    
    # T3: Rule Uppercase
    base3 = gen_passphrase().lower()
    targets.append({
        'base': base3, 'expected': base3.upper(),
        'addr': priv_to_btc_address(passphrase_to_priv(base3.upper(), False)),
        'type': 'Rule: Uppercase'
    })
    
    # T4: Rule Capitalize + Inverse
    base4 = gen_passphrase().lower()
    mut4 = base4.capitalize()
    targets.append({
        'base': base4, 'expected': mut4 + " (INVERSE)",
        'addr': priv_to_btc_address(passphrase_to_priv(mut4, True)),
        'type': 'Rule: Capitalize + Inverse'
    })
    
    # T5: Rule Append !
    base5 = gen_passphrase()
    mut5 = base5 + "!"
    targets.append({
        'base': base5, 'expected': mut5,
        'addr': priv_to_btc_address(passphrase_to_priv(mut5, False)),
        'type': 'Rule: Append !'
    })
    
    # T6: Rule Append ! + Inverse
    base6 = gen_passphrase()
    mut6 = base6 + "!"
    targets.append({
        'base': base6, 'expected': mut6 + " (INVERSE)",
        'addr': priv_to_btc_address(passphrase_to_priv(mut6, True)),
        'type': 'Rule: Append ! + Inverse'
    })
    
    # Additional padding targets to reach 10
    for i in range(4):
        b = gen_passphrase()
        is_inv = random.choice([True, False])
        mut = b.upper()
        targets.append({
            'base': b, 'expected': mut + (" (INVERSE)" if is_inv else ""),
            'addr': priv_to_btc_address(passphrase_to_priv(mut, is_inv)),
            'type': f'Random Rule Uppercase + Inv:{is_inv}'
        })

    print(f"=== BRAINWALLET AUTOMATED TEST SUITE ===")
    ok = run_test(hydra, targets)
    if ok:
        print("\nALL 10 TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED.")
        sys.exit(1)

if __name__ == '__main__':
    main()
