#!/usr/bin/env python3
"""testhex.py — 10 tests mode HEX (monolithique, stdlib)
Usage : python3 testhex.py [./Hydra]
"""
import hashlib,hmac,os,random,struct,subprocess,sys

def resolve_hydra(explicit=None):
    if explicit: return explicit
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = ('Hydra.exe', 'Hydra') if os.name == 'nt' else ('Hydra', 'Hydra.exe')
    for c in candidates:
        p = os.path.join(base, '..', c)
        if os.path.exists(p): return p
    return './Hydra'

P=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

def _add(A,B):
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

def pub(k): px,py=smul(k); return(b'\x02'if py%2==0 else b'\x03')+px.to_bytes(32,'big'),px,py
def pub_uncompressed(k): _,px,py=pub(k); return b'\x04'+px.to_bytes(32,'big')+py.to_bytes(32,'big')
def sha2(d): return hashlib.sha256(d).digest()
def sha2d(d): return sha2(sha2(d))
def h160(d): r=hashlib.new('ripemd160'); r.update(sha2(d)); return r.digest()

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

def btc_legacy(k): p,_,__=pub(k); h=h160(p); pl=b'\x00'+h; return b58e(pl+sha2d(pl)[:4])
def btc_segwit(k):
    p,_,__=pub(k); h=h160(p); d=[0]+_cv(h,8,5)
    he=[ord(x)>>5 for x in'bc']+[0]+[ord(x)&31 for x in'bc']
    p2=_pm(he+d+[0]*6)^1; cs=[(p2>>5*(5-i))&31 for i in range(6)]
    return 'bc1'+''.join(_BC[x] for x in d+cs)
def eth_addr(k): _,px,py=pub(k); return'0x'+keccak256(px.to_bytes(32,'big')+py.to_bytes(32,'big'))[12:].hex()

HYDRA=resolve_hydra()
NUM_TESTS=10; NUM_UNKN=8
ATYPES=['btc_legacy','btc_segwit','eth','pubkey_compressed','pubkey_uncompressed']

def gen(i):
    k=random.randint(1,N-1); kh=k.to_bytes(32,'big').hex()
    at=ATYPES[i%len(ATYPES)]
    if at=='btc_legacy':
        addr=btc_legacy(k)
    elif at=='btc_segwit':
        addr=btc_segwit(k)
    elif at=='eth':
        addr=eth_addr(k)
    elif at=='pubkey_compressed':
        addr=pub(k)[0].hex()
    else:
        addr=pub_uncompressed(k).hex()
    s=random.randint(0,64-NUM_UNKN)
    return dict(i=i,kh=kh,addr=addr,at=at,mask=kh[:s]+'#'*NUM_UNKN+kh[s+NUM_UNKN:],s=s)

def run(t):
    print(f"\n{'='*62}\nTest #{t['i']+1:02d} | {t['at']}\n  Clé    : {t['kh']}\n  Masque : {t['mask']} (pos {t['s']}..{t['s']+NUM_UNKN-1})\n  Target : {t['addr']}")
    try: r=subprocess.run([HYDRA,t['mask'],t['addr']],cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), capture_output=True,text=True,timeout=300)
    except subprocess.TimeoutExpired: print("  ❌ TIMEOUT"); return False
    out=r.stdout+r.stderr
    if'VICTORY'in out:
        print("  ✅ PASS")
        for l in out.splitlines():
            if any(k in l for k in('VICTORY','Clé','priv','key','Key')): print(f"     {l.strip()}")
        return True
    print(f"  ❌ FAIL (rc={r.returncode})\n  {out[-300:].strip()}"); return False

if __name__=='__main__':
    print(f"=== Hydra HEX test suite — {NUM_TESTS} tests × {NUM_UNKN} nibbles ===\nBinaire : {HYDRA}")
    if not os.path.exists(HYDRA): sys.exit(f"ERREUR : {HYDRA} introuvable")
    passed=sum(run(gen(i)) for i in range(NUM_TESTS))
    print(f"\n{'='*62}\nRésultat : {passed}/{NUM_TESTS}"); sys.exit(0 if passed==NUM_TESTS else 1)
