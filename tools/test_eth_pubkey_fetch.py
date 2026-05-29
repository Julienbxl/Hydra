#!/usr/bin/env python3
"""
test_eth_pubkey_fetch.py
Récupère la pubkey d'une adresse ETH depuis ses transactions signées.

Mécanisme :
  1. Blockscout API → liste des transactions de l'adresse
  2. Pour chaque tx sortante : récupérer r, s, v + hash tx non-signé
  3. ecrecover(tx_hash, r, s, v) → pubkey 64 bytes (x||y)
  4. keccak256(pubkey)[12:] → adresse ETH → vérification

Usage : python3 test_eth_pubkey_fetch.py <eth_address>
"""

import sys, ssl, json, urllib.request, hashlib

# ── Keccak-256 (stdlib only) ──────────────────────────────────────────────────
def keccak256(data: bytes) -> bytes:
    RC = [
        0x0000000000000001,0x0000000000008082,0x800000000000808A,0x8000000080008000,
        0x000000000000808B,0x0000000080000001,0x8000000080008081,0x8000000000008009,
        0x000000000000008A,0x0000000000000088,0x0000000080008009,0x000000008000000A,
        0x000000008000808B,0x800000000000008B,0x8000000000008089,0x8000000000008003,
        0x8000000000008002,0x8000000000000080,0x000000000000800A,0x800000008000000A,
        0x8000000080008081,0x8000000000008080,0x0000000080000001,0x8000000080008008,
    ]
    ROT = [0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14]
    PI  = [0,10,20,5,15,16,1,11,21,6,7,17,2,12,22,23,8,18,3,13,14,24,9,19,4]
    rotl64 = lambda x, n: ((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF
    M = 0xFFFFFFFFFFFFFFFF

    msg = bytearray(data) + b'\x01'
    while len(msg) % 136: msg += b'\x00'
    msg[-1] |= 0x80

    st = [0] * 25
    for bs in range(0, len(msg), 136):
        for i in range(17):
            st[i] ^= int.from_bytes(msg[bs+i*8:bs+i*8+8], 'little')
        for rnd in range(24):
            C = [st[x]^st[x+5]^st[x+10]^st[x+15]^st[x+20] for x in range(5)]
            D = [C[(x+4)%5]^rotl64(C[(x+1)%5],1) for x in range(5)]
            st = [st[i]^D[i%5] for i in range(25)]
            B = [0]*25
            for i in range(25): B[PI[i]] = rotl64(st[i], ROT[i])
            st = [B[x+5*y]^((~B[(x+1)%5+5*y])&M&B[(x+2)%5+5*y])
                  for y in range(5) for x in range(5)]
            st[0] ^= RC[rnd]
    return b''.join(st[i].to_bytes(8, 'little') for i in range(4))

# ── secp256k1 params ──────────────────────────────────────────────────────────
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx= 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy= 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

def modinv(a, m): return pow(a, -1, m)

def point_add(P1, P2):
    if P1 is None: return P2
    if P2 is None: return P1
    x1,y1 = P1; x2,y2 = P2
    if x1 == x2:
        if y1 != y2: return None
        m = (3*x1*x1 * modinv(2*y1, P)) % P
    else:
        m = ((y2-y1) * modinv(x2-x1, P)) % P
    x3 = (m*m - x1 - x2) % P
    return x3, (m*(x1-x3) - y1) % P

def scalar_mul(k, point):
    r = None; q = point
    while k:
        if k & 1: r = point_add(r, q)
        q = point_add(q, q); k >>= 1
    return r

G = (Gx, Gy)

# ── ecrecover ─────────────────────────────────────────────────────────────────
def ecrecover(msg_hash: bytes, v: int, r: int, s: int):
    """
    Recover public key from ECDSA signature.
    v : recovery id, 0 or 1 (Ethereum uses 27/28, subtract 27 first)
    Returns (x, y) or None if invalid.
    """
    if r <= 0 or r >= N or s <= 0 or s >= N:
        return None

    # R = point with x = r (+ N if v & 2, extremely rare)
    x = r + (N if (v & 2) else 0)
    if x >= P:
        return None

    # Compute y from x using curve equation y^2 = x^3 + 7 mod P
    y_sq = (pow(x, 3, P) + 7) % P
    y = pow(y_sq, (P + 1) // 4, P)
    if (y & 1) != (v & 1):
        y = P - y

    R = (x, y)

    # Q = r^-1 * (s*R - hash*G)
    r_inv = modinv(r, N)
    e = int.from_bytes(msg_hash, 'big')

    # Q = r_inv * (s * R - e * G)
    sR = scalar_mul(s, R)
    eG = scalar_mul((-e) % N, G)
    Q  = scalar_mul(r_inv, point_add(sR, eG))
    return Q

def pubkey_to_eth_addr(x: int, y: int) -> str:
    pub64 = x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
    h = keccak256(pub64)
    return '0x' + h[12:].hex()

# ── RLP encoding (minimal, for EIP-155 transactions) ─────────────────────────
def rlp_encode_int(n: int) -> bytes:
    if n == 0: return b'\x80'
    b = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    if len(b) == 1 and b[0] < 0x80: return b
    return bytes([0x80 + len(b)]) + b

def rlp_encode_bytes(b: bytes) -> bytes:
    if len(b) == 0: return b'\x80'
    if len(b) == 1 and b[0] < 0x80: return b
    if len(b) <= 55: return bytes([0x80 + len(b)]) + b
    len_b = len(b).to_bytes((len(b).bit_length() + 7) // 8, 'big')
    return bytes([0xB7 + len(len_b)]) + len_b + b

def rlp_encode_list(items: list) -> bytes:
    payload = b''.join(items)
    if len(payload) <= 55:
        return bytes([0xC0 + len(payload)]) + payload
    len_b = len(payload).to_bytes((len(payload).bit_length() + 7) // 8, 'big')
    return bytes([0xF7 + len(len_b)]) + len_b + payload

def compute_tx_hash_legacy(tx: dict, forced_chain_id=None) -> bytes:
    """
    Compute the signing hash for a legacy (type 0) or EIP-155 transaction.
    EIP-155 : sign(nonce, gasPrice, gasLimit, to, value, data, chainId, 0, 0)
    Legacy   : sign(nonce, gasPrice, gasLimit, to, value, data)
    """
    chain_id = forced_chain_id if forced_chain_id is not None else tx.get('chainId', None)
    if isinstance(chain_id, str):
        chain_id = int(chain_id, 16) if chain_id.startswith('0x') else int(chain_id)

    nonce     = int(tx['nonce'], 16) if isinstance(tx['nonce'], str) else tx['nonce']
    gas_price = int(tx['gasPrice'], 16) if isinstance(tx['gasPrice'], str) else tx['gasPrice']
    gas_limit = int(tx['gas'], 16) if isinstance(tx['gas'], str) else tx['gas']
    to        = bytes.fromhex(tx['to'][2:]) if tx['to'] else b''
    value     = int(tx['value'], 16) if isinstance(tx['value'], str) else tx['value']
    data      = bytes.fromhex(tx['input'][2:]) if tx.get('input') and tx['input'] != '0x' else b''

    items = [
        rlp_encode_int(nonce),
        rlp_encode_int(gas_price),
        rlp_encode_int(gas_limit),
        rlp_encode_bytes(to),
        rlp_encode_int(value),
        rlp_encode_bytes(data),
    ]

    if chain_id and chain_id > 0:
        # EIP-155 replay protection
        items += [rlp_encode_int(chain_id), rlp_encode_int(0), rlp_encode_int(0)]

    return keccak256(rlp_encode_list(items))

def compute_tx_hash_eip1559(tx: dict) -> bytes:
    """
    Compute the signing hash for an EIP-1559 (type 2) transaction.
    sign(chain_id, nonce, max_priority_fee, max_fee, gas_limit, to, value, data, access_list)
    """
    chain_id          = int(tx.get('chainId', '0x1'), 16)
    nonce             = int(tx['nonce'], 16)
    max_priority_fee  = int(tx.get('maxPriorityFeePerGas', '0x0'), 16)
    max_fee           = int(tx.get('maxFeePerGas', '0x0'), 16)
    gas_limit         = int(tx['gas'], 16)
    to                = bytes.fromhex(tx['to'][2:]) if tx['to'] else b''
    value             = int(tx['value'], 16)
    data              = bytes.fromhex(tx['input'][2:]) if tx.get('input') and tx['input'] != '0x' else b''
    access_list       = b'\xc0'  # empty list RLP

    payload = rlp_encode_list([
        rlp_encode_int(chain_id),
        rlp_encode_int(nonce),
        rlp_encode_int(max_priority_fee),
        rlp_encode_int(max_fee),
        rlp_encode_int(gas_limit),
        rlp_encode_bytes(to),
        rlp_encode_int(value),
        rlp_encode_bytes(data),
        access_list,
    ])
    return keccak256(b'\x02' + payload)

# ── Blockscout API ────────────────────────────────────────────────────────────
BLOCKSCOUT = "https://eth.blockscout.com"

def blockscout_get(path: str) -> dict:
    url = f"{BLOCKSCOUT}/api?{path.lstrip('?&')}"
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Hydra"})
    with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
        return json.loads(r.read())

def blockscout_rpc(method: str, params: list) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1}).encode()
    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        f"{BLOCKSCOUT}/api/eth-rpc",
        data=body,
        headers={"User-Agent": "Hydra", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
        return json.loads(r.read())

def fetch_tx_detail(txhash: str) -> dict:
    """Fetch full tx detail including r, s, v via eth_getTransactionByHash"""
    data = blockscout_rpc("eth_getTransactionByHash", [txhash])
    return data.get('result', {})

def fetch_outgoing_txs(addr: str) -> list:
    """Fetch confirmed outgoing transactions signed by addr (normal txs)"""
    data = blockscout_get(
        f"module=account&action=txlist&address={addr}"
        f"&startblock=0&endblock=99999999&sort=asc&page=1&offset=10"
    )
    status = data.get('status')
    txs = data.get('result', [])

    print(f"   Blockscout status: {status}")
    print(f"   Raw result type  : {type(txs).__name__}")
    if isinstance(txs, list) and txs:
        print(f"   First tx keys    : {list(txs[0].keys()) if isinstance(txs[0], dict) else txs[0]}")
    elif isinstance(txs, str):
        print(f"   Message          : {txs}")

    # Blockscout can return a string like "No transactions found" when empty
    if not isinstance(txs, list):
        return []

    addr_low = addr.lower()
    # Keep all txs where this address is the signer (from field)
    # We don't filter on isError — a failed tx still has a valid signature
    out = [tx for tx in txs
           if isinstance(tx, dict)
           and tx.get('from','').lower() == addr_low]
    print(f"   Signed by addr   : {len(out)} / {len(txs)}")
    return out

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    addr = sys.argv[1] if len(sys.argv) > 1 else "0xe8d2E34d18D3547ba69ce6bA78171e877B093977"

    print(f"Testing ETH pubkey fetch for: {addr}")
    print()

    print("1. Fetching outgoing transactions via Blockscout...")
    txs = fetch_outgoing_txs(addr)
    print(f"   Outgoing txs found : {len(txs)}")
    if not txs:
        print("   -> No outgoing tx found. Pubkey unknown.")
        sys.exit(1)
    print()

    # Try each tx until we recover the pubkey successfully
    pubkey = None
    for i, tx in enumerate(txs[:5]):  # Try first 5
        txhash = tx['hash']
        print(f"2. Fetching full tx detail for {txhash[:20]}...")
        tx_full = fetch_tx_detail(txhash)
        if not tx_full:
            print("   -> Empty response, trying next tx")
            continue

        # Extract v, r, s
        v_raw = int(tx_full.get('v', '0x0'), 16)
        r     = int(tx_full.get('r', '0x0'), 16)
        s     = int(tx_full.get('s', '0x0'), 16)
        tx_type = int(tx_full.get('type', '0x0'), 16)

        print(f"   tx type : {tx_type}")
        print(f"   v={v_raw}, r={hex(r)[:18]}..., s={hex(s)[:18]}...")

        # Compute recovery id from v
        # EIP-155 : v = recovery_id + chain_id*2 + 35
        # Legacy   : v = recovery_id + 27
        # EIP-1559 : v = recovery_id (0 or 1)
        if tx_type == 2:
            rec_id = v_raw & 1  # EIP-1559 : v is already 0 or 1
            signing_chain_id = int(tx_full.get('chainId', '0x1'), 16)
        elif v_raw in (0, 1):
            rec_id = v_raw
            signing_chain_id = int(tx_full.get('chainId', '0x1'), 16)
        elif v_raw in (27, 28):
            rec_id = v_raw - 27
            signing_chain_id = 0
        else:
            # EIP-155 : v = rec_id + chain_id*2 + 35
            chain_id = int(tx_full.get('chainId', '0x1'), 16)
            signing_chain_id = chain_id
            rec_id = v_raw - chain_id * 2 - 35
            if rec_id not in (0, 1):
                print(f"   -> Unexpected rec_id={rec_id}, skipping")
                continue

        print(f"   rec_id  : {rec_id}")

        # Compute tx signing hash
        print(f"3. Computing signing hash (type {tx_type})...")
        try:
            if tx_type == 2:
                msg_hash = compute_tx_hash_eip1559(tx_full)
            else:
                msg_hash = compute_tx_hash_legacy(tx_full, signing_chain_id)
            print(f"   signing hash : {msg_hash.hex()}")
        except Exception as e:
            print(f"   -> Hash computation failed: {e}")
            continue

        # ecrecover
        print(f"4. ecrecover(hash, rec_id={rec_id}, r, s)...")
        Q = ecrecover(msg_hash, rec_id, r, s)
        if Q is None:
            print("   -> ecrecover failed, trying next tx")
            continue

        x, y = Q
        recovered_addr = pubkey_to_eth_addr(x, y)
        print(f"   Recovered address : {recovered_addr}")

        if recovered_addr.lower() == addr.lower():
            pubkey = (x, y)
            print(f"\n5. MATCH! Pubkey recovered successfully.")
            print(f"   pubkey_x      = {hex(x)}")
            print(f"   pubkey_y      = {hex(y)}")
            print(f"   pubkey_y_parity = {y & 1}")

            # Format for Hydra
            xb = x.to_bytes(32, 'big')
            limbs = []
            for i in range(4):
                v = int.from_bytes(xb[24-i*8:32-i*8], 'big')
                limbs.append(f"0x{v:016x}")
            print(f"\n   Hydra ETH_PUBKEY target:")
            print(f"   pubkey_y_parity = {y & 1}")
            print(f"   pubkey_x        = {hex(x)[:18]}...{hex(x)[-8:]}")
            print(f"   limbs[0..3]     = {limbs}")
            break
        else:
            print(f"   -> Address mismatch ({recovered_addr} != {addr})")
            print(f"      → Probably RLP encoding issue, trying next tx")

    if pubkey is None:
        print("\n-> Could not recover pubkey from any transaction.")
        sys.exit(1)
