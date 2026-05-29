import sys
import hashlib
import mmh3  # pip install mmh3
import base58 # pip install base58
import bech32 # pip install bech32
import binascii
import os

# --- PARAMÈTRES FIXES ---
BLOOM_K_HASHES = 16
BLOOM_FILTER_FILE = "bloom.bin"
SEED = 0x9747b28c

def check_bloom_filter(h160_bytes, bloom_data):
    """
    Vérifie un hash de 20 octets contre les données du filtre de Bloom.
    """
    if len(h160_bytes) != 20:
        print(f"[Attention] Le hash fait {len(h160_bytes)} octets (attendu: 20). Filtre ignoré pour cette adresse.")
        return False

    bloom_size_bytes = len(bloom_data)
    bloom_m_bits = bloom_size_bytes * 8

    # Double hachage Murmur3
    h1 = mmh3.hash(h160_bytes, seed=SEED, signed=False)
    h2 = mmh3.hash(h160_bytes, seed=h1, signed=False)

    for i in range(BLOOM_K_HASHES):
        bit_pos = (h1 + i * h2) % bloom_m_bits
        byte_index = bit_pos // 8
        bit_index_in_byte = bit_pos % 8
        
        if not (bloom_data[byte_index] & (1 << bit_index_in_byte)):
            return False 

    return True 

def decode_btc_base58(addr):
    """Décode une adresse Legacy (1...) ou P2SH (3...)."""
    try:
        decoded = base58.b58decode_check(addr)
        # On retire le byte de version (le 1er) pour avoir le H160
        return decoded[1:]
    except Exception as e:
        print(f"[Erreur] Base58 invalide : {e}")
        return None

def decode_btc_bech32(addr):
    """Décode une adresse SegWit native (bc1q...)."""
    try:
        # Décodage Bech32 (HRP + data 5-bits)
        hrp, data = bech32.bech32_decode(addr)
        
        if hrp != 'bc' or data is None:
            print("[Erreur] Bech32 invalide ou mauvais HRP.")
            return None
            
        # Conversion des données 5-bits en 8-bits
        # data[0] est la version du témoin (witness version), on le saute pour avoir le hash
        decoded_data = bech32.convertbits(data[1:], 5, 8, False)
        
        if decoded_data is None:
            print("[Erreur] Conversion des bits Bech32 échouée.")
            return None
            
        return bytes(decoded_data)
    except Exception as e:
        print(f"[Erreur] Bech32 exception : {e}")
        return None

def decode_eth_address(addr):
    """Décode une adresse Ethereum hexadécimale."""
    if addr.startswith('0x'):
        addr = addr[2:]
    try:
        if len(addr) != 40:
            print("[Erreur] Une adresse ETH doit faire 40 caractères hexadécimaux.")
            return None
        return binascii.unhexlify(addr)
    except Exception as e:
        print(f"[Erreur] Impossible de décoder l'adresse Ethereum : {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <adresse_btc_ou_eth>")
        sys.exit(1)
        
    address = sys.argv[1]
    h160 = None
    
    # --- LOGIQUE DE DÉTECTION MISE À JOUR ---
    if address.startswith('bc1'):
        print(f"[*] Détection : Bitcoin SegWit (Bech32)")
        h160 = decode_btc_bech32(address)
    elif address.startswith('1') or address.startswith('3'):
        print(f"[*] Détection : Bitcoin Legacy/P2SH (Base58)")
        h160 = decode_btc_base58(address)
    elif address.startswith('0x'):
        print(f"[*] Détection : Ethereum (Hex)")
        h160 = decode_eth_address(address)
    else:
        # Fallback
        try:
            print(f"[*] Détection : Tentative Base58 par défaut")
            h160 = decode_btc_base58(address)
        except:
            print("Adresse non reconnue.")
            sys.exit(1)
        
    if not h160:
        sys.exit(1)

    print(f"[*] Hash Target (Hex) : {h160.hex()}")

    # Chargement du fichier Bloom Filter
    if not os.path.exists(BLOOM_FILTER_FILE):
        print(f"[ERREUR CRITIQUE] Fichier '{BLOOM_FILTER_FILE}' non trouvé.")
        sys.exit(1)

    try:
        # Lecture optimisée : on lit tout en mémoire (si < quelques Go ça passe)
        # Sinon il faudrait utiliser mmap pour les très gros fichiers
        with open(BLOOM_FILTER_FILE, "rb") as f:
            bloom_data = f.read()
            
        file_size = len(bloom_data)
        print(f"[*] Chargement du filtre '{BLOOM_FILTER_FILE}' OK.")
        print(f"    -> Taille : {file_size / (1024*1024):.2f} Mo")
        
    except Exception as e:
        print(f"[Erreur] Lecture du fichier impossible : {e}")
        sys.exit(1)

    # Vérification
    is_hit = check_bloom_filter(h160, bloom_data)

    print("\n--- RÉSULTAT DU DIAGNOSTIC ---")
    if is_hit:
        print(" ✅  HIT POSITIF")
        print("    -> L'adresse est PRÉSENTE dans le filtre (ou collision).")
    else:
        print(" ❌  MISS NÉGATIF")
        print("    -> L'adresse est ABSENTE du filtre.")