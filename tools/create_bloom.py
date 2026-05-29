import sys
import mmh3
import base58
import bech32  # pip install bech32
import os
from tqdm import tqdm
from bitarray import bitarray

# --- CONFIGURATION (must match include/Bloom.h) ---
TARGET_SIZE_GB = 2        # Taille recommandée pour 100M adresses
NUM_HASHES = 16
SEED1 = 0x9747b28c

# Calcul automatique
FILTER_BITS = int(TARGET_SIZE_GB * 1024 * 1024 * 1024 * 8)
FILTER_BYTES = FILTER_BITS // 8

def create_bloom_filter(input_files, output_filepath):
    print(f"[*] Configuration : {TARGET_SIZE_GB} GB / k={NUM_HASHES}")
    print(f"[*] Allocation de {FILTER_BYTES / (1024**3):.2f} Go de RAM...")
    
    # IMPORTANT : endian='little' pour compatibilité GPU
    ba = bitarray(FILTER_BITS, endian='little')
    ba.setall(0)
    
    # 1. Calcul du nombre total de lignes (pour la barre de progression)
    total_lines = 0
    valid_files = []
    
    print("[*] Analyse des fichiers d'entrée...")
    for fp in input_files:
        if os.path.exists(fp):
            print(f"    -> Ajout : {fp}")
            valid_files.append(fp)
            # Comptage rapide des lignes
            with open(fp, 'r') as f:
                total_lines += sum(1 for line in f if line.strip())
        else:
            print(f"    [!] Fichier introuvable ignoré : {fp}")

    if not valid_files:
        print("[ERREUR] Aucun fichier d'entrée valide.")
        return

    print(f"[*] Traitement cumulé de {total_lines:,} adresses...")
    
    count_added = 0
    count_ignored = 0
    
    # Une seule barre de progression pour l'ensemble des fichiers
    with tqdm(total=total_lines, unit="addr") as pbar:
        for fp in valid_files:
            with open(fp, 'r') as f:
                for line in f:
                    address_str = line.strip()
                    if not address_str: 
                        pbar.update(1)
                        continue
                    
                    data_to_hash = None
                    try:
                        # --- LOGIQUE DE DÉTECTION ---
                        
                        # 1. ETH (0x...)
                        if address_str.startswith('0x'):
                            data_to_hash = bytes.fromhex(address_str[2:])
                        
                        # 2. BTC Legacy (1...)
                        elif address_str.startswith('1'):
                            data_to_hash = base58.b58decode_check(address_str)[1:]
                        
                        # 3. BTC Segwit (bc1...)
                        elif address_str.startswith('bc1'):
                            # Décodage Bech32
                            hrp, data = bech32.bech32_decode(address_str)
                            if data is not None and len(data) > 0:
                                # CORRECTION : On prend data[1:] (le hash) AVANT conversion
                                # data[0] est la version (0), data[1:] est le hash 160
                                decoded_data = bech32.convertbits(data[1:], 5, 8, False)
                                
                                if decoded_data is not None:
                                    witness_program = bytes(decoded_data)
                                    
                                    if len(witness_program) == 20:
                                        data_to_hash = witness_program # P2WPKH (OK)

                        # Vérification Finale (Taille GPU)
                        if data_to_hash is None or len(data_to_hash) != 20:
                            count_ignored += 1
                            pbar.update(1)
                            continue

                        # --- HASHING & INSERTION ---
                        h1 = mmh3.hash(data_to_hash, SEED1, signed=False)
                        h2 = mmh3.hash(data_to_hash, h1, signed=False)
                        if h2 == 0:
                            h2 = 1
                        
                        for i in range(NUM_HASHES):
                            bit_pos = (h1 + i * h2) % FILTER_BITS
                            ba[bit_pos] = 1
                        
                        count_added += 1
                        
                    except Exception:
                        count_ignored += 1
                    
                    pbar.update(1)

    print(f"\n[*] Écriture du fichier {output_filepath}...")
    with open(output_filepath, 'wb') as f:
        ba.tofile(f)
        
    print(f"[SUCCESS] Terminé.")
    print(f"    - Adresses ajoutées : {count_added:,}")
    print(f"    - Lignes ignorées   : {count_ignored:,} (P2WSH, erreurs, vides...)")

if __name__ == '__main__':
    # Logique d'arguments : 
    # Au moins 2 arguments (1 entrée + 1 sortie)
    # Usage : python script.py in1.txt in2.txt in3.txt output.blf
    
    if len(sys.argv) < 3:
        print("Usage: python tools/create_bloom.py <input1> [input2 ...] resources/bloom.bin")
        sys.exit(1)
        
    # Tous les arguments sauf le dernier sont des entrées
    inputs = sys.argv[1:-1]
    # Le dernier est la sortie
    output = sys.argv[-1]
    
    create_bloom_filter(inputs, output)
