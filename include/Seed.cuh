#pragma once
/*
 * ======================================================================================
 * HYDRA - Seed.cuh  (Mode BIP39 : brute force mots manquants)
 * ======================================================================================
 *
 * Chaque thread teste une combinaison de mots X :
 *
 *   1. Reconstruit la phrase mnémonique depuis les indices de mots
 *   2. PBKDF2-HMAC-SHA512 (2048 itérations) → seed 64 bytes
 *   3. BIP32 master key : HMAC-SHA512("Bitcoin seed", seed)
 *   4. BIP44 dérivation : m/44'/0'/0'/0/0 (BTC) ou m/44'/60'/0'/0/0 (ETH)
 *      5 niveaux × HMAC-SHA512
 *   5. ECC : scalar_mul → clé publique
 *   6. Hash160 (BTC) ou Keccak (ETH) → compare
 *
 * ======================================================================================
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include "HydraCommon.h"
#include "Bloom.h"
#include "ECC.h"
#include "Hash.cuh"
#include "PBKDF2.h"

// ============================================================================
// DICTIONNAIRE BIP39 EN CONSTANT MEMORY
// ============================================================================
// Importé depuis BIP39_Dict.h (côté CPU), uploadé via cudaMemcpyToSymbol
__constant__ uint8_t  d_BIP39_BLOB[11068];
__constant__ uint16_t d_BIP39_OFFS[2048];
__constant__ uint8_t  d_BIP39_LENS[2048];

// ============================================================================
// CONFIG SEED
// ============================================================================
#define SEED_MAX_WORDS   24   // max mots dans une phrase BIP39
#define SEED_MAX_X        6   // max positions inconnues (utilisateur assume)

// Structure passée au kernel — décrit le masque de la phrase
struct SeedMask {
    // Mots connus : word_indices[i] = index BIP39 (0..2047), ou 0xFFFF si inconnu
    uint16_t word_indices[SEED_MAX_WORDS];
    uint8_t  num_words;           // 12, 15, 18, 21, ou 24
    uint8_t  num_unknown;         // nombre de positions X
    uint8_t  unknown_pos[SEED_MAX_X];  // positions des X (0..num_words-1)

    // === Checksum BIP39 optimisé ===
    uint8_t  checksum_bits;       // 4 pour 12 mots, 8 pour 24 mots
    bool     last_word_unknown;   // true si le dernier mot est parmi les X
    uint8_t  required_checksum;   // checksum attendu si dernier mot connu (0xFF sinon)
    uint8_t  electrum_mode;       // 0=BIP39, 1=Electrum standard ("01"), 2=Electrum segwit ("100")
    uint8_t  pad[1];

    uint64_t total_candidates;    // candidats RÉELS après optimisation checksum
};


// ============================================================================
// HELPERS DEVICE BIP39
// ============================================================================

// Calcule le checksum BIP39 (SHA256 des bytes entropy, 4 ou 8 bits selon longueur)
__device__ __forceinline__ uint8_t bip39_checksum(
    const uint8_t *entropy, int entropy_bytes)
{
    // SHA256 de l'entropy (entropy_bytes = 16 pour 12 mots, 32 pour 24 mots)
    // On n'a besoin que des premiers bits du hash
    // Réutilise compute_sha256 inline
    uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint32_t w[64] = {0};

    // Charge entropy (max 32 bytes = 8 mots 32 bits)
    int nw = (entropy_bytes + 3) / 4;
    for(int i=0;i<nw;i++){
        int base = i*4;
        w[i] = ((uint32_t)entropy[base]<<24)
              |((uint32_t)(base+1<entropy_bytes?entropy[base+1]:0)<<16)
              |((uint32_t)(base+2<entropy_bytes?entropy[base+2]:0)<<8)
              |((uint32_t)(base+3<entropy_bytes?entropy[base+3]:0));
    }
    // Padding
    w[nw] = 0x80000000;
    w[15] = entropy_bytes * 8;

    for(int i=16;i<64;i++){
        uint32_t s0 = (w[i-15]>>7|(w[i-15]<<25))^(w[i-15]>>18|(w[i-15]<<14))^(w[i-15]>>3);
        uint32_t s1 = (w[i-2]>>17|(w[i-2]<<15))^(w[i-2]>>19|(w[i-2]<<13))^(w[i-2]>>10);
        w[i] = w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for(int i=0;i<64;i++){
        uint32_t S1=(e>>6|(e<<26))^(e>>11|(e<<21))^(e>>25|(e<<7));
        uint32_t ch=(e&f)^(~e&g);
        uint32_t t1=hh+S1+ch+k[i]+w[i];
        uint32_t S0=(a>>2|(a<<30))^(a>>13|(a<<19))^(a>>22|(a<<10));
        uint32_t maj=(a&b)^(a&c)^(b&c);
        hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+S0+maj;
    }
    // 4 bits checksum pour 12 mots (128 bits entropy), 8 bits pour 24 mots (256 bits)
    // First digest word = SHA256 IV[0] + working variable a.
    // Keep the IV explicit so this remains correct if the local state layout changes.
    uint32_t first = 0x6a09e667u + a;
    return (entropy_bytes == 16) ? (uint8_t)(first >> 28) : (uint8_t)(first >> 24);
}

// Extrait l'index du mot w (0..num_words-1) depuis entropy+checksum
__device__ __forceinline__ uint16_t bip39_word_index(
    const uint8_t *entropy, uint8_t csum, int word_idx, int num_words)
{
    int start_bit = word_idx * 11;
    uint16_t val = 0;
    int entropy_bits = (num_words == 12) ? 128 : 256;
    for(int i=start_bit; i<start_bit+11; i++){
        uint8_t bit;
        if(i < entropy_bits){
            bit = (entropy[i/8] >> (7-(i%8))) & 1;
        } else {
            int cb = i - entropy_bits;
            // Pour 12 mots: 4 bits checksum dans les bits hauts
            bit = (csum >> (3-cb)) & 1;
        }
        val = (val<<1)|bit;
    }
    return val;
}

// Reconstruit la mnemonic string depuis les word_indices
__device__ __forceinline__ uint32_t build_mnemonic(
    const uint16_t *word_indices, int num_words,
    uint8_t *out_buf)
{
    uint32_t pos = 0;
    for(int w=0; w<num_words; w++){
        uint16_t idx = word_indices[w];
        uint16_t off = d_BIP39_OFFS[idx];
        uint8_t  len = d_BIP39_LENS[idx];
        for(int k=0;k<len;k++) out_buf[pos++] = d_BIP39_BLOB[off+k];
        if(w < num_words-1) out_buf[pos++] = ' ';
    }
    return pos;
}

// ============================================================================
// BIP32 : dérivation d'un niveau
// child_key[64] = HMAC-SHA512(parent_key[32] || parent_chain[32], data[37])
// Pour hardened : data = 0x00 || parent_priv[32] || index_be32
// Pour normal   : data = compressed_pubkey[33] || index_be32
// ============================================================================
// ============================================================================
// secp256k1 ordre n (constant device)
// ============================================================================
__device__ __constant__ uint8_t SECP256K1_N[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

// child_priv = (tweak[0..31] + parent_priv[0..31]) mod n
// tweak et parent sont en big-endian byte[32]
__device__ __forceinline__ void add_mod_n(
    const uint8_t tweak[32],
    const uint8_t parent[32],
    uint8_t       out[32])
{
    uint32_t carry = 0;
    uint8_t  tmp[32];
    #pragma unroll
    for (int i = 31; i >= 0; i--) {
        uint32_t s = (uint32_t)tweak[i] + (uint32_t)parent[i] + carry;
        tmp[i] = (uint8_t)(s & 0xFF);
        carry  = s >> 8;
    }
    bool ge = (carry > 0);
    if (!ge) {
        for (int i = 0; i < 32; i++) {
            if (tmp[i] > SECP256K1_N[i]) { ge = true;  break; }
            if (tmp[i] < SECP256K1_N[i]) { break; }
        }
    }
    if (ge) {
        uint32_t borrow = 0;
        for (int i = 31; i >= 0; i--) {
            int32_t s  = (int32_t)tmp[i] - (int32_t)SECP256K1_N[i] - (int32_t)borrow;
            out[i]     = (uint8_t)((s + 256) & 0xFF);
            borrow     = (s < 0) ? 1 : 0;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 32; i++) out[i] = tmp[i];
    }
}

__device__ __noinline__ void bip32_derive_child_hardened(
    const uint8_t parent_priv[32],
    const uint8_t parent_chain[32],
    uint32_t      index,          // avec 0x80000000 déjà appliqué
    uint8_t       child_priv[32],
    uint8_t       child_chain[32])
{
    // data = 0x00 || parent_priv[32] || index_be32  (37 bytes)
    uint8_t data[37];
    data[0] = 0x00;
    #pragma unroll
    for (int i = 0; i < 32; i++) data[i+1] = parent_priv[i];
    data[33] = (uint8_t)(index >> 24);
    data[34] = (uint8_t)(index >> 16);
    data[35] = (uint8_t)(index >>  8);
    data[36] = (uint8_t)(index);

    // HMAC-SHA512(key=chain[32], msg=data[37]) — bypass optimisé
    uint8_t out[64];
    hmac_sha512_bip32(parent_chain, data, out);

    // child_priv = (out[0..31] + parent_priv) mod n
    add_mod_n(out, parent_priv, child_priv);

    // child_chain = out[32..63]
    #pragma unroll
    for (int i = 0; i < 32; i++) child_chain[i] = out[32+i];
}

// Dérivation normale : nécessite la clé publique compressée du parent
__device__ __noinline__ void bip32_derive_child_normal(
    const uint8_t parent_priv[32],
    const uint8_t parent_chain[32],
    uint32_t      index,
    uint8_t       child_priv[32],
    uint8_t       child_chain[32])
{
    // ECC : parent_priv → pubkey compressée 33 bytes
    uint64_t priv_limbs[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        priv_limbs[3-i] =
              ((uint64_t)parent_priv[i*8+0] << 56) | ((uint64_t)parent_priv[i*8+1] << 48)
            | ((uint64_t)parent_priv[i*8+2] << 40) | ((uint64_t)parent_priv[i*8+3] << 32)
            | ((uint64_t)parent_priv[i*8+4] << 24) | ((uint64_t)parent_priv[i*8+5] << 16)
            | ((uint64_t)parent_priv[i*8+6] <<  8) | ((uint64_t)parent_priv[i*8+7]);
    }
    uint64_t ax[4], ay[4];
    scalarMulBaseAffine(priv_limbs, ax, ay);
    fieldNormalize(ax); fieldNormalize(ay);

    // data = pubkey[33] || index_be32  (37 bytes)
    uint8_t data[37];
    data[0] = (ay[0] & 1) ? 0x03 : 0x02;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t w = ax[3-i];
        data[1+i*8+0]=(uint8_t)(w>>56); data[1+i*8+1]=(uint8_t)(w>>48);
        data[1+i*8+2]=(uint8_t)(w>>40); data[1+i*8+3]=(uint8_t)(w>>32);
        data[1+i*8+4]=(uint8_t)(w>>24); data[1+i*8+5]=(uint8_t)(w>>16);
        data[1+i*8+6]=(uint8_t)(w>> 8); data[1+i*8+7]=(uint8_t)(w);
    }
    data[33] = (uint8_t)(index >> 24);
    data[34] = (uint8_t)(index >> 16);
    data[35] = (uint8_t)(index >>  8);
    data[36] = (uint8_t)(index);

    // HMAC-SHA512(key=chain[32], msg=data[37]) — bypass optimisé
    uint8_t out[64];
    hmac_sha512_bip32(parent_chain, data, out);

    // child_priv = (out[0..31] + parent_priv) mod n
    add_mod_n(out, parent_priv, child_priv);

    // child_chain = out[32..63]
    #pragma unroll
    for (int i = 0; i < 32; i++) child_chain[i] = out[32+i];
}

// ============================================================================
// BIP44 : m/44'/coin'/0'/0/0
// coin = 0 pour BTC, 60 pour ETH
// Dérivation : 5 niveaux
//   0 : hardened 44'
//   1 : hardened coin'
//   2 : hardened 0'
//   3 : normal 0
//   4 : normal 0
// ============================================================================
__device__ __noinline__ void bip44_derive(
    const uint8_t master_priv[32],
    const uint8_t master_chain[32],
    uint32_t coin_type,   // 0=BTC, 60=ETH
    uint8_t out_priv[32],
    uint8_t out_chain[32])
{
    uint8_t k0[32],c0[32],k1[32],c1[32],k2[32],c2[32],k3[32],c3[32];

    // m/44'
    bip32_derive_child_hardened(master_priv,  master_chain,  0x8000002C, k0, c0);
    // m/44'/coin'
    bip32_derive_child_hardened(k0, c0, 0x80000000|coin_type, k1, c1);
    // m/44'/coin'/0'
    bip32_derive_child_hardened(k1, c1, 0x80000000, k2, c2);
    // m/44'/coin'/0'/0
    bip32_derive_child_normal(k2, c2, 0, k3, c3);
    // m/44'/coin'/0'/0/0
    bip32_derive_child_normal(k3, c3, 0, out_priv, out_chain);
}

// ============================================================================
// HELPER DEVICE : décode global_idx → word_indices[]
// Commun aux deux kernels
// ============================================================================
__device__ __forceinline__ void decode_word_indices(
    const SeedMask* __restrict__ mask,
    uint64_t global_idx,
    uint16_t word_indices[SEED_MAX_WORDS])
{
    for(int w = 0; w < mask->num_words; w++)
        word_indices[w] = mask->word_indices[w];

    uint64_t idx = global_idx;
    if(mask->last_word_unknown){
        // Dernier inconnu : encodé sur 2^(11-cs_bits) valeurs (bits entropy seulement)
        const uint32_t ent_vals = 1u << (11u - mask->checksum_bits);
        uint16_t ent_part = (uint16_t)(idx % ent_vals);
        // Stocker les bits entropy dans les bits hauts (cs_bits bas = 0, seront forcés)
        word_indices[mask->unknown_pos[mask->num_unknown - 1]] =
            (uint16_t)(ent_part << mask->checksum_bits);
        idx /= ent_vals;
        // Les N-1 autres inconnus : base 2048 normale
        for(int x = mask->num_unknown - 2; x >= 0; x--){
            word_indices[mask->unknown_pos[x]] = (uint16_t)(idx % 2048);
            idx /= 2048;
        }
    } else {
        for(int x = mask->num_unknown - 1; x >= 0; x--){
            word_indices[mask->unknown_pos[x]] = (uint16_t)(idx % 2048);
            idx /= 2048;
        }
    }
}

// ============================================================================
// HELPER DEVICE : reconstruit les bytes d'entropy depuis word_indices
// ============================================================================
__device__ __forceinline__ void build_entropy(
    const uint16_t word_indices[SEED_MAX_WORDS],
    int num_words, int ent_bits,
    uint8_t entropy[32])
{
    for(int i = 0; i < 32; i++) entropy[i] = 0;
    for(int w = 0; w < num_words; w++){
        uint16_t wi = word_indices[w];
        for(int b = 0; b < 11; b++){
            int bit_pos = w * 11 + b;
            if(bit_pos >= ent_bits) break;
            uint8_t bit = (wi >> (10 - b)) & 1;
            entropy[bit_pos / 8] |= (bit << (7 - (bit_pos % 8)));
        }
    }
}

// ============================================================================
// KERNEL 1 — FILTRE CHECKSUM BIP39
// Ultra-léger (~20 registres), haute occupancy, élimine 15/16 ou 255/256
// des candidats AVANT le PBKDF2.
//
// Cas A (last_word_unknown = true) :
//   Le dernier mot X est déduit du checksum → fixé directement, toujours valide.
//   Espace réduit à 2048^(N-1) × 2^(11-cs_bits).
//
// Cas B (last_word_unknown = false) :
//   Le dernier mot est connu → son checksum est dans required_checksum.
//   On vérifie que les mots inconnus produisent ce checksum → 1/16 ou 1/256 passent.
// ============================================================================
__global__ void hydra_checksum_kernel(
    const SeedMask*  __restrict__ mask,
    uint64_t         wave_offset,
    int              wave_size,
    uint64_t* __restrict__ valid_indices,
    int*      __restrict__ valid_count,
    int              valid_capacity)
{
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if(tid >= wave_size) return;

    uint64_t global_idx = wave_offset + (uint64_t)tid;
    if(global_idx >= mask->total_candidates) return;

    uint16_t word_indices[SEED_MAX_WORDS];
    decode_word_indices(mask, global_idx, word_indices);

    const int ent_bits  = (int)mask->checksum_bits * 32; // 4*32=128 ou 8*32=256
    const int ent_bytes = ent_bits / 8;
    const uint8_t cs_mask = (uint8_t)((1u << mask->checksum_bits) - 1u);

    if(mask->last_word_unknown){
        // Cas A : calculer checksum depuis les N-1 mots connus/fixés,
        // puis forcer le dernier mot à la valeur correcte.
        // Les 11-cs_bits bits de poids fort du dernier mot sont déjà décodés
        // (les bits entropy), seuls les cs_bits de checksum sont fixés.
        uint8_t entropy[32];
        // Entropy : bits des mots 0..num_words-2 + bits entropy du dernier mot
        // Le dernier mot X a déjà ses bits entropy décodés dans decode_word_indices
        // mais son checksum est incorrect → on le recalcule.
        build_entropy(word_indices, mask->num_words, ent_bits, entropy);
        uint8_t csum = bip39_checksum(entropy, ent_bytes);
        // Forcer les cs_bits bas du dernier mot au checksum calculé
        uint16_t last = word_indices[mask->num_words - 1];
        word_indices[mask->num_words - 1] = (last & ~(uint16_t)cs_mask) | (uint16_t)(csum & cs_mask);
        // Ce candidat est toujours valide → pas de filtre, on passe directement
        // (mais on doit sauver l'index avec le dernier mot corrigé — donc on encode)
        // Pour transmettre le dernier mot fixé, on encode l'index corrigé :
        // Recalcule global_idx avec le bon dernier mot
        // idx = (global_idx / base_last) * 2048 + word_indices[last_pos]
        // Plus simple : stocker les word_indices directement serait idéal,
        // mais pour rester compatible avec le kernel2 qui décode depuis l'index,
        // on reconstruit l'index corrigé.
        {
            // Réencoder : le dernier inconnu est en base ent_vals = 2^(11-cs_bits)
            const uint32_t ent_vals = 1u << (11u - mask->checksum_bits);
            // ent_part = bits entropy du mot final (après forçage checksum)
            uint16_t ent_part = word_indices[mask->num_words - 1] >> mask->checksum_bits;
            uint64_t corrected = (global_idx / (uint64_t)ent_vals) * (uint64_t)ent_vals
                               + (uint64_t)ent_part;
            int pos = atomicAdd(valid_count, 1);
            if (pos < valid_capacity) valid_indices[pos] = corrected;
        }
    } else {
        // Cas B : dernier mot connu → vérifier que l'entropy produit le bon checksum
        uint8_t entropy[32];
        build_entropy(word_indices, mask->num_words, ent_bits, entropy);
        uint8_t csum = bip39_checksum(entropy, ent_bytes);
        if((csum & cs_mask) != mask->required_checksum) return; // 15/16 ou 255/256 eliminés
        int pos = atomicAdd(valid_count, 1);
        if (pos < valid_capacity) valid_indices[pos] = global_idx;
    }
}

// ============================================================================
// ============================================================================
// KERNEL 2 — PBKDF2 + BIP44 + ECC + COMPARE
// Reçoit uniquement les indices validés par le kernel checksum.
// ============================================================================

// ============================================================================
// CONSTANTES DEVICE — MODE PASSPHRASE
// H_ipad et H_opad calculés depuis la mnemonic fixe, partagés par tous les threads
// Initialisés par un kernel setup avant le batch principal
// ============================================================================
__device__ uint64_t d_pass_H_ipad[8];
__device__ uint64_t d_pass_H_opad[8];

// ============================================================================
// KERNEL SETUP PASSPHRASE — calcule H_ipad/H_opad depuis la mnemonic
// Lancé avec 1 seul thread avant le batch principal
// ============================================================================
__global__ void hydra_passphrase_setup(
    const uint8_t* __restrict__ d_mnemonic,
    uint32_t                    mnemonic_len)
{
    if(threadIdx.x != 0 || blockIdx.x != 0) return;
    pbkdf2_compute_key_states(d_mnemonic, mnemonic_len,
                              d_pass_H_ipad, d_pass_H_opad);
}

// ============================================================================
// KERNEL 2a PASSPHRASE — PBKDF2 avec salt dynamique (mnemonic fixe + passphrase)
//
// Input  : d_passphrases[batch × MAX_PASS_LEN], d_pass_lens[batch]
// Output : d_seeds[batch × 64]
// H_ipad/H_opad lus depuis d_pass_H_ipad/H_opad (device global, initialisés par setup)
// ============================================================================
#define MAX_PASS_LEN 96   // ≤ 96 bytes → salt+counter tient en 1 bloc SHA512

__global__ __launch_bounds__(256, 2)
void hydra_k2a_passphrase(
    const uint8_t*  __restrict__ d_chunk,
    const uint32_t* __restrict__ d_offsets,
    int             batch_size,
    const GpuRule*  __restrict__ d_rules,
    int             num_rules,
    uint8_t*        __restrict__ d_rejected,
    uint8_t*        __restrict__ d_seeds)         // [batch * max(1, num_rules) × 64]
{
    // Use a flat 1D grid to avoid the 65535 limit on gridDim.y
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_items = batch_size * (num_rules > 0 ? num_rules : 1);
    if (out_idx >= total_items) return;

    const int rule_idx = (num_rules > 0) ? (out_idx % num_rules) : 0;
    const int word_idx = (num_rules > 0) ? (out_idx / num_rules) : out_idx;

    // Charger H_ipad/H_opad depuis device global (précalculés une fois)
    uint64_t H_ipad[8], H_opad[8];
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        H_ipad[i] = d_pass_H_ipad[i];
        H_opad[i] = d_pass_H_opad[i];
    }

    const uint32_t offset = d_offsets[word_idx];
    uint32_t pass_len = (uint32_t)d_chunk[offset];
    const uint8_t* passphrase = d_chunk + offset + 1;

    __shared__ uint8_t s_pass[256 * MAX_PASS_LEN]; // Assuming max 256 threads per block
    uint8_t* local_pass = &s_pass[threadIdx.x * MAX_PASS_LEN];

    if (num_rules > 0) {
        bool valid = apply_rule(passphrase, pass_len, local_pass, pass_len, &d_rules[rule_idx], MAX_PASS_LEN);
        d_rejected[out_idx] = valid ? 0 : 1;
        if (!valid) {
            uint8_t* dst = d_seeds + (size_t)out_idx * 64;
            #pragma unroll
            for (int b = 0; b < 64; b++) dst[b] = 0;
            return;
        }
        passphrase = local_pass;
    } else {
        d_rejected[out_idx] = 0;
    }

    // Itération 1 : H_ipad/H_opad + salt("mnemonic" + passphrase + counter)
    uint64_t U_first[8];
    pbkdf2_first_iter_pass(H_ipad, H_opad, passphrase, pass_len, U_first);

    // Itérations 2..2048 : bypass pur (inchangé)
    uint8_t seed[64];
    pbkdf2_iterate(H_ipad, H_opad, U_first, seed);

    // Stocker seed
    uint8_t* dst = d_seeds + (size_t)out_idx * 64;
    #pragma unroll
    for(int b = 0; b < 64; b++) dst[b] = seed[b];
}

// ============================================================================
// KERNEL 2a — PBKDF2 (word_indices → seed[64])
// Objectif : ~106 reg, 0 spill
// ============================================================================
__global__ __launch_bounds__(256, 2)
void hydra_k2a_pbkdf2(
    const SeedMask*  __restrict__ mask,
    const uint64_t*  __restrict__ valid_indices,
    int              valid_count,
    uint8_t*         __restrict__ d_seeds)   // [valid_count × 64]
{
    const int tid    = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(blockDim.x * gridDim.x);

    for(int i = tid; i < valid_count; i += stride)
    {
        uint64_t global_idx = valid_indices[i];

        uint16_t word_indices[SEED_MAX_WORDS];
        decode_word_indices(mask, global_idx, word_indices);

        if(mask->last_word_unknown){
            const int ent_bits  = (int)mask->checksum_bits * 32;
            const int ent_bytes = ent_bits / 8;
            const uint8_t cs_mask_v = (uint8_t)((1u << mask->checksum_bits) - 1u);
            uint8_t entropy[32];
            build_entropy(word_indices, mask->num_words, ent_bits, entropy);
            uint8_t csum = bip39_checksum(entropy, ent_bytes);
            uint16_t last = word_indices[mask->num_words - 1];
            word_indices[mask->num_words - 1] = (last & ~(uint16_t)cs_mask_v)
                                               | (uint16_t)(csum & cs_mask_v);
        }

        // Buffer mnemonic : 9 bytes/mot max, 12-24 mots → 220 bytes max
        uint8_t mnemonic[220];
        uint32_t mlen = build_mnemonic(word_indices, mask->num_words, mnemonic);

        // PBKDF2 via pbkdf2_setup + pbkdf2_iterate (bypass optimisé)
        const uint8_t salt[8] = {0x6d,0x6e,0x65,0x6d,0x6f,0x6e,0x69,0x63};
        uint64_t H_ipad[8], H_opad[8], U_first[8];
        pbkdf2_setup(mnemonic, mlen, salt, 8, H_ipad, H_opad, U_first);

        uint8_t seed[64];
        pbkdf2_iterate(H_ipad, H_opad, U_first, seed);

        uint8_t* dst = d_seeds + (size_t)i * 64;
        #pragma unroll
        for(int b = 0; b < 64; b++) dst[b] = seed[b];
    }
}

// ============================================================================
// KERNEL 2b — BIP32 master + 3 niveaux hardened SEULEMENT (SANS ECC)
// m/44'/coin'/0'  — aucun scalarMul ici
// Objectif : ~80-90 reg, ~400 bytes stack, 0 spill
// Output : d_intermed[i*64..+63] = priv[32] || chain[32] après m/44'/coin'/0'
// ============================================================================
__global__ __launch_bounds__(256, 3)
void hydra_k2b_hardened(
    uint32_t          coin_type,
    const uint8_t*    __restrict__ d_seeds,     // [valid_count × 64]
    uint8_t*          __restrict__ d_intermed,  // [valid_count × 64]
    int               valid_count)
{
    const int tid    = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(blockDim.x * gridDim.x);

    for(int i = tid; i < valid_count; i += stride)
    {
        const uint8_t* seed = d_seeds + (size_t)i * 64;

        // BIP32 master key : HMAC-SHA512("Bitcoin seed", seed)
        // Clé fixe → H_ipad/H_opad précalculés en constant memory
        uint8_t master_node[64];
        hmac_sha512_bip32_master(seed, master_node);

        uint8_t k0[32], c0[32];
        for(int b=0;b<32;b++) k0[b] = master_node[b];
        for(int b=0;b<32;b++) c0[b] = master_node[32+b];

        // m/44'  (hardened)
        uint8_t k1[32], c1[32];
        bip32_derive_child_hardened(k0, c0, 0x8000002C, k1, c1);

        // m/44'/coin'  (hardened)
        uint8_t k2[32], c2[32];
        bip32_derive_child_hardened(k1, c1, 0x80000000u | coin_type, k2, c2);

        // m/44'/coin'/0'  (hardened)
        uint8_t k3[32], c3[32];
        bip32_derive_child_hardened(k2, c2, 0x80000000u, k3, c3);

        // Écriture priv[32] || chain[32]
        uint8_t* dst = d_intermed + (size_t)i * 64;
        #pragma unroll
        for(int b=0;b<32;b++) dst[b]    = k3[b];
        #pragma unroll
        for(int b=0;b<32;b++) dst[32+b] = c3[b];
    }
}

// ============================================================================
// KERNEL 2c — 2 niveaux normaux (ECC×2) + ECC final + Hash + compare/Bloom
// Input : d_intermed = priv[32] || chain[32] après m/44'/coin'/0'
// Effectue : m/44'/coin'/0'/0  → normal(ECC)
//            m/44'/coin'/0'/0/0 → normal(ECC)
//            ECC final leaf_priv → pubkey → Hash160/Keccak → compare
// Objectif : ~166 reg (mesuré), ~300 bytes stack, 0 spill
// ============================================================================

// ============================================================================

// ============================================================================
// ============================================================================
// MODE ELECTRUM V2
// Différences vs BIP39 :
//   Checksum : HMAC-SHA512("Seed version", mnemonic) doit commencer par "01"
//              (standard P2PKH) ou "100" (segwit P2WPKH)
//   PBKDF2   : salt = "electrum" (pas "mnemonic")
//   Dérivation: m/0/0 (normal, non hardened — BIP32 direct)
// ============================================================================

// États HMAC pour la clé fixe "Seed version" (12 bytes)
// Calculés une fois par hydra_electrum_setup avant le batch
__device__ uint64_t d_electrum_H_ipad[8];
__device__ uint64_t d_electrum_H_opad[8];

// ============================================================================
// SETUP KERNEL — calcule H_ipad/H_opad depuis "Seed version"
// Lancé avec 1 seul thread avant le batch principal
// ============================================================================
__global__ void hydra_electrum_setup()
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // "Seed version" = 12 bytes
    const uint8_t key[12] = {0x53,0x65,0x65,0x64,0x20,0x76,0x65,0x72,0x73,0x69,0x6f,0x6e};
    pbkdf2_hmac_key_states_bip39(key, 12, d_electrum_H_ipad, d_electrum_H_opad);
}

// ============================================================================
// Helper : inner SHA512 depuis un état H_base (après 128 bytes absorbés),
// message de longueur variable (peut dépasser 111 bytes pour 18+ mots)
// ============================================================================
__device__ __forceinline__ void electrum_sha512_inner(
    const uint64_t H_base[8],
    const uint8_t* msg, uint32_t msg_len,
    uint64_t H_out[8])
{
    if (msg_len <= 111) {
        sha512_compress_state_variable(H_base, msg, msg_len, H_out);
    } else {
        // Phrase longue (18+ mots) : utilise le contexte général
        Hydra_SHA512_CTX ctx;
        #pragma unroll
        for (int i = 0; i < 8; i++) ctx.h[i] = H_base[i];
        ctx.buffer_len = 0;
        ctx.total_len  = 128;   // H_base représente 128 bytes déjà absorbés
        sha512_update(&ctx, msg, msg_len);
        uint8_t tmp[64];
        sha512_final(&ctx, tmp);
        #pragma unroll
        for (int i = 0; i < 8; i++) H_out[i] = sha512_load_be64(tmp + i*8);
    }
}

// ============================================================================
// KERNEL FILTRE ELECTRUM — remplace hydra_checksum_kernel
// Calcule HMAC-SHA512("Seed version", mnemonic) et vérifie le préfixe :
//   mode 1 : "01"  → premier byte == 0x01
//   mode 2 : "100" → premier byte == 0x10 && nibble haut du deuxième == 0x0
// Filtre ~255/256 (mode 1) ou ~4095/4096 (mode 2) des candidats
// ============================================================================
__global__ __launch_bounds__(256, 1)
void hydra_electrum_filter(
    const SeedMask*  __restrict__ mask,
    uint64_t         wave_offset,
    int              wave_size,
    uint64_t* __restrict__ valid_indices,
    int*      __restrict__ valid_count,
    int              valid_capacity)
{
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= wave_size) return;

    uint64_t global_idx = wave_offset + (uint64_t)tid;
    if (global_idx >= mask->total_candidates) return;

    uint16_t word_indices[SEED_MAX_WORDS];
    decode_word_indices(mask, global_idx, word_indices);

    // Mnemonic string (max 220 bytes pour 24 mots)
    uint8_t mnemonic[220];
    uint32_t mlen = build_mnemonic(word_indices, mask->num_words, mnemonic);

    // HMAC-SHA512("Seed version", mnemonic)
    uint64_t H_ipad[8], H_opad[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) H_ipad[i] = d_electrum_H_ipad[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) H_opad[i] = d_electrum_H_opad[i];

    // Inner = SHA512(H_ipad || mnemonic)
    uint64_t H_inner[8];
    electrum_sha512_inner(H_ipad, mnemonic, mlen, H_inner);

    // Outer = SHA512(H_opad || inner[64])
    uint64_t H_result[8];
    sha512_compress_state(H_opad, H_inner, H_result);

    // Les bytes du résultat en big-endian : H_result[0] >> 56 = byte 0
    const uint8_t b0 = (uint8_t)(H_result[0] >> 56);
    const uint8_t b1 = (uint8_t)(H_result[0] >> 48);

    bool valid;
    if (mask->electrum_mode == 2) {
        // Préfixe hex "100" : byte0=0x10, nibble haut byte1=0x0
        valid = (b0 == 0x10) && ((b1 & 0xF0) == 0x00);
    } else {
        // Préfixe hex "01"  : byte0=0x01
        valid = (b0 == 0x01);
    }

    if (valid) {
        int pos = atomicAdd(valid_count, 1);
        if (pos < valid_capacity) valid_indices[pos] = global_idx;
    }
}

// ============================================================================
// KERNEL K2a ELECTRUM — PBKDF2 avec salt "electrum" (au lieu de "mnemonic")
// Identique à hydra_k2a_pbkdf2 sauf le sel
// ============================================================================
__global__ __launch_bounds__(256, 2)
void hydra_electrum_k2a_pbkdf2(
    const SeedMask*  __restrict__ mask,
    const uint64_t*  __restrict__ valid_indices,
    int              valid_count,
    uint8_t*         __restrict__ d_seeds)  // [valid_count × 64]
{
    const int tid    = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(blockDim.x * gridDim.x);

    for (int i = tid; i < valid_count; i += stride)
    {
        uint64_t global_idx = valid_indices[i];

        uint16_t word_indices[SEED_MAX_WORDS];
        decode_word_indices(mask, global_idx, word_indices);

        // Mnemonic string
        uint8_t mnemonic[220];
        uint32_t mlen = build_mnemonic(word_indices, mask->num_words, mnemonic);

        // PBKDF2-HMAC-SHA512 avec salt "electrum"
        const uint8_t salt[8] = {0x65,0x6c,0x65,0x63,0x74,0x72,0x75,0x6d};  // "electrum"
        uint64_t H_ipad[8], H_opad[8], U_first[8];
        pbkdf2_setup(mnemonic, mlen, salt, 8, H_ipad, H_opad, U_first);

        uint8_t seed[64];
        pbkdf2_iterate(H_ipad, H_opad, U_first, seed);

        uint8_t* dst = d_seeds + (size_t)i * 64;
        #pragma unroll
        for (int b = 0; b < 64; b++) dst[b] = seed[b];
    }
}

// ============================================================================
// KERNEL K2b ELECTRUM — BIP32 master key uniquement (pas de hardened)
// Input  : d_seeds[valid_count × 64]
// Output : d_intermed[valid_count × 64] = master_priv[32] || master_chain[32]
// Le kernel K2c BIP44 réutilisé ensuite effectue m/0/0 (2 derivations normales)
// ============================================================================
__global__ __launch_bounds__(256, 3)
void hydra_electrum_k2b(
    const uint8_t* __restrict__ d_seeds,     // [valid_count × 64]
    uint8_t*       __restrict__ d_intermed,  // [valid_count × 64]
    int            valid_count)
{
    const int tid    = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(blockDim.x * gridDim.x);

    for (int i = tid; i < valid_count; i += stride)
    {
        const uint8_t* seed = d_seeds    + (size_t)i * 64;
        uint8_t*       dst  = d_intermed + (size_t)i * 64;
        // Calcule directement la master key BIP32 → sortie pour K2c
        hmac_sha512_bip32_master(seed, dst);
    }
}
