#pragma once
/*
 * ======================================================================================
 * HYDRA - Wif.cuh  (Mode WIF : brute force caractères # inconnus)
 * ======================================================================================
 *
 * WIF compressé   = 52 caractères Base58Check
 * Décodé          = 38 bytes : [0x80][clé privée 32B][0x01][checksum 4B]
 * WIF uncompressed= 51 caractères Base58Check
 * Décodé          = 37 bytes : [0x80][clé privée 32B][checksum 4B]
 *
 * Pipeline :
 *   K1 : décode Base58 → SHA256×2 → vérifie checksum 4 bytes → filtre ÷2^32
 *   K2 : extrait key[32] → ECC → Hash160 → compare
 *
 * Optimisation Base58 (v2) :
 *   Le décodage Base58 classique coûte 52×38 multiply-add même si seules
 *   N positions sont inconnues. On précalcule sur CPU :
 *     - base_bytes[38] : décodage avec toutes les inconnues = 0
 *     - weights[N][38] : contribution de chaque position inconnue pour val=1
 *   Sur GPU : wif_bytes = base + Σ(val[i] × weight[i])
 *   Coût réduit : N×38 multiply-add au lieu de 52×38 → gain ~10× pour N=5.
 *
 * ======================================================================================
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include "HydraCommon.h"
#include "Bloom.h"
#include "ECC.h"
#include "Hash.cuh"

// ============================================================================
// CONSTANTES
// ============================================================================
#define WIF_COMPRESSED_LEN      52
#define WIF_UNCOMPRESSED_LEN    51
#define WIF_MAX_LEN             52
#define WIF_COMPRESSED_BYTES    38
#define WIF_UNCOMPRESSED_BYTES  37
#define WIF_MAX_BYTES           38
#define WIF_KEY_OFFSET           1
#define WIF_MAX_UNKN            16

// Alphabet Base58 : index 0='1', ..., 57='z'
__constant__ uint8_t B58_VAL[128] = {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,  // 0x00-0x0F
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,  // 0x10-0x1F
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,  // 0x20-0x2F
    255,  0,  1,  2,  3,  4,  5,  6,  7,  8,255,255,255,255,255,255, // 0x30-0x3F
    255,  9, 10, 11, 12, 13, 14, 15, 16,255, 17, 18, 19, 20, 21,255, // 0x40-0x4F
     22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,255,255,255,255,255, // 0x50-0x5F
    255, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,255, 44, 45, 46, // 0x60-0x6F
     47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,255,255,255,255,255  // 0x70-0x7F
};

// ============================================================================
// CONSTANT MEMORY — Précomputation Base58
// Uploadé depuis CPU via precompute_wif_b58() + cudaMemcpyToSymbol
// ============================================================================
__constant__ uint8_t c_wif_base[WIF_MAX_BYTES];                    // max 38 bytes
__constant__ uint8_t c_wif_weights[WIF_MAX_UNKN][WIF_MAX_BYTES];   // 16x38 = 608 bytes
// Total : 646 bytes, negligible.

// ============================================================================
// STRUCT WifMask
// ============================================================================
struct WifMask {
    uint8_t  known_b58[WIF_MAX_LEN];    // valeur Base58 (0-57), 0xFF = inconnu
    uint8_t  num_chars;                  // 52 compressed, 51 uncompressed
    uint8_t  num_unknown;                // nombre de #
    uint8_t  decoded_bytes;              // 38 compressed, 37 uncompressed
    uint8_t  payload_len;                // 34 compressed, 33 uncompressed
    uint8_t  checksum_offset;            // payload_len
    uint8_t  is_compressed;              // 1 compressed, 0 uncompressed
    uint8_t  unknown_pos[WIF_MAX_UNKN];  // positions des # dans known_b58
    uint8_t  _pad[5];
    uint64_t total_candidates;           // 58^num_unknown
};

// ============================================================================
// CPU : précompute base_bytes et weights, upload en constant memory
//
// base_bytes[38] = décodage B58 avec toutes les inconnues = 0
// weights[x][38] = contribution de unknown_pos[x] pour val = 1
//
// Appeler APRÈS parse_wif_mask(), AVANT le lancement des kernels.
// ============================================================================
static void precompute_wif_b58(const WifMask& mask)
{
    // ── 1. Base : décoder avec toutes les inconnues = 0 ──────────────────────
    uint8_t b58_base[WIF_MAX_LEN] = {};
    for(int i = 0; i < mask.num_chars; i++)
        b58_base[i] = (mask.known_b58[i] == 0xFF) ? 0 : mask.known_b58[i];

    uint32_t tmp[WIF_MAX_BYTES] = {};
    for(int i = 0; i < mask.num_chars; i++){
        uint32_t carry = b58_base[i];
        for(int j = mask.decoded_bytes - 1; j >= 0; j--){
            carry += 58u * tmp[j];
            tmp[j]  = carry & 0xFF;
            carry >>= 8;
        }
    }
    uint8_t base_bytes[WIF_MAX_BYTES] = {};
    for(int i = 0; i < mask.decoded_bytes; i++) base_bytes[i] = (uint8_t)tmp[i];
    cudaMemcpyToSymbol(c_wif_base, base_bytes, WIF_MAX_BYTES);

    // ── 2. Poids : pour chaque position inconnue, décoder b58[pos]=1, reste=0 ──
    uint8_t weights[WIF_MAX_UNKN][WIF_MAX_BYTES] = {};
    for(int x = 0; x < mask.num_unknown; x++){
        // b58_one : tout à 0 sauf la position inconnue x = 1
        uint32_t tmp2[WIF_MAX_BYTES] = {};
        // Contribution directe de la position unknown_pos[x] avec valeur 1 :
        // poids = 58^(num_chars-1-pos)
        // On recalcule via le même algo de décodage avec val=1 à cette position
        uint8_t b58_one[WIF_MAX_LEN] = {};
        b58_one[mask.unknown_pos[x]] = 1;
        for(int i = 0; i < mask.num_chars; i++){
            uint32_t carry = b58_one[i];
            for(int j = mask.decoded_bytes - 1; j >= 0; j--){
                carry += 58u * tmp2[j];
                tmp2[j]  = carry & 0xFF;
                carry >>= 8;
            }
        }
        for(int i = 0; i < mask.decoded_bytes; i++) weights[x][i] = (uint8_t)tmp2[i];
    }
    cudaMemcpyToSymbol(c_wif_weights, weights, WIF_MAX_UNKN * WIF_MAX_BYTES);
}

// ============================================================================
// DEVICE : décodage Base58 optimisé
//
// wif_bytes = c_wif_base + Σ(val[x] × c_wif_weights[x])
// Coût : num_unknown × 38 multiply-add (vs 52×38 classique)
// ============================================================================
__device__ __forceinline__ void wif_decode_b58_fast(
    const WifMask* __restrict__ mask,
    uint64_t       global_idx,
    uint8_t        out[WIF_MAX_BYTES])
{
    // Récupérer les valeurs des positions inconnues depuis global_idx
    uint8_t vals[WIF_MAX_UNKN];
    uint64_t idx = global_idx;
    for(int x = (int)mask->num_unknown - 1; x >= 0; x--){
        vals[x] = (uint8_t)(idx % 58);
        idx    /= 58;
    }

    // Initialiser avec la base (positions connues précompilées)
    uint32_t acc[WIF_MAX_BYTES];
    #pragma unroll
    for(int i = 0; i < WIF_MAX_BYTES; i++) acc[i] = c_wif_base[i];

    // Ajouter la contribution de chaque position inconnue
    for(int x = 0; x < (int)mask->num_unknown; x++){
        const uint32_t v = vals[x];
        if(v == 0) continue;  // contribution nulle → skip
        uint32_t carry = 0;
        for(int j = mask->decoded_bytes - 1; j >= 0; j--){
            uint32_t s = acc[j] + c_wif_weights[x][j] * v + carry;
            acc[j]  = s & 0xFF;
            carry   = s >> 8;
        }
        // carry sortant est nul (la somme tient dans 38 bytes pour un WIF valide)
    }

    #pragma unroll
    for(int i = 0; i < WIF_MAX_BYTES; i++) out[i] = (uint8_t)acc[i];
}

template<bool COMPRESSED>
__device__ __forceinline__ void wif_decode_b58_fast_fixed(
    const WifMask* __restrict__ mask,
    uint64_t       global_idx,
    uint8_t        out[WIF_MAX_BYTES])
{
    constexpr int BYTES = COMPRESSED ? WIF_COMPRESSED_BYTES : WIF_UNCOMPRESSED_BYTES;

    // Récupérer les valeurs des positions inconnues depuis global_idx.
    uint8_t vals[WIF_MAX_UNKN];
    uint64_t idx = global_idx;
    for(int x = (int)mask->num_unknown - 1; x >= 0; x--){
        vals[x] = (uint8_t)(idx % 58);
        idx    /= 58;
    }

    // BYTES est une constante de compilation ici. Le chemin compressed retrouve
    // donc les mêmes bornes fixes que l'ancien WIF-only kernel.
    uint32_t acc[WIF_MAX_BYTES];
    #pragma unroll
    for(int i = 0; i < BYTES; i++) acc[i] = c_wif_base[i];

    for(int x = 0; x < (int)mask->num_unknown; x++){
        const uint32_t v = vals[x];
        if(v == 0) continue;
        uint32_t carry = 0;
        for(int j = BYTES - 1; j >= 0; j--){
            uint32_t s = acc[j] + c_wif_weights[x][j] * v + carry;
            acc[j]  = s & 0xFF;
            carry   = s >> 8;
        }
    }

    #pragma unroll
    for(int i = 0; i < BYTES; i++) out[i] = (uint8_t)acc[i];
}

// ============================================================================
// HELPER : SHA256 sur 34 bytes (payload WIF sans checksum)
// Format fixe : bytes[0..33] = 0x80 + key[32] + 0x01
// ============================================================================
__device__ __forceinline__ void sha256_34bytes(
    const uint8_t data[34],
    uint8_t       out[32])
{
    uint32_t W[16];
    #pragma unroll
    for(int i = 0; i < 8; i++){
        W[i] = ((uint32_t)data[i*4+0] << 24)
             | ((uint32_t)data[i*4+1] << 16)
             | ((uint32_t)data[i*4+2] <<  8)
             | ((uint32_t)data[i*4+3]);
    }
    W[8] = ((uint32_t)data[32] << 24)
         | ((uint32_t)data[33] << 16)
         | 0x00008000u;
    #pragma unroll
    for(int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 272u;  // 34 * 8

    uint32_t st[8];
    SHA256Initialize(st);
    SHA256Transform(st, W);

    #pragma unroll
    for(int i = 0; i < 8; i++){
        out[i*4+0] = (uint8_t)(st[i] >> 24);
        out[i*4+1] = (uint8_t)(st[i] >> 16);
        out[i*4+2] = (uint8_t)(st[i] >>  8);
        out[i*4+3] = (uint8_t)(st[i]);
    }
}

// ============================================================================
// HELPER : SHA256 sur 33 bytes (payload WIF uncompressed sans checksum)
// Format : bytes[0..32] = 0x80 + key[32]
// ============================================================================
__device__ __forceinline__ void sha256_33bytes(
    const uint8_t data[33],
    uint8_t       out[32])
{
    uint32_t W[16];
    #pragma unroll
    for(int i = 0; i < 8; i++){
        W[i] = ((uint32_t)data[i*4+0] << 24)
             | ((uint32_t)data[i*4+1] << 16)
             | ((uint32_t)data[i*4+2] <<  8)
             | ((uint32_t)data[i*4+3]);
    }
    W[8] = ((uint32_t)data[32] << 24) | 0x00800000u;
    #pragma unroll
    for(int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 264u;  // 33 * 8

    uint32_t st[8];
    SHA256Initialize(st);
    SHA256Transform(st, W);

    #pragma unroll
    for(int i = 0; i < 8; i++){
        out[i*4+0] = (uint8_t)(st[i] >> 24);
        out[i*4+1] = (uint8_t)(st[i] >> 16);
        out[i*4+2] = (uint8_t)(st[i] >>  8);
        out[i*4+3] = (uint8_t)(st[i]);
    }
}

// ============================================================================
// HELPER : SHA256 sur 32 bytes (second passage du double-SHA256)
// ============================================================================
__device__ __forceinline__ void sha256_32bytes(
    const uint8_t data[32],
    uint8_t       out[32])
{
    uint32_t W[16];
    #pragma unroll
    for(int i = 0; i < 8; i++){
        W[i] = ((uint32_t)data[i*4+0] << 24)
             | ((uint32_t)data[i*4+1] << 16)
             | ((uint32_t)data[i*4+2] <<  8)
             | ((uint32_t)data[i*4+3]);
    }
    W[8] = 0x80000000u;
    #pragma unroll
    for(int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 256u;

    uint32_t st[8];
    SHA256Initialize(st);
    SHA256Transform(st, W);

    #pragma unroll
    for(int i = 0; i < 8; i++){
        out[i*4+0] = (uint8_t)(st[i] >> 24);
        out[i*4+1] = (uint8_t)(st[i] >> 16);
        out[i*4+2] = (uint8_t)(st[i] >>  8);
        out[i*4+3] = (uint8_t)(st[i]);
    }
}

// ============================================================================
// HELPER : decode_wif_indices (conservé pour K2 — utilisé rarement)
// ============================================================================
__device__ __forceinline__ void decode_wif_indices(
    const WifMask* __restrict__ mask,
    uint64_t       global_idx,
    uint8_t        b58vals[WIF_MAX_LEN])
{
    #pragma unroll
    for(int i = 0; i < WIF_MAX_LEN; i++) b58vals[i] = mask->known_b58[i];
    uint64_t idx = global_idx;
    for(int x = (int)mask->num_unknown - 1; x >= 0; x--){
        b58vals[mask->unknown_pos[x]] = (uint8_t)(idx % 58);
        idx /= 58;
    }
}

// ============================================================================
// HELPER : décodage B58 classique (conservé pour K2)
// ============================================================================
__device__ __forceinline__ void wif_decode_b58(
    const WifMask* __restrict__ mask,
    const uint8_t b58vals[WIF_MAX_LEN],
    uint8_t       out[WIF_MAX_BYTES])
{
    uint32_t tmp[WIF_MAX_BYTES];
    #pragma unroll
    for(int i = 0; i < WIF_MAX_BYTES; i++) tmp[i] = 0;
    for(int i = 0; i < mask->num_chars; i++){
        uint32_t carry = b58vals[i];
        for(int j = mask->decoded_bytes - 1; j >= 0; j--){
            carry  += 58u * tmp[j];
            tmp[j]  = carry & 0xFF;
            carry >>= 8;
        }
    }
    #pragma unroll
    for(int i = 0; i < WIF_MAX_BYTES; i++) out[i] = (uint8_t)tmp[i];
}

// ============================================================================
// KERNEL 1 — FILTRE CHECKSUM WIF  (version optimisée)
//
// Utilise wif_decode_b58_fast() au lieu de wif_decode_b58()
// Coût décodage : N×38 multiply-add (vs 52×38 avant)
// Gain mesurable surtout pour N ≥ 6.
// ============================================================================
template<bool COMPRESSED>
__global__ void hydra_wif_checksum_kernel_t(
    const WifMask* __restrict__ mask,
    uint64_t       wave_offset,
    int            wave_size,
    uint64_t*      __restrict__ valid_indices,
    int*           __restrict__ valid_count,
    int            valid_capacity)
{
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if(tid >= wave_size) return;

    uint64_t global_idx = wave_offset + (uint64_t)tid;
    if(global_idx >= mask->total_candidates) return;

    // 1. Décodage Base58 optimisé (N×38 ops au lieu de 52×38)
    uint8_t wif_bytes[WIF_MAX_BYTES];
    wif_decode_b58_fast_fixed<COMPRESSED>(mask, global_idx, wif_bytes);

    // 2. Sanity check : byte[0] = 0x80, optional compressed marker byte[33] = 0x01
    if(wif_bytes[0] != 0x80u) return;
    if(COMPRESSED && wif_bytes[33] != 0x01u) return;

    // 3. SHA256(SHA256(payload)) → checksum
    uint8_t h1[32], h2[32];
    if(COMPRESSED)
        sha256_34bytes(wif_bytes, h1);
    else
        sha256_33bytes(wif_bytes, h1);
    sha256_32bytes(h1, h2);

    // 4. Comparer les 4 bytes de checksum (early exit sur premier mismatch)
    constexpr int co = COMPRESSED ? 34 : 33;
    if(h2[0] != wif_bytes[co + 0]) return;
    if(h2[1] != wif_bytes[co + 1]) return;
    if(h2[2] != wif_bytes[co + 2]) return;
    if(h2[3] != wif_bytes[co + 3]) return;

    // 5. Checksum valide → enregistrer
    int pos = atomicAdd(valid_count, 1);
    if (pos < valid_capacity) valid_indices[pos] = global_idx;
}

// ============================================================================
// KERNEL 2 — ECC + HASH160 + COMPARE
// Reçoit uniquement les indices validés par K1 (quasi aucun en pratique)
// Utilise le décodage classique (pas besoin d'optimiser ce chemin rare)
// ============================================================================
__global__ __launch_bounds__(256, 1)
void hydra_wif_ecc_kernel(
    const WifMask*    __restrict__ mask,
    const TargetData* __restrict__ target,
    HydraResult*      __restrict__ result,
    const uint64_t*   __restrict__ valid_indices,
    int               valid_count)
{
    const int tid    = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(blockDim.x * gridDim.x);

    for(int i = tid; i < valid_count; i += stride)
    {
        if(atomicAdd(&result->found, 0) != 0) return;

        uint64_t global_idx = valid_indices[i];

        // Décodage classique (K2 est appelé ~0 fois par run, pas critique)
        uint8_t b58vals[WIF_MAX_LEN];
        decode_wif_indices(mask, global_idx, b58vals);
        uint8_t wif_bytes[WIF_MAX_BYTES];
        wif_decode_b58(mask, b58vals, wif_bytes);

        // Extraire la clé privée (bytes 1..32)
        uint64_t priv_limbs[4];
        #pragma unroll
        for(int j = 0; j < 4; j++){
            const uint8_t* p = wif_bytes + 1 + (3 - j) * 8;
            priv_limbs[j] = ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48)
                          | ((uint64_t)p[2] << 40) | ((uint64_t)p[3] << 32)
                          | ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16)
                          | ((uint64_t)p[6] <<  8) | ((uint64_t)p[7]);
        }

        // ECC : clé privée → clé publique
        uint64_t ax[4], ay[4];
        scalarMulBaseAffine(priv_limbs, ax, ay);
        fieldNormalize(ax); fieldNormalize(ay);

        if(target->type == TargetType::BTC_PUBKEY){
            if(ax[0] == target->pubkey_x[0] &&
               ax[1] == target->pubkey_x[1] &&
               ax[2] == target->pubkey_x[2] &&
               ax[3] == target->pubkey_x[3] &&
               (uint8_t)(ay[0] & 1ULL) == target->pubkey_y_parity){
                if(atomicCAS(&result->found, 0, 1) == 0)
                    result->index = global_idx;
                return;
            }
            continue;
        }

        uint8_t h160[20];
        if(mask->is_compressed){
            const uint8_t par = (ay[0] & 1) ? 0x03 : 0x02;
            getHash160_33_from_limbs(par, ax, h160);
        } else {
            uint8_t pub65[65];
            pub65[0] = 0x04;
            #pragma unroll
            for(int li = 0; li < 4; li++){
                #pragma unroll
                for(int b = 0; b < 8; b++){
                    pub65[1 + (3 - li) * 8 + (7 - b)] = (uint8_t)(ax[li] >> (b * 8));
                    pub65[33 + (3 - li) * 8 + (7 - b)] = (uint8_t)(ay[li] >> (b * 8));
                }
            }
            getHash160_65bytes(pub65, h160);
        }

        bool _hit = false;
        if(is_any_bloom(target)){
            if(bloom_want_btc(target))
                _hit = bloom_check(h160, target->d_bloom_filter, target->bloom_m_bits);
            if(!_hit && bloom_want_eth(target)){
                uint8_t eth20[20];
                getEthAddr_from_limbs(ax, ay, eth20);
                _hit = bloom_check(eth20, target->d_bloom_filter, target->bloom_m_bits);
            }
        } else {
            _hit = hash20_matches(h160, target->hash20);
        }

        if(_hit){
            if(atomicCAS(&result->found, 0, 1) == 0)
                result->index = global_idx;
            return;
        }
    }
}
