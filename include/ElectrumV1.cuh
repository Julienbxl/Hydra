#pragma once

#include <stdint.h>
#include "HydraCommon.h"
#include "Bloom.h"
#include "Hash.cuh"
#include "ECC.h"

#define ELECTRUM_V1_WORDS        12
#define ELECTRUM_V1_MAX_X         6
#define ELECTRUM_V1_WORD_COUNT 1626
#define ELECTRUM_V1_LOOKAHEAD    20
#define ELECTRUM_V1_THREADS     128

struct ElectrumV1Mask {
    uint16_t word_indices[ELECTRUM_V1_WORDS];
    uint8_t  num_unknown;
    uint8_t  unknown_pos[ELECTRUM_V1_MAX_X];
    uint8_t  pad[1];
    uint64_t total_candidates;
    uint32_t lookahead;
    uint32_t single_path;
    uint32_t path_change;
    uint32_t path_index;
};

struct ElectrumV1Result {
    int      found;
    uint32_t change;
    uint32_t address_index;
    uint32_t pad;
    uint64_t index;
};

__device__ __forceinline__ uint8_t ev1_hex_digit(uint8_t x) {
    return (x < 10) ? (uint8_t)('0' + x) : (uint8_t)('a' + (x - 10));
}

__device__ __forceinline__ void ev1_u32_to_hex_be(uint32_t v, uint8_t* out) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint8_t b = (uint8_t)(v >> (24 - i * 8));
        out[i * 2 + 0] = ev1_hex_digit((uint8_t)(b >> 4));
        out[i * 2 + 1] = ev1_hex_digit((uint8_t)(b & 15));
    }
}

__device__ __forceinline__ void ev1_candidate_words(
    const ElectrumV1Mask* mask, uint64_t candidate, uint16_t words[ELECTRUM_V1_WORDS])
{
    #pragma unroll
    for (int i = 0; i < ELECTRUM_V1_WORDS; ++i) words[i] = mask->word_indices[i];
    for (int x = (int)mask->num_unknown - 1; x >= 0; --x) {
        words[mask->unknown_pos[x]] = (uint16_t)(candidate % ELECTRUM_V1_WORD_COUNT);
        candidate /= ELECTRUM_V1_WORD_COUNT;
    }
}

__device__ __forceinline__ void ev1_decode_seed_ascii(
    const uint16_t words[ELECTRUM_V1_WORDS], uint8_t seed_ascii[32])
{
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        const uint32_t w1 = words[group * 3 + 0];
        const uint32_t w2 = words[group * 3 + 1];
        const uint32_t w3 = words[group * 3 + 2];
        const uint32_t n  = ELECTRUM_V1_WORD_COUNT;
        const uint32_t d2 = (w2 + n - w1) % n;
        const uint32_t d3 = (w3 + n - w2) % n;
        const uint32_t chunk = w1 + n * d2 + n * n * d3;
        ev1_u32_to_hex_be(chunk, seed_ascii + group * 8);
    }
}

__device__ __forceinline__ void ev1_sha256_len64(const uint8_t msg[64], uint8_t out[32]) {
    uint32_t M[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        M[i] = ((uint32_t)msg[i*4+0] << 24) | ((uint32_t)msg[i*4+1] << 16)
             | ((uint32_t)msg[i*4+2] << 8)  |  (uint32_t)msg[i*4+3];
    }
    SHA256_64bytes_words_to_bytes(M, out);
}

__device__ __forceinline__ void ev1_stretch_seed(
    const uint8_t seed_ascii[32], uint8_t master_priv[32])
{
    uint32_t seed_words[8], x_words[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        seed_words[i] = ((uint32_t)seed_ascii[i * 4 + 0] << 24) |
                        ((uint32_t)seed_ascii[i * 4 + 1] << 16) |
                        ((uint32_t)seed_ascii[i * 4 + 2] << 8) |
                        (uint32_t)seed_ascii[i * 4 + 3];
        x_words[i] = seed_words[i];
    }
    #pragma unroll 1
    for (int round = 0; round < 100000; ++round) {
        uint32_t M[16], next_words[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            M[i] = x_words[i];
            M[8 + i] = seed_words[i];
        }
        SHA256_64bytes_words(M, next_words);
        #pragma unroll
        for (int i = 0; i < 8; ++i) x_words[i] = next_words[i];
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        master_priv[i * 4 + 0] = (uint8_t)(x_words[i] >> 24);
        master_priv[i * 4 + 1] = (uint8_t)(x_words[i] >> 16);
        master_priv[i * 4 + 2] = (uint8_t)(x_words[i] >> 8);
        master_priv[i * 4 + 3] = (uint8_t)(x_words[i]);
    }
}

__device__ __forceinline__ void ev1_be32_to_limbs_le(const uint8_t be[32], uint64_t le[4]) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint8_t* b = be + j * 8;
        le[3-j] = ((uint64_t)b[0]<<56) | ((uint64_t)b[1]<<48) | ((uint64_t)b[2]<<40) | ((uint64_t)b[3]<<32)
                | ((uint64_t)b[4]<<24) | ((uint64_t)b[5]<<16) | ((uint64_t)b[6]<<8)  |  (uint64_t)b[7];
    }
}

__device__ __forceinline__ void ev1_limbs_to_be32(const uint64_t le[4], uint8_t be[32]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint64_t w = le[3-i];
        be[i*8+0] = (uint8_t)(w >> 56);
        be[i*8+1] = (uint8_t)(w >> 48);
        be[i*8+2] = (uint8_t)(w >> 40);
        be[i*8+3] = (uint8_t)(w >> 32);
        be[i*8+4] = (uint8_t)(w >> 24);
        be[i*8+5] = (uint8_t)(w >> 16);
        be[i*8+6] = (uint8_t)(w >> 8);
        be[i*8+7] = (uint8_t)(w);
    }
}

__device__ __forceinline__ uint32_t ev1_write_u32_dec(uint32_t v, uint8_t* out) {
    uint8_t tmp[10];
    uint32_t n = 0;
    do {
        tmp[n++] = (uint8_t)('0' + (v % 10u));
        v /= 10u;
    } while (v != 0u);
    for (uint32_t i = 0; i < n; ++i) out[i] = tmp[n - 1u - i];
    return n;
}

__device__ __forceinline__ void ev1_sequence_hash(
    const uint8_t mpk_xy[64], uint32_t change, uint32_t address_index, uint8_t seq[32])
{
    uint8_t msg[96];
    uint32_t pos = ev1_write_u32_dec(address_index, msg);
    msg[pos++] = ':';
    msg[pos++] = (uint8_t)('0' + change);
    msg[pos++] = ':';
    #pragma unroll
    for (int i = 0; i < 64; ++i) msg[pos + i] = mpk_xy[i];
    uint8_t first[32];
    bw_sha256(msg, pos + 64, first);
    bw_sha256(first, 32, seq);
}

__device__ __forceinline__ void ev1_make_pubkey65(const uint64_t x[4], const uint64_t y[4], uint8_t pub65[65]) {
    pub65[0] = 0x04;
    ev1_limbs_to_be32(x, pub65 + 1);
    ev1_limbs_to_be32(y, pub65 + 33);
}

__device__ __forceinline__ bool ev1_target_match(
    const TargetData* target, const uint64_t x[4], const uint64_t y[4])
{
    uint8_t pub65[65], h160[20];
    ev1_make_pubkey65(x, y, pub65);
    getHash160_65bytes(pub65, h160);
    if (is_any_bloom(target)) {
        return bloom_want_btc(target) && bloom_check(h160, target->d_bloom_filter, target->bloom_m_bits);
    }
    if (target->type == TargetType::BTC) {
        #pragma unroll
        for (int i = 0; i < 20; ++i) if (h160[i] != target->hash20[i]) return false;
        return true;
    }
    return false;
}

__global__ __launch_bounds__(ELECTRUM_V1_THREADS, 4)
void hydra_electrumv1_kernel(
    const ElectrumV1Mask* __restrict__ mask,
    const TargetData*     __restrict__ target,
    uint64_t offset,
    int count,
    ElectrumV1Result*     __restrict__ result)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    if (atomicAdd(&result->found, 0) != 0) return;

    const uint64_t candidate = offset + (uint64_t)tid;
    uint16_t words[ELECTRUM_V1_WORDS];
    uint8_t seed_ascii[32], master_priv[32], mpk_xy[64];
    ev1_candidate_words(mask, candidate, words);
    ev1_decode_seed_ascii(words, seed_ascii);
    ev1_stretch_seed(seed_ascii, master_priv);

    uint64_t master_k[4], mx[4], my[4];
    ev1_be32_to_limbs_le(master_priv, master_k);
    scalarMulBaseAffine(master_k, mx, my);
    fieldNormalize(mx);
    fieldNormalize(my);
    ev1_limbs_to_be32(mx, mpk_xy);
    ev1_limbs_to_be32(my, mpk_xy + 32);

    if (!mask->single_path) {
        const uint32_t lookahead = mask->lookahead;
        for (uint32_t change = 0; change <= 1; ++change) {
            for (uint32_t addr = 0; addr < lookahead; ++addr) {
                uint8_t seq[32], child_priv[32];
                ev1_sequence_hash(mpk_xy, change, addr, seq);
                add_mod_n(seq, master_priv, child_priv);

                uint64_t child_k[4], cx[4], cy[4];
                ev1_be32_to_limbs_le(child_priv, child_k);
                scalarMulBaseAffine(child_k, cx, cy);
                fieldNormalize(cx);
                fieldNormalize(cy);
                if (ev1_target_match(target, cx, cy)) {
                    if (atomicCAS(&result->found, 0, 1) == 0) {
                        result->index = candidate;
                        result->change = change;
                        result->address_index = addr;
                    }
                    return;
                }
            }
        }
    } else {
        const uint32_t change = mask->path_change;
        const uint32_t addr = mask->path_index;
        uint8_t seq[32], child_priv[32];
        ev1_sequence_hash(mpk_xy, change, addr, seq);
        add_mod_n(seq, master_priv, child_priv);

        uint64_t child_k[4], cx[4], cy[4];
        ev1_be32_to_limbs_le(child_priv, child_k);
        scalarMulBaseAffine(child_k, cx, cy);
        fieldNormalize(cx);
        fieldNormalize(cy);
        if (ev1_target_match(target, cx, cy)) {
            if (atomicCAS(&result->found, 0, 1) == 0) {
                result->index = candidate;
                result->change = change;
                result->address_index = addr;
            }
            return;
        }
    }
}

__global__ __launch_bounds__(ELECTRUM_V1_THREADS, 4)
void hydra_electrumv1_stretch_kernel(
    const ElectrumV1Mask* __restrict__ mask,
    uint64_t offset,
    int count,
    uint8_t* __restrict__ d_master_priv)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const uint64_t candidate = offset + (uint64_t)tid;
    uint16_t words[ELECTRUM_V1_WORDS];
    uint8_t seed_ascii[32];
    ev1_candidate_words(mask, candidate, words);
    ev1_decode_seed_ascii(words, seed_ascii);
    ev1_stretch_seed(seed_ascii, d_master_priv + (size_t)tid * 32);
}

__global__ __launch_bounds__(ELECTRUM_V1_THREADS, 4)
void hydra_electrumv1_scan_kernel(
    const ElectrumV1Mask* __restrict__ mask,
    const TargetData*     __restrict__ target,
    const uint8_t*        __restrict__ d_master_priv,
    uint64_t offset,
    int count,
    ElectrumV1Result*     __restrict__ result)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    if (atomicAdd(&result->found, 0) != 0) return;

    const uint8_t* master_priv = d_master_priv + (size_t)tid * 32;
    uint8_t mpk_xy[64];
    uint64_t master_k[4], mx[4], my[4];
    ev1_be32_to_limbs_le(master_priv, master_k);
    scalarMulBaseAffine(master_k, mx, my);
    fieldNormalize(mx);
    fieldNormalize(my);
    ev1_limbs_to_be32(mx, mpk_xy);
    ev1_limbs_to_be32(my, mpk_xy + 32);

    const uint32_t change_begin = mask->single_path ? mask->path_change : 0;
    const uint32_t change_end   = mask->single_path ? (mask->path_change + 1) : 2;
    const uint32_t addr_begin   = mask->single_path ? mask->path_index : 0;
    const uint32_t addr_end     = mask->single_path ? (mask->path_index + 1) : mask->lookahead;
    for (uint32_t change = change_begin; change < change_end; ++change) {
        for (uint32_t addr = addr_begin; addr < addr_end; ++addr) {
            uint8_t seq[32], child_priv[32];
            ev1_sequence_hash(mpk_xy, change, addr, seq);
            add_mod_n(seq, master_priv, child_priv);

            uint64_t child_k[4], cx[4], cy[4];
            ev1_be32_to_limbs_le(child_priv, child_k);
            scalarMulBaseAffine(child_k, cx, cy);
            fieldNormalize(cx);
            fieldNormalize(cy);
            if (ev1_target_match(target, cx, cy)) {
                if (atomicCAS(&result->found, 0, 1) == 0) {
                    result->index = offset + (uint64_t)tid;
                    result->change = change;
                    result->address_index = addr;
                }
                return;
            }
        }
    }
}
