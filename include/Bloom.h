#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "HydraCommon.h"

// =================================================================================
// BLOOM FILTER CHECK (device)
// Compatible avec le filtre créé par tools/create_bloom.py
// Murmur3-32 double hash, k=BLOOM_K_HASHES probes
// =================================================================================

#ifndef BLOOM_K_HASHES
#define BLOOM_K_HASHES 16
#endif

__device__ __forceinline__
uint32_t murmur3_bloom(const uint8_t h160[20], uint32_t seed) {
    // Charge les 20 bytes comme 5 mots uint32 little-endian
    uint32_t k0 = (uint32_t)h160[0]  | ((uint32_t)h160[1]<<8)  | ((uint32_t)h160[2]<<16)  | ((uint32_t)h160[3]<<24);
    uint32_t k1 = (uint32_t)h160[4]  | ((uint32_t)h160[5]<<8)  | ((uint32_t)h160[6]<<16)  | ((uint32_t)h160[7]<<24);
    uint32_t k2 = (uint32_t)h160[8]  | ((uint32_t)h160[9]<<8)  | ((uint32_t)h160[10]<<16) | ((uint32_t)h160[11]<<24);
    uint32_t k3 = (uint32_t)h160[12] | ((uint32_t)h160[13]<<8) | ((uint32_t)h160[14]<<16) | ((uint32_t)h160[15]<<24);
    uint32_t k4 = (uint32_t)h160[16] | ((uint32_t)h160[17]<<8) | ((uint32_t)h160[18]<<16) | ((uint32_t)h160[19]<<24);

    const uint32_t c1 = 0xcc9e2d51u;
    const uint32_t c2 = 0x1b873593u;
    uint32_t h = seed;

    #define MUR_BLOCK(k) do { \
        uint32_t _k = (k); \
        _k *= c1; _k = (_k<<15)|(_k>>(32-15)); _k *= c2; \
        h ^= _k; h = (h<<13)|(h>>(32-13)); h = h*5u + 0xe6546b64u; \
    } while(0)

    MUR_BLOCK(k0); MUR_BLOCK(k1); MUR_BLOCK(k2); MUR_BLOCK(k3); MUR_BLOCK(k4);
    #undef MUR_BLOCK

    h ^= 20u;                                  // finalize length
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// Retourne true si h160 est potentiellement dans le filtre
__device__ __forceinline__
bool bloom_check(const uint8_t h160[20], const uint64_t* __restrict__ d_bloom, uint64_t bloom_m_bits) {
    uint32_t h1 = murmur3_bloom(h160, 0x9747b28cu);
    uint32_t h2 = murmur3_bloom(h160, h1);

    // Empêche le cas critique où h2 est nul
    if (h2 == 0) h2 = 1;

    // bloom_m_bits is always a power of 2 (create_bloom.py: TARGET_GB * 8 GBits)
    // & (m-1) replaces % m : saves ~20 cycles per probe (~5-8% faster in bloom mode)
    const uint64_t mask = bloom_m_bits - 1ULL;

    #pragma unroll
    for (int i = 0; i < BLOOM_K_HASHES; i++) {
        uint64_t bit_pos  = (h1 + (uint64_t)i * h2) & mask;
        uint64_t word_idx = bit_pos >> 6;   // / 64
        uint32_t bit_idx  = (uint32_t)(bit_pos & 63ULL);  // % 64
        if ((d_bloom[word_idx] & (1ULL << bit_idx)) == 0) return false;
    }
    return true;
}

// =============================================================================
// Helpers inline pour dispatch bloom
// =============================================================================
__device__ __forceinline__
bool is_any_bloom(const TargetData* t) {
    return t->type == TargetType::BLOOM
        || t->type == TargetType::BLOOM_BTC
        || t->type == TargetType::BLOOM_ETH;
}

__device__ __forceinline__
bool bloom_want_btc(const TargetData* t) {
    return t->type == TargetType::BLOOM || t->type == TargetType::BLOOM_BTC;
}

__device__ __forceinline__
bool bloom_want_eth(const TargetData* t) {
    return t->type == TargetType::BLOOM || t->type == TargetType::BLOOM_ETH;
}
