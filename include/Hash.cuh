#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// Les fonctions inline sont dans .cuh. Ici on garde les implémentations
// pour le code "lent" qui n'a pas besoin d'être inliné.

// Constantes pour les fonctions lentes
__device__ __constant__ uint32_t K_SLOW[64] = {
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1,
    0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786,
    0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147,
    0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85, 0xA2BFE8A1, 0xA81A664B,
    0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A,
    0x5B9CCA4F, 0x682E6FF3, 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2};

__device__ __constant__ uint32_t IV_SLOW[8] = {
    0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul,
    0x510e527ful, 0x9b05688cul, 0x1f83d9ab,   0x5be0cd19ul};

// Ré-implémentation locale des helpers pour éviter dépendance au header
__device__ __forceinline__ uint32_t _ror32(uint32_t x, int n) {
#if __CUDA_ARCH__ >= 350
  return __funnelshift_r(x, x, n);
#else
  return (x >> n) | (x << (32 - n));
#endif
}
__device__ __forceinline__ uint32_t _bigS0(uint32_t x) {
  return _ror32(x, 2) ^ _ror32(x, 13) ^ _ror32(x, 22);
}
__device__ __forceinline__ uint32_t _bigS1(uint32_t x) {
  return _ror32(x, 6) ^ _ror32(x, 11) ^ _ror32(x, 25);
}
__device__ __forceinline__ uint32_t _smallS0(uint32_t x) {
  return _ror32(x, 7) ^ _ror32(x, 18) ^ (x >> 3);
}
__device__ __forceinline__ uint32_t _smallS1(uint32_t x) {
  return _ror32(x, 17) ^ _ror32(x, 19) ^ (x >> 10);
}
__device__ __forceinline__ uint32_t _Ch(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}
__device__ __forceinline__ uint32_t _Maj(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) | (x & z) | (y & z);
}
__device__ __forceinline__ uint32_t _bswap32(uint32_t x) {
  return ((x & 0x000000FFu) << 24) | ((x & 0x0000FF00u) << 8) |
         ((x & 0x00FF0000u) >> 8) | ((x & 0xFF000000u) >> 24);
}
__device__ __forceinline__ uint32_t _pack_be4(uint8_t a, uint8_t b, uint8_t c,
                                              uint8_t d) {
  return ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)c << 8) |
         ((uint32_t)d);
}

__device__ __forceinline__ void SHA256Initialize(uint32_t s[8]) {
#pragma unroll
  for (int i = 0; i < 8; i++)
    s[i] = IV_SLOW[i];
}

__device__ __forceinline__ void SHA256Transform(uint32_t state[8],
                                                uint32_t W_in[64]) {
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  uint32_t w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    w[i] = W_in[i];
#pragma unroll 64
  for (int t = 0; t < 64; ++t) {
    if (t >= 16) {
      uint32_t s0 = _smallS0(w[(t + 1) & 15]);
      uint32_t s1 = _smallS1(w[(t + 14) & 15]);
      w[t & 15] = w[t & 15] + s1 + w[(t + 9) & 15] + s0;
    }
    uint32_t Wt = w[t & 15];
    uint32_t T1 = h + _bigS1(e) + _Ch(e, f, g) + K_SLOW[t] + Wt;
    uint32_t T2 = _bigS0(a) + _Maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + T1;
    d = c;
    c = b;
    b = a;
    a = T1 + T2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

__device__ __forceinline__ void SHA256_64bytes_words(const uint32_t msg_words[16],
                                                     uint32_t out_state[8]) {
  uint32_t state[8];
  SHA256Initialize(state);
  uint32_t M[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    M[i] = msg_words[i];
  SHA256Transform(state, M);
#pragma unroll
  for (int i = 0; i < 16; ++i)
    M[i] = 0;
  M[0] = 0x80000000u;
  M[15] = 64u * 8u;
  SHA256Transform(state, M);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    out_state[i] = state[i];
}

__device__ __forceinline__ void SHA256_64bytes_words_to_bytes(
    const uint32_t msg_words[16], uint8_t out[32]) {
  uint32_t state[8];
  SHA256_64bytes_words(msg_words, state);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i * 4 + 0] = (uint8_t)(state[i] >> 24);
    out[i * 4 + 1] = (uint8_t)(state[i] >> 16);
    out[i * 4 + 2] = (uint8_t)(state[i] >> 8);
    out[i * 4 + 3] = (uint8_t)(state[i]);
  }
}

__device__ __forceinline__ void RIPEMD160Initialize(uint32_t s[5]) {
  s[0] = 0x67452301ul;
  s[1] = 0xEFCDAB89ul;
  s[2] = 0x98BADCFEul;
  s[3] = 0x10325476ul;
  s[4] = 0xC3D2E1F0ul;
}

#define ROL(x, n) ((x >> (32 - n)) | (x << n))
#define f1(x, y, z) (x ^ y ^ z)
#define f2(x, y, z) ((x & y) | (~x & z))
#define f3(x, y, z) ((x | ~y) ^ z)
#define f4(x, y, z) ((x & z) | (~z & y))
#define f5(x, y, z) (x ^ (y | ~z))
#define RPRound(a, b, c, d, e, f, x, k, r)                                     \
  u = a + f + x + k;                                                           \
  a = ROL(u, r) + e;                                                           \
  c = ROL(c, 10);

#define R11(a, b, c, d, e, x, r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0, r)
#define R21(a, b, c, d, e, x, r)                                               \
  RPRound(a, b, c, d, e, f2(b, c, d), x, 0x5A827999ul, r)
#define R31(a, b, c, d, e, x, r)                                               \
  RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6ED9EBA1ul, r)
#define R41(a, b, c, d, e, x, r)                                               \
  RPRound(a, b, c, d, e, f4(b, c, d), x, 0x8F1BBCDCul, r)
#define R51(a, b, c, d, e, x, r)                                               \
  RPRound(a, b, c, d, e, f5(b, c, d), x, 0xA953FD4Eul, r)
#define R12(a, b, c, d, e, x, r)                                               \
  RPRound(a, b, c, d, e, f5(b, c, d), x, 0x50A28BE6ul, r)
#define R22(a, b, c, d, e, x, r)                                               \
  RPRound(a, b, c, d, e, f4(b, c, d), x, 0x5C4DD124ul, r)
#define R32(a, b, c, d, e, x, r)                                               \
  RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6D703EF3ul, r)
#define R42(a, b, c, d, e, x, r)                                               \
  RPRound(a, b, c, d, e, f2(b, c, d), x, 0x7A6D76E9ul, r)
#define R52(a, b, c, d, e, x, r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0, r)

__device__ __forceinline__ void RIPEMD160Transform(uint32_t s[5], uint32_t *w) {
  uint32_t u;
  uint32_t a1 = s[0], b1 = s[1], c1 = s[2], d1 = s[3], e1 = s[4];
  uint32_t a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;
  R11(a1, b1, c1, d1, e1, w[0], 11);
  R12(a2, b2, c2, d2, e2, w[5], 8);
  R11(e1, a1, b1, c1, d1, w[1], 14);
  R12(e2, a2, b2, c2, d2, w[14], 9);
  R11(d1, e1, a1, b1, c1, w[2], 15);
  R12(d2, e2, a2, b2, c2, w[7], 9);
  R11(c1, d1, e1, a1, b1, w[3], 12);
  R12(c2, d2, e2, a2, b2, w[0], 11);
  R11(b1, c1, d1, e1, a1, w[4], 5);
  R12(b2, c2, d2, e2, a2, w[9], 13);
  R11(a1, b1, c1, d1, e1, w[5], 8);
  R12(a2, b2, c2, d2, e2, w[2], 15);
  R11(e1, a1, b1, c1, d1, w[6], 7);
  R12(e2, a2, b2, c2, d2, w[11], 15);
  R11(d1, e1, a1, b1, c1, w[7], 9);
  R12(d2, e2, a2, b2, c2, w[4], 5);
  R11(c1, d1, e1, a1, b1, w[8], 11);
  R12(c2, d2, e2, a2, b2, w[13], 7);
  R11(b1, c1, d1, e1, a1, w[9], 13);
  R12(b2, c2, d2, e2, a2, w[6], 7);
  R11(a1, b1, c1, d1, e1, w[10], 14);
  R12(a2, b2, c2, d2, e2, w[15], 8);
  R11(e1, a1, b1, c1, d1, w[11], 15);
  R12(e2, a2, b2, c2, d2, w[8], 11);
  R11(d1, e1, a1, b1, c1, w[12], 6);
  R12(d2, e2, a2, b2, c2, w[1], 14);
  R11(c1, d1, e1, a1, b1, w[13], 7);
  R12(c2, d2, e2, a2, b2, w[10], 14);
  R11(b1, c1, d1, e1, a1, w[14], 9);
  R12(b2, c2, d2, e2, a2, w[3], 12);
  R11(a1, b1, c1, d1, e1, w[15], 8);
  R12(a2, b2, c2, d2, e2, w[12], 6);

  R21(e1, a1, b1, c1, d1, w[7], 7);
  R22(e2, a2, b2, c2, d2, w[6], 9);
  R21(d1, e1, a1, b1, c1, w[4], 6);
  R22(d2, e2, a2, b2, c2, w[11], 13);
  R21(c1, d1, e1, a1, b1, w[13], 8);
  R22(c2, d2, e2, a2, b2, w[3], 15);
  R21(b1, c1, d1, e1, a1, w[1], 13);
  R22(b2, c2, d2, e2, a2, w[7], 7);
  R21(a1, b1, c1, d1, e1, w[10], 11);
  R22(a2, b2, c2, d2, e2, w[0], 12);
  R21(e1, a1, b1, c1, d1, w[6], 9);
  R22(e2, a2, b2, c2, d2, w[13], 8);
  R21(d1, e1, a1, b1, c1, w[15], 7);
  R22(d2, e2, a2, b2, c2, w[5], 9);
  R21(c1, d1, e1, a1, b1, w[3], 15);
  R22(c2, d2, e2, a2, b2, w[10], 11);
  R21(b1, c1, d1, e1, a1, w[12], 7);
  R22(b2, c2, d2, e2, a2, w[14], 7);
  R21(a1, b1, c1, d1, e1, w[0], 12);
  R22(a2, b2, c2, d2, e2, w[15], 7);
  R21(e1, a1, b1, c1, d1, w[9], 15);
  R22(e2, a2, b2, c2, d2, w[8], 12);
  R21(d1, e1, a1, b1, c1, w[5], 9);
  R22(d2, e2, a2, b2, c2, w[12], 7);
  R21(c1, d1, e1, a1, b1, w[2], 11);
  R22(c2, d2, e2, a2, b2, w[4], 6);
  R21(b1, c1, d1, e1, a1, w[14], 7);
  R22(b2, c2, d2, e2, a2, w[9], 15);
  R21(a1, b1, c1, d1, e1, w[11], 13);
  R22(a2, b2, c2, d2, e2, w[1], 13);
  R21(e1, a1, b1, c1, d1, w[8], 12);
  R22(e2, a2, b2, c2, d2, w[2], 11);

  R31(d1, e1, a1, b1, c1, w[3], 11);
  R32(d2, e2, a2, b2, c2, w[15], 9);
  R31(c1, d1, e1, a1, b1, w[10], 13);
  R32(c2, d2, e2, a2, b2, w[5], 7);
  R31(b1, c1, d1, e1, a1, w[14], 6);
  R32(b2, c2, d2, e2, a2, w[1], 15);
  R31(a1, b1, c1, d1, e1, w[4], 7);
  R32(a2, b2, c2, d2, e2, w[3], 11);
  R31(e1, a1, b1, c1, d1, w[9], 14);
  R32(e2, a2, b2, c2, d2, w[7], 8);
  R31(d1, e1, a1, b1, c1, w[15], 9);
  R32(d2, e2, a2, b2, c2, w[14], 6);
  R31(c1, d1, e1, a1, b1, w[8], 13);
  R32(c2, d2, e2, a2, b2, w[6], 6);
  R31(b1, c1, d1, e1, a1, w[1], 15);
  R32(b2, c2, d2, e2, a2, w[9], 14);
  R31(a1, b1, c1, d1, e1, w[2], 14);
  R32(a2, b2, c2, d2, e2, w[11], 12);
  R31(e1, a1, b1, c1, d1, w[7], 8);
  R32(e2, a2, b2, c2, d2, w[8], 13);
  R31(d1, e1, a1, b1, c1, w[0], 13);
  R32(d2, e2, a2, b2, c2, w[12], 5);
  R31(c1, d1, e1, a1, b1, w[6], 6);
  R32(c2, d2, e2, a2, b2, w[2], 14);
  R31(b1, c1, d1, e1, a1, w[13], 5);
  R32(b2, c2, d2, e2, a2, w[10], 13);
  R31(a1, b1, c1, d1, e1, w[11], 12);
  R32(a2, b2, c2, d2, e2, w[0], 13);
  R31(e1, a1, b1, c1, d1, w[5], 7);
  R32(e2, a2, b2, c2, d2, w[4], 7);
  R31(d1, e1, a1, b1, c1, w[12], 5);
  R32(d2, e2, a2, b2, c2, w[13], 5);

  R41(c1, d1, e1, a1, b1, w[1], 11);
  R42(c2, d2, e2, a2, b2, w[8], 15);
  R41(b1, c1, d1, e1, a1, w[9], 12);
  R42(b2, c2, d2, e2, a2, w[6], 5);
  R41(a1, b1, c1, d1, e1, w[11], 14);
  R42(a2, b2, c2, d2, e2, w[4], 8);
  R41(e1, a1, b1, c1, d1, w[10], 15);
  R42(e2, a2, b2, c2, d2, w[1], 11);
  R41(d1, e1, a1, b1, c1, w[0], 14);
  R42(d2, e2, a2, b2, c2, w[3], 14);
  R41(c1, d1, e1, a1, b1, w[8], 15);
  R42(c2, d2, e2, a2, b2, w[11], 14);
  R41(b1, c1, d1, e1, a1, w[12], 9);
  R42(b2, c2, d2, e2, a2, w[15], 6);
  R41(a1, b1, c1, d1, e1, w[4], 8);
  R42(a2, b2, c2, d2, e2, w[0], 14);
  R41(e1, a1, b1, c1, d1, w[13], 9);
  R42(e2, a2, b2, c2, d2, w[5], 6);
  R41(d1, e1, a1, b1, c1, w[3], 14);
  R42(d2, e2, a2, b2, c2, w[12], 9);
  R41(c1, d1, e1, a1, b1, w[7], 5);
  R42(c2, d2, e2, a2, b2, w[2], 12);
  R41(b1, c1, d1, e1, a1, w[15], 6);
  R42(b2, c2, d2, e2, a2, w[13], 9);
  R41(a1, b1, c1, d1, e1, w[14], 8);
  R42(a2, b2, c2, d2, e2, w[9], 12);
  R41(e1, a1, b1, c1, d1, w[5], 6);
  R42(e2, a2, b2, c2, d2, w[7], 5);
  R41(d1, e1, a1, b1, c1, w[6], 5);
  R42(d2, e2, a2, b2, c2, w[10], 15);
  R41(c1, d1, e1, a1, b1, w[2], 12);
  R42(c2, d2, e2, a2, b2, w[14], 8);

  R51(b1, c1, d1, e1, a1, w[4], 9);
  R52(b2, c2, d2, e2, a2, w[12], 8);
  R51(a1, b1, c1, d1, e1, w[0], 15);
  R52(a2, b2, c2, d2, e2, w[15], 5);
  R51(e1, a1, b1, c1, d1, w[5], 5);
  R52(e2, a2, b2, c2, d2, w[10], 12);
  R51(d1, e1, a1, b1, c1, w[9], 11);
  R52(d2, e2, a2, b2, c2, w[4], 9);
  R51(c1, d1, e1, a1, b1, w[7], 6);
  R52(c2, d2, e2, a2, b2, w[1], 12);
  R51(b1, c1, d1, e1, a1, w[12], 8);
  R52(b2, c2, d2, e2, a2, w[5], 5);
  R51(a1, b1, c1, d1, e1, w[2], 13);
  R52(a2, b2, c2, d2, e2, w[8], 14);
  R51(e1, a1, b1, c1, d1, w[10], 12);
  R52(e2, a2, b2, c2, d2, w[7], 6);
  R51(d1, e1, a1, b1, c1, w[14], 5);
  R52(d2, e2, a2, b2, c2, w[6], 8);
  R51(c1, d1, e1, a1, b1, w[1], 12);
  R52(c2, d2, e2, a2, b2, w[2], 13);
  R51(b1, c1, d1, e1, a1, w[3], 13);
  R52(b2, c2, d2, e2, a2, w[13], 6);
  R51(a1, b1, c1, d1, e1, w[8], 14);
  R52(a2, b2, c2, d2, e2, w[14], 5);
  R51(e1, a1, b1, c1, d1, w[11], 11);
  R52(e2, a2, b2, c2, d2, w[0], 15);
  R51(d1, e1, a1, b1, c1, w[6], 8);
  R52(d2, e2, a2, b2, c2, w[3], 13);
  R51(c1, d1, e1, a1, b1, w[15], 5);
  R52(c2, d2, e2, a2, b2, w[9], 11);
  R51(b1, c1, d1, e1, a1, w[13], 6);
  R52(b2, c2, d2, e2, a2, w[11], 11);

  uint32_t t = s[0];
  s[0] = s[1] + c1 + d2;
  s[1] = s[2] + d1 + e2;
  s[2] = s[3] + e1 + a2;
  s[3] = s[4] + a1 + b2;
  s[4] = t + b1 + c2;
}

__device__ __forceinline__ void getSHA256_33bytes(const uint8_t *pubkey33,
                                                  uint8_t sha[32]) {
  uint32_t M[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    M[i] = 0;
#pragma unroll
  for (int i = 0; i < 33; ++i) {
    M[i >> 2] |= (uint32_t)pubkey33[i] << (24 - ((i & 3) << 3));
  }
  M[8] |= (uint32_t)0x80 << (24 - ((33 & 3) << 3));
  M[14] = 0;
  M[15] = 33u * 8u;
  uint32_t state[8];
  SHA256Initialize(state);
  SHA256Transform(state, M);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sha[4 * i + 0] = (uint8_t)(state[i] >> 24);
    sha[4 * i + 1] = (uint8_t)(state[i] >> 16);
    sha[4 * i + 2] = (uint8_t)(state[i] >> 8);
    sha[4 * i + 3] = (uint8_t)(state[i] >> 0);
  }
}

__device__ __forceinline__ void getRIPEMD160_32bytes(const uint8_t *sha,
                                                     uint8_t ripemd[20]) {
  uint8_t block[64] = {0};
  for (int i = 0; i < 32; i++)
    block[i] = sha[i];
  block[32] = 0x80;
  const uint32_t bitLen = 256;
  block[56] = bitLen & 0xFF;
  block[57] = (bitLen >> 8) & 0xFF;
  block[58] = (bitLen >> 16) & 0xFF;
  block[59] = (bitLen >> 24) & 0xFF;
  uint32_t W[16];
  for (int i = 0; i < 16; i++) {
    W[i] = ((uint32_t)block[4 * i + 3] << 24) |
           ((uint32_t)block[4 * i + 2] << 16) |
           ((uint32_t)block[4 * i + 1] << 8) | ((uint32_t)block[4 * i]);
  }
  uint32_t state[5];
  RIPEMD160Initialize(state);
  RIPEMD160Transform(state, W);
  for (int i = 0; i < 5; i++) {
    ripemd[4 * i] = (state[i] >> 0) & 0xFF;
    ripemd[4 * i + 1] = (state[i] >> 8) & 0xFF;
    ripemd[4 * i + 2] = (state[i] >> 16) & 0xFF;
    ripemd[4 * i + 3] = (state[i] >> 24) & 0xFF;
  }
}

__device__ __forceinline__ void getHash160_33bytes(const uint8_t *pubkey33, uint8_t *hash20) {
  uint8_t sha256[32];
  getSHA256_33bytes(pubkey33, sha256);
  getRIPEMD160_32bytes(sha256, hash20);
}

__device__ __forceinline__ void getSHA256_65bytes(const uint8_t *pubkey65,
                                                  uint8_t sha[32]) {
  uint32_t state[8];
  SHA256Initialize(state);

  uint32_t M[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    M[i] = ((uint32_t)pubkey65[i * 4 + 0] << 24) |
           ((uint32_t)pubkey65[i * 4 + 1] << 16) |
           ((uint32_t)pubkey65[i * 4 + 2] << 8) |
           ((uint32_t)pubkey65[i * 4 + 3]);
  }
  SHA256Transform(state, M);

#pragma unroll
  for (int i = 0; i < 16; ++i)
    M[i] = 0;
  M[0] = ((uint32_t)pubkey65[64] << 24) | 0x00800000u;
  M[15] = 65u * 8u;
  SHA256Transform(state, M);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sha[4 * i + 0] = (uint8_t)(state[i] >> 24);
    sha[4 * i + 1] = (uint8_t)(state[i] >> 16);
    sha[4 * i + 2] = (uint8_t)(state[i] >> 8);
    sha[4 * i + 3] = (uint8_t)(state[i] >> 0);
  }
}

__device__ __forceinline__ void getHash160_65bytes(const uint8_t *pubkey65,
                                                   uint8_t *hash20) {
  uint8_t sha256[32];
  getSHA256_65bytes(pubkey65, sha256);
  getRIPEMD160_32bytes(sha256, hash20);
}

__device__ __forceinline__ void
SHA256_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4],
                     uint32_t out_state[8]) {
  const uint64_t v3 = x_be_limbs[3];
  const uint64_t v2 = x_be_limbs[2];
  const uint64_t v1 = x_be_limbs[1];
  const uint64_t v0 = x_be_limbs[0];
  uint32_t M[16];
  M[0] = _pack_be4(prefix02_03, (uint8_t)(v3 >> 56), (uint8_t)(v3 >> 48),
                   (uint8_t)(v3 >> 40));
  M[1] = _pack_be4((uint8_t)(v3 >> 32), (uint8_t)(v3 >> 24),
                   (uint8_t)(v3 >> 16), (uint8_t)(v3 >> 8));
  M[2] = _pack_be4((uint8_t)(v3 >> 0), (uint8_t)(v2 >> 56), (uint8_t)(v2 >> 48),
                   (uint8_t)(v2 >> 40));
  M[3] = _pack_be4((uint8_t)(v2 >> 32), (uint8_t)(v2 >> 24),
                   (uint8_t)(v2 >> 16), (uint8_t)(v2 >> 8));
  M[4] = _pack_be4((uint8_t)(v2 >> 0), (uint8_t)(v1 >> 56), (uint8_t)(v1 >> 48),
                   (uint8_t)(v1 >> 40));
  M[5] = _pack_be4((uint8_t)(v1 >> 32), (uint8_t)(v1 >> 24),
                   (uint8_t)(v1 >> 16), (uint8_t)(v1 >> 8));
  M[6] = _pack_be4((uint8_t)(v1 >> 0), (uint8_t)(v0 >> 56), (uint8_t)(v0 >> 48),
                   (uint8_t)(v0 >> 40));
  M[7] = _pack_be4((uint8_t)(v0 >> 32), (uint8_t)(v0 >> 24),
                   (uint8_t)(v0 >> 16), (uint8_t)(v0 >> 8));
  M[8] = _pack_be4((uint8_t)(v0 >> 0), 0x80u, 0x00u, 0x00u);
#pragma unroll
  for (int i = 9; i < 16; ++i)
    M[i] = 0;
  M[15] = 33u * 8u;
  uint32_t st[8];
  SHA256Initialize(st);
  SHA256Transform(st, M);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    out_state[i] = st[i];
}

__device__ __forceinline__ void
RIPEMD160_from_SHA256_state(const uint32_t sha_state_be[8],
                            uint8_t ripemd20[20]) {
  uint32_t W[16];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    W[i] = _bswap32(sha_state_be[i]);
  W[8] = 0x00000080u;
#pragma unroll
  for (int i = 9; i < 14; ++i)
    W[i] = 0;
  W[14] = 256u;
  W[15] = 0u;
  uint32_t s[5];
  RIPEMD160Initialize(s);
  RIPEMD160Transform(s, W);
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    ripemd20[4 * i + 0] = (uint8_t)(s[i] >> 0);
    ripemd20[4 * i + 1] = (uint8_t)(s[i] >> 8);
    ripemd20[4 * i + 2] = (uint8_t)(s[i] >> 16);
    ripemd20[4 * i + 3] = (uint8_t)(s[i] >> 24);
  }
}

__device__ __forceinline__ void getHash160_33_from_limbs(uint8_t prefix02_03,
                                         const uint64_t x_be_limbs[4],
                                         uint8_t out20[20]) {
  uint32_t sha_state[8];
  SHA256_33_from_limbs(prefix02_03, x_be_limbs, sha_state);
  RIPEMD160_from_SHA256_state(sha_state, out20);
}


// ============================================================================
// ETH : Keccak-256 (version Barracuda — scalarisée, fully unrolled)
// ============================================================================

// Constantes Keccak round
__device__ __constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

// bswap64 inline (indépendant de Common.h)
__device__ __forceinline__ uint64_t _bswap64_hydra(uint64_t x) {
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t lo = (uint32_t)x;
    return ((uint64_t)__byte_perm(lo, 0, 0x0123) << 32) | __byte_perm(hi, 0, 0x0123);
}

// rotl64 inline
__device__ __forceinline__ uint64_t _rotl64_hydra(uint64_t x, int s) {
    return (x << s) | (x >> (64 - s));
}

// Keccak256 ETH depuis x_le[4] et y_le[4] (little-endian, format ECC.h)
// Retourne les 20 derniers bytes du hash dans out_bytes[20]
// Adapté de Barracuda/Hash.h : keccak256_eth_last20_from_xy_words
__device__ __forceinline__ void getEthAddr_from_limbs(
    const uint64_t x_le[4],
    const uint64_t y_le[4],
    uint8_t out_bytes[20])
{
    // État 5×5 scalarisé (zéro-initialisé)
    uint64_t a00=0,a10=0,a20=0,a30=0,a40=0;
    uint64_t a01=0,a11=0,a21=0,a31=0,a41=0;
    uint64_t a02=0,a12=0,a22=0,a32=0,a42=0;
    uint64_t a03=0,a13=0,a23=0,a33=0,a43=0;
    uint64_t a04=0,a14=0,a24=0,a34=0,a44=0;

    // Absorb X (big-endian) : x_le[3] = MSB, x_le[0] = LSB
    a00 ^= _bswap64_hydra(x_le[3]);
    a10 ^= _bswap64_hydra(x_le[2]);
    a20 ^= _bswap64_hydra(x_le[1]);
    a30 ^= _bswap64_hydra(x_le[0]);
    // Absorb Y (big-endian)
    a40 ^= _bswap64_hydra(y_le[3]);
    a01 ^= _bswap64_hydra(y_le[2]);
    a11 ^= _bswap64_hydra(y_le[1]);
    a21 ^= _bswap64_hydra(y_le[0]);

    // Padding Keccak (ETH = Keccak, pas SHA3 → 0x01)
    a31 ^= 0x01ULL;
    a13 ^= 0x8000000000000000ULL;  // rate=136 bytes → lane[17] = st[a13] bit63

    const uint64_t MASK = 0xFFFFFFFFFFFFFFFFULL;

    // unroll 4 : bon compromis vitesse/temps de compilation (vs unroll 24 qui explose)
    #pragma unroll 4
    for (int r = 0; r < 24; ++r) {
        // θ
        uint64_t C0=a00^a01^a02^a03^a04;
        uint64_t C1=a10^a11^a12^a13^a14;
        uint64_t C2=a20^a21^a22^a23^a24;
        uint64_t C3=a30^a31^a32^a33^a34;
        uint64_t C4=a40^a41^a42^a43^a44;
        uint64_t D0=_rotl64_hydra(C1,1)^C4;
        uint64_t D1=_rotl64_hydra(C2,1)^C0;
        uint64_t D2=_rotl64_hydra(C3,1)^C1;
        uint64_t D3=_rotl64_hydra(C4,1)^C2;
        uint64_t D4=_rotl64_hydra(C0,1)^C3;
        a00^=D0; a01^=D0; a02^=D0; a03^=D0; a04^=D0;
        a10^=D1; a11^=D1; a12^=D1; a13^=D1; a14^=D1;
        a20^=D2; a21^=D2; a22^=D2; a23^=D2; a24^=D2;
        a30^=D3; a31^=D3; a32^=D3; a33^=D3; a34^=D3;
        a40^=D4; a41^=D4; a42^=D4; a43^=D4; a44^=D4;

        // ρ+π (toutes les rotations de Barracuda)
        uint64_t B00=a00;
        uint64_t B10=_rotl64_hydra(a11,44); uint64_t B20=_rotl64_hydra(a22,43);
        uint64_t B30=_rotl64_hydra(a33,21); uint64_t B40=_rotl64_hydra(a44,14);
        uint64_t B01=_rotl64_hydra(a30,28); uint64_t B11=_rotl64_hydra(a41,20);
        uint64_t B21=_rotl64_hydra(a02, 3); uint64_t B31=_rotl64_hydra(a13,45);
        uint64_t B41=_rotl64_hydra(a24,61);
        uint64_t B02=_rotl64_hydra(a10, 1); uint64_t B12=_rotl64_hydra(a21, 6);
        uint64_t B22=_rotl64_hydra(a32,25); uint64_t B32=_rotl64_hydra(a43, 8);
        uint64_t B42=_rotl64_hydra(a04,18);
        uint64_t B03=_rotl64_hydra(a40,27); uint64_t B13=_rotl64_hydra(a01,36);
        uint64_t B23=_rotl64_hydra(a12,10); uint64_t B33=_rotl64_hydra(a23,15);
        uint64_t B43=_rotl64_hydra(a34,56);
        uint64_t B04=_rotl64_hydra(a20,62); uint64_t B14=_rotl64_hydra(a31,55);
        uint64_t B24=_rotl64_hydra(a42,39); uint64_t B34=_rotl64_hydra(a03,41);
        uint64_t B44=_rotl64_hydra(a14, 2);

        // χ
        uint64_t t0,t1,t2,t3,t4;
        t0=B00^((~B10&MASK)&B20); t1=B10^((~B20&MASK)&B30); t2=B20^((~B30&MASK)&B40);
        t3=B30^((~B40&MASK)&B00); t4=B40^((~B00&MASK)&B10);
        a00=t0; a10=t1; a20=t2; a30=t3; a40=t4;

        t0=B01^((~B11&MASK)&B21); t1=B11^((~B21&MASK)&B31); t2=B21^((~B31&MASK)&B41);
        t3=B31^((~B41&MASK)&B01); t4=B41^((~B01&MASK)&B11);
        a01=t0; a11=t1; a21=t2; a31=t3; a41=t4;

        t0=B02^((~B12&MASK)&B22); t1=B12^((~B22&MASK)&B32); t2=B22^((~B32&MASK)&B42);
        t3=B32^((~B42&MASK)&B02); t4=B42^((~B02&MASK)&B12);
        a02=t0; a12=t1; a22=t2; a32=t3; a42=t4;

        t0=B03^((~B13&MASK)&B23); t1=B13^((~B23&MASK)&B33); t2=B23^((~B33&MASK)&B43);
        t3=B33^((~B43&MASK)&B03); t4=B43^((~B03&MASK)&B13);
        a03=t0; a13=t1; a23=t2; a33=t3; a43=t4;

        t0=B04^((~B14&MASK)&B24); t1=B14^((~B24&MASK)&B34); t2=B24^((~B34&MASK)&B44);
        t3=B34^((~B44&MASK)&B04); t4=B44^((~B04&MASK)&B14);
        a04=t0; a14=t1; a24=t2; a34=t3; a44=t4;

        // ι
        a00 ^= KECCAK_RC[r];
    }

    // Squeeze : bytes 12..31 du hash Keccak (= adresse ETH)
    // Les lanes Keccak sont little-endian : byte k = (lane >> (8*(k%8))) & 0xFF
    // bytes 0-7=a00, 8-15=a10, 16-23=a20, 24-31=a30
    // On veut bytes 12..31 → octets 4..7 de a10, tous de a20, tous de a30

    // Octets 12-15 : bits 32-63 de a10 (little-endian in-lane)
    out_bytes[ 0] = (uint8_t)(a10 >> 32);
    out_bytes[ 1] = (uint8_t)(a10 >> 40);
    out_bytes[ 2] = (uint8_t)(a10 >> 48);
    out_bytes[ 3] = (uint8_t)(a10 >> 56);
    // Octets 16-23 : a20 complet
    out_bytes[ 4] = (uint8_t)(a20      );
    out_bytes[ 5] = (uint8_t)(a20 >>  8);
    out_bytes[ 6] = (uint8_t)(a20 >> 16);
    out_bytes[ 7] = (uint8_t)(a20 >> 24);
    out_bytes[ 8] = (uint8_t)(a20 >> 32);
    out_bytes[ 9] = (uint8_t)(a20 >> 40);
    out_bytes[10] = (uint8_t)(a20 >> 48);
    out_bytes[11] = (uint8_t)(a20 >> 56);
    // Octets 24-31 : a30 complet
    out_bytes[12] = (uint8_t)(a30      );
    out_bytes[13] = (uint8_t)(a30 >>  8);
    out_bytes[14] = (uint8_t)(a30 >> 16);
    out_bytes[15] = (uint8_t)(a30 >> 24);
    out_bytes[16] = (uint8_t)(a30 >> 32);
    out_bytes[17] = (uint8_t)(a30 >> 40);
    out_bytes[18] = (uint8_t)(a30 >> 48);
    out_bytes[19] = (uint8_t)(a30 >> 56);
}
