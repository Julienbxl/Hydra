/*
 * ======================================================================================
 * ECC_PTX.h - LAZY REDUCTION EDITION (Fastest)
 * ======================================================================================
 * - Math   : CUDAMath Scalarized
 * - Add    : Lazy Carry-Injection (Gain ~5-10% sur l'addition)
 * - Inv    : Bernstein-Yang Optimized
 * - Point  : Low Register Pressure
 * ======================================================================================
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// =================================================================================
// 1. MACROS PTX
// =================================================================================

// Addition avec Carry
#define UADD(r, a, b)                                                          \
  asm volatile("addc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
#define UADDO(r, a, b)                                                         \
  asm volatile("add.cc.u64 %0, %1, %2;"                                        \
               : "=l"(r)                                                       \
               : "l"(a), "l"(b)                                                \
               : "memor"                                                       \
                 "y");
#define UADDC(r, a, b)                                                         \
  asm volatile("addc.cc.u64 %0, %1, %2;"                                       \
               : "=l"(r)                                                       \
               : "l"(a), "l"(b)                                                \
               : "memor"                                                       \
                 "y");

// In-Place (Accumulateurs)
#define UADDO1(c, a)                                                           \
  asm volatile("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define UADDC1(c, a)                                                           \
  asm volatile("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define UADD1(c, a) asm volatile("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

// Soustraction
#define USUB(r, a, b)                                                          \
  asm volatile("subc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
#define USUBO(r, a, b)                                                         \
  asm volatile("sub.cc.u64 %0, %1, %2;"                                        \
               : "=l"(r)                                                       \
               : "l"(a), "l"(b)                                                \
               : "memor"                                                       \
                 "y");
#define USUBC(r, a, b)                                                         \
  asm volatile("subc.cc.u64 %0, %1, %2;"                                       \
               : "=l"(r)                                                       \
               : "l"(a), "l"(b)                                                \
               : "memor"                                                       \
                 "y");

// In-Place
#define USUBO1(c, a)                                                           \
  asm volatile("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define USUBC1(c, a)                                                           \
  asm volatile("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define USUB1(c, a) asm volatile("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

// Multiplication (CUDAMath Primitives)
#define CM_UMULLO(lo, a, b)                                                    \
  asm volatile("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define CM_UMULHI(hi, a, b)                                                    \
  asm volatile("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define CM_MADDO(r, a, b, c)                                                   \
  asm volatile("mad.hi.cc.u64 %0, %1, %2, %3;"                                 \
               : "=l"(r)                                                       \
               : "l"(a), "l"(b), "l"(c)                                        \
               : "memory");
#define CM_MADDC(r, a, b, c)                                                   \
  asm volatile("madc.hi.cc.u64 %0, %1, %2, %3;"                                \
               : "=l"(r)                                                       \
               : "l"(a), "l"(b), "l"(c)                                        \
               : "memory");
#define CM_MADD(r, a, b, c)                                                    \
  asm volatile("madc.hi.u64 %0, %1, %2, %3;"                                   \
               : "=l"(r)                                                       \
               : "l"(a), "l"(b), "l"(c));

// Bitwise
#define __sleft128(a, b, n) (((b) << (n)) | ((a) >> (64 - (n))))
#define __sright128(a, b, n) (((a) >> (n)) | ((b) << (64 - (n))))

// =================================================================================
// 2. CONSTANTES
// =================================================================================

__constant__ uint64_t SECP_P_LE[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL};

__constant__ uint64_t SECP_GX_LE[4] = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL,
    0x79BE667EF9DCBBACULL};

__constant__ uint64_t SECP_GY_LE[4] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL,
    0x483ADA7726A3C465ULL};

// K = 2^32 + 977 (Pour le Carry Folding)
static constexpr uint64_t SECP_K = 0x1000003D1ULL;

// =================================================================================
// 3. ADD / SUB MODULAIRE (LAZY CARRY FOLDING - CORRIGÉ)
// =================================================================================

__device__ __forceinline__ int cmp256(const uint64_t a[4],
                                      const uint64_t b[4]) {
  if (a[3] != b[3])
    return (a[3] < b[3]) ? -1 : 1;
  if (a[2] != b[2])
    return (a[2] < b[2]) ? -1 : 1;
  if (a[1] != b[1])
    return (a[1] < b[1]) ? -1 : 1;
  if (a[0] != b[0])
    return (a[0] < b[0]) ? -1 : 1;
  return 0;
}

__device__ __forceinline__ void fieldSub(const uint64_t a[4],
                                         const uint64_t b[4], uint64_t r[4]) {
  uint64_t t0, t1, t2, t3;
  uint64_t bor; // sera 0 ou 0xFFFF..FFFF

  asm volatile("sub.cc.u64  %0, %5, %9;\n\t"
               "subc.cc.u64 %1, %6, %10;\n\t"
               "subc.cc.u64 %2, %7, %11;\n\t"
               "subc.cc.u64 %3, %8, %12;\n\t"
               "subc.u64    %4, 0, 0;\n\t"
               : "=l"(t0), "=l"(t1), "=l"(t2), "=l"(t3), "=l"(bor)
               : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(b[0]),
                 "l"(b[1]), "l"(b[2]), "l"(b[3]));

  // bor est 0 ou 0xFFFF.. ; on veut un bit 0/1
  const uint64_t bbit = bor & 1ULL; // 0 ou 1
  const uint64_t m = 0ULL - bbit;   // 0x000.. ou 0xFFF..

  const uint64_t p0 = SECP_P_LE[0] & m;
  const uint64_t p1 = SECP_P_LE[1] & m;
  const uint64_t p2 = SECP_P_LE[2] & m;
  const uint64_t p3 = SECP_P_LE[3] & m;

  // r = t + (p & m), carry chain complète
  asm volatile("add.cc.u64  %0, %4, %8;\n\t"
               "addc.cc.u64 %1, %5, %9;\n\t"
               "addc.cc.u64 %2, %6, %10;\n\t"
               "addc.u64    %3, %7, %11;\n\t"
               : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3])
               : "l"(t0), "l"(t1), "l"(t2), "l"(t3), "l"(p0), "l"(p1), "l"(p2),
                 "l"(p3));
}

// OPTIMISATION LAZY : Carry Injection (Version Corrigée)
__device__ void fieldAdd(const uint64_t a[4], const uint64_t b[4],
                         uint64_t r[4]) {
  uint64_t c_out;

  // 1. Addition 256 bits
  UADDO(r[0], a[0], b[0]);
  UADDC(r[1], a[1], b[1]);
  UADDC(r[2], a[2], b[2]);

  // Correction de l'erreur : On calcule r[3] ET on capture le carry
  // manuellement
  UADDC(r[3], a[3], b[3]);
  asm volatile("addc.u64 %0, 0, 0;"
               : "=l"(c_out)); // Capture le bit de retenue (0 ou 1)

  // 2. Carry Folding (Lazy Reduction)
  // Si c_out=1 (débordement 256 bits), on ajoute K (0x1000003D1)
  uint64_t k_val = c_out * 0x1000003D1ULL;

  UADDO1(r[0], k_val);
  UADDC1(r[1], 0ULL);
  UADDC1(r[2], 0ULL);
  UADD1(r[3], 0ULL); // UADD1 à 2 arguments est correct ici
}
__device__ __forceinline__ void fieldNeg(const uint64_t *a, uint64_t *r) {
  uint64_t t0, t1, t2, t3;
  USUBO(t0, SECP_P_LE[0], a[0]);
  USUBC(t1, SECP_P_LE[1], a[1]);
  USUBC(t2, SECP_P_LE[2], a[2]);
  USUB(t3, SECP_P_LE[3], a[3]);
  const uint64_t nz = a[0] | a[1] | a[2] | a[3];
  const uint64_t mask = 0ULL - (uint64_t)(nz != 0);
  r[0] = t0 & mask;
  r[1] = t1 & mask;
  r[2] = t2 & mask;
  r[3] = t3 & mask;
}

// =================================================================================
// 4. MULTIPLICATION & CARRÉ (SCALARIZED CUDAMATH LOGIC)
// =================================================================================

// Macro interne scalariée pour UMult
#define CM_UMult_Scalar(r0, r1, r2, r3, r4, a, b)                              \
  {                                                                            \
    CM_UMULLO(r0, a[0], b);                                                    \
    CM_UMULLO(r1, a[1], b);                                                    \
    CM_MADDO(r1, a[0], b, r1);                                                 \
    CM_UMULLO(r2, a[2], b);                                                    \
    CM_MADDC(r2, a[1], b, r2);                                                 \
    CM_UMULLO(r3, a[3], b);                                                    \
    CM_MADDC(r3, a[2], b, r3);                                                 \
    CM_MADD(r4, a[3], b, 0ULL);                                                \
  }

// Macro interne scalarisée pour UMultSpecial (Réduction) - VERSION MUL DIRECTE
// Calcule [a0,a1,a2,a3] * SECP_K (0x1000003D1) avec chaîne MAD
#define CM_UMultSpecial_Scalar(r0, r1, r2, r3, r4, a0, a1, a2, a3, a4)         \
  {                                                                            \
    CM_UMULLO(r0, a0, SECP_K);                                                 \
    CM_UMULLO(r1, a1, SECP_K);                                                 \
    CM_MADDO(r1, a0, SECP_K, r1);                                              \
    CM_UMULLO(r2, a2, SECP_K);                                                 \
    CM_MADDC(r2, a1, SECP_K, r2);                                              \
    CM_UMULLO(r3, a3, SECP_K);                                                 \
    CM_MADDC(r3, a2, SECP_K, r3);                                              \
    CM_MADD(r4, a3, SECP_K, 0ULL);                                             \
  }

// LAZY MUL - Version fusionnée avec une seule passe de réduction
__device__ __forceinline__ void fieldMul(const uint64_t *a, const uint64_t *b,
                                         uint64_t *r) {
  uint64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;
  uint64_t t0, t1, t2, t3, t4;

  // 1. Multiplication 256x256 -> 512 (Scalarized)
  CM_UMult_Scalar(r0, r1, r2, r3, r4, a, b[0]);

  CM_UMult_Scalar(t0, t1, t2, t3, t4, a, b[1]);
  UADDO1(r1, t0);
  UADDC1(r2, t1);
  UADDC1(r3, t2);
  UADDC1(r4, t3);
  UADD1(r5, t4);

  CM_UMult_Scalar(t0, t1, t2, t3, t4, a, b[2]);
  UADDO1(r2, t0);
  UADDC1(r3, t1);
  UADDC1(r4, t2);
  UADDC1(r5, t3);
  UADD1(r6, t4);

  CM_UMult_Scalar(t0, t1, t2, t3, t4, a, b[3]);
  UADDO1(r3, t0);
  UADDC1(r4, t1);
  UADDC1(r5, t2);
  UADDC1(r6, t3);
  UADD1(r7, t4);

  // 2. Réduction 512 -> 320 (via UMultSpecial MUL)
  CM_UMultSpecial_Scalar(t0, t1, t2, t3, t4, r4, r5, r6, r7, 0ULL);
  UADDO1(r0, t0);
  UADDC1(r1, t1);
  UADDC1(r2, t2);
  UADDC1(r3, t3);

  // 3. Réduction LAZY 320 -> 256 (UNE SEULE PASSE)
  // Capture carry et multiplie par K en une opération fusionnée
  UADD1(t4, 0ULL);
  uint64_t al, ah;
  CM_UMULLO(al, t4, SECP_K);
  CM_UMULHI(ah, t4, SECP_K);

  // Addition finale - le résultat peut être légèrement > P mais c'est OK (Lazy)
  UADDO(r[0], r0, al);
  UADDC(r[1], r1, ah);
  UADDC(r[2], r2, 0ULL);
  UADD(r[3], r3, 0ULL);

  // ⚠️ PAS de second carry folding ! Résultat dans [0, ~1.02×P]
}

// LAZY SQR - Version fusionnée avec une seule passe de réduction
__device__ __forceinline__ void fieldSqr(const uint64_t *up, uint64_t *rp) {
  uint64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;
  uint64_t SL, SH;
  uint64_t r01L, r01H, r02L, r02H, r03L, r03H;
  uint64_t r12L, r12H, r13L, r13H;
  uint64_t r23L, r23H;

  // 1. Square Optimisé (termes diagonaux + termes croisés ×2)
  CM_UMULLO(SL, up[0], up[0]);
  CM_UMULHI(SH, up[0], up[0]);
  CM_UMULLO(r01L, up[0], up[1]);
  CM_UMULHI(r01H, up[0], up[1]);
  CM_UMULLO(r02L, up[0], up[2]);
  CM_UMULHI(r02H, up[0], up[2]);
  CM_UMULLO(r03L, up[0], up[3]);
  CM_UMULHI(r03H, up[0], up[3]);

  r0 = SL;
  r1 = r01L;
  r2 = r02L;
  r3 = r03L;

  UADDO1(r1, SH);
  UADDC1(r2, r01H);
  UADDC1(r3, r02H);
  UADD(r4, r03H, 0ULL);

  CM_UMULLO(SL, up[1], up[1]);
  CM_UMULHI(SH, up[1], up[1]);
  CM_UMULLO(r12L, up[1], up[2]);
  CM_UMULHI(r12H, up[1], up[2]);
  CM_UMULLO(r13L, up[1], up[3]);
  CM_UMULHI(r13H, up[1], up[3]);

  UADDO1(r1, r01L);
  UADDC1(r2, SL);
  UADDC1(r3, r12L);
  UADDC1(r4, r13L);
  UADD(r5, r13H, 0ULL);
  UADDO1(r2, r01H);
  UADDC1(r3, SH);
  UADDC1(r4, r12H);
  UADD1(r5, 0ULL);

  CM_UMULLO(SL, up[2], up[2]);
  CM_UMULHI(SH, up[2], up[2]);
  CM_UMULLO(r23L, up[2], up[3]);
  CM_UMULHI(r23H, up[2], up[3]);

  UADDO1(r2, r02L);
  UADDC1(r3, r12L);
  UADDC1(r4, SL);
  UADDC1(r5, r23L);
  UADD(r6, r23H, 0ULL);
  UADDO1(r3, r02H);
  UADDC1(r4, r12H);
  UADDC1(r5, SH);
  UADD1(r6, 0ULL);

  CM_UMULLO(SL, up[3], up[3]);
  CM_UMULHI(SH, up[3], up[3]);
  UADDO1(r3, r03L);
  UADDC1(r4, r13L);
  UADDC1(r5, r23L);
  UADDC1(r6, SL);
  UADD(r7, SH, 0ULL);
  UADDO1(r4, r03H);
  UADDC1(r5, r13H);
  UADDC1(r6, r23H);
  UADD1(r7, 0ULL);

  // 2. Reduce 512 -> 320 (via UMultSpecial MUL)
  uint64_t t0, t1, t2, t3, t4;
  CM_UMultSpecial_Scalar(t0, t1, t2, t3, t4, r4, r5, r6, r7, 0ULL);
  UADDO1(r0, t0);
  UADDC1(r1, t1);
  UADDC1(r2, t2);
  UADDC1(r3, t3);

  // 3. Réduction LAZY 320 -> 256 (UNE SEULE PASSE)
  UADD1(t4, 0ULL);
  uint64_t al, ah;
  CM_UMULLO(al, t4, SECP_K);
  CM_UMULHI(ah, t4, SECP_K);

  // Addition finale - résultat dans [0, ~1.02×P]
  UADDO(rp[0], r0, al);
  UADDC(rp[1], r1, ah);
  UADDC(rp[2], r2, 0ULL);
  UADD(rp[3], r3, 0ULL);

  // ⚠️ PAS de second carry folding ! (Lazy)
}

// OPTIMISATION 1.3 : FUSED SUB-MUL (SCALARISÉ - sans tableau temporaire)
__device__ __forceinline__ void fieldSubMul(const uint64_t *a,
                                            const uint64_t *b,
                                            const uint64_t *c, uint64_t *r) {
  // 1. Soustraction (Résultat gardé en registres s0..s3)
  uint64_t s0, s1, s2, s3, bor;
  asm volatile("sub.cc.u64  %0, %5, %9;\n\t"
               "subc.cc.u64 %1, %6, %10;\n\t"
               "subc.cc.u64 %2, %7, %11;\n\t"
               "subc.cc.u64 %3, %8, %12;\n\t"
               "subc.u64    %4, 0, 0;\n\t"
               : "=l"(s0), "=l"(s1), "=l"(s2), "=l"(s3), "=l"(bor)
               : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(b[0]),
                 "l"(b[1]), "l"(b[2]), "l"(b[3]));
  const uint64_t m = 0ULL - (bor & 1ULL);
  asm volatile("add.cc.u64  %0, %0, %4;\n\t"
               "addc.cc.u64 %1, %1, %5;\n\t"
               "addc.cc.u64 %2, %2, %6;\n\t"
               "addc.u64    %3, %3, %7;\n\t"
               : "+l"(s0), "+l"(s1), "+l"(s2), "+l"(s3)
               : "l"(SECP_P_LE[0] & m), "l"(SECP_P_LE[1] & m),
                 "l"(SECP_P_LE[2] & m), "l"(SECP_P_LE[3] & m));

  // 2. Multiplication SCALARISÉE directe (sans tableau intermédiaire)
  uint64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;
  uint64_t t0, t1, t2, t3, t4;

  // Multiplication s * c[0]
  CM_UMULLO(r0, s0, c[0]);
  CM_UMULLO(r1, s1, c[0]);
  CM_MADDO(r1, s0, c[0], r1);
  CM_UMULLO(r2, s2, c[0]);
  CM_MADDC(r2, s1, c[0], r2);
  CM_UMULLO(r3, s3, c[0]);
  CM_MADDC(r3, s2, c[0], r3);
  CM_MADD(r4, s3, c[0], 0ULL);

  // Multiplication s * c[1]
  CM_UMULLO(t0, s0, c[1]);
  CM_UMULLO(t1, s1, c[1]);
  CM_MADDO(t1, s0, c[1], t1);
  CM_UMULLO(t2, s2, c[1]);
  CM_MADDC(t2, s1, c[1], t2);
  CM_UMULLO(t3, s3, c[1]);
  CM_MADDC(t3, s2, c[1], t3);
  CM_MADD(t4, s3, c[1], 0ULL);
  UADDO1(r1, t0);
  UADDC1(r2, t1);
  UADDC1(r3, t2);
  UADDC1(r4, t3);
  UADD1(r5, t4);

  // Multiplication s * c[2]
  CM_UMULLO(t0, s0, c[2]);
  CM_UMULLO(t1, s1, c[2]);
  CM_MADDO(t1, s0, c[2], t1);
  CM_UMULLO(t2, s2, c[2]);
  CM_MADDC(t2, s1, c[2], t2);
  CM_UMULLO(t3, s3, c[2]);
  CM_MADDC(t3, s2, c[2], t3);
  CM_MADD(t4, s3, c[2], 0ULL);
  UADDO1(r2, t0);
  UADDC1(r3, t1);
  UADDC1(r4, t2);
  UADDC1(r5, t3);
  UADD1(r6, t4);

  // Multiplication s * c[3]
  CM_UMULLO(t0, s0, c[3]);
  CM_UMULLO(t1, s1, c[3]);
  CM_MADDO(t1, s0, c[3], t1);
  CM_UMULLO(t2, s2, c[3]);
  CM_MADDC(t2, s1, c[3], t2);
  CM_UMULLO(t3, s3, c[3]);
  CM_MADDC(t3, s2, c[3], t3);
  CM_MADD(t4, s3, c[3], 0ULL);
  UADDO1(r3, t0);
  UADDC1(r4, t1);
  UADDC1(r5, t2);
  UADDC1(r6, t3);
  UADD1(r7, t4);

  // Réduction 512 -> 320
  CM_UMultSpecial_Scalar(t0, t1, t2, t3, t4, r4, r5, r6, r7, 0ULL);
  UADDO1(r0, t0);
  UADDC1(r1, t1);
  UADDC1(r2, t2);
  UADDC1(r3, t3);

  // Réduction LAZY 320 -> 256
  UADD1(t4, 0ULL);
  uint64_t al, ah;
  CM_UMULLO(al, t4, SECP_K);
  CM_UMULHI(ah, t4, SECP_K);

  UADDO(r[0], r0, al);
  UADDC(r[1], r1, ah);
  UADDC(r[2], r2, 0ULL);
  UADD(r[3], r3, 0ULL);
}

// =================================================================================
// WRAPPERS & UTILS
// =================================================================================

__device__ __forceinline__ void fieldNormalize(uint64_t x[4]) {
  uint64_t t0, t1, t2, t3;
  uint64_t bor; // 0 ou 0xFFFF..FFFF

  asm volatile("sub.cc.u64  %0, %5, %9;\n\t"
               "subc.cc.u64 %1, %6, %10;\n\t"
               "subc.cc.u64 %2, %7, %11;\n\t"
               "subc.cc.u64 %3, %8, %12;\n\t"
               "subc.u64    %4, 0, 0;\n\t"
               : "=l"(t0), "=l"(t1), "=l"(t2), "=l"(t3), "=l"(bor)
               : "l"(x[0]), "l"(x[1]), "l"(x[2]), "l"(x[3]), "l"(SECP_P_LE[0]),
                 "l"(SECP_P_LE[1]), "l"(SECP_P_LE[2]), "l"(SECP_P_LE[3]));

  // bor = 0  => x>=p (pas d'emprunt) -> on veut garder t (= x-p)
  // bor = -1 => x<p  (emprunt)       -> on veut garder x
  const uint64_t lt = bor & 1ULL;           // 1 si x<p
  const uint64_t keep = 0ULL - (1ULL - lt); // 0xFFFF.. si x>=p, sinon 0

  x[0] = (x[0] & ~keep) | (t0 & keep);
  x[1] = (x[1] & ~keep) | (t1 & keep);
  x[2] = (x[2] & ~keep) | (t2 & keep);
  x[3] = (x[3] & ~keep) | (t3 & keep);
}

__device__ __forceinline__ void _ModMult(uint64_t *r, const uint64_t *a) {
  fieldMul(r, a, r);
}
__device__ __forceinline__ void _ModMult(uint64_t *r, uint64_t *a,
                                         uint64_t *b) {
  fieldMul(a, b, r);
}
__device__ __forceinline__ void _ModSqr(uint64_t *r, const uint64_t *a) {
  fieldSqr(a, r);
}
__device__ __forceinline__ void ModSub256(uint64_t *r, uint64_t *a,
                                          uint64_t *b) {
  fieldSub(a, b, r);
}
__device__ __forceinline__ void ModNeg256(uint64_t *r, uint64_t *a) {
  fieldNeg(a, r);
}

// VERSION CORRIGÉE POUR LAZY REDUCTION
// Calcule a - b et retourne la parité exacte du résultat normalisé
__device__ __forceinline__ void ModSub256isOdd(uint64_t *a, uint64_t *b,
                                               uint8_t *parity) {
  uint64_t T[4];
  fieldSub(a, b, T);
  fieldNormalize(T); // INDISPENSABLE avec Lazy Reduc !
  *parity = (uint8_t)(T[0] & 1);
}

// =================================================================================
// 5. INVERSION MODULAIRE (Bernstein-Yang Optimized)
// =================================================================================
#define BY_NBBLOCK 5
#define BY_IsPositive(x) (((int64_t)(x[4])) >= 0LL)
#define BY_IsNegative(x) (((int64_t)(x[4])) < 0LL)
#define BY_IsZero(a) ((a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define BY_IsOne(a)                                                            \
  ((a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) &&     \
   (a[0] == 1ULL))
static constexpr uint64_t BY_MM64 = 0xD838091DD2253531ULL;
static constexpr uint64_t BY_MSK62 = 0x3FFFFFFFFFFFFFFFULL;

template <typename T> __device__ __forceinline__ void by_swap(T &a, T &b) {
  T t = a;
  a = b;
  b = t;
}

__device__ __forceinline__ void BY_AddP(uint64_t r[5]) {
  UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL);
  UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL);
  UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL);
  UADDC1(r[3], 0xFFFFFFFFFFFFFFFFULL);
  UADD1(r[4], 0ULL);
}
__device__ __forceinline__ void BY_SubP(uint64_t r[5]) {
  USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL);
  USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL);
  USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL);
  USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL);
  USUB1(r[4], 0ULL);
}
__device__ __forceinline__ void BY_Neg(uint64_t r[5]) {
  USUBO(r[0], 0ULL, r[0]);
  USUBC(r[1], 0ULL, r[1]);
  USUBC(r[2], 0ULL, r[2]);
  USUBC(r[3], 0ULL, r[3]);
  USUB(r[4], 0ULL, r[4]);
}
__device__ __forceinline__ void BY_Load(uint64_t r[5], const uint64_t a[5]) {
#pragma unroll
  for (int i = 0; i < 5; i++)
    r[i] = a[i];
}
__device__ __forceinline__ uint32_t BY_ctz(uint64_t x) {
  uint32_t n;
  asm("{\n\t .reg .u64 tmp;\n\t brev.b64 tmp, %1;\n\t clz.b64 %0, tmp;\n\t}"
      : "=r"(n)
      : "l"(x));
  return n;
}
__device__ __forceinline__ void BY_ShiftR62(uint64_t r[5]) {
  r[0] = (r[1] << 2) | (r[0] >> 62);
  r[1] = (r[2] << 2) | (r[1] >> 62);
  r[2] = (r[3] << 2) | (r[2] >> 62);
  r[3] = (r[4] << 2) | (r[3] >> 62);
  r[4] = (int64_t)(r[4]) >> 62;
}
__device__ __forceinline__ void
BY_ShiftR62_Carry(uint64_t dest[5], const uint64_t r[5], uint64_t carry) {
  dest[0] = (r[1] << 2) | (r[0] >> 62);
  dest[1] = (r[2] << 2) | (r[1] >> 62);
  dest[2] = (r[3] << 2) | (r[2] >> 62);
  dest[3] = (r[4] << 2) | (r[3] >> 62);
  dest[4] = (carry << 2) | (uint64_t)((int64_t)r[4] >> 62);
}
__device__ __forceinline__ uint64_t BY_IMultC(uint64_t *r, uint64_t *a,
                                              int64_t b) {
  uint64_t t[BY_NBBLOCK], carry;
  if (b < 0) {
    b = -b;
    USUBO(t[0], 0ULL, a[0]);
    USUBC(t[1], 0ULL, a[1]);
    USUBC(t[2], 0ULL, a[2]);
    USUBC(t[3], 0ULL, a[3]);
    USUB(t[4], 0ULL, a[4]);
  } else {
    BY_Load(t, a);
  }
  CM_UMULLO(r[0], t[0], b);
  CM_UMULLO(r[1], t[1], b);
  CM_MADDO(r[1], t[0], b, r[1]);
  CM_UMULLO(r[2], t[2], b);
  CM_MADDC(r[2], t[1], b, r[2]);
  CM_UMULLO(r[3], t[3], b);
  CM_MADDC(r[3], t[2], b, r[3]);
  CM_UMULLO(r[4], t[4], b);
  CM_MADDC(r[4], t[3], b, r[4]);
  asm volatile("madc.hi.s64 %0, %1, %2, %3;"
               : "=l"(carry)
               : "l"(t[4]), "l"(b), "l"(0ULL));
  return carry;
}
__device__ __forceinline__ void BY_IMult(uint64_t *r, uint64_t *a, int64_t b) {
  uint64_t t[BY_NBBLOCK];
  if (b < 0) {
    b = -b;
    USUBO(t[0], 0ULL, a[0]);
    USUBC(t[1], 0ULL, a[1]);
    USUBC(t[2], 0ULL, a[2]);
    USUBC(t[3], 0ULL, a[3]);
    USUB(t[4], 0ULL, a[4]);
  } else {
    BY_Load(t, a);
  }
  CM_UMULLO(r[0], t[0], b);
  CM_UMULLO(r[1], t[1], b);
  CM_MADDO(r[1], t[0], b, r[1]);
  CM_UMULLO(r[2], t[2], b);
  CM_MADDC(r[2], t[1], b, r[2]);
  CM_UMULLO(r[3], t[3], b);
  CM_MADDC(r[3], t[2], b, r[3]);
  CM_UMULLO(r[4], t[4], b);
  CM_MADD(r[4], t[3], b, r[4]);
}
// Original BY_MatrixVecMul
__device__ __forceinline__ void BY_MatrixVecMul(uint64_t u[5], uint64_t v[5],
                                                int64_t _11, int64_t _12,
                                                int64_t _21, int64_t _22) {
  uint64_t t1[BY_NBBLOCK], t2[BY_NBBLOCK], t3[BY_NBBLOCK], t4[BY_NBBLOCK];
  BY_IMult(t1, u, _11);
  BY_IMult(t2, v, _12);
  BY_IMult(t3, u, _21);
  BY_IMult(t4, v, _22);
  UADDO(u[0], t1[0], t2[0]);
  UADDC(u[1], t1[1], t2[1]);
  UADDC(u[2], t1[2], t2[2]);
  UADDC(u[3], t1[3], t2[3]);
  UADD(u[4], t1[4], t2[4]);
  UADDO(v[0], t3[0], t4[0]);
  UADDC(v[1], t3[1], t4[1]);
  UADDC(v[2], t3[2], t4[2]);
  UADDC(v[3], t3[3], t4[3]);
  UADD(v[4], t3[4], t4[4]);
}

// Original BY_MatrixVecMulHalf
__device__ __forceinline__ void
BY_MatrixVecMulHalf(uint64_t dest[5], uint64_t u[5], uint64_t v[5], int64_t _11,
                    int64_t _12, uint64_t *carry) {
  uint64_t t1[BY_NBBLOCK], t2[BY_NBBLOCK], c1, c2, cout;
  c1 = BY_IMultC(t1, u, _11);
  c2 = BY_IMultC(t2, v, _12);
  asm volatile("add.cc.u64  %0, %6, %11;\n\t"
               "addc.cc.u64 %1, %7, %12;\n\t"
               "addc.cc.u64 %2, %8, %13;\n\t"
               "addc.cc.u64 %3, %9, %14;\n\t"
               "addc.cc.u64 %4, %10, %15;\n\t"
               "addc.u64    %5, 0, 0;\n\t"
               : "=l"(dest[0]), "=l"(dest[1]), "=l"(dest[2]), "=l"(dest[3]),
                 "=l"(dest[4]), "=l"(cout)
               : "l"(t1[0]), "l"(t1[1]), "l"(t1[2]), "l"(t1[3]), "l"(t1[4]),
                 "l"(t2[0]), "l"(t2[1]), "l"(t2[2]), "l"(t2[3]), "l"(t2[4]));
  *carry = c1 + c2 + cout;
}

// Original BY_MulP
__device__ __forceinline__ void BY_MulP(uint64_t *r, uint64_t a) {
  uint64_t ah, al;
  CM_UMULLO(al, a, 0x1000003D1ULL);
  CM_UMULHI(ah, a, 0x1000003D1ULL);
  USUBO(r[0], 0ULL, al);
  USUBC(r[1], 0ULL, ah);
  USUBC(r[2], 0ULL, 0ULL);
  USUBC(r[3], 0ULL, 0ULL);
  USUB(r[4], a, 0ULL);
}

// Original BY_AddCh
__device__ __forceinline__ uint64_t BY_AddCh(uint64_t r[5], const uint64_t a[5],
                                             uint64_t carry) {
  uint64_t carryOut;
  asm volatile("add.cc.u64 %0, %0, %6;\n\t"
               "addc.cc.u64 %1, %1, %7;\n\t"
               "addc.cc.u64 %2, %2, %8;\n\t"
               "addc.cc.u64 %3, %3, %9;\n\t"
               "addc.cc.u64 %4, %4, %10;\n\t"
               "addc.u64    %5, %11, 0;\n\t"
               : "+l"(r[0]), "+l"(r[1]), "+l"(r[2]), "+l"(r[3]), "+l"(r[4]),
                 "=l"(carryOut)
               : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(a[4]),
                 "l"(carry));
  return carryOut;
}

// BY_DivStep62 - calcule 62 étapes de division pour Bernstein-Yang
__device__ __forceinline__ void BY_DivStep62(uint64_t u[5], uint64_t v[5],
                                             int32_t *pos, int64_t *uu,
                                             int64_t *uv, int64_t *vu,
                                             int64_t *vv) {
  *uu = 1;
  *uv = 0;
  *vu = 0;
  *vv = 1;
  uint32_t bitCount = 62, zeros;
  uint64_t u0 = u[0], v0 = v[0], uh, vh;
  while (*pos > 0 && (u[*pos] | v[*pos]) == 0)
    (*pos)--;
  if (*pos == 0) {
    uh = u[0];
    vh = v[0];
  } else {
    uint32_t s = __clzll(u[*pos] | v[*pos]);
    if (s == 0) {
      uh = u[*pos];
      vh = v[*pos];
    } else {
      uh = __sleft128(u[*pos - 1], u[*pos], s);
      vh = __sleft128(v[*pos - 1], v[*pos], s);
    }
  }
  while (true) {
    zeros = BY_ctz(v0 | (1ULL << bitCount));
    v0 >>= zeros;
    vh >>= zeros;
    *uu <<= zeros;
    *uv <<= zeros;
    bitCount -= zeros;
    if (bitCount == 0)
      break;
    if (vh < uh) {
      by_swap(uh, vh);
      by_swap(u0, v0);
      by_swap(*uu, *vu);
      by_swap(*uv, *vv);
    }
    vh -= uh;
    v0 -= u0;
    *vv -= *uv;
    *vu -= *uu;
  }
}

// BY_ModInv5 - Register optimized (3 temp arrays au lieu de 4)
__device__ __noinline__ void BY_ModInv5(uint64_t *R) {
  int64_t uu, uv, vu, vv;
  uint64_t mr0, ms0, carryR, carryS;
  int32_t pos = BY_NBBLOCK - 1;
  uint64_t u[BY_NBBLOCK], v[BY_NBBLOCK], r[BY_NBBLOCK], s[BY_NBBLOCK];

  // OPTIMISATION: 3 tableaux au lieu de 4
  // tr reste séparé (on en a besoin jusqu'au shift final)
  // tmp est partagé pour r0 et ts (jamais utilisés simultanément)
  // s0 reste séparé
  uint64_t tr[BY_NBBLOCK], tmp[BY_NBBLOCK], s0[BY_NBBLOCK];

  u[0] = 0xFFFFFFFEFFFFFC2F;
  u[1] = 0xFFFFFFFFFFFFFFFF;
  u[2] = 0xFFFFFFFFFFFFFFFF;
  u[3] = 0xFFFFFFFFFFFFFFFF;
  u[4] = 0;
  BY_Load(v, R);
  r[0] = 0;
  s[0] = 1;
  r[1] = r[2] = r[3] = r[4] = 0;
  s[1] = s[2] = s[3] = s[4] = 0;
  while (true) {
    BY_DivStep62(u, v, &pos, &uu, &uv, &vu, &vv);
    BY_MatrixVecMul(u, v, uu, uv, vu, vv);
    if (BY_IsNegative(u)) {
      BY_Neg(u);
      uu = -uu;
      uv = -uv;
    }
    if (BY_IsNegative(v)) {
      BY_Neg(v);
      vu = -vu;
      vv = -vv;
    }
    BY_ShiftR62(u);
    BY_ShiftR62(v);

    // tr = uu*r + uv*s
    BY_MatrixVecMulHalf(tr, r, s, uu, uv, &carryR);
    mr0 = (tr[0] * BY_MM64) & BY_MSK62;
    // tmp = r0 = MulP(mr0)
    BY_MulP(tmp, mr0);
    // tr += r0
    carryR = BY_AddCh(tr, tmp, carryR);

    if (BY_IsZero(v)) {
      BY_ShiftR62_Carry(r, tr, carryR);
      break;
    } else {
      // tmp = ts = vu*r + vv*s (réutilise tmp, on n'en a plus besoin)
      BY_MatrixVecMulHalf(tmp, r, s, vu, vv, &carryS);
      ms0 = (tmp[0] * BY_MM64) & BY_MSK62;
      // s0 = MulP(ms0)
      BY_MulP(s0, ms0);
      // tmp(ts) += s0
      carryS = BY_AddCh(tmp, s0, carryS);
    }
    BY_ShiftR62_Carry(r, tr, carryR);
    BY_ShiftR62_Carry(s, tmp, carryS);
  }
  if (!BY_IsOne(u)) {
    R[0] = R[1] = R[2] = R[3] = R[4] = 0;
    return;
  }
  while (BY_IsNegative(r))
    BY_AddP(r);
  while (!BY_IsNegative(r))
    BY_SubP(r);
  BY_AddP(r);
  BY_Load(R, r);
}
__device__ __forceinline__ void fieldInv(const uint64_t a[4], uint64_t r[4]) {
  if ((a[0] | a[1] | a[2] | a[3]) == 0ULL) {
    r[0] = r[1] = r[2] = r[3] = 0;
    return;
  }
  uint64_t t[5] = {a[0], a[1], a[2], a[3], 0};
  BY_ModInv5(t);
  r[0] = t[0];
  r[1] = t[1];
  r[2] = t[2];
  r[3] = t[3];
}
__device__ __forceinline__ void _ModInv(uint64_t *R) { fieldInv(R, R); }

// =================================================================================
// 6. EC POINT LOGIC (LOW REGISTER PRESSURE)
// =================================================================================

struct ECPointA {
  uint64_t X[4];
  uint64_t Y[4];
  bool infinity;
};

__device__ __forceinline__ void pointSetInfinity(ECPointA &P) {
  P.infinity = true;
  P.X[0] = P.X[1] = P.X[2] = P.X[3] = 0;
  P.Y[0] = P.Y[1] = P.Y[2] = P.Y[3] = 0;
}
__device__ __forceinline__ void pointSetG(ECPointA &P) {
  P.infinity = false;
  P.X[0] = SECP_GX_LE[0];
  P.X[1] = SECP_GX_LE[1];
  P.X[2] = SECP_GX_LE[2];
  P.X[3] = SECP_GX_LE[3];
  P.Y[0] = SECP_GY_LE[0];
  P.Y[1] = SECP_GY_LE[1];
  P.Y[2] = SECP_GY_LE[2];
  P.Y[3] = SECP_GY_LE[3];
}
__device__ __forceinline__ void fieldCopy(const uint64_t a[4],
                                          uint64_t out[4]) {
  out[0] = a[0];
  out[1] = a[1];
  out[2] = a[2];
  out[3] = a[3];
}

// Optimisation Low-Regs : On réutilise agressivement t0, t1, t2, t3
__device__ void pointDoubleAffine(const ECPointA &P, ECPointA &R) {
  if (P.infinity) {
    pointSetInfinity(R);
    return;
  }

  uint64_t t0[4], t1[4], t2[4], t3[4];
  // t0 = lambda, t1 = temp, t2 = temp, t3 = temp

  // 1. Calcul Lambda = (3x^2) / (2y)
  fieldSqr(P.X, t0);    // t0 = x^2
  fieldAdd(t0, t0, t1); // t1 = 2x^2 (Utilise Lazy Add)
  fieldAdd(t1, t0, t0); // t0 = 3x^2

  fieldAdd(P.Y, P.Y, t1); // t1 = 2y
  fieldInv(t1, t2);       // t2 = 1/2y
  fieldMul(t0, t2, t1);   // t1 = Lambda

  // 2. Calcul x3 = lambda^2 - 2x
  fieldSqr(t1, t2);       // t2 = lambda^2
  fieldAdd(P.X, P.X, t3); // t3 = 2x
  fieldSub(t2, t3, R.X);  // R.X = x3 (STOCKÉ)

  // 3. Calcul y3 = lambda(x - x3) - y
  fieldSub(P.X, R.X, t2); // t2 = x - x3
  fieldMul(t1, t2, t3);   // t3 = lambda * (x - x3)
  fieldSub(t3, P.Y, R.Y); // R.Y = y3 (STOCKÉ)

  R.infinity = false;
}

__device__ void pointAddAffine(const ECPointA &P, const ECPointA &Q,
                               ECPointA &R) {
  if (P.infinity) {
    R = Q;
    return;
  }
  if (Q.infinity) {
    R = P;
    return;
  }

  if (cmp256(P.X, Q.X) == 0) {
    if (cmp256(P.Y, Q.Y) == 0)
      pointDoubleAffine(P, R);
    else
      pointSetInfinity(R);
    return;
  }

  uint64_t t0[4], t1[4], t2[4], t3[4];
  // t0 = lambda, t1 = dx, t2 = dy, t3 = temp

  // 1. Calcul Lambda = (y2 - y1) / (x2 - x1)
  fieldSub(Q.X, P.X, t1); // t1 = dx
  fieldSub(Q.Y, P.Y, t2); // t2 = dy
  fieldInv(t1, t3);       // t3 = 1/dx
  fieldMul(t2, t3, t0);   // t0 = Lambda

  // 2. Calcul x3 = lambda^2 - x1 - x2
  fieldSqr(t0, t1);       // t1 = lambda^2
  fieldSub(t1, P.X, t2);  // t2 = lambda^2 - x1
  fieldSub(t2, Q.X, R.X); // R.X = x3 (STOCKÉ)

  // 3. Calcul y3 = lambda(x1 - x3) - y1
  fieldSub(P.X, R.X, t1); // t1 = x1 - x3
  fieldMul(t0, t1, t2);   // t2 = lambda * (...)
  fieldSub(t2, P.Y, R.Y); // R.Y = y3 (STOCKÉ)

  R.infinity = false;
}

__device__ void scalarMulBaseAffine(const uint64_t scalar_le[4],
                                    uint64_t outX[4], uint64_t outY[4]) {
  ECPointA R;
  pointSetInfinity(R);
  int msb = -1;
  for (int limb = 3; limb >= 0; --limb) {
    if (scalar_le[limb] != 0) {
      msb = limb * 64 + 63 - __clzll(scalar_le[limb]);
      break;
    }
  }
  if (msb == -1) {
    outX[0] = 0;
    outY[0] = 0;
    return;
  }
  for (int bi = msb; bi >= 0; --bi) {
    if (!R.infinity) {
      ECPointA tmp;
      pointDoubleAffine(R, tmp);
      R = tmp;
    }
    if ((scalar_le[bi >> 6] >> (bi & 63)) & 1ULL) {
      ECPointA Gp;
      pointSetG(Gp);
      if (R.infinity)
        R = Gp;
      else {
        ECPointA tmp;
        pointAddAffine(R, Gp, tmp);
        R = tmp;
      }
    }
  }
  fieldCopy(R.X, outX);
  fieldCopy(R.Y, outY);
}

// ============================================================================
// ADDITION SCALAIRE 256 BITS OPTIMISÉE PTX
// r += a (où 'a' est un uint64_t, propagé sur 256 bits)
// ============================================================================
__device__ __forceinline__ void scalarAdd256_PTX(uint64_t* r, uint64_t a) {
    asm volatile (
        "add.cc.u64      %0, %0, %4;\n\t"  // r[0] += a;   (Set Carry)
        "addc.cc.u64     %1, %1, 0;\n\t"   // r[1] += 0 + Carry
        "addc.cc.u64     %2, %2, 0;\n\t"   // r[2] += 0 + Carry
        "addc.u64        %3, %3, 0;\n\t"   // r[3] += 0 + Carry
        : "+l"(r[0]), "+l"(r[1]), "+l"(r[2]), "+l"(r[3])
        : "l"(a)
        : "memory"
    );
}

// Version pour ajouter un grand nombre (256 bits + 256 bits)
// r += a
__device__ __forceinline__ void scalarAdd256_Full_PTX(uint64_t* r, const uint64_t* a) {
    asm volatile (
        "add.cc.u64      %0, %0, %4;\n\t"  // r[0] += a[0]
        "addc.cc.u64     %1, %1, %5;\n\t"  // r[1] += a[1] + Carry
        "addc.cc.u64     %2, %2, %6;\n\t"  // r[2] += a[2] + Carry
        "addc.u64        %3, %3, %7;\n\t"  // r[3] += a[3] + Carry
        : "+l"(r[0]), "+l"(r[1]), "+l"(r[2]), "+l"(r[3])
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3])
        : "memory"
    );
}

__global__ void scalarMulKernelBase(const uint64_t *scalars_in, uint64_t *outX,
                                    uint64_t *outY, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;
  scalarMulBaseAffine(scalars_in + idx * 4, outX + idx * 4, outY + idx * 4);
}

