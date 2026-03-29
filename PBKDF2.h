/*
 * PBKDF2.h - VERSION OPTIMISÉE (ROLLING BUFFER)
 * Optimisation registres pour maximiser l'occupancy GPU
 */

#ifndef PBKDF2_H
#define PBKDF2_H

#include <cuda_runtime.h>
#include <stdint.h>

#define SHA512_BLOCK_SIZE  128
#define SHA512_DIGEST_SIZE 64

// ============================================================================
// 1. CONSTANTES & MACROS SÉCURISÉES
// ============================================================================

// rotr64 PTX : décompose en deux __funnelshift_r sur les moitiés 32-bit
// __funnelshift_r(hi, lo, n) = bits [hi:lo] décalés de n → instruction SHFR native
__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t lo = (uint32_t)(x);
    uint32_t r_hi, r_lo;
    if (n < 32) {
        r_hi = __funnelshift_r(hi, lo, n);
        r_lo = __funnelshift_r(lo, hi, n);
    } else {
        r_hi = __funnelshift_r(lo, hi, n - 32);
        r_lo = __funnelshift_r(hi, lo, n - 32);
    }
    return ((uint64_t)r_hi << 32) | (uint64_t)r_lo;
}

#define SHA512_SHR(x, n)    ((x) >> (n))
// LOP3.LUT : instruction matérielle NVIDIA, calcule toute fonction logique 3-variables en 1 cycle
// Ch(x,y,z)  = (x&y)^(~x&z)      LUT = 0xCA  (vérifié par table de vérité)
// Maj(x,y,z) = (x&y)^(x&z)^(y&z) LUT = 0xE8  (vérifié par table de vérité)
__device__ __forceinline__ uint64_t SHA512_CH(uint64_t x, uint64_t y, uint64_t z) {
    uint32_t r_lo, r_hi;
    asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(r_lo) : "r"((uint32_t)x),       "r"((uint32_t)y),       "r"((uint32_t)z));
    asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(r_hi) : "r"((uint32_t)(x>>32)), "r"((uint32_t)(y>>32)), "r"((uint32_t)(z>>32)));
    return ((uint64_t)r_hi << 32) | r_lo;
}
__device__ __forceinline__ uint64_t SHA512_MAJ(uint64_t x, uint64_t y, uint64_t z) {
    uint32_t r_lo, r_hi;
    asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r_lo) : "r"((uint32_t)x),       "r"((uint32_t)y),       "r"((uint32_t)z));
    asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r_hi) : "r"((uint32_t)(x>>32)), "r"((uint32_t)(y>>32)), "r"((uint32_t)(z>>32)));
    return ((uint64_t)r_hi << 32) | r_lo;
}
#define SHA512_SUM0(x)      (rotr64((x), 28) ^ rotr64((x), 34) ^ rotr64((x), 39))
#define SHA512_SUM1(x)      (rotr64((x), 14) ^ rotr64((x), 18) ^ rotr64((x), 41))
#define SHA512_SIG0(x)      (rotr64((x), 1)  ^ rotr64((x), 8)  ^ SHA512_SHR((x), 7))
#define SHA512_SIG1(x)      (rotr64((x), 19) ^ rotr64((x), 61) ^ SHA512_SHR((x), 6))

static __constant__ uint64_t K_SHA512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

struct Hydra_SHA512_CTX {
    uint64_t h[8];
    uint8_t  buffer[SHA512_BLOCK_SIZE];
    uint32_t buffer_len;
    uint64_t total_len;
};

// ============================================================================
// 2. HELPER FUNCTIONS
// ============================================================================

// sha512_load_be64 PTX : __byte_perm = bswap32 natif (1 instruction)
__device__ __forceinline__ uint64_t sha512_load_be64(const uint8_t *p) {
    uint32_t hi, lo;
    __builtin_memcpy(&hi, p,   4);
    __builtin_memcpy(&lo, p+4, 4);
    return ((uint64_t)__byte_perm(hi, 0, 0x0123) << 32)
         |  (uint64_t)__byte_perm(lo, 0, 0x0123);
}

// sha512_store_be64 PTX : __byte_perm = bswap32 natif (1 instruction)
__device__ __forceinline__ void sha512_store_be64(uint8_t *p, uint64_t v) {
    uint32_t hi = __byte_perm((uint32_t)(v >> 32), 0, 0x0123);
    uint32_t lo = __byte_perm((uint32_t)(v),       0, 0x0123);
    __builtin_memcpy(p,   &hi, 4);
    __builtin_memcpy(p+4, &lo, 4);
}

// ============================================================================
// 3. SHA-512 CORE OPTIMISÉ (TAMPON ROULANT)
// ============================================================================

__device__ __noinline__ void sha512_transform(Hydra_SHA512_CTX *ctx, const uint8_t block[SHA512_BLOCK_SIZE]) {
    // TAMPON ROULANT : 16 mots au lieu de 80 -> Gain de 512 octets de registres
    uint64_t w[16];
    
    // Chargement initial
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        w[i] = sha512_load_be64(block + 8 * i);
    }
    
    uint64_t a = ctx->h[0], b = ctx->h[1], c = ctx->h[2], d = ctx->h[3];
    uint64_t e = ctx->h[4], f = ctx->h[5], g = ctx->h[6], h = ctx->h[7];
    
    // Boucle principale sans déroulement excessif
    #pragma unroll 1 
    for (int i = 0; i < 80; ++i) {
        uint64_t wi;
        
        // Calcul du message schedule à la volée avec indexation circulaire
        if (i < 16) {
            wi = w[i];
        } else {
            // w[t] = SIG1(w[t-2]) + w[t-7] + SIG0(w[t-15]) + w[t-16]
            // On utilise les masques & 15 pour le modulo 16
            int idx = i & 15;
            uint64_t s1 = SHA512_SIG1(w[(i + 14) & 15]); // i-2
            uint64_t s0 = SHA512_SIG0(w[(i + 1)  & 15]); // i-15
            wi = s1 + w[(i + 9) & 15] + s0 + w[idx];     // i-7, i-16 (qui est w[idx] avant maj)
            w[idx] = wi; // Mise à jour du buffer circulaire
        }
        
        uint64_t t1 = h + SHA512_SUM1(e) + SHA512_CH(e, f, g) + K_SHA512[i] + wi;
        uint64_t t2 = SHA512_SUM0(a) + SHA512_MAJ(a, b, c);
        
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    
    ctx->h[0] += a; ctx->h[1] += b; ctx->h[2] += c; ctx->h[3] += d;
    ctx->h[4] += e; ctx->h[5] += f; ctx->h[6] += g; ctx->h[7] += h;
}

// sha512_transform_state_hot : version générique INLINE, tous contextes
// unroll 8 = sweet spot registres/IPC pour SHA512 80 rounds
__device__ __forceinline__ void sha512_transform_state_hot(uint64_t H[8], const uint64_t w_in[16]) {
    uint64_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = w_in[i];

    uint64_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint64_t e = H[4], f = H[5], g = H[6], h = H[7];

    #pragma unroll 8
    for (int i = 0; i < 80; ++i) {
        uint64_t wi;
        if (i < 16) { wi = w[i]; }
        else {
            int idx = i & 15;
            uint64_t s1 = SHA512_SIG1(w[(i + 14) & 15]);
            uint64_t s0 = SHA512_SIG0(w[(i + 1)  & 15]);
            wi = s1 + w[(i + 9) & 15] + s0 + w[idx];
            w[idx] = wi;
        }
        uint64_t t1 = h + SHA512_SUM1(e) + SHA512_CH(e, f, g) + K_SHA512[i] + wi;
        uint64_t t2 = SHA512_SUM0(a) + SHA512_MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

// sha512_compress_pbkdf2_hot : version OPTIMISÉE avec padding precomputation
// UNIQUEMENT valide quand w[8..15] = {0x8000..., 0,0,0,0,0,0, 1536}
// c'est-à-dire les blocs PBKDF2 de 64 bytes (U → compress_state)
__device__ __forceinline__ void sha512_compress_pbkdf2_hot(uint64_t H[8], const uint64_t w_in[16]) {
    uint64_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = w_in[i];

    uint64_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint64_t e = H[4], f = H[5], g = H[6], h = H[7];

    #define SHA512_ROUND_P(wi_val, k_val) do { \
        uint64_t t1 = h + SHA512_SUM1(e) + SHA512_CH(e, f, g) + (k_val) + (wi_val); \
        uint64_t t2 = SHA512_SUM0(a) + SHA512_MAJ(a, b, c); \
        h = g; g = f; f = e; e = d + t1; \
        d = c; c = b; b = a; a = t1 + t2; \
    } while(0)

    // Tours 0-15
    #pragma unroll
    for (int i = 0; i < 16; ++i) { SHA512_ROUND_P(w[i], K_SHA512[i]); }

    // Tours 16-23 : padding precomputation
    // Valide car w[8..15] = {0x8000..., 0,0,0,0,0,0, 1536} garanti par l'appelant
    uint64_t wi;
    wi = SHA512_SIG0(w[1]) + w[0];                                               // Tour 16 : SIG1(0)=0, W[9]=0
    w[0] = wi; SHA512_ROUND_P(wi, K_SHA512[16]);
    wi = SHA512_SIG1(1536ULL) + SHA512_SIG0(w[2]) + w[1];                       // Tour 17 : SIG1(1536)=cst, W[10]=0
    w[1] = wi; SHA512_ROUND_P(wi, K_SHA512[17]);
    wi = SHA512_SIG1(w[0]) + SHA512_SIG0(w[3]) + w[2];                          // Tour 18 : W[11]=0
    w[2] = wi; SHA512_ROUND_P(wi, K_SHA512[18]);
    wi = SHA512_SIG1(w[1]) + SHA512_SIG0(w[4]) + w[3];                          // Tour 19 : W[12]=0
    w[3] = wi; SHA512_ROUND_P(wi, K_SHA512[19]);
    wi = SHA512_SIG1(w[2]) + SHA512_SIG0(w[5]) + w[4];                          // Tour 20 : W[13]=0
    w[4] = wi; SHA512_ROUND_P(wi, K_SHA512[20]);
    wi = SHA512_SIG1(w[3]) + SHA512_SIG0(w[6]) + w[5];                          // Tour 21 : W[14]=0
    w[5] = wi; SHA512_ROUND_P(wi, K_SHA512[21]);
    wi = SHA512_SIG1(w[4]) + 1536ULL + SHA512_SIG0(w[7]) + w[6];               // Tour 22 : W[15]=1536
    w[6] = wi; SHA512_ROUND_P(wi, K_SHA512[22]);
    wi = SHA512_SIG1(w[5]) + w[0] + SHA512_SIG0(0x8000000000000000ULL) + w[7]; // Tour 23 : SIG0(0x8000...)=cst
    w[7] = wi; SHA512_ROUND_P(wi, K_SHA512[23]);

    // Tours 24-79
    #pragma unroll 8
    for (int i = 24; i < 80; ++i) {
        int idx = i & 15;
        uint64_t s1 = SHA512_SIG1(w[(i + 14) & 15]);
        uint64_t s0 = SHA512_SIG0(w[(i + 1)  & 15]);
        wi = s1 + w[(i + 9) & 15] + s0 + w[idx];
        w[idx] = wi;
        SHA512_ROUND_P(wi, K_SHA512[i]);
    }

    #undef SHA512_ROUND_P

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

// sha512_transform_state : wrapper __noinline__ pour les appels hors boucle chaude
__device__ __noinline__ void sha512_transform_state(uint64_t H[8], const uint64_t w_in[16]) {
    sha512_transform_state_hot(H, w_in);
}

__device__ __forceinline__ void sha512_init(Hydra_SHA512_CTX *ctx) {
    ctx->h[0] = 0x6a09e667f3bcc908ULL; ctx->h[1] = 0xbb67ae8584caa73bULL;
    ctx->h[2] = 0x3c6ef372fe94f82bULL; ctx->h[3] = 0xa54ff53a5f1d36f1ULL;
    ctx->h[4] = 0x510e527fade682d1ULL; ctx->h[5] = 0x9b05688c2b3e6c1fULL;
    ctx->h[6] = 0x1f83d9abfb41bd6bULL; ctx->h[7] = 0x5be0cd19137e2179ULL;
    ctx->buffer_len = 0; ctx->total_len = 0;
}

__device__ void sha512_update(Hydra_SHA512_CTX *ctx, const uint8_t *data, uint32_t len) {
    if (len == 0) return;
    
    if (ctx->buffer_len > 0) {
        uint32_t space = SHA512_BLOCK_SIZE - ctx->buffer_len;
        uint32_t to_copy = (len < space) ? len : space;
        
        #pragma unroll
        for (uint32_t i = 0; i < to_copy; ++i) ctx->buffer[ctx->buffer_len + i] = data[i];
        
        ctx->buffer_len += to_copy;
        data += to_copy;
        len -= to_copy;
        
        if (ctx->buffer_len == SHA512_BLOCK_SIZE) {
            sha512_transform(ctx, ctx->buffer);
            ctx->total_len += SHA512_BLOCK_SIZE;
            ctx->buffer_len = 0;
        }
    }
    
    while (len >= SHA512_BLOCK_SIZE) {
        sha512_transform(ctx, data);
        ctx->total_len += SHA512_BLOCK_SIZE;
        data += SHA512_BLOCK_SIZE;
        len -= SHA512_BLOCK_SIZE;
    }
    
    if (len > 0) {
        #pragma unroll
        for (uint32_t i = 0; i < len; ++i) ctx->buffer[i] = data[i];
        ctx->buffer_len = len;
    }
}

__device__ void sha512_final(Hydra_SHA512_CTX *ctx, uint8_t out[SHA512_DIGEST_SIZE]) {
    uint64_t total_bits = (ctx->total_len + ctx->buffer_len) * 8ULL;
    
    ctx->buffer[ctx->buffer_len++] = 0x80;
    
    if (ctx->buffer_len > (SHA512_BLOCK_SIZE - 16)) {
        while (ctx->buffer_len < SHA512_BLOCK_SIZE) ctx->buffer[ctx->buffer_len++] = 0x00;
        sha512_transform(ctx, ctx->buffer);
        ctx->buffer_len = 0;
    }
    
    while (ctx->buffer_len < (SHA512_BLOCK_SIZE - 16)) ctx->buffer[ctx->buffer_len++] = 0x00;
    
    uint8_t *p = ctx->buffer + SHA512_BLOCK_SIZE - 16;
    for (int i = 0; i < 8; ++i) p[i] = 0;
    p += 8;
    sha512_store_be64(p, total_bits);
    
    sha512_transform(ctx, ctx->buffer);
    
    #pragma unroll
    for (int i = 0; i < 8; ++i) sha512_store_be64(out + 8*i, ctx->h[i]);
}

__device__ __forceinline__ void sha512_hash(const uint8_t *data, uint32_t len, uint8_t out[SHA512_DIGEST_SIZE]) {
    Hydra_SHA512_CTX ctx;
    sha512_init(&ctx);
    sha512_update(&ctx, data, len);
    sha512_final(&ctx, out);
}

// ============================================================================
// 4. HMAC & PBKDF2 (OPTIMISÉS)
// ============================================================================

__device__ void hmac_sha512_prepare_keys(const uint8_t *key, uint32_t key_len,
                                         uint8_t k_ipad[SHA512_BLOCK_SIZE],
                                         uint8_t k_opad[SHA512_BLOCK_SIZE]) {
    uint8_t key_block[SHA512_BLOCK_SIZE];
    #pragma unroll
    for (int i = 0; i < SHA512_BLOCK_SIZE; ++i) key_block[i] = 0;
    
    if (key_len > SHA512_BLOCK_SIZE) {
        uint8_t hashed[SHA512_DIGEST_SIZE];
        sha512_hash(key, key_len, hashed);
        #pragma unroll
        for (int i = 0; i < SHA512_DIGEST_SIZE; ++i) key_block[i] = hashed[i];
    } else {
        #pragma unroll
        for (uint32_t i = 0; i < key_len; ++i) key_block[i] = key[i];
    }
    
    #pragma unroll
    for (int i = 0; i < SHA512_BLOCK_SIZE; ++i) {
        k_ipad[i] = key_block[i] ^ 0x36;
        k_opad[i] = key_block[i] ^ 0x5c;
    }
}

__device__ void hmac_sha512_with_prepared_keys(const uint8_t k_ipad[SHA512_BLOCK_SIZE],
                                               const uint8_t k_opad[SHA512_BLOCK_SIZE],
                                               const uint8_t *msg, uint32_t msg_len,
                                               uint8_t out[SHA512_DIGEST_SIZE]) {
    Hydra_SHA512_CTX ctx;
    uint8_t inner[SHA512_DIGEST_SIZE];
    
    sha512_init(&ctx);
    sha512_update(&ctx, k_ipad, SHA512_BLOCK_SIZE);
    if(msg_len > 0) sha512_update(&ctx, msg, msg_len);
    sha512_final(&ctx, inner);
    
    sha512_init(&ctx);
    sha512_update(&ctx, k_opad, SHA512_BLOCK_SIZE);
    sha512_update(&ctx, inner, SHA512_DIGEST_SIZE);
    sha512_final(&ctx, out);
}

__device__ void hmac_sha512(const uint8_t *key, uint32_t key_len,
                           const uint8_t *msg, uint32_t msg_len,
                           uint8_t out[SHA512_DIGEST_SIZE]) {
    uint8_t k_ipad[SHA512_BLOCK_SIZE];
    uint8_t k_opad[SHA512_BLOCK_SIZE];
    hmac_sha512_prepare_keys(key, key_len, k_ipad, k_opad);
    hmac_sha512_with_prepared_keys(k_ipad, k_opad, msg, msg_len, out);
}

// ============================================================================
// sha512_compress_state : cœur du bypass
//
// Équivaut à : sha512_update(ctx_base, msg64, 64) + sha512_final(out, 64)
// MAIS sans copie de struct ni gestion de buffer.
//
// Prérequis :
//   - H_base[8] = état SHA512 après avoir processé exactement 128 bytes (ipad ou opad)
//   - msg[8]    = message de 64 bytes en uint64_t big-endian
//
// Résultat dans H_out[8].
//
// Pourquoi ça marche :
//   total_len = 128 (déjà processé) + 64 (message) = 192 bytes = 1536 bits
//   Le message (64 bytes = 8 mots) tient dans W[0..7].
//   Le padding occupe W[8..15] : W[8]=0x8000..., W[9..14]=0, W[15]=1536.
//   C'est exactement ce que sha512_final calculerait — on le forge directement.
// ============================================================================
__device__ __forceinline__ void sha512_compress_state(
    const uint64_t H_base[8],
    const uint64_t msg[8],        // 64 bytes en big-endian uint64_t
    uint64_t       H_out[8])
{
    uint64_t W[16];
    #pragma unroll
    for (int i = 0; i < 8;  i++) W[i] = msg[i];
    W[8]  = 0x8000000000000000ULL;
    W[9]  = 0; W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0; W[14] = 0;
    W[15] = 1536ULL;   // (128 + 64) * 8 bits

    #pragma unroll
    for (int i = 0; i < 8; i++) H_out[i] = H_base[i];
    sha512_compress_pbkdf2_hot(H_out, W);  // w[8..15] = {0x8000...,0×6,1536} garanti ici
}
// ============================================================================
// sha512_compress_state_37 : bypass pour msg de exactement 37 bytes
//
// Cas BIP32 derive child : data = 0x00||priv[32]||index[4] = 37 bytes
// total_len = 128 (H_base) + 37 = 165 bytes = 1320 bits
//
// W[0..3] = msg[0..31]   (4 mots complets)
// W[4]    = msg[32..36] || 0x80 || 0x00 || 0x00  (5 bytes msg + padding)
// W[5..14]= 0
// W[15]   = 1320
// ============================================================================
__device__ __forceinline__ void sha512_compress_state_37(
    const uint64_t H_base[8],
    const uint8_t  msg[37],   // exactement 37 bytes
    uint64_t       H_out[8])
{
    uint64_t W[16];

    // W[0..3] : 4 mots complets (32 bytes)
    #pragma unroll
    for (int i = 0; i < 4; i++) W[i] = sha512_load_be64(msg + i*8);

    // W[4] : 5 bytes restants + 0x80 + 2 zéros
    W[4] = ((uint64_t)msg[32] << 56)
         | ((uint64_t)msg[33] << 48)
         | ((uint64_t)msg[34] << 40)
         | ((uint64_t)msg[35] << 32)
         | ((uint64_t)msg[36] << 24)
         | 0x0000000000800000ULL;  // 0x80 en position byte 5, zéros après

    // W[5..14] : zéros
    #pragma unroll
    for (int i = 5; i < 15; i++) W[i] = 0ULL;

    // W[15] : longueur totale en bits = (128 + 37) * 8 = 1320
    W[15] = 1320ULL;

    #pragma unroll
    for (int i = 0; i < 8; i++) H_out[i] = H_base[i];
    sha512_transform_state_hot(H_out, W);
}

// ============================================================================
// hmac_sha512_bip32 : HMAC-SHA512 optimisé pour BIP32
//
// Remplace hmac_sha512(chain[32], msg[37], out) dans les dérivations BIP32.
// Utilise sha512_compress_state_37 et sha512_compress_state (existant).
//
// Pipeline bypass complet — zéro Hydra_SHA512_CTX sur le stack, zéro buffer :
//   1. Précalcul H_ipad, H_opad depuis chain[32]
//   2. inner = sha512_compress_state_37(H_ipad, msg)   ← 1 transform
//   3. out   = sha512_compress_state(H_opad, inner)    ← 1 transform
//      (inner est 64 bytes → sha512_compress_state existant)
//
// Note : chain[32] est ≤ 128 bytes → pas de hachage préalable nécessaire.
// ============================================================================
__device__ __forceinline__ void hmac_sha512_bip32(
    const uint8_t chain[32],   // clé HMAC = parent chain code
    const uint8_t msg[37],     // data BIP32 = 0x00||priv||index ou pubkey||index
    uint8_t       out[64])
{
    // Précalcul H_ipad, H_opad :
    // chain[32] padded à 128 bytes avec zéros → XOR 0x36 / 0x5C
    // Puis SHA512 absorbe ce bloc de 128 bytes → état H après 1 transform
    //
    // On forge directement le bloc W[16] de k_ipad/k_opad sans buffer byte[] :
    // W[i] = (chain_word[i] ^ 0x3636363636363636) pour i < 4
    //        (0x3636363636363636)                  pour i >= 4  (padding zéro XOR 0x36)

    uint64_t H_ipad[8], H_opad[8];

    // Charger chain[32] en 4 mots uint64_t big-endian
    uint64_t c[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) c[i] = sha512_load_be64(chain + i*8);

    // Forger et absorber k_ipad (chain XOR 0x36, puis 0x36...36 pour les 96 bytes restants)
    {
        uint64_t W[16];
        const uint64_t pad36 = 0x3636363636363636ULL;
        const uint64_t pad5c = 0x5c5c5c5c5c5c5c5cULL;

        #pragma unroll
        for (int i = 0; i < 4; i++) W[i] = c[i] ^ pad36;
        #pragma unroll
        for (int i = 4; i < 16; i++) W[i] = pad36;

        // H_ipad = SHA512_init + 1 transform(W_ipad)
        H_ipad[0] = 0x6a09e667f3bcc908ULL; H_ipad[1] = 0xbb67ae8584caa73bULL;
        H_ipad[2] = 0x3c6ef372fe94f82bULL; H_ipad[3] = 0xa54ff53a5f1d36f1ULL;
        H_ipad[4] = 0x510e527fade682d1ULL; H_ipad[5] = 0x9b05688c2b3e6c1fULL;
        H_ipad[6] = 0x1f83d9abfb41bd6bULL; H_ipad[7] = 0x5be0cd19137e2179ULL;
        sha512_transform_state_hot(H_ipad, W);

        // H_opad = SHA512_init + 1 transform(W_opad)
        #pragma unroll
        for (int i = 0; i < 4; i++) W[i] = c[i] ^ pad5c;
        #pragma unroll
        for (int i = 4; i < 16; i++) W[i] = pad5c;

        H_opad[0] = 0x6a09e667f3bcc908ULL; H_opad[1] = 0xbb67ae8584caa73bULL;
        H_opad[2] = 0x3c6ef372fe94f82bULL; H_opad[3] = 0xa54ff53a5f1d36f1ULL;
        H_opad[4] = 0x510e527fade682d1ULL; H_opad[5] = 0x9b05688c2b3e6c1fULL;
        H_opad[6] = 0x1f83d9abfb41bd6bULL; H_opad[7] = 0x5be0cd19137e2179ULL;
        sha512_transform_state_hot(H_opad, W);
    }

    // inner = SHA512(H_ipad || msg[37])
    uint64_t H_inner[8];
    sha512_compress_state_37(H_ipad, msg, H_inner);

    // out = SHA512(H_opad || inner[64])  — inner est 64 bytes → sha512_compress_state
    sha512_compress_state(H_opad, H_inner, H_inner);

    // Sérialiser H_inner → out[64]
    #pragma unroll
    for (int i = 0; i < 8; i++) sha512_store_be64(out + i*8, H_inner[i]);
}

// ============================================================================
// hmac_sha512_bip32_master : HMAC-SHA512 pour BIP32 master key
//
// Remplace hmac_sha512("Bitcoin seed"[12], seed[64], out).
// La clé "Bitcoin seed" est fixe → H_ipad et H_opad sont des constantes
// précalculées. Zéro calcul de key schedule à l'exécution.
//
// Précalcul offline de H_ipad et H_opad pour key = "Bitcoin seed" (12 bytes) :
//   k_ipad = "Bitcoin seed" || 0x00×116 XOR 0x36×128
//   k_opad = "Bitcoin seed" || 0x00×116 XOR 0x5C×128
// ============================================================================

// États SHA512 après absorption de k_ipad et k_opad pour "Bitcoin seed"
// Précalculés une fois pour toutes (valeurs vérifiées par vecteur de test BIP32)
// SHA512 state after absorbing k_ipad = ("Bitcoin seed" || 0x00×116) XOR 0x36×128
// Verified : HMAC("Bitcoin seed", abandon×11 about seed) → correct master_priv
__device__ __constant__ uint64_t BIP32_MASTER_H_IPAD[8] = {
    0x2e2af459060c1873ULL, 0x7894b868dc88433aULL,
    0xdd1a797ef1a1933aULL, 0xe6486d04fcb412a7ULL,
    0xfbcc67b9a396caa0ULL, 0xa2970b146f49b65eULL,
    0xfdf1daabc66f6248ULL, 0x2ff99c812ada6dc3ULL,
};
// SHA512 state after absorbing k_opad = ("Bitcoin seed" || 0x00×116) XOR 0x5C×128
__device__ __constant__ uint64_t BIP32_MASTER_H_OPAD[8] = {
    0xbbd27bac212e9dbdULL, 0xdd0bc55e7e4037c1ULL,
    0xdfdd3d6890bd6424ULL, 0x2902de663032b34cULL,
    0xa30f8aa6f67899fcULL, 0x69a566c30f88378fULL,
    0x0500247985ecb694ULL, 0xf6d70307c6b2d337ULL,
};

__device__ __forceinline__ void hmac_sha512_bip32_master(
    const uint8_t seed[64],   // BIP32 seed (output de PBKDF2)
    uint8_t       out[64])
{
    // inner = SHA512(H_ipad_const || seed[64])  — seed est 64 bytes exact
    uint64_t seed_w[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) seed_w[i] = sha512_load_be64(seed + i*8);

    uint64_t H_inner[8];
    sha512_compress_state(BIP32_MASTER_H_IPAD, seed_w, H_inner);

    // out = SHA512(H_opad_const || inner[64])
    sha512_compress_state(BIP32_MASTER_H_OPAD, H_inner, H_inner);

    #pragma unroll
    for (int i = 0; i < 8; i++) sha512_store_be64(out + i*8, H_inner[i]);
}




// ============================================================================
// sha512_compress_state_variable : bypass pour msg de longueur variable
//
// Équivaut à : sha512_update(ctx_base, msg, msg_len) + sha512_final(H_out)
// SANS jamais allouer Hydra_SHA512_CTX sur le stack.
//
// Prérequis :
//   - H_base[8] = état SHA512 après avoir processé exactement 128 bytes
//   - msg[msg_len] avec msg_len <= 111 bytes  (tient dans 1 bloc avec padding)
//   - total_len_before = 128 (bytes déjà absorbés par H_base)
//
// Le message + padding + longueur tient dans un seul bloc de 128 bytes car :
//   msg_len <= 111 → msg + 0x80 + zéros + 16-byte-length ≤ 128 bytes
//
// Pour BIP39 PBKDF2 :
//   msg = salt[salt_len] || 0x00000001
//   salt = "mnemonic" + passphrase (ou "mnemonic" seul pour passphrase vide)
//   msg_len = salt_len + 4  ≤ 100 + 4 = 104 → toujours ≤ 111 ✓
// ============================================================================
__device__ __forceinline__ void sha512_compress_state_variable(
    const uint64_t H_base[8],
    const uint8_t* msg,
    uint32_t       msg_len,   // <= 111 bytes
    uint64_t       H_out[8])
{
    // total_len = 128 (H_base) + msg_len
    const uint64_t total_bits = (uint64_t)(128 + msg_len) * 8ULL;

    // Forger W[16] en assemblant msg byte par byte dans un buffer 128B temporaire
    // puis charger chaque mot en big-endian
    uint8_t blk[128];

    // Copier msg dans blk
    #pragma unroll 1
    for (uint32_t i = 0; i < msg_len; i++) blk[i] = msg[i];

    // Padding : 0x80 immédiatement après le message
    blk[msg_len] = 0x80;

    // Zéros jusqu'à blk[111] inclus (les 16 derniers bytes sont réservés à la longueur)
    #pragma unroll 1
    for (uint32_t i = msg_len + 1; i < 112; i++) blk[i] = 0x00;

    // SHA-512 utilise un champ de longueur de 128 bits (16 bytes), big-endian
    // Les 8 premiers bytes (longueur haute) sont toujours 0 pour nous
    blk[112] = 0; blk[113] = 0; blk[114] = 0; blk[115] = 0;
    blk[116] = 0; blk[117] = 0; blk[118] = 0; blk[119] = 0;

    // Les 8 bytes bas = total_bits en big-endian
    blk[120] = (uint8_t)(total_bits >> 56);
    blk[121] = (uint8_t)(total_bits >> 48);
    blk[122] = (uint8_t)(total_bits >> 40);
    blk[123] = (uint8_t)(total_bits >> 32);
    blk[124] = (uint8_t)(total_bits >> 24);
    blk[125] = (uint8_t)(total_bits >> 16);
    blk[126] = (uint8_t)(total_bits >>  8);
    blk[127] = (uint8_t)(total_bits      );

    // Charger W[16] depuis blk en big-endian
    uint64_t W[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) W[i] = sha512_load_be64(blk + i*8);

    // Lancer le compress
    #pragma unroll
    for (int i = 0; i < 8; i++) H_out[i] = H_base[i];
    sha512_transform_state_hot(H_out, W);
}

// ============================================================================
// PBKDF2 SPLIT — pipeline multi-kernel
//
// pbkdf2_setup()   : ~50 reg — mnemonic → H_ipad[8] + H_opad[8] + U_first[8]
// pbkdf2_iterate() : ~80 reg — boucle 2..2048 bypass pur → seed[64]
// ============================================================================

// ============================================================================
// pbkdf2_hmac_key_states_bip39 : bypass complet pour H_ipad/H_opad
//
// Remplace hmac_sha512_prepare_keys + sha512_update(k_ipad/k_opad).
// Zéro k_ipad[128], zéro k_opad[128], zéro Hydra_SHA512_CTX.
//
// Pour BIP39 : password = mnemonic, pwd_len ≤ 215 bytes (24 mots × 9 chars)
//   → key_block = password (pas de pré-hachage car pwd_len ≤ 128... FAUX,
//     un mnemonic 24 mots PEUT dépasser 128 bytes !)
//
// Cas pwd_len ≤ 128 : key_block = password padded avec 0 → 128 bytes
//   W_ipad[i] = key_block_word[i] XOR 0x3636...  (i < nw), 0x3636... sinon
//   H_ipad = SHA512_init + transform(W_ipad)  — 1 seule transform, 0 bytes stack
//
// Cas pwd_len > 128 : key_block = SHA512(password), 64 bytes → padded à 128
//   Nécessite sha512_hash(password) d'abord (1 ou 2 transforms selon taille)
//   puis forge W_ipad/W_opad depuis les 64 bytes du hash
//
// Pour les mnemonics 12 mots (47-95 bytes) : toujours ≤ 128 → 1 transform
// Pour les mnemonics 24 mots (103-215 bytes) : peut dépasser 128 → branche SHA512
// ============================================================================
__device__ __forceinline__ void pbkdf2_hmac_key_states_bip39(
    const uint8_t* password, uint32_t pwd_len,
    uint64_t H_ipad_out[8],
    uint64_t H_opad_out[8])
{
    const uint64_t PAD36 = 0x3636363636363636ULL;
    const uint64_t PAD5C = 0x5c5c5c5c5c5c5c5cULL;

    if (pwd_len <= 128) {
        // Cas direct : password tient dans 1 bloc → forge W[16] en uint64_t
        // key_block = password[pwd_len] || 0x00...(128-pwd_len)
        // W_ipad[i] = key_block_be64[i] XOR 0x3636...
        // On charge pwd en big-endian mot par mot, en masquant les bytes hors pwd_len

        uint64_t W[16];

        // Nombre de mots complets
        uint32_t full_words = pwd_len / 8;
        uint32_t tail       = pwd_len % 8;

        // Mots complets : charger directement depuis password
        #pragma unroll 1
        for (uint32_t wi = 0; wi < full_words; wi++) {
            W[wi] = sha512_load_be64(password + wi*8) ^ PAD36;
        }

        // Mot partiel (si tail > 0) : assembler byte par byte
        if (tail > 0 && full_words < 16) {
            uint64_t word = 0;
            const uint8_t* p = password + full_words*8;
            #pragma unroll 1
            for (uint32_t b = 0; b < tail; b++)
                word |= ((uint64_t)p[b] << (56 - b*8));
            W[full_words] = word ^ PAD36;
        }

        // Mots restants : pure constante 0x36 (zéros XOR 0x36)
        #pragma unroll 1
        for (uint32_t wi = (tail > 0 ? full_words + 1 : full_words); wi < 16; wi++)
            W[wi] = PAD36;

        // H_ipad = SHA512_init + transform(W_ipad)
        H_ipad_out[0] = 0x6a09e667f3bcc908ULL; H_ipad_out[1] = 0xbb67ae8584caa73bULL;
        H_ipad_out[2] = 0x3c6ef372fe94f82bULL; H_ipad_out[3] = 0xa54ff53a5f1d36f1ULL;
        H_ipad_out[4] = 0x510e527fade682d1ULL; H_ipad_out[5] = 0x9b05688c2b3e6c1fULL;
        H_ipad_out[6] = 0x1f83d9abfb41bd6bULL; H_ipad_out[7] = 0x5be0cd19137e2179ULL;
        sha512_transform_state_hot(H_ipad_out, W);

        // W_opad = même chose avec PAD5C
        #pragma unroll 1
        for (uint32_t wi = 0; wi < full_words; wi++)
            W[wi] = sha512_load_be64(password + wi*8) ^ PAD5C;
        if (tail > 0 && full_words < 16) {
            uint64_t word = 0;
            const uint8_t* p = password + full_words*8;
            #pragma unroll 1
            for (uint32_t b = 0; b < tail; b++)
                word |= ((uint64_t)p[b] << (56 - b*8));
            W[full_words] = word ^ PAD5C;
        }
        #pragma unroll 1
        for (uint32_t wi = (tail > 0 ? full_words + 1 : full_words); wi < 16; wi++)
            W[wi] = PAD5C;

        H_opad_out[0] = 0x6a09e667f3bcc908ULL; H_opad_out[1] = 0xbb67ae8584caa73bULL;
        H_opad_out[2] = 0x3c6ef372fe94f82bULL; H_opad_out[3] = 0xa54ff53a5f1d36f1ULL;
        H_opad_out[4] = 0x510e527fade682d1ULL; H_opad_out[5] = 0x9b05688c2b3e6c1fULL;
        H_opad_out[6] = 0x1f83d9abfb41bd6bULL; H_opad_out[7] = 0x5be0cd19137e2179ULL;
        sha512_transform_state_hot(H_opad_out, W);

    } else {
        // Cas mnemonic long (24 mots > 128 bytes) :
        // key_block = SHA512(password)[64], padded à 128 bytes avec zéros
        // On forge W_ipad/W_opad depuis ces 64 bytes (8 mots, les 8 suivants = PAD36/5C)
        uint8_t hashed[64];
        sha512_hash(password, pwd_len, hashed);  // sha512_hash utilise Hydra_SHA512_CTX
                                                  // mais c'est __noinline__ → stack séparé
                                                  // acceptable : chemin rare (24 mots longs)

        uint64_t W[16];
        #pragma unroll
        for (int i = 0; i < 8; i++) W[i] = sha512_load_be64(hashed + i*8) ^ PAD36;
        #pragma unroll
        for (int i = 8; i < 16; i++) W[i] = PAD36;

        H_ipad_out[0] = 0x6a09e667f3bcc908ULL; H_ipad_out[1] = 0xbb67ae8584caa73bULL;
        H_ipad_out[2] = 0x3c6ef372fe94f82bULL; H_ipad_out[3] = 0xa54ff53a5f1d36f1ULL;
        H_ipad_out[4] = 0x510e527fade682d1ULL; H_ipad_out[5] = 0x9b05688c2b3e6c1fULL;
        H_ipad_out[6] = 0x1f83d9abfb41bd6bULL; H_ipad_out[7] = 0x5be0cd19137e2179ULL;
        sha512_transform_state_hot(H_ipad_out, W);

        #pragma unroll
        for (int i = 0; i < 8; i++) W[i] = sha512_load_be64(hashed + i*8) ^ PAD5C;
        #pragma unroll
        for (int i = 8; i < 16; i++) W[i] = PAD5C;

        H_opad_out[0] = 0x6a09e667f3bcc908ULL; H_opad_out[1] = 0xbb67ae8584caa73bULL;
        H_opad_out[2] = 0x3c6ef372fe94f82bULL; H_opad_out[3] = 0xa54ff53a5f1d36f1ULL;
        H_opad_out[4] = 0x510e527fade682d1ULL; H_opad_out[5] = 0x9b05688c2b3e6c1fULL;
        H_opad_out[6] = 0x1f83d9abfb41bd6bULL; H_opad_out[7] = 0x5be0cd19137e2179ULL;
        sha512_transform_state_hot(H_opad_out, W);
    }
}

__device__ void pbkdf2_setup(
    const uint8_t* password, uint32_t pwd_len,
    const uint8_t* salt,     uint32_t salt_len,
    uint64_t H_ipad_out[8],
    uint64_t H_opad_out[8],
    uint64_t U_out[8])
{
    // H_ipad / H_opad : bypass total, zéro Hydra_SHA512_CTX, zéro k_ipad[128]/k_opad[128]
    pbkdf2_hmac_key_states_bip39(password, pwd_len, H_ipad_out, H_opad_out);

    // Itération 1 : salt || 0x00000001 — bypass via sha512_compress_state_variable
    uint8_t msg_buf[116];  // salt(≤107) + counter(4) + marge
    uint32_t pos = 0;
    for(uint32_t i = 0; i < salt_len; i++) msg_buf[pos++] = salt[i];
    msg_buf[pos++] = 0; msg_buf[pos++] = 0; msg_buf[pos++] = 0; msg_buf[pos++] = 1;

    sha512_compress_state_variable(H_ipad_out, msg_buf, pos, U_out);
    sha512_compress_state(H_opad_out, U_out, U_out);
}

__device__ void pbkdf2_iterate(
    const uint64_t H_ipad[8],
    const uint64_t H_opad[8],
    const uint64_t U_first[8],
    uint8_t        seed_out[64])
{
    uint64_t U[8], T[8];
    #pragma unroll
    for(int i = 0; i < 8; i++) { U[i] = U_first[i]; T[i] = U_first[i]; }

    // Boucle chaude : bypass pur, ~80 registres
    for(uint32_t iter = 2; iter <= 2048; ++iter) {
        uint64_t H_inner[8];
        sha512_compress_state(H_ipad, U, H_inner);
        sha512_compress_state(H_opad, H_inner, U);
        #pragma unroll
        for(int i = 0; i < 8; i++) T[i] ^= U[i];
    }

    #pragma unroll
    for(int i = 0; i < 8; i++) sha512_store_be64(seed_out + i*8, T[i]);
}

// ============================================================================
// PBKDF2 PASSPHRASE — fonctions spécialisées pour le mode passphrase
//
// En mode passphrase :
//   password = mnemonic (fixe pour toute la session)
//   salt     = "mnemonic" + passphrase (variable par thread)
//
// On précalcule H_ipad_pw[8] + H_opad_pw[8] une seule fois depuis le password,
// puis chaque thread ne fait que :
//   1. pbkdf2_first_iter_pass(H_ipad_pw, H_opad_pw, passphrase, pass_len)
//   2. pbkdf2_iterate(H_ipad_pw, H_opad_pw, U_first, seed)
// ============================================================================

// Précalcul des états HMAC depuis le password (mnemonic).
// Appelé une seule fois côté GPU avant le batch.
// Résultat stocké en device global et partagé par tous les threads.
__device__ __noinline__ void pbkdf2_compute_key_states(
    const uint8_t* password, uint32_t pwd_len,
    uint64_t H_ipad_out[8],
    uint64_t H_opad_out[8])
{
    uint8_t k_ipad[SHA512_BLOCK_SIZE];
    uint8_t k_opad[SHA512_BLOCK_SIZE];
    hmac_sha512_prepare_keys(password, pwd_len, k_ipad, k_opad);

    Hydra_SHA512_CTX ctx;
    sha512_init(&ctx);
    sha512_update(&ctx, k_ipad, SHA512_BLOCK_SIZE);
    #pragma unroll
    for(int i = 0; i < 8; i++) H_ipad_out[i] = ctx.h[i];

    sha512_init(&ctx);
    sha512_update(&ctx, k_opad, SHA512_BLOCK_SIZE);
    #pragma unroll
    for(int i = 0; i < 8; i++) H_opad_out[i] = ctx.h[i];
}

// Première itération PBKDF2 pour un salt variable (passphrase).
// salt_full = "mnemonic" + passphrase + counter_be32(1)
// Contrainte : pass_len ≤ 96 pour tenir en 1 bloc SHA512 → bypass direct.
//
// W layout (après H_ipad, 128 bytes déjà absorbés) :
//   Message = "mnemonic"[8] + passphrase[pass_len] + 0x00000001[4]
//   total_msg_bytes = 8 + pass_len + 4
//   Padding SHA512 : 0x80 + zéros + length_be64
//   total_bytes = 128 + 8 + pass_len + 4
//   W[15] = total_bytes * 8
__device__ __forceinline__ void pbkdf2_first_iter_pass(
    const uint64_t H_ipad[8],
    const uint64_t H_opad[8],
    const uint8_t* passphrase,
    uint32_t       pass_len,    // ≤ 96 bytes
    uint64_t       U_out[8])
{
    // Construire le message : "mnemonic" + passphrase + 0x00000001
    // On forge directement W[16] en big-endian uint64_t

    // Longueur totale du message : 8 + pass_len + 4
    const uint32_t msg_len   = 8 + pass_len + 4;
    const uint64_t total_len = (uint64_t)(128 + msg_len);

    // Construire W[16] en assemblant byte par byte dans un buffer temporaire
    // puis charger en big-endian
    uint8_t msg[128] = {};  // zéro-initialisé
    // "mnemonic"
    msg[0]=0x6d; msg[1]=0x6e; msg[2]=0x65; msg[3]=0x6d;
    msg[4]=0x6f; msg[5]=0x6e; msg[6]=0x69; msg[7]=0x63;
    // passphrase
    for(uint32_t i = 0; i < pass_len; i++) msg[8+i] = passphrase[i];
    // counter = 0x00000001 (big-endian)
    msg[8+pass_len+0] = 0x00;
    msg[8+pass_len+1] = 0x00;
    msg[8+pass_len+2] = 0x00;
    msg[8+pass_len+3] = 0x01;
    // padding 0x80
    msg[msg_len] = 0x80;
    // longueur en bits à W[15] (les 8 derniers bytes du bloc 128B)
    // W[15] = total_len * 8, les 7 bytes précédents sont zéros (déjà)
    const uint64_t total_bits = total_len * 8;

    // Charger W[0..14] depuis msg, W[15] = total_bits
    uint64_t W[16];
    #pragma unroll
    for(int i = 0; i < 15; i++) W[i] = sha512_load_be64(msg + i*8);
    W[15] = total_bits;

    // inner = SHA512(H_ipad || msg)
    uint64_t H_inner[8];
    #pragma unroll
    for(int i = 0; i < 8; i++) H_inner[i] = H_ipad[i];
    sha512_transform_state_hot(H_inner, W);

    // outer = SHA512(H_opad || inner[64])  — inner est 64 bytes → sha512_compress_state
    sha512_compress_state(H_opad, H_inner, U_out);
}



__device__ void pbkdf2_hmac_sha512(const uint8_t *password, uint32_t pwd_len,
                                   const uint8_t *salt, uint32_t salt_len,
                                   uint32_t iterations,
                                   uint8_t *out_dk, uint32_t dkLen) {
    if (iterations == 0 || dkLen == 0) return;

    constexpr size_t MSG_BUF_MAX = 192;
    uint8_t msg_buf[MSG_BUF_MAX];
    if (salt_len + 4 > MSG_BUF_MAX) return;

    uint8_t k_ipad[SHA512_BLOCK_SIZE];
    uint8_t k_opad[SHA512_BLOCK_SIZE];
    hmac_sha512_prepare_keys(password, pwd_len, k_ipad, k_opad);

    // Précalcul des états H après avoir absorbé ipad/opad (128 bytes chacun)
    // Ces 8 uint64_t remplacent la copie de struct Hydra_SHA512_CTX (280 bytes)
    uint64_t H_ipad[8], H_opad[8];
    {
        Hydra_SHA512_CTX ctx;
        sha512_init(&ctx);
        sha512_update(&ctx, k_ipad, SHA512_BLOCK_SIZE);
        #pragma unroll
        for (int i = 0; i < 8; i++) H_ipad[i] = ctx.h[i];

        sha512_init(&ctx);
        sha512_update(&ctx, k_opad, SHA512_BLOCK_SIZE);
        #pragma unroll
        for (int i = 0; i < 8; i++) H_opad[i] = ctx.h[i];
    }

    const uint32_t hLen = SHA512_DIGEST_SIZE;  // 64
    uint32_t num_blocks    = (dkLen + hLen - 1) / hLen;
    uint32_t last_blk_len  = dkLen - (num_blocks - 1) * hLen;

    // U et T en uint64_t pour éviter les conversions dans la boucle chaude
    uint64_t U[8], T[8];

    for (uint32_t block_idx = 1; block_idx <= num_blocks; ++block_idx) {

        // ----------------------------------------------------------------
        // Itération 1 : salt || counter → longueur variable → on doit
        // utiliser sha512_update/final classique (pas de bypass possible
        // car le message n'est pas forcément aligné sur 64 bytes)
        // ----------------------------------------------------------------
        {
            uint32_t pos = 0;
            for (uint32_t i = 0; i < salt_len; ++i) msg_buf[pos++] = salt[i];
            msg_buf[pos++] = (uint8_t)((block_idx >> 24) & 0xFF);
            msg_buf[pos++] = (uint8_t)((block_idx >> 16) & 0xFF);
            msg_buf[pos++] = (uint8_t)((block_idx >>  8) & 0xFF);
            msg_buf[pos++] = (uint8_t)( block_idx        & 0xFF);

        {
            // inner : H_ipad + msg_buf (longueur variable) → bypass complet
            // msg_buf = salt + counter, toujours ≤ 111 bytes (BIP39)
            sha512_compress_state_variable(H_ipad, msg_buf, pos, U);
            // outer : H_opad + inner (64 bytes) → bypass standard
            sha512_compress_state(H_opad, U, U);
        }
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) T[i] = U[i];

        // ----------------------------------------------------------------
        // Itérations 2..N : U est toujours 64 bytes → bypass total
        // Zéro copie de struct, zéro gestion de buffer
        // ----------------------------------------------------------------
        for (uint32_t iter = 2; iter <= iterations; ++iter) {
            // inner
            uint64_t H_inner[8];
            sha512_compress_state(H_ipad, U, H_inner);
            // outer
            sha512_compress_state(H_opad, H_inner, U);
            #pragma unroll
            for (int i = 0; i < 8; i++) T[i] ^= U[i];
        }

        // Écriture du résultat (conversion big-endian → bytes)
        uint8_t T_bytes[64];
        #pragma unroll
        for (int i = 0; i < 8; i++) sha512_store_be64(T_bytes + i*8, T[i]);
        uint32_t offset  = (block_idx - 1) * hLen;
        uint32_t to_copy = (block_idx == num_blocks) ? last_blk_len : hLen;
        for (uint32_t i = 0; i < to_copy; ++i) out_dk[offset + i] = T_bytes[i];
    }
}

#endif // PBKDF2_H
