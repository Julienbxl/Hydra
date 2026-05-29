#pragma once
/*
 * ======================================================================================
 * HYDRA - BSGS common definitions
 * ======================================================================================
 *
 * Shared structures for the Gray-first BSGS engine.
 *
 * Production path:
 *   - HEX + exact secp256k1 public key
 *   - Bloom + bucketed baby table
 *   - tiled Gray giant steps with Montgomery batch inversion
 *
 * ======================================================================================
 */

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "HydraCommon.h"
#include "ECC.h"
#include "Wif.cuh"

// ============================================================================
// CONSTANTS
// ============================================================================

#ifndef BLOOM_K_HASHES
#define BLOOM_K_HASHES 16
#endif

#ifndef BSGS_MAX_UNKN
#define BSGS_MAX_UNKN 64
#endif

#ifndef BSGS_MAX_RADIX
#define BSGS_MAX_RADIX 58
#endif

// ============================================================================
// TYPES
// ============================================================================

enum class BsgsInputKind : uint32_t {
    HEX              = 0,
    WIF_COMPRESSED   = 1,
    WIF_UNCOMPRESSED = 2
};

enum class BsgsLookupBackend : uint32_t {
    BLOOM_BUCKETED = 0
};

enum class BsgsRunFlags : uint32_t {
    NONE               = 0,
    GIANT_NEGATIVE     = 1u << 0, // giant dictionaries use Y -> p - Y
    WIF_CARRY_PROBES   = 1u << 1  // DEPRECATED: replaced by exact carry in BsgsBabyEntry
};

__host__ __device__ __forceinline__ uint32_t bsgs_flag(BsgsRunFlags f)
{
    return static_cast<uint32_t>(f);
}

static constexpr uint8_t BSGS_POINT_INFINITY = 1u;
static constexpr uint32_t BSGS_RADIX58_TILE = 58u;
static constexpr uint32_t BSGS_RADIX58_ROWS_PER_THREAD = 1u;
static constexpr uint32_t BSGS_RADIX_ITEMS_PER_THREAD = 64u;

struct BsgsPoint {
    uint64_t x[4];
    uint64_t y[4];
    uint8_t  flags; // BSGS_POINT_INFINITY when set
    uint8_t  _pad[7];
};

static_assert(sizeof(BsgsPoint) == 72, "BsgsPoint layout changed unexpectedly");

// One unknown digit contribution in scalar space.
//
// scalar_weight is little-endian uint64_t[4], matching ECC.h scalar format.
// For HEX, radix is 16 and scalar_weight = 16^n mod secp256k1_n.
// For WIF, scalar_weight comes from the Base58 integer contribution projected
// into the private-key bytes, with carry handled separately.
struct BsgsUnknown {
    uint16_t pos;              // source mask position
    uint16_t group_pos;        // position inside baby or giant group
    uint16_t radix;            // 16 for HEX, 58 for WIF
    uint16_t _pad;
    uint64_t scalar_weight[4]; // little-endian scalar limbs
    uint64_t wif_low_weight;   // low checksum/compression bits before >> wif_shift
};

// CPU/GPU plan uploaded once per BSGS run.
struct BsgsPlan {
    BsgsInputKind     input_kind;
    BsgsLookupBackend lookup_backend;
    uint32_t          flags;
    uint32_t          radix;

    uint32_t total_unknown;
    uint32_t baby_unknown;
    uint32_t giant_unknown;
    uint32_t wif_shift;        // 40 compressed, 32 uncompressed, 0 for HEX
    uint32_t wif_split_initial_baby;
    uint32_t wif_split_non_tail;
    uint32_t wif_split_scalar;
    uint32_t wif_split_auto;

    uint64_t baby_count;
    uint64_t giant_count;
    uint32_t baby_low_bits;
    uint32_t baby_high_bits;
    uint64_t baby_high_count;
    uint32_t giant_low_bits;
    uint32_t giant_high_bits;
    uint64_t giant_high_count;

    uint64_t k_base[4];        // little-endian scalar limbs
    uint64_t p_start_x[4];     // little-endian field limbs
    uint64_t p_start_y[4];     // little-endian field limbs

    uint64_t wif_low_base[2];  // enough for low 40 bits; only [0] used for now
    BsgsPoint wif_top_carry_contrib; // (2^256 mod n) * G, used by WIF key-window carry probes
    BsgsPoint wif_carry_probe[9];    // top*2^256*G - low*G, index = top*3 + low
    WifMask wif_mask;                 // WIF only: Base58 layout used for checksum-on-hit
    uint8_t wif_base_bytes[WIF_MAX_BYTES];
    uint8_t wif_weight_bytes[BSGS_MAX_UNKN][WIF_MAX_BYTES];

    BsgsUnknown baby[BSGS_MAX_UNKN];
    BsgsUnknown giant[BSGS_MAX_UNKN];

    // Precomputed contribution points:
    //   contrib[group_pos][digit] = digit * scalar_weight[group_pos] * G
    //
    // For the giant side, the production path may upload negative points
    // directly (Y -> p - Y), so the kernel can keep using normal additions.
    BsgsPoint baby_contrib[BSGS_MAX_UNKN][BSGS_MAX_RADIX];
    BsgsPoint giant_contrib[BSGS_MAX_UNKN][BSGS_MAX_RADIX];
    BsgsPoint baby_low_dict[LOW_SIZE];
    BsgsPoint giant_low_dict[LOW_SIZE];
};

__host__ __device__ __forceinline__ uint32_t bsgs_radix_bit_width(uint32_t radix)
{
    return radix == 4u ? 2u : 4u;
}

__host__ __device__ __forceinline__ uint32_t bsgs_side_bit_width(
    const BsgsUnknown* unknowns,
    uint32_t unknown_count)
{
    uint32_t bits = 0;
    for (uint32_t i = 0; i < unknown_count; i++)
        bits += bsgs_radix_bit_width(unknowns[i].radix);
    return bits;
}

__host__ __device__ __forceinline__ bool bsgs_bit_to_group_digit(
    const BsgsUnknown* unknowns,
    uint32_t unknown_count,
    uint32_t abs_bit,
    uint32_t& group,
    uint32_t& digit)
{
    uint32_t base = 0;
    for (int g = (int)unknown_count - 1; g >= 0; g--) {
        const uint32_t width = bsgs_radix_bit_width(unknowns[g].radix);
        if (abs_bit < base + width) {
            group = (uint32_t)g;
            digit = 1u << (abs_bit - base);
            return true;
        }
        base += width;
    }
    group = 0;
    digit = 0;
    return false;
}

// Phase 1 exact baby entry.
//
// Compact point identity used by the BSGS lookup path. The Bloom and bucket
// filters are still built from the full normalized point key; the stored entry
// keeps only a 64-bit fingerprint, which is verified again on CPU after a hit.
// idx_a reconstructs the baby-side digits/delta.
//
// WIF carry fields (zero for HEX mode, wif_shift == 0):
//   carry_a     = floor((wif_low_base[0] + baby_low_sum) / 2^wif_shift)
//                 Encoded as uint8 — maximum observed value is ~100 for any valid WIF.
//   wif_low_res = (wif_low_base[0] + baby_low_sum) & ((1<<wif_shift)-1)
//                 Stored big-endian on 5 bytes (covers both shift=40 and shift=32 cases).
//
// On the giant side, the exact carry is reconstructed without any probes:
//   carry_b     = giant_low_sum >> wif_shift
//   residue_b   = giant_low_sum & mask
//   interaction = (residue_ab + residue_b) >> wif_shift   [always 0 or 1]
//   carry_total = carry_a + carry_b + interaction          [mathematically exact]
struct BsgsBabyEntry {
    uint64_t key_fp;       // low 64 bits of normalized point fingerprint
    uint64_t idx_a;
    uint32_t bucket;       // precomputed bucket index for scatter without full X
    uint32_t key_fp_hi;    // high 32 bits of normalized point fingerprint
    uint8_t  carry_a;      // WIF only: floor((base_low + baby_low_sum) >> wif_shift); 0 for HEX
    uint8_t  wif_low_res[5]; // WIF only: residue big-endian 5 bytes; all-zero for HEX
    uint8_t  _pad[2];
};

static_assert(sizeof(BsgsBabyEntry) == 32, "BsgsBabyEntry must stay compact/predictable");

struct BsgsHex8Entry {
    uint32_t key_fp;
    uint32_t idx_a;
};

static_assert(sizeof(BsgsHex8Entry) == 8, "BsgsHex8Entry must stay 8 bytes");

struct BsgsHit {
    uint64_t idx_a;
    uint64_t idx_b;
    uint64_t lookup_x[4];
    uint8_t  lookup_y_parity;
    uint8_t  carry;            // WIF only: 0,1,2. HEX uses 0.
    uint8_t  _pad[6];
};

static_assert(sizeof(BsgsHit) == 56, "BsgsHit must stay compact/predictable");

#ifndef BSGS_GRAY_ITEMS_PER_THREAD
#define BSGS_GRAY_ITEMS_PER_THREAD 64
#endif

#ifndef BSGS_LOW_TILE
#define BSGS_LOW_TILE 128
#endif

#define BSGS_LOW_MAX_TILES (((LOW_SIZE - 1) + BSGS_LOW_TILE - 1) / BSGS_LOW_TILE)

struct BsgsCandidate {
    int      found;
    uint8_t  carry;
    uint8_t  _pad0[3];
    uint64_t idx_a;
    uint64_t idx_b;
};

// This struct intentionally does not allocate in its constructor so it remains a
// plain aggregate that can be zero-initialized in Hydra.cu.
struct BsgsGpuBuffers {
    BsgsPlan*       d_plan = nullptr;
    BsgsBabyEntry*  d_baby_entries = nullptr;
    BsgsBabyEntry*  d_bucketed_entries = nullptr;
    BsgsHex8Entry*  d_hex8_bucketed_entries = nullptr;
    BsgsHit*        d_hits = nullptr;
    uint32_t*       d_hit_count = nullptr;
    BsgsCandidate*  d_candidate = nullptr;
    uint64_t*       d_bsgs_bloom = nullptr;
    uint32_t*       d_bucket_offsets = nullptr;
    uint32_t*       d_bucket_counts = nullptr;
    uint32_t*       d_bucket_cursor = nullptr;

    uint64_t baby_capacity = 0;
    uint64_t hit_capacity = 0;
    uint64_t bloom_m_bits = 0;
    uint32_t bucket_count = 0;
    uint32_t bucket_bits = 0;
    bool hex8_mode = false;
};

static inline void bsgs_free_buffers(BsgsGpuBuffers& buffers);

// ============================================================================
// HOST HELPERS
// ============================================================================

static inline cudaError_t bsgs_alloc_bloom_buffers(
    BsgsGpuBuffers& buffers,
    uint64_t baby_capacity,
    uint64_t hit_capacity,
    uint64_t bloom_m_bits)
{
    if (baby_capacity > (SIZE_MAX / sizeof(BsgsBabyEntry)))
        return cudaErrorMemoryAllocation;
    if (hit_capacity > (SIZE_MAX / sizeof(BsgsHit)))
        return cudaErrorMemoryAllocation;
    if (bloom_m_bits == 0 || (bloom_m_bits & (bloom_m_bits - 1ULL)) != 0)
        return cudaErrorInvalidValue;
    if (bloom_m_bits > (SIZE_MAX / sizeof(uint64_t)) * 64ULL)
        return cudaErrorMemoryAllocation;

    buffers = {};
    buffers.baby_capacity = baby_capacity;
    buffers.hit_capacity = hit_capacity;
    buffers.bloom_m_bits = bloom_m_bits;

    auto cleanup_on_error = [&buffers](cudaError_t err) -> cudaError_t {
        if (err == cudaSuccess) return err;
        if (buffers.d_bsgs_bloom)   cudaFree(buffers.d_bsgs_bloom);
        if (buffers.d_candidate)    cudaFree(buffers.d_candidate);
        if (buffers.d_hit_count)    cudaFree(buffers.d_hit_count);
        if (buffers.d_hits)         cudaFree(buffers.d_hits);
        if (buffers.d_baby_entries) cudaFree(buffers.d_baby_entries);
        if (buffers.d_plan)         cudaFree(buffers.d_plan);
        buffers = {};
        return err;
    };

    cudaError_t err = cudaMalloc(&buffers.d_plan, sizeof(BsgsPlan));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&buffers.d_baby_entries, (size_t)baby_capacity * sizeof(BsgsBabyEntry));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_hits, (size_t)hit_capacity * sizeof(BsgsHit));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_hit_count, sizeof(uint32_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_candidate, sizeof(BsgsCandidate));
    if (err != cudaSuccess) return cleanup_on_error(err);

    const size_t bloom_words = (size_t)(bloom_m_bits >> 6);
    err = cudaMalloc(&buffers.d_bsgs_bloom, bloom_words * sizeof(uint64_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    uint32_t zero = 0;
    err = cudaMemcpy(buffers.d_hit_count, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return cleanup_on_error(err);

    BsgsCandidate candidate = {};
    err = cudaMemcpy(buffers.d_candidate, &candidate, sizeof(BsgsCandidate), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMemset(buffers.d_bsgs_bloom, 0, bloom_words * sizeof(uint64_t));
    if (err != cudaSuccess) return cleanup_on_error(err);
    return cleanup_on_error(err);
}

static inline cudaError_t bsgs_alloc_bloom_bucketed_buffers(
    BsgsGpuBuffers& buffers,
    uint64_t baby_capacity,
    uint64_t hit_capacity,
    uint64_t bloom_m_bits,
    uint32_t bucket_bits)
{
    if (bucket_bits == 0 || bucket_bits > 30)
        return cudaErrorInvalidValue;

    cudaError_t err = bsgs_alloc_bloom_buffers(
        buffers, baby_capacity, hit_capacity, bloom_m_bits);
    if (err != cudaSuccess) return err;

    auto cleanup_on_error = [&buffers](cudaError_t e) -> cudaError_t {
        if (e == cudaSuccess) return e;
        bsgs_free_buffers(buffers);
        return e;
    };

    buffers.bucket_bits = bucket_bits;
    buffers.bucket_count = 1u << bucket_bits;

    err = cudaMalloc(&buffers.d_bucket_offsets,
                     (size_t)(buffers.bucket_count + 1u) * sizeof(uint32_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMemset(buffers.d_bucket_offsets, 0,
                     (size_t)(buffers.bucket_count + 1u) * sizeof(uint32_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_bucket_counts,
                     (size_t)buffers.bucket_count * sizeof(uint32_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_bucket_cursor,
                     (size_t)buffers.bucket_count * sizeof(uint32_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_bucketed_entries,
                     (size_t)baby_capacity * sizeof(BsgsBabyEntry));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMemset(buffers.d_bucket_counts, 0,
                     (size_t)buffers.bucket_count * sizeof(uint32_t));
    return cleanup_on_error(err);
}

static inline cudaError_t bsgs_alloc_hex8_bucketed_buffers(
    BsgsGpuBuffers& buffers,
    uint64_t baby_capacity,
    uint64_t hit_capacity,
    uint64_t bloom_m_bits,
    uint32_t bucket_bits)
{
    if (bucket_bits == 0 || bucket_bits > 30)
        return cudaErrorInvalidValue;
    if (baby_capacity > (uint64_t)UINT32_MAX + 1ULL)
        return cudaErrorInvalidValue;
    if (hit_capacity > (SIZE_MAX / sizeof(BsgsHit)))
        return cudaErrorMemoryAllocation;
    if (bloom_m_bits == 0 || (bloom_m_bits & (bloom_m_bits - 1ULL)) != 0)
        return cudaErrorInvalidValue;
    if (bloom_m_bits > (SIZE_MAX / sizeof(uint64_t)) * 64ULL)
        return cudaErrorMemoryAllocation;

    buffers = {};
    buffers.baby_capacity = baby_capacity;
    buffers.hit_capacity = hit_capacity;
    buffers.bloom_m_bits = bloom_m_bits;
    buffers.bucket_bits = bucket_bits;
    buffers.bucket_count = 1u << bucket_bits;
    buffers.hex8_mode = true;

    auto cleanup_on_error = [&buffers](cudaError_t err) -> cudaError_t {
        if (err == cudaSuccess) return err;
        bsgs_free_buffers(buffers);
        return err;
    };

    cudaError_t err = cudaMalloc(&buffers.d_plan, sizeof(BsgsPlan));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_hits, (size_t)hit_capacity * sizeof(BsgsHit));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_hit_count, sizeof(uint32_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_candidate, sizeof(BsgsCandidate));
    if (err != cudaSuccess) return cleanup_on_error(err);

    const size_t bloom_words = (size_t)(bloom_m_bits >> 6);
    err = cudaMalloc(&buffers.d_bsgs_bloom, bloom_words * sizeof(uint64_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_bucket_offsets,
                     (size_t)(buffers.bucket_count + 1u) * sizeof(uint32_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_bucket_counts,
                     (size_t)buffers.bucket_count * sizeof(uint32_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_bucket_cursor,
                     (size_t)buffers.bucket_count * sizeof(uint32_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMalloc(&buffers.d_hex8_bucketed_entries,
                     (size_t)baby_capacity * sizeof(BsgsHex8Entry));
    if (err != cudaSuccess) return cleanup_on_error(err);

    uint32_t zero = 0;
    err = cudaMemcpy(buffers.d_hit_count, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return cleanup_on_error(err);

    BsgsCandidate candidate = {};
    err = cudaMemcpy(buffers.d_candidate, &candidate, sizeof(BsgsCandidate), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMemset(buffers.d_bsgs_bloom, 0, bloom_words * sizeof(uint64_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMemset(buffers.d_bucket_offsets, 0,
                     (size_t)(buffers.bucket_count + 1u) * sizeof(uint32_t));
    if (err != cudaSuccess) return cleanup_on_error(err);

    err = cudaMemset(buffers.d_bucket_counts, 0,
                     (size_t)buffers.bucket_count * sizeof(uint32_t));
    return cleanup_on_error(err);
}

static inline void bsgs_free_buffers(BsgsGpuBuffers& buffers)
{
    if (buffers.d_bucket_cursor)  cudaFree(buffers.d_bucket_cursor);
    if (buffers.d_bucket_counts)  cudaFree(buffers.d_bucket_counts);
    if (buffers.d_bucket_offsets) cudaFree(buffers.d_bucket_offsets);
    if (buffers.d_bucketed_entries) cudaFree(buffers.d_bucketed_entries);
    if (buffers.d_hex8_bucketed_entries) cudaFree(buffers.d_hex8_bucketed_entries);
    if (buffers.d_bsgs_bloom)   cudaFree(buffers.d_bsgs_bloom);
    if (buffers.d_candidate)    cudaFree(buffers.d_candidate);
    if (buffers.d_hit_count)    cudaFree(buffers.d_hit_count);
    if (buffers.d_hits)         cudaFree(buffers.d_hits);
    if (buffers.d_baby_entries) cudaFree(buffers.d_baby_entries);
    if (buffers.d_plan)         cudaFree(buffers.d_plan);
    buffers = {};
}

static inline void bsgs_release_raw_baby_entries(BsgsGpuBuffers& buffers)
{
    if (buffers.d_baby_entries) {
        cudaFree(buffers.d_baby_entries);
        buffers.d_baby_entries = nullptr;
    }
}

static inline cudaError_t bsgs_upload_plan(
    const BsgsPlan& plan,
    BsgsGpuBuffers& buffers)
{
    return cudaMemcpy(buffers.d_plan, &plan, sizeof(BsgsPlan), cudaMemcpyHostToDevice);
}

// ============================================================================
// DEVICE HELPERS / PHASE 1 KERNELS
// ============================================================================

__device__ __forceinline__ void bsgs_point_to_ecpoint(const BsgsPoint& src, ECPointA& dst)
{
    if (src.flags & BSGS_POINT_INFINITY) {
        pointSetInfinity(dst);
        return;
    }
    dst.infinity = false;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        dst.X[i] = src.x[i];
        dst.Y[i] = src.y[i];
    }
}

__device__ __forceinline__ bool bsgs_limbs_is_zero4(const uint64_t a[4])
{
    return (a[0] | a[1] | a[2] | a[3]) == 0ULL;
}

__device__ __forceinline__ void bsgs_limbs_copy4(const uint64_t src[4], uint64_t dst[4])
{
    #pragma unroll
    for (int i = 0; i < 4; i++) dst[i] = src[i];
}

__device__ __forceinline__ void bsgs_point_add_or_sub(
    ECPointA& acc,
    const BsgsPoint& contrib,
    bool add)
{
    ECPointA q, next;
    bsgs_point_to_ecpoint(contrib, q);
    if (q.infinity) return;
    if (!add) fieldNeg(q.Y, q.Y);
    pointAddAffine(acc, q, next);
    acc = next;
}

__device__ __forceinline__ void bsgs_add_wif_cross_carry(
    ECPointA& acc,
    uint64_t carry,
    bool negative)
{
    if (carry == 0) return;
    ECPointA g;
    pointSetG(g);
    if (negative) fieldNeg(g.Y, g.Y);
    for (uint64_t i = 0; i < carry; i++) {
        ECPointA next;
        pointAddAffine(acc, g, next);
        acc = next;
    }
}

__device__ __forceinline__ void bsgs_add_wif_top_carry(
    ECPointA& acc,
    const BsgsPlan* __restrict__ plan,
    uint64_t carry,
    bool add)
{
    if (carry == 0) return;
    for (uint64_t i = 0; i < carry; i++)
        bsgs_point_add_or_sub(acc, plan->wif_top_carry_contrib, add);
}

__device__ __forceinline__ void bsgs_adjust_wif_top_carry(
    ECPointA& acc,
    const BsgsPlan* __restrict__ plan,
    int64_t delta,
    bool giant)
{
    if (delta == 0) return;
    if (delta > 0) {
        bsgs_add_wif_top_carry(acc, plan, (uint64_t)delta, giant);
    } else {
        bsgs_add_wif_top_carry(acc, plan, (uint64_t)(-delta), !giant);
    }
}

__device__ __forceinline__ uint64_t bsgs_low_mask_for_shift(uint32_t shift)
{
    return (shift >= 64u) ? UINT64_MAX : ((1ULL << shift) - 1ULL);
}

__device__ __forceinline__ uint64_t bsgs_add_word_to_le4_carry(uint64_t a[4], uint64_t v)
{
    uint64_t old = a[0];
    a[0] += v;
    uint64_t carry = (a[0] < old) ? 1ULL : 0ULL;
    #pragma unroll
    for (int i = 1; i < 4; i++) {
        if (carry == 0) break;
        old = a[i];
        a[i] += carry;
        carry = (a[i] < old) ? 1ULL : 0ULL;
    }
    return carry;
}

__device__ __forceinline__ uint64_t bsgs_add_le4_carry(uint64_t a[4], const uint64_t b[4])
{
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t old = a[i];
        a[i] += b[i];
        const uint64_t c1 = (a[i] < old) ? 1ULL : 0ULL;
        old = a[i];
        a[i] += carry;
        const uint64_t c2 = (a[i] < old) ? 1ULL : 0ULL;
        carry = c1 + c2;
    }
    return carry;
}

__device__ __forceinline__ void bsgs_wif_component_limbs(
    const BsgsUnknown& u,
    uint32_t digit,
    uint32_t wif_shift,
    uint64_t out[4])
{
    uint64_t carry = 0;
    const uint64_t d = (uint64_t)digit;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint64_t lo = u.scalar_weight[i] * d;
        const uint64_t hi = __umul64hi(u.scalar_weight[i], d);
        const uint64_t with_carry = lo + carry;
        carry = hi + ((with_carry < lo) ? 1ULL : 0ULL);
        out[i] = with_carry;
    }

    const uint64_t digit_low = (uint64_t)digit * u.wif_low_weight;
    const uint64_t digit_carry = digit_low >> wif_shift;
    if (digit_carry != 0)
        bsgs_add_word_to_le4_carry(out, digit_carry);
}

__device__ __forceinline__ uint64_t bsgs_compute_wif_side_top_carry(
    const BsgsPlan* __restrict__ plan,
    bool giant,
    const uint8_t digits[BSGS_MAX_UNKN],
    uint64_t low_sum)
{
    if (plan->wif_shift == 0) return 0;

    const uint32_t unknown_count = giant ? plan->giant_unknown : plan->baby_unknown;
    const BsgsUnknown* unknowns = giant ? plan->giant : plan->baby;

    uint64_t acc[4] = {0, 0, 0, 0};
    uint64_t top = 0;
    #pragma unroll 1
    for (uint32_t group = 0; group < unknown_count; group++) {
        const uint32_t digit = digits[group];
        if (digit == 0) continue;
        uint64_t component[4];
        bsgs_wif_component_limbs(unknowns[group], digit, plan->wif_shift, component);
        top += bsgs_add_le4_carry(acc, component);
    }

    const uint64_t cross_carry = low_sum >> plan->wif_shift;
    if (cross_carry != 0)
        top += bsgs_add_word_to_le4_carry(acc, cross_carry);

    return top;
}

__device__ __forceinline__ void bsgs_build_acc_from_binary_gray(
    const BsgsPlan* __restrict__ plan,
    bool giant,
    uint64_t binary_idx,
    ECPointA& acc)
{
    if (giant) {
        acc.infinity = false;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            acc.X[i] = plan->p_start_x[i];
            acc.Y[i] = plan->p_start_y[i];
        }
    } else {
        pointSetInfinity(acc);
    }

    const uint32_t unknown_count = giant ? plan->giant_unknown : plan->baby_unknown;
    const BsgsUnknown* unknowns = giant ? plan->giant : plan->baby;
    const uint32_t bit_count = bsgs_side_bit_width(unknowns, unknown_count);
    const uint64_t gray = binary_idx ^ (binary_idx >> 1);
    for (uint32_t bit = 0; bit < bit_count; bit++) {
        if (((gray >> bit) & 1ULL) == 0) continue;
        uint32_t group = 0;
        uint32_t digit = 0;
        if (!bsgs_bit_to_group_digit(unknowns, unknown_count, bit, group, digit)) continue;
        const BsgsPoint& q = giant ? plan->giant_contrib[group][digit]
                                   : plan->baby_contrib[group][digit];
        bsgs_point_add_or_sub(acc, q, true);
    }
}

__device__ __forceinline__ void bsgs_build_acc_from_radix_index(
    const BsgsPlan* __restrict__ plan,
    bool giant,
    uint64_t idx,
    ECPointA& acc)
{
    if (giant) {
        acc.infinity = false;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            acc.X[i] = plan->p_start_x[i];
            acc.Y[i] = plan->p_start_y[i];
        }
    } else {
        pointSetInfinity(acc);
    }

    const uint32_t unknown_count = giant ? plan->giant_unknown : plan->baby_unknown;
    const BsgsUnknown* unknowns = giant ? plan->giant : plan->baby;
    const BsgsPoint (*contrib)[BSGS_MAX_RADIX] = giant ? plan->giant_contrib : plan->baby_contrib;
    uint64_t low_sum = 0;
    const uint64_t low_mask = bsgs_low_mask_for_shift(plan->wif_shift);

    for (int group = (int)unknown_count - 1; group >= 0; group--) {
        const uint32_t radix = unknowns[group].radix ? unknowns[group].radix : plan->radix;
        const uint32_t digit = (uint32_t)(idx % radix);
        idx /= radix;
        // Accumulate only the residue of (digit * wif_low_weight).
        // The floor part (digit_carry = (digit*lw)>>shift) is already baked into
        // each contrib point by bsgs_build_contrib_points_cpu, so we must NOT
        // add it again here. Only the residue drives the inter-term interaction.
        if (plan->wif_shift != 0)
            low_sum += ((uint64_t)digit * unknowns[group].wif_low_weight) & low_mask;
        if (digit == 0) continue;
        bsgs_point_add_or_sub(acc, contrib[group][digit], true);
    }
    // cross_carry and top_carry are intentionally NOT applied here:
    // the residue-based correction is computed at match time from the stored
    // baby carry_a / wif_low_res fields (see bsgs_try_bucketed_match_with_carry).
}

__device__ __forceinline__ void bsgs_adjust_wif_cross_carry(
    ECPointA& acc,
    int64_t delta,
    bool giant)
{
    if (delta == 0) return;
    if (delta > 0) {
        bsgs_add_wif_cross_carry(acc, (uint64_t)delta, giant);
    } else {
        bsgs_add_wif_cross_carry(acc, (uint64_t)(-delta), !giant);
    }
}

__device__ __forceinline__ uint64_t bsgs_init_radix_digits(
    const BsgsPlan* __restrict__ plan,
    bool giant,
    uint64_t idx,
    uint8_t digits[BSGS_MAX_UNKN])
{
    const uint32_t unknown_count = giant ? plan->giant_unknown : plan->baby_unknown;
    const BsgsUnknown* unknowns = giant ? plan->giant : plan->baby;
    const uint64_t low_mask = bsgs_low_mask_for_shift(plan->wif_shift);
    uint64_t low_sum = 0;

    for (int group = (int)unknown_count - 1; group >= 0; group--) {
        const uint32_t radix = unknowns[group].radix ? unknowns[group].radix : plan->radix;
        const uint8_t digit = (uint8_t)(idx % radix);
        idx /= radix;
        digits[group] = digit;
        // Residue only — the floor part lives in the contrib point already.
        if (plan->wif_shift != 0)
            low_sum += ((uint64_t)digit * unknowns[group].wif_low_weight) & low_mask;
    }
    return low_sum;
}

__device__ __forceinline__ void bsgs_step_radix_incremental(
    const BsgsPlan* __restrict__ plan,
    bool giant,
    uint8_t digits[BSGS_MAX_UNKN],
    uint64_t& low_sum,
    uint64_t& top_sum,
    ECPointA& acc)
{
    const uint32_t unknown_count = giant ? plan->giant_unknown : plan->baby_unknown;
    const BsgsUnknown* unknowns = giant ? plan->giant : plan->baby;
    const BsgsPoint (*contrib)[BSGS_MAX_RADIX] = giant ? plan->giant_contrib : plan->baby_contrib;
    const uint64_t low_mask = bsgs_low_mask_for_shift(plan->wif_shift);

    for (int group = (int)unknown_count - 1; group >= 0; group--) {
        const uint32_t old_digit = digits[group];
        const uint32_t new_digit = old_digit + 1u;

        if (old_digit != 0)
            bsgs_point_add_or_sub(acc, contrib[group][old_digit], false);
        // Remove residue of old digit, add residue of new digit.
        if (plan->wif_shift != 0)
            low_sum -= ((uint64_t)old_digit * unknowns[group].wif_low_weight) & low_mask;

        const uint32_t radix = unknowns[group].radix ? unknowns[group].radix : plan->radix;
        if (new_digit < radix) {
            digits[group] = (uint8_t)new_digit;
            bsgs_point_add_or_sub(acc, contrib[group][new_digit], true);
            if (plan->wif_shift != 0)
                low_sum += ((uint64_t)new_digit * unknowns[group].wif_low_weight) & low_mask;
            break;
        }

        digits[group] = 0;
    }
    if (plan->wif_shift != 0) {
        const uint64_t new_side_carry = low_sum >> plan->wif_shift;
        bsgs_adjust_wif_cross_carry(acc, (int64_t)new_side_carry - (int64_t)top_sum, giant);
        top_sum = new_side_carry;
    }
}

__device__ __forceinline__ void bsgs_step_radix58_tile_base(
    const BsgsPlan* __restrict__ plan,
    bool giant,
    uint8_t digits[BSGS_MAX_UNKN],
    uint64_t& low_sum,
    uint64_t& top_sum,
    ECPointA& acc)
{
    const uint32_t unknown_count = giant ? plan->giant_unknown : plan->baby_unknown;
    if (unknown_count < 2u) return;

    const BsgsUnknown* unknowns = giant ? plan->giant : plan->baby;
    const BsgsPoint (*contrib)[BSGS_MAX_RADIX] = giant ? plan->giant_contrib : plan->baby_contrib;
    const uint64_t low_mask = bsgs_low_mask_for_shift(plan->wif_shift);

    // A radix-58 tile fixes the least-significant group at digit 0 and scans
    // that digit inside the batch. Moving to the next tile is therefore +58:
    // carry into the group immediately above the low digit, without touching
    // the low group itself.
    for (int group = (int)unknown_count - 2; group >= 0; group--) {
        const uint32_t old_digit = digits[group];
        const uint32_t new_digit = old_digit + 1u;

        if (old_digit != 0)
            bsgs_point_add_or_sub(acc, contrib[group][old_digit], false);
        if (plan->wif_shift != 0)
            low_sum -= ((uint64_t)old_digit * unknowns[group].wif_low_weight) & low_mask;

        const uint32_t radix = unknowns[group].radix ? unknowns[group].radix : plan->radix;
        if (new_digit < radix) {
            digits[group] = (uint8_t)new_digit;
            bsgs_point_add_or_sub(acc, contrib[group][new_digit], true);
            if (plan->wif_shift != 0)
                low_sum += ((uint64_t)new_digit * unknowns[group].wif_low_weight) & low_mask;
            break;
        }

        digits[group] = 0;
    }

    if (plan->wif_shift != 0) {
        const uint64_t new_side_carry = low_sum >> plan->wif_shift;
        bsgs_adjust_wif_cross_carry(acc, (int64_t)new_side_carry - (int64_t)top_sum, giant);
        top_sum = new_side_carry;
    }
}

__device__ __forceinline__ void bsgs_step_binary_gray(
    const BsgsPlan* __restrict__ plan,
    bool giant,
    uint32_t unknown_count,
    uint64_t current_binary_idx,
    ECPointA& acc)
{
    const uint64_t g0 = current_binary_idx ^ (current_binary_idx >> 1);
    const uint64_t next = current_binary_idx + 1ULL;
    const uint64_t g1 = next ^ (next >> 1);
    const uint64_t diff = g0 ^ g1;
    const uint32_t bit = (uint32_t)__ffsll((long long)diff) - 1u;
    const bool add = ((g1 >> bit) & 1ULL) != 0;
    const BsgsUnknown* unknowns = giant ? plan->giant : plan->baby;
    uint32_t group = 0;
    uint32_t digit = 0;
    if (!bsgs_bit_to_group_digit(unknowns, unknown_count, bit, group, digit)) return;
    const BsgsPoint& q = giant ? plan->giant_contrib[group][digit]
                               : plan->baby_contrib[group][digit];
    bsgs_point_add_or_sub(acc, q, add);
}

__device__ __forceinline__ uint64_t bsgs_mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__device__ __forceinline__ uint64_t bsgs_key_fingerprint(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags)
{
    uint64_t fp = 0x9e3779b97f4a7c15ULL;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        fp ^= bsgs_mix64(x[i] + 0x9e3779b97f4a7c15ULL + (fp << 6) + (fp >> 2));
    fp ^= ((uint64_t)y_parity << 8) | (uint64_t)flags;
    return bsgs_mix64(fp);
}

__device__ __forceinline__ uint32_t bsgs_key_fingerprint_hi(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags)
{
    uint64_t fp = 0xd1b54a32d192ed03ULL;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        fp ^= bsgs_mix64(x[3 - i] + 0x94d049bb133111ebULL + (fp << 7) + (fp >> 3));
    fp ^= ((uint64_t)flags << 40) | ((uint64_t)y_parity << 32) | 0x517cc1b727220a95ULL;
    return (uint32_t)bsgs_mix64(fp);
}

__device__ __forceinline__ void bsgs_extract_point_key(
    const ECPointA& point,
    uint64_t x[4],
    uint8_t& y_parity,
    uint8_t& flags)
{
    if (point.infinity) {
        #pragma unroll
        for (int i = 0; i < 4; i++) x[i] = 0;
        y_parity = 0;
        flags = BSGS_POINT_INFINITY;
        return;
    }

    uint64_t y[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        x[i] = point.X[i];
        y[i] = point.Y[i];
    }
    fieldNormalize(x);
    fieldNormalize(y);
    y_parity = (uint8_t)(y[0] & 1ULL);
    flags = 0;
}

__device__ __forceinline__ void bsgs_store_baby_entry_from_key(
    BsgsBabyEntry& out,
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags,
    uint64_t idx_a)
{
    out.key_fp = bsgs_key_fingerprint(x, y_parity, flags);
    out.key_fp_hi = bsgs_key_fingerprint_hi(x, y_parity, flags);
    out.idx_a = idx_a;
    out.bucket = 0;
    // WIF carry fields default to zero; filled by the WIF baby kernel when wif_shift != 0.
    out.carry_a = 0;
    #pragma unroll
    for (int b = 0; b < 5; b++) out.wif_low_res[b] = 0;
    #pragma unroll
    for (int b = 0; b < 2; b++) out._pad[b] = 0;
}

__device__ __forceinline__ void bsgs_store_baby_entry(
    BsgsBabyEntry& out,
    const ECPointA& point,
    uint64_t idx_a)
{
    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_extract_point_key(point, x, y_parity, flags);
    bsgs_store_baby_entry_from_key(out, x, y_parity, flags, idx_a);
}

__device__ __forceinline__ uint32_t bsgs_rotl32(uint32_t x, int r)
{
    return (x << r) | (x >> (32 - r));
}

__device__ __forceinline__ uint32_t bsgs_fmix32(uint32_t h)
{
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__ uint32_t bsgs_murmur_mix_block(uint32_t h, uint32_t k)
{
    k *= 0xcc9e2d51u;
    k = bsgs_rotl32(k, 15);
    k *= 0x1b873593u;
    h ^= k;
    h = bsgs_rotl32(h, 13);
    return h * 5u + 0xe6546b64u;
}

__device__ __forceinline__ uint32_t bsgs_bloom_hash_key(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags,
    uint32_t seed)
{
    uint32_t h = seed;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        h = bsgs_murmur_mix_block(h, (uint32_t)x[i]);
        h = bsgs_murmur_mix_block(h, (uint32_t)(x[i] >> 32));
    }
    const uint32_t tail = (uint32_t)y_parity | ((uint32_t)flags << 8);
    h = bsgs_murmur_mix_block(h, tail);
    h ^= 36u;
    return bsgs_fmix32(h);
}

__device__ __forceinline__ void bsgs_bloom_add_key(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags,
    uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits)
{
    uint32_t h1 = bsgs_bloom_hash_key(x, y_parity, flags, 0x9747b28cu);
    uint32_t h2 = bsgs_bloom_hash_key(x, y_parity, flags, h1);
    if (h2 == 0) h2 = 1;

    const uint64_t mask = bloom_m_bits - 1ULL;
    #pragma unroll
    for (int i = 0; i < BLOOM_K_HASHES; i++) {
        const uint64_t bit_pos = (h1 + (uint64_t)i * h2) & mask;
        const uint64_t word_idx = bit_pos >> 6;
        const uint32_t bit_idx = (uint32_t)(bit_pos & 63ULL);
        atomicOr((unsigned long long*)&bloom[word_idx], 1ULL << bit_idx);
    }
}

__device__ __forceinline__ bool bsgs_bloom_check_key(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags,
    const uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits)
{
    uint32_t h1 = bsgs_bloom_hash_key(x, y_parity, flags, 0x9747b28cu);
    uint32_t h2 = bsgs_bloom_hash_key(x, y_parity, flags, h1);
    if (h2 == 0) h2 = 1;

    const uint64_t mask = bloom_m_bits - 1ULL;
    #pragma unroll
    for (int i = 0; i < BLOOM_K_HASHES; i++) {
        const uint64_t bit_pos = (h1 + (uint64_t)i * h2) & mask;
        const uint64_t word_idx = bit_pos >> 6;
        const uint32_t bit_idx = (uint32_t)(bit_pos & 63ULL);
        const uint64_t word = __ldg(&bloom[word_idx]);
        if ((word & (1ULL << bit_idx)) == 0) return false;
    }
    return true;
}

__device__ __forceinline__ uint32_t bsgs_bucket_index_key(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags,
    uint32_t bucket_mask)
{
    return bsgs_bloom_hash_key(x, y_parity, flags, 0x6d2b79f5u) & bucket_mask;
}

__device__ __forceinline__ uint32_t bsgs_hex8_fingerprint_key(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags)
{
    const uint64_t fp = bsgs_key_fingerprint(x, y_parity, flags);
    const uint32_t hi = bsgs_key_fingerprint_hi(x, y_parity, flags);
    return (uint32_t)fp ^ (uint32_t)(fp >> 32) ^ hi;
}

__device__ __forceinline__ void bsgs_store_hex8_entry_from_key(
    BsgsHex8Entry& out,
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags,
    uint64_t idx_a)
{
    out.key_fp = bsgs_hex8_fingerprint_key(x, y_parity, flags);
    out.idx_a = (uint32_t)idx_a;
}

__global__ void bsgs_hex8_baby_count_kernel(
    const BsgsPlan* __restrict__ plan,
    uint64_t baby_count,
    uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    uint32_t* __restrict__ bucket_counts,
    uint32_t bucket_mask)
{
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= baby_count || tid >= plan->baby_count) return;

    ECPointA acc;
    bsgs_build_acc_from_radix_index(plan, false, tid, acc);

    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_extract_point_key(acc, x, y_parity, flags);

    const uint32_t bucket = bsgs_bucket_index_key(x, y_parity, flags, bucket_mask);
    bsgs_bloom_add_key(x, y_parity, flags, bloom, bloom_m_bits);
    atomicAdd(&bucket_counts[bucket], 1u);
}

__global__ void bsgs_hex8_baby_scatter_kernel(
    const BsgsPlan* __restrict__ plan,
    BsgsHex8Entry* __restrict__ bucketed_entries,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t* __restrict__ bucket_cursor,
    uint64_t baby_count,
    uint32_t bucket_mask)
{
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= baby_count || tid >= plan->baby_count) return;

    ECPointA acc;
    bsgs_build_acc_from_radix_index(plan, false, tid, acc);

    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_extract_point_key(acc, x, y_parity, flags);

    BsgsHex8Entry entry;
    bsgs_store_hex8_entry_from_key(entry, x, y_parity, flags, tid);

    const uint32_t bucket = bsgs_bucket_index_key(x, y_parity, flags, bucket_mask);
    const uint32_t slot = bucket_offsets[bucket] + atomicAdd(&bucket_cursor[bucket], 1u);
    bucketed_entries[slot] = entry;
}

__global__ void bsgs_baby_bloom_bucket_count_kernel(
    const BsgsPlan* __restrict__ plan,
    BsgsBabyEntry*  __restrict__ baby_entries,
    uint64_t baby_count,
    uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    uint32_t* __restrict__ bucket_counts,
    uint32_t bucket_mask)
{
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= baby_count || tid >= plan->baby_count) return;

    ECPointA acc;
    bsgs_build_acc_from_radix_index(plan, false, tid, acc);

    // WIF: store exact carry_a and residue for giant-side reconstruction.
    // low_sum here is the residue sum: sum_i((digit_i * wif_low_weight_i) & mask).
    // The floor parts are already embedded in the contrib points.
    uint64_t carry_a = 0;
    uint64_t residue = 0;
    if (plan->wif_shift != 0) {
        const uint64_t low_mask = bsgs_low_mask_for_shift(plan->wif_shift);
        const uint32_t unknown_count = plan->baby_unknown;
        const BsgsUnknown* unknowns = plan->baby;
        uint64_t baby_ls = 0;
        uint64_t tmp_idx = tid;
        for (int group = (int)unknown_count - 1; group >= 0; group--) {
            const uint32_t radix = unknowns[group].radix ? unknowns[group].radix : plan->radix;
            const uint8_t digit = (uint8_t)(tmp_idx % radix);
            tmp_idx /= radix;
            baby_ls += ((uint64_t)digit * unknowns[group].wif_low_weight) & low_mask;
        }
        // combined = base_low_residue + baby_residue_sum
        // base_low is already < 2^shift, so no need to mask it.
        const uint64_t combined = plan->wif_low_base[0] + baby_ls;
        carry_a = combined >> plan->wif_shift;
        residue = combined & low_mask;
        bsgs_add_wif_cross_carry(acc, carry_a, false);
    }

    BsgsBabyEntry entry;
    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_extract_point_key(acc, x, y_parity, flags);
    bsgs_store_baby_entry_from_key(entry, x, y_parity, flags, tid);

    if (plan->wif_shift != 0) {
        entry.carry_a = (uint8_t)carry_a;
        entry.wif_low_res[0] = (uint8_t)(residue >> 32);
        entry.wif_low_res[1] = (uint8_t)(residue >> 24);
        entry.wif_low_res[2] = (uint8_t)(residue >> 16);
        entry.wif_low_res[3] = (uint8_t)(residue >>  8);
        entry.wif_low_res[4] = (uint8_t)(residue      );
    }

    const uint32_t bucket = bsgs_bucket_index_key(x, y_parity, flags, bucket_mask);
    entry.bucket = bucket;
    baby_entries[tid] = entry;
    bsgs_bloom_add_key(x, y_parity, flags, bloom, bloom_m_bits);
    atomicAdd(&bucket_counts[bucket], 1u);
}

__global__ void bsgs_baby_bloom_bucket_emit_kernel(
    const BsgsPlan* __restrict__ plan,
    BsgsBabyEntry*  __restrict__ baby_entries,
    uint64_t baby_offset,
    uint64_t baby_count,
    uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    uint32_t bucket_mask,
    uint32_t add_to_bloom)
{
    const uint64_t local = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t idx_a = baby_offset + local;
    if (local >= baby_count || idx_a >= plan->baby_count) return;

    ECPointA acc;
    bsgs_build_acc_from_radix_index(plan, false, idx_a, acc);

    uint64_t carry_a = 0;
    uint64_t residue = 0;
    if (plan->wif_shift != 0) {
        const uint64_t low_mask = bsgs_low_mask_for_shift(plan->wif_shift);
        const uint32_t unknown_count = plan->baby_unknown;
        const BsgsUnknown* unknowns = plan->baby;
        uint64_t baby_ls = 0;
        uint64_t tmp_idx = idx_a;
        for (int group = (int)unknown_count - 1; group >= 0; group--) {
            const uint32_t radix = unknowns[group].radix ? unknowns[group].radix : plan->radix;
            const uint8_t digit = (uint8_t)(tmp_idx % radix);
            tmp_idx /= radix;
            baby_ls += ((uint64_t)digit * unknowns[group].wif_low_weight) & low_mask;
        }
        const uint64_t combined = plan->wif_low_base[0] + baby_ls;
        carry_a = combined >> plan->wif_shift;
        residue = combined & low_mask;
        bsgs_add_wif_cross_carry(acc, carry_a, false);
    }

    BsgsBabyEntry entry;
    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_extract_point_key(acc, x, y_parity, flags);
    bsgs_store_baby_entry_from_key(entry, x, y_parity, flags, idx_a);

    if (plan->wif_shift != 0) {
        entry.carry_a = (uint8_t)carry_a;
        entry.wif_low_res[0] = (uint8_t)(residue >> 32);
        entry.wif_low_res[1] = (uint8_t)(residue >> 24);
        entry.wif_low_res[2] = (uint8_t)(residue >> 16);
        entry.wif_low_res[3] = (uint8_t)(residue >>  8);
        entry.wif_low_res[4] = (uint8_t)(residue      );
    }

    const uint32_t bucket = bsgs_bucket_index_key(x, y_parity, flags, bucket_mask);
    entry.bucket = bucket;
    baby_entries[local] = entry;

    if (add_to_bloom)
        bsgs_bloom_add_key(x, y_parity, flags, bloom, bloom_m_bits);
}

__global__ void bsgs_baby_bloom_bucket_count_tiled_kernel(
    const BsgsPlan* __restrict__ plan,
    BsgsBabyEntry*  __restrict__ baby_entries,
    uint64_t baby_count,
    uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    uint32_t* __restrict__ bucket_counts,
    uint32_t bucket_mask)
{
    const uint64_t high_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (high_idx >= plan->baby_high_count) return;

    ECPointA base;
    pointSetInfinity(base);

    const uint64_t high_gray = high_idx ^ (high_idx >> 1);
    for (uint32_t bit = 0; bit < plan->baby_high_bits; bit++) {
        if (((high_gray >> bit) & 1ULL) == 0) continue;
        const uint32_t abs_bit = plan->baby_low_bits + bit;
        uint32_t group = 0;
        uint32_t digit = 0;
        if (!bsgs_bit_to_group_digit(plan->baby, plan->baby_unknown, abs_bit, group, digit)) continue;
        bsgs_point_add_or_sub(base, plan->baby_contrib[group][digit], true);
    }

    const uint64_t base_idx = high_gray << plan->baby_low_bits;
    if (base_idx < baby_count && base_idx < plan->baby_count) {
        BsgsBabyEntry entry;
        uint64_t x[4];
        uint8_t y_parity = 0;
        uint8_t flags = 0;
        bsgs_extract_point_key(base, x, y_parity, flags);
        bsgs_store_baby_entry_from_key(entry, x, y_parity, flags, base_idx);
        const uint32_t bucket = bsgs_bucket_index_key(x, y_parity, flags, bucket_mask);
        entry.bucket = bucket;
        baby_entries[base_idx] = entry;
        bsgs_bloom_add_key(x, y_parity, flags, bloom, bloom_m_bits);
        atomicAdd(&bucket_counts[bucket], 1u);
    }

    if (LOW_SIZE <= 1) return;

    if (base.infinity) {
        for (uint32_t k = 1; k < LOW_SIZE; k++) {
            const uint64_t idx_a = base_idx | (uint64_t)k;
            if (idx_a >= baby_count || idx_a >= plan->baby_count) return;
            BsgsBabyEntry entry;
            ECPointA p;
            bsgs_point_to_ecpoint(plan->baby_low_dict[k], p);
            uint64_t x[4];
            uint8_t y_parity = 0;
            uint8_t flags = 0;
            bsgs_extract_point_key(p, x, y_parity, flags);
            bsgs_store_baby_entry_from_key(entry, x, y_parity, flags, idx_a);
            const uint32_t bucket = bsgs_bucket_index_key(x, y_parity, flags, bucket_mask);
            entry.bucket = bucket;
            baby_entries[idx_a] = entry;
            bsgs_bloom_add_key(x, y_parity, flags, bloom, bloom_m_bits);
            atomicAdd(&bucket_counts[bucket], 1u);
        }
        return;
    }

    uint64_t bx[4], by[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        bx[i] = base.X[i];
        by[i] = base.Y[i];
    }
    fieldNormalize(bx);
    fieldNormalize(by);

    int      tile_begin_compact[BSGS_LOW_MAX_TILES];
    uint64_t tile_prod       [BSGS_LOW_MAX_TILES][4];
    uint64_t tile_prefix_prod[BSGS_LOW_MAX_TILES][4];
    uint64_t tile_inv_prod   [BSGS_LOW_MAX_TILES][4];
    int      nt_tiles = 0;

    #pragma unroll 1
    for (int tile_begin = 1; tile_begin < LOW_SIZE; tile_begin += BSGS_LOW_TILE) {
        const int tile_end =
            (tile_begin + BSGS_LOW_TILE < LOW_SIZE)
                ? (tile_begin + BSGS_LOW_TILE)
                : LOW_SIZE;

        bool has_prod = false;
        uint64_t prod[4];

        #pragma unroll 1
        for (int k = tile_begin; k < tile_end; k++) {
            const BsgsPoint& low = plan->baby_low_dict[k];
            if (low.flags & BSGS_POINT_INFINITY) continue;

            uint64_t dx[4];
            fieldSub(low.x, bx, dx);
            if (bsgs_limbs_is_zero4(dx)) continue;

            if (!has_prod) {
                bsgs_limbs_copy4(dx, prod);
                has_prod = true;
            } else {
                uint64_t tmp[4];
                fieldMul(prod, dx, tmp);
                bsgs_limbs_copy4(tmp, prod);
            }
        }

        if (!has_prod) continue;
        tile_begin_compact[nt_tiles] = tile_begin;
        bsgs_limbs_copy4(prod, tile_prod[nt_tiles]);
        if (nt_tiles == 0) bsgs_limbs_copy4(prod, tile_prefix_prod[0]);
        else fieldMul(tile_prefix_prod[nt_tiles - 1], prod, tile_prefix_prod[nt_tiles]);
        nt_tiles++;
    }

    if (nt_tiles == 0) return;

    uint64_t inv_total[4];
    fieldNormalize(tile_prefix_prod[nt_tiles - 1]);
    fieldInv(tile_prefix_prod[nt_tiles - 1], inv_total);

    {
        uint64_t running_tile_inv[4];
        bsgs_limbs_copy4(inv_total, running_tile_inv);
        for (int t = nt_tiles - 1; t >= 0; t--) {
            if (t > 0) {
                fieldMul(running_tile_inv, tile_prefix_prod[t - 1], tile_inv_prod[t]);
                uint64_t tmp[4];
                fieldMul(running_tile_inv, tile_prod[t], tmp);
                bsgs_limbs_copy4(tmp, running_tile_inv);
            } else {
                bsgs_limbs_copy4(running_tile_inv, tile_inv_prod[0]);
            }
        }
    }

    for (int t = nt_tiles - 1; t >= 0; t--) {
        const int tile_begin = tile_begin_compact[t];
        const int tile_end =
            (tile_begin + BSGS_LOW_TILE < LOW_SIZE)
                ? (tile_begin + BSGS_LOW_TILE)
                : LOW_SIZE;

        int      valid_idx[BSGS_LOW_TILE];
        uint64_t acc_prod [BSGS_LOW_TILE][4];
        int      nv = 0;

        #pragma unroll 1
        for (int k = tile_begin; k < tile_end; k++) {
            const BsgsPoint& low = plan->baby_low_dict[k];
            if (low.flags & BSGS_POINT_INFINITY) continue;

            uint64_t dx[4];
            fieldSub(low.x, bx, dx);
            if (bsgs_limbs_is_zero4(dx)) continue;

            valid_idx[nv] = k;
            if (nv == 0) bsgs_limbs_copy4(dx, acc_prod[0]);
            else fieldMul(acc_prod[nv - 1], dx, acc_prod[nv]);
            nv++;
        }

        if (nv == 0) continue;

        uint64_t running_inv[4];
        bsgs_limbs_copy4(tile_inv_prod[t], running_inv);

        for (int pos = nv - 1; pos >= 0; pos--) {
            const int k = valid_idx[pos];
            const uint64_t idx_a = base_idx | (uint64_t)k;
            if (idx_a >= baby_count || idx_a >= plan->baby_count) continue;
            const BsgsPoint& low = plan->baby_low_dict[k];

            uint64_t dx_inv[4];
            if (pos > 0) fieldMul(running_inv, acc_prod[pos - 1], dx_inv);
            else bsgs_limbs_copy4(running_inv, dx_inv);

            uint64_t dx[4];
            fieldSub(low.x, bx, dx);
            if (pos > 0) {
                uint64_t tmp_run[4];
                fieldMul(running_inv, dx, tmp_run);
                bsgs_limbs_copy4(tmp_run, running_inv);
            }

            uint64_t dy[4];
            fieldSub(low.y, by, dy);

            uint64_t lam[4], lam2[4], x3[4], y3[4], tmp[4];
            fieldMul(dy, dx_inv, lam);
            fieldSqr(lam, lam2);
            fieldSub(lam2, bx, x3);
            fieldSub(x3, low.x, x3);
            fieldSub(bx, x3, tmp);
            fieldMul(lam, tmp, y3);
            fieldSub(y3, by, y3);
            fieldNormalize(x3);
            fieldNormalize(y3);

            BsgsBabyEntry entry;
            const uint8_t y_parity = (uint8_t)(y3[0] & 1ULL);
            bsgs_store_baby_entry_from_key(entry, x3, y_parity, 0, idx_a);
            const uint32_t bucket = bsgs_bucket_index_key(x3, y_parity, 0, bucket_mask);
            entry.bucket = bucket;
            baby_entries[idx_a] = entry;
            bsgs_bloom_add_key(x3, y_parity, 0, bloom, bloom_m_bits);
            atomicAdd(&bucket_counts[bucket], 1u);
        }
    }
}

__global__ void bsgs_bucket_scatter_kernel(
    const BsgsBabyEntry* __restrict__ baby_entries,
    BsgsBabyEntry* __restrict__ bucketed_entries,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t* __restrict__ bucket_cursor,
    uint64_t baby_count,
    uint32_t bucket_mask)
{
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= baby_count) return;

    const BsgsBabyEntry entry = baby_entries[tid];
    const uint32_t bucket = entry.bucket & bucket_mask;
    const uint32_t slot = bucket_offsets[bucket] + atomicAdd(&bucket_cursor[bucket], 1u);
    bucketed_entries[slot] = entry;
}

__device__ __forceinline__ bool bsgs_point_matches_baby_entry(
    const ECPointA& point,
    const BsgsBabyEntry& entry)
{
    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_extract_point_key(point, x, y_parity, flags);
    return bsgs_key_fingerprint(x, y_parity, flags) == entry.key_fp &&
           bsgs_key_fingerprint_hi(x, y_parity, flags) == entry.key_fp_hi;
}

__device__ __forceinline__ void bsgs_normalize_point_key(
    const ECPointA& point,
    uint64_t x[4],
    uint8_t& y_parity,
    uint8_t& flags)
{
    if (point.infinity) {
        #pragma unroll
        for (int i = 0; i < 4; i++) x[i] = 0;
        y_parity = 0;
        flags = BSGS_POINT_INFINITY;
        return;
    }

    uint64_t y[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        x[i] = point.X[i];
        y[i] = point.Y[i];
    }
    fieldNormalize(x);
    fieldNormalize(y);
    y_parity = (uint8_t)(y[0] & 1ULL);
    flags = 0;
}

__device__ __forceinline__ int bsgs_compare_key_to_entry(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags,
    const BsgsBabyEntry& entry)
{
    const uint64_t fp = bsgs_key_fingerprint(x, y_parity, flags);
    const uint32_t fp_hi = bsgs_key_fingerprint_hi(x, y_parity, flags);
    if (fp < entry.key_fp) return -1;
    if (fp > entry.key_fp) return 1;
    if (fp_hi < entry.key_fp_hi) return -1;
    if (fp_hi > entry.key_fp_hi) return 1;
    return 0;
}

__device__ __forceinline__ bool bsgs_try_bucketed_match(
    const ECPointA& point,
    const BsgsBabyEntry* __restrict__ baby_entries,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t bucket_mask,
    const uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    BsgsHit* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t hit_capacity,
    uint64_t idx_b,
    uint8_t carry,
    BsgsCandidate* __restrict__ candidate,
    bool stop_on_match)
{
    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_normalize_point_key(point, x, y_parity, flags);

    if (!bsgs_bloom_check_key(x, y_parity, flags, bloom, bloom_m_bits))
        return false;

    const uint32_t bucket = bsgs_bucket_index_key(x, y_parity, flags, bucket_mask);
    const uint32_t begin = bucket_offsets[bucket];
    const uint32_t end = bucket_offsets[bucket + 1u];
    bool matched = false;
    for (uint32_t i = begin; i < end; i++) {
        if (bsgs_compare_key_to_entry(x, y_parity, flags, baby_entries[i]) == 0) {
            matched = true;
            const uint32_t hit_slot = atomicAdd(hit_count, 1u);
            if (hit_slot < hit_capacity) {
                hits[hit_slot].idx_a = baby_entries[i].idx_a;
                hits[hit_slot].idx_b = idx_b;
                #pragma unroll
                for (int limb = 0; limb < 4; limb++) hits[hit_slot].lookup_x[limb] = x[limb];
                hits[hit_slot].lookup_y_parity = y_parity;
                hits[hit_slot].carry = carry;
            }
            if (stop_on_match && atomicCAS(&candidate->found, 0, 1) == 0) {
                candidate->idx_a = baby_entries[i].idx_a;
                candidate->idx_b = idx_b;
                candidate->carry = carry;
            }
            if (stop_on_match) return true;
        }
    }
    return stop_on_match && matched;
}

__device__ __forceinline__ bool bsgs_try_bucketed_match_hex8(
    const ECPointA& point,
    const BsgsHex8Entry* __restrict__ baby_entries,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t bucket_mask,
    const uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    BsgsHit* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t hit_capacity,
    uint64_t idx_b,
    BsgsCandidate* __restrict__ candidate,
    bool stop_on_match)
{
    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_normalize_point_key(point, x, y_parity, flags);

    if (!bsgs_bloom_check_key(x, y_parity, flags, bloom, bloom_m_bits))
        return false;

    const uint32_t key_fp = bsgs_hex8_fingerprint_key(x, y_parity, flags);
    const uint32_t bucket = bsgs_bucket_index_key(x, y_parity, flags, bucket_mask);
    const uint32_t begin = bucket_offsets[bucket];
    const uint32_t end = bucket_offsets[bucket + 1u];
    bool matched = false;
    for (uint32_t i = begin; i < end; i++) {
        const BsgsHex8Entry e = baby_entries[i];
        if (e.key_fp != key_fp) continue;
        matched = true;
        const uint32_t hit_slot = atomicAdd(hit_count, 1u);
        if (hit_slot < hit_capacity) {
            hits[hit_slot].idx_a = (uint64_t)e.idx_a;
            hits[hit_slot].idx_b = idx_b;
            #pragma unroll
            for (int limb = 0; limb < 4; limb++) hits[hit_slot].lookup_x[limb] = x[limb];
            hits[hit_slot].lookup_y_parity = y_parity;
            hits[hit_slot].carry = 0;
        }
        if (stop_on_match && atomicCAS(&candidate->found, 0, 1) == 0) {
            candidate->idx_a = (uint64_t)e.idx_a;
            candidate->idx_b = idx_b;
            candidate->carry = 0;
        }
        if (stop_on_match) return true;
    }
    return stop_on_match && matched;
}

__device__ __forceinline__ bool bsgs_try_bucketed_match(
    const ECPointA& point,
    const BsgsHex8Entry* __restrict__ baby_entries,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t bucket_mask,
    const uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    BsgsHit* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t hit_capacity,
    uint64_t idx_b,
    uint8_t,
    BsgsCandidate* __restrict__ candidate,
    bool stop_on_match)
{
    return bsgs_try_bucketed_match_hex8(
        point, baby_entries, bucket_offsets, bucket_mask, bloom, bloom_m_bits,
        hits, hit_count, hit_capacity, idx_b, candidate, stop_on_match);
}

__device__ __forceinline__ void bsgs_wif_add_weight_digit_bytes(
    const BsgsPlan* __restrict__ plan,
    uint32_t weight_idx,
    uint32_t digit,
    uint32_t acc[WIF_MAX_BYTES])
{
    if (digit == 0) return;
    uint32_t carry = 0;
    for (int j = (int)plan->wif_mask.decoded_bytes - 1; j >= 0; j--) {
        const uint32_t s =
            acc[j] + (uint32_t)plan->wif_weight_bytes[weight_idx][j] * digit + carry;
        acc[j] = s & 0xFFu;
        carry = s >> 8;
    }
}

__device__ __forceinline__ void bsgs_wif_decode_candidate_bytes(
    const BsgsPlan* __restrict__ plan,
    uint64_t idx_a,
    uint64_t idx_b,
    uint8_t out[WIF_MAX_BYTES])
{
    uint32_t acc[WIF_MAX_BYTES];
    #pragma unroll
    for (int i = 0; i < WIF_MAX_BYTES; i++)
        acc[i] = plan->wif_base_bytes[i];

    uint8_t baby_digits[BSGS_MAX_UNKN];
    uint64_t a = idx_a;
    for (int group = (int)plan->baby_unknown - 1; group >= 0; group--) {
        const uint32_t radix = plan->baby[group].radix ? plan->baby[group].radix : plan->radix;
        baby_digits[group] = (uint8_t)(a % radix);
        a /= radix;
    }
    for (uint32_t group = 0; group < plan->baby_unknown; group++)
        bsgs_wif_add_weight_digit_bytes(plan, group, baby_digits[group], acc);

    uint8_t giant_digits[BSGS_MAX_UNKN];
    uint64_t b = idx_b;
    for (int group = (int)plan->giant_unknown - 1; group >= 0; group--) {
        const uint32_t radix = plan->giant[group].radix ? plan->giant[group].radix : plan->radix;
        giant_digits[group] = (uint8_t)(b % radix);
        b /= radix;
    }
    for (uint32_t group = 0; group < plan->giant_unknown; group++)
        bsgs_wif_add_weight_digit_bytes(
            plan, plan->baby_unknown + group, giant_digits[group], acc);

    #pragma unroll
    for (int i = 0; i < WIF_MAX_BYTES; i++)
        out[i] = (uint8_t)acc[i];
}

__device__ __noinline__ bool bsgs_wif_candidate_checksum_ok(
    const BsgsPlan* __restrict__ plan,
    uint64_t idx_a,
    uint64_t idx_b)
{
    if (plan->wif_shift == 0) return true;

    uint8_t raw[WIF_MAX_BYTES];
    bsgs_wif_decode_candidate_bytes(plan, idx_a, idx_b, raw);
    if (raw[0] != 0x80u) return false;

    uint8_t h1[32], h2[32];
    if (plan->wif_mask.is_compressed) {
        if (raw[33] != 0x01u) return false;
        sha256_34bytes(raw, h1);
    } else {
        sha256_33bytes(raw, h1);
    }
    sha256_32bytes(h1, h2);

    const uint32_t co = plan->wif_mask.checksum_offset;
    return raw[co + 0] == h2[0] &&
           raw[co + 1] == h2[1] &&
           raw[co + 2] == h2[2] &&
           raw[co + 3] == h2[3];
}

// ---------------------------------------------------------------------------
// bsgs_try_bucketed_match_with_carry
//
// WIF exact-carry lookup — no carry probes, no batch inversion on probe points.
//
// The giant kernel passes giant_low_sum (unmasked accumulator of all giant
// wif_low_weight contributions) computed for free during bsgs_step_radix_incremental.
//
// Carry reconstruction (mathematically proven exact, ∈ {0,1,2} for any valid WIF):
//
//   stored in BsgsBabyEntry:
//     carry_a    = floor((wif_low_base[0] + baby_low_sum) / 2^wif_shift)
//     residue_ab = (wif_low_base[0] + baby_low_sum) & mask   [5 bytes BE]
//
//   computed per giant step:
//     carry_b    = giant_low_sum >> wif_shift
//     residue_b  = giant_low_sum & mask
//     interaction = (residue_ab + residue_b) >> wif_shift     [0 or 1]
//
//   carry_total = carry_a + carry_b + interaction              [exact, ≤ 2]
//
// For HEX mode (wif_shift == 0) the WIF fields are all-zero and carry_total = 0,
// so this function degenerates to the original zero-carry bucketed match.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool bsgs_try_bucketed_match_with_carry_probe(
    const BsgsPlan* __restrict__ plan,
    const ECPointA& probe_point,
    const BsgsBabyEntry* __restrict__ baby_entries,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t bucket_mask,
    const uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    BsgsHit* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t hit_capacity,
    uint64_t idx_b,
    BsgsCandidate* __restrict__ candidate,
    uint64_t giant_low_sum,
    uint32_t probe)
{
    const uint32_t shift  = plan->wif_shift;
    const uint64_t mask   = bsgs_low_mask_for_shift(shift);
    const uint64_t carry_b   = (shift != 0) ? (giant_low_sum >> shift)   : 0ULL;
    const uint64_t residue_b = (shift != 0) ? (giant_low_sum &  mask)    : 0ULL;

    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_normalize_point_key(probe_point, x, y_parity, flags);

    if (!bsgs_bloom_check_key(x, y_parity, flags, bloom, bloom_m_bits))
        return false;

    const uint32_t bucket = bsgs_bucket_index_key(x, y_parity, flags, bucket_mask);
    const uint32_t begin = bucket_offsets[bucket];
    const uint32_t end   = bucket_offsets[bucket + 1u];
    bool matched = false;

    for (uint32_t i = begin; i < end; i++) {
        const BsgsBabyEntry& e = baby_entries[i];
        if (bsgs_compare_key_to_entry(x, y_parity, flags, e) != 0) continue;

        uint64_t carry_total = 0;
        if (shift != 0) {
            const uint64_t residue_ab =
                ((uint64_t)e.wif_low_res[0] << 32) |
                ((uint64_t)e.wif_low_res[1] << 24) |
                ((uint64_t)e.wif_low_res[2] << 16) |
                ((uint64_t)e.wif_low_res[3] <<  8) |
                 (uint64_t)e.wif_low_res[4];
            const uint64_t interaction = (residue_ab + residue_b) >> shift;
            if (interaction != probe) continue;
            carry_total = (uint64_t)e.carry_a + carry_b + interaction;
        }

        if (shift != 0) {
            if (!bsgs_wif_candidate_checksum_ok(plan, e.idx_a, idx_b))
                continue;
        }

        matched = true;
        const uint32_t hit_slot = atomicAdd(hit_count, 1u);
        if (hit_slot < hit_capacity) {
            hits[hit_slot].idx_a = e.idx_a;
            hits[hit_slot].idx_b = idx_b;
            #pragma unroll
            for (int limb = 0; limb < 4; limb++) hits[hit_slot].lookup_x[limb] = x[limb];
            hits[hit_slot].lookup_y_parity = y_parity;
            hits[hit_slot].carry = (uint8_t)carry_total;
        }
        if (shift == 0 && atomicCAS(&candidate->found, 0, 1) == 0) {
            candidate->idx_a = e.idx_a;
            candidate->idx_b = idx_b;
            candidate->carry = (uint8_t)carry_total;
            return true;
        }

        // WIF masks can include checksum-only Base58 characters. Those
        // digits may produce duplicate ECC points with different WIF
        // checksums, so the GPU must emit every exact point duplicate in
        // the bucket and let the CPU WIF checksum/pubkey verifier select
        // the valid candidate. Returning on the first duplicate causes
        // false negatives for masks such as "##...###".
    }
    return matched;
}

__device__ __forceinline__ bool bsgs_try_bucketed_match_with_carry(
    const BsgsPlan* __restrict__ plan,
    const ECPointA& point,
    const BsgsBabyEntry* __restrict__ baby_entries,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t bucket_mask,
    const uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    BsgsHit* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t hit_capacity,
    uint64_t idx_b,
    BsgsCandidate* __restrict__ candidate,
    uint64_t giant_low_sum = 0ULL)
{
    const uint32_t shift  = plan->wif_shift;
    const uint32_t probe_count = (shift != 0) ? 2u : 1u;
    bool matched = false;

    for (uint32_t probe = 0; probe < probe_count; probe++) {
        ECPointA probe_point = point;
        if (probe != 0)
            bsgs_add_wif_cross_carry(probe_point, probe, true);
        matched |= bsgs_try_bucketed_match_with_carry_probe(
            plan, probe_point, baby_entries, bucket_offsets, bucket_mask,
            bloom, bloom_m_bits,
            hits, hit_count, hit_capacity, idx_b, candidate, giant_low_sum, probe);
    }
    return matched;
}

template<uint32_t WIF_SHIFT>
__device__ __forceinline__ bool bsgs_try_bucketed_match_with_carry_probe_wif(
    const BsgsPlan* __restrict__ plan,
    const ECPointA& probe_point,
    const BsgsBabyEntry* __restrict__ baby_entries,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t bucket_mask,
    const uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    BsgsHit* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t hit_capacity,
    uint64_t idx_b,
    BsgsCandidate* __restrict__ candidate,
    uint64_t giant_low_sum,
    uint32_t probe)
{
    const uint64_t mask = (WIF_SHIFT == 64u) ? ~0ULL : ((1ULL << WIF_SHIFT) - 1ULL);
    const uint64_t carry_b = giant_low_sum >> WIF_SHIFT;
    const uint64_t residue_b = giant_low_sum & mask;

    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_normalize_point_key(probe_point, x, y_parity, flags);

    if (!bsgs_bloom_check_key(x, y_parity, flags, bloom, bloom_m_bits))
        return false;

    const uint32_t bucket = bsgs_bucket_index_key(x, y_parity, flags, bucket_mask);
    const uint32_t begin = bucket_offsets[bucket];
    const uint32_t end = bucket_offsets[bucket + 1u];
    bool matched = false;

    for (uint32_t i = begin; i < end; i++) {
        const BsgsBabyEntry& e = baby_entries[i];
        if (bsgs_compare_key_to_entry(x, y_parity, flags, e) != 0) continue;

        const uint64_t residue_ab =
            ((uint64_t)e.wif_low_res[0] << 32) |
            ((uint64_t)e.wif_low_res[1] << 24) |
            ((uint64_t)e.wif_low_res[2] << 16) |
            ((uint64_t)e.wif_low_res[3] <<  8) |
             (uint64_t)e.wif_low_res[4];
        const uint64_t interaction = (residue_ab + residue_b) >> WIF_SHIFT;
        if (interaction != probe) continue;
        const uint64_t carry_total = (uint64_t)e.carry_a + carry_b + interaction;

        if (!bsgs_wif_candidate_checksum_ok(plan, e.idx_a, idx_b))
            continue;

        matched = true;
        const uint32_t hit_slot = atomicAdd(hit_count, 1u);
        if (hit_slot < hit_capacity) {
            hits[hit_slot].idx_a = e.idx_a;
            hits[hit_slot].idx_b = idx_b;
            #pragma unroll
            for (int limb = 0; limb < 4; limb++) hits[hit_slot].lookup_x[limb] = x[limb];
            hits[hit_slot].lookup_y_parity = y_parity;
            hits[hit_slot].carry = (uint8_t)carry_total;
        }
    }
    return matched;
}

template<uint32_t WIF_SHIFT>
__device__ __forceinline__ bool bsgs_try_bucketed_match_with_carry_wif(
    const BsgsPlan* __restrict__ plan,
    const ECPointA& point,
    const BsgsBabyEntry* __restrict__ baby_entries,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t bucket_mask,
    const uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    BsgsHit* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t hit_capacity,
    uint64_t idx_b,
    BsgsCandidate* __restrict__ candidate,
    uint64_t giant_low_sum)
{
    bool matched = false;
    #pragma unroll
    for (uint32_t probe = 0; probe < 2u; probe++) {
        ECPointA probe_point = point;
        if (probe != 0)
            bsgs_add_wif_cross_carry(probe_point, probe, true);
        matched |= bsgs_try_bucketed_match_with_carry_probe_wif<WIF_SHIFT>(
            plan, probe_point, baby_entries, bucket_offsets, bucket_mask,
            bloom, bloom_m_bits,
            hits, hit_count, hit_capacity, idx_b, candidate, giant_low_sum, probe);
    }
    return matched;
}

template<uint32_t WIF_SHIFT>
__device__ __forceinline__ bool bsgs_emit_bloom_hit_with_carry_wif(
    const BsgsPlan* __restrict__ plan,
    const ECPointA& point,
    const uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    BsgsHit* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t hit_capacity,
    uint64_t idx_b,
    BsgsCandidate* __restrict__ candidate)
{
    (void)plan;
    bool matched = false;
    #pragma unroll
    for (uint32_t probe = 0; probe < 2u; probe++) {
        ECPointA probe_point = point;
        if (probe != 0)
            bsgs_add_wif_cross_carry(probe_point, probe, true);

        uint64_t x[4];
        uint8_t y_parity = 0;
        uint8_t flags = 0;
        bsgs_normalize_point_key(probe_point, x, y_parity, flags);
        if (!bsgs_bloom_check_key(x, y_parity, flags, bloom, bloom_m_bits))
            continue;

        matched = true;
        const uint32_t hit_slot = atomicAdd(hit_count, 1u);
        if (hit_slot < hit_capacity) {
            hits[hit_slot].idx_a = 0;
            hits[hit_slot].idx_b = idx_b;
            #pragma unroll
            for (int limb = 0; limb < 4; limb++) hits[hit_slot].lookup_x[limb] = x[limb];
            hits[hit_slot].lookup_y_parity = y_parity;
            hits[hit_slot].carry = (uint8_t)probe;
        }
    }
    return matched;
}

template<uint32_t WIF_SHIFT, bool BLOOM_ONLY>
__device__ __noinline__ bool bsgs_resolve_radix58_candidate_wif(
    const BsgsPlan* __restrict__ plan,
    const uint64_t bx[4],
    const uint64_t by[4],
    const uint64_t dx_inv[4],
    const BsgsPoint& low,
    const BsgsBabyEntry* __restrict__ baby_entries,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t bucket_mask,
    const uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    BsgsHit* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t hit_capacity,
    uint64_t idx,
    BsgsCandidate* __restrict__ candidate,
    uint64_t base_low_sum,
    uint64_t base_top_sum,
    uint64_t low_weight,
    uint64_t low_mask,
    uint32_t k)
{
    uint64_t lx[4], ly[4];
    #pragma unroll
    for (int limb = 0; limb < 4; limb++) {
        lx[limb] = low.x[limb];
        ly[limb] = low.y[limb];
    }

    uint64_t dy[4];
    fieldSub(ly, by, dy);

    uint64_t lam[4], lam2[4], x3[4], y3[4], tmp[4];
    fieldMul(dy, dx_inv, lam);
    fieldSqr(lam, lam2);
    fieldSub(lam2, bx, x3);
    fieldSub(x3, lx, x3);
    fieldSub(bx, x3, tmp);
    fieldMul(lam, tmp, y3);
    fieldSub(y3, by, y3);
    fieldNormalize(x3);
    fieldNormalize(y3);

    ECPointA candidate_point;
    candidate_point.infinity = false;
    #pragma unroll
    for (int limb = 0; limb < 4; limb++) {
        candidate_point.X[limb] = x3[limb];
        candidate_point.Y[limb] = y3[limb];
    }

    const uint64_t low_sum = base_low_sum + (((uint64_t)k * low_weight) & low_mask);
    const uint64_t top_sum = low_sum >> WIF_SHIFT;
    bsgs_adjust_wif_cross_carry(candidate_point, (int64_t)top_sum - (int64_t)base_top_sum, true);

    if constexpr (BLOOM_ONLY) {
        return bsgs_emit_bloom_hit_with_carry_wif<WIF_SHIFT>(
            plan, candidate_point, bloom, bloom_m_bits,
            hits, hit_count, hit_capacity, idx, candidate);
    } else {
        return bsgs_try_bucketed_match_with_carry_wif<WIF_SHIFT>(
            plan, candidate_point, baby_entries, bucket_offsets, bucket_mask,
            bloom, bloom_m_bits,
            hits, hit_count, hit_capacity, idx, candidate, low_sum);
    }
}


// ============================================================================
// COMMON HOST LAUNCHERS
// ============================================================================

static inline cudaError_t bsgs_launch_baby_bloom_bucket_count(
    const BsgsGpuBuffers& buffers,
    uint64_t baby_count,
    int threads = 256)
{
    if (baby_count > buffers.baby_capacity) return cudaErrorInvalidValue;
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_bucket_counts || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    const int blocks = (int)((baby_count + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_baby_bloom_bucket_count_kernel<<<blocks, threads>>>(
        buffers.d_plan, buffers.d_baby_entries, baby_count,
        buffers.d_bsgs_bloom, buffers.bloom_m_bits,
        buffers.d_bucket_counts, buffers.bucket_count - 1u);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_hex8_baby_count(
    const BsgsGpuBuffers& buffers,
    uint64_t baby_count,
    int threads = 256)
{
    if (baby_count > buffers.baby_capacity) return cudaErrorInvalidValue;
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_bucket_counts || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    const int blocks = (int)((baby_count + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_hex8_baby_count_kernel<<<blocks, threads>>>(
        buffers.d_plan, baby_count,
        buffers.d_bsgs_bloom, buffers.bloom_m_bits,
        buffers.d_bucket_counts, buffers.bucket_count - 1u);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_baby_bloom_bucket_emit(
    const BsgsGpuBuffers& buffers,
    uint64_t baby_offset,
    uint64_t baby_count,
    uint32_t bucket_mask,
    bool add_to_bloom,
    int threads = 256)
{
    if (baby_count > buffers.baby_capacity) return cudaErrorInvalidValue;
    if (!buffers.d_baby_entries) return cudaErrorInvalidValue;
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    const int blocks = (int)((baby_count + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_baby_bloom_bucket_emit_kernel<<<blocks, threads>>>(
        buffers.d_plan, buffers.d_baby_entries, baby_offset, baby_count,
        buffers.d_bsgs_bloom, buffers.bloom_m_bits,
        bucket_mask, add_to_bloom ? 1u : 0u);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_baby_bloom_bucket_count_tiled(
    const BsgsGpuBuffers& buffers,
    uint64_t baby_count,
    int threads = 256)
{
    if (baby_count > buffers.baby_capacity) return cudaErrorInvalidValue;
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_bucket_counts || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    const uint64_t high_count = (baby_count + (uint64_t)LOW_SIZE - 1ULL) / (uint64_t)LOW_SIZE;
    const int blocks = (int)((high_count + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_baby_bloom_bucket_count_tiled_kernel<<<blocks, threads>>>(
        buffers.d_plan, buffers.d_baby_entries, baby_count,
        buffers.d_bsgs_bloom, buffers.bloom_m_bits,
        buffers.d_bucket_counts, buffers.bucket_count - 1u);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_bucket_scatter(
    const BsgsGpuBuffers& buffers,
    uint64_t baby_count,
    int threads = 256)
{
    if (baby_count > buffers.baby_capacity) return cudaErrorInvalidValue;
    if (!buffers.d_bucketed_entries || !buffers.d_bucket_offsets ||
        !buffers.d_bucket_cursor || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    const int blocks = (int)((baby_count + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_bucket_scatter_kernel<<<blocks, threads>>>(
        buffers.d_baby_entries, buffers.d_bucketed_entries,
        buffers.d_bucket_offsets, buffers.d_bucket_cursor,
        baby_count, buffers.bucket_count - 1u);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_hex8_baby_scatter(
    const BsgsGpuBuffers& buffers,
    uint64_t baby_count,
    int threads = 256)
{
    if (baby_count > buffers.baby_capacity) return cudaErrorInvalidValue;
    if (!buffers.d_hex8_bucketed_entries || !buffers.d_bucket_offsets ||
        !buffers.d_bucket_cursor || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    const int blocks = (int)((baby_count + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_hex8_baby_scatter_kernel<<<blocks, threads>>>(
        buffers.d_plan, buffers.d_hex8_bucketed_entries,
        buffers.d_bucket_offsets, buffers.d_bucket_cursor,
        baby_count, buffers.bucket_count - 1u);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}
