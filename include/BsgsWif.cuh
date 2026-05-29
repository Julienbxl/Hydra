#pragma once
#include "BsgsCommon.cuh"

__global__ void bsgs_giant_bloom_bucketed_kernel(
    const BsgsPlan*       __restrict__ plan,
    const BsgsBabyEntry*  __restrict__ baby_entries,
    const uint32_t*       __restrict__ bucket_offsets,
    uint32_t                          bucket_mask,
    const uint64_t*       __restrict__ bloom,
    uint64_t                          bloom_m_bits,
    BsgsHit*              __restrict__ hits,
    uint32_t*             __restrict__ hit_count,
    uint32_t                          hit_capacity,
    uint64_t                          giant_offset,
    uint64_t                          giant_limit,
    BsgsCandidate*        __restrict__ candidate)
{
    const uint64_t local_tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (local_tid >= giant_limit) return;
    const uint64_t tid = giant_offset + local_tid;
    if (tid >= plan->giant_count) return;
    if (atomicAdd(&candidate->found, 0) != 0) return;

    ECPointA acc;
    bsgs_build_acc_from_radix_index(plan, true, tid, acc);

    // Compute giant_low_sum for exact WIF carry reconstruction (free for HEX: wif_shift==0).
    uint64_t giant_low_sum = 0;
    if (plan->wif_shift != 0) {
        uint8_t digits_tmp[BSGS_MAX_UNKN];
        giant_low_sum = bsgs_init_radix_digits(plan, true, tid, digits_tmp);
    }

    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_normalize_point_key(acc, x, y_parity, flags);
    if (plan->wif_shift == 0) {
        bsgs_try_bucketed_match(
            acc, baby_entries, bucket_offsets, bucket_mask,
            bloom, bloom_m_bits,
            hits, hit_count, hit_capacity, tid, 0, candidate, true);
    } else {
        bsgs_try_bucketed_match_with_carry(
            plan, acc, baby_entries, bucket_offsets, bucket_mask,
            bloom, bloom_m_bits,
            hits, hit_count, hit_capacity, tid, candidate, giant_low_sum);
    }
}

__global__ void bsgs_giant_bloom_bucketed_radix_kernel(
    const BsgsPlan*       __restrict__ plan,
    const BsgsBabyEntry*  __restrict__ baby_entries,
    const uint32_t*       __restrict__ bucket_offsets,
    uint32_t                          bucket_mask,
    const uint64_t*       __restrict__ bloom,
    uint64_t                          bloom_m_bits,
    BsgsHit*              __restrict__ hits,
    uint32_t*             __restrict__ hit_count,
    uint32_t                          hit_capacity,
    uint64_t                          giant_offset,
    uint64_t                          giant_limit,
    BsgsCandidate*        __restrict__ candidate)
{
    const uint64_t worker = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t local_start = worker * (uint64_t)BSGS_RADIX_ITEMS_PER_THREAD;
    if (local_start >= giant_limit) return;

    uint64_t local_end = local_start + (uint64_t)BSGS_RADIX_ITEMS_PER_THREAD;
    if (local_end > giant_limit) local_end = giant_limit;

    const uint64_t start_idx = giant_offset + local_start;
    if (start_idx >= plan->giant_count) return;

    ECPointA acc;
    bsgs_build_acc_from_radix_index(plan, true, start_idx, acc);

    uint8_t digits[BSGS_MAX_UNKN];
    uint64_t low_sum = bsgs_init_radix_digits(plan, true, start_idx, digits);
    uint64_t top_sum = (plan->wif_shift != 0) ? (low_sum >> plan->wif_shift) : 0ULL;
    if (top_sum != 0)
        bsgs_add_wif_cross_carry(acc, top_sum, true);

    for (uint64_t local = local_start; local < local_end; local++) {
        const uint64_t tid = giant_offset + local;
        if (tid >= plan->giant_count) return;
        if (plan->wif_shift == 0 && atomicAdd(&candidate->found, 0) != 0) return;

        if (bsgs_try_bucketed_match_with_carry(
                plan, acc, baby_entries, bucket_offsets, bucket_mask,
                bloom, bloom_m_bits,
                hits, hit_count, hit_capacity, tid, candidate, low_sum))
            if (plan->wif_shift == 0) return;

        if (local + 1ULL < local_end)
            bsgs_step_radix_incremental(plan, true, digits, low_sum, top_sum, acc);
    }
}


template<uint32_t WIF_SHIFT, bool BLOOM_ONLY>
__global__ void bsgs_giant_bloom_bucketed_radix58_tiled_wif_kernel(
    const BsgsPlan*       __restrict__ plan,
    const BsgsBabyEntry*  __restrict__ baby_entries,
    const uint32_t*       __restrict__ bucket_offsets,
    uint32_t                          bucket_mask,
    const uint64_t*       __restrict__ bloom,
    uint64_t                          bloom_m_bits,
    BsgsHit*              __restrict__ hits,
    uint32_t*             __restrict__ hit_count,
    uint32_t                          hit_capacity,
    uint64_t                          giant_offset,
    uint64_t                          giant_limit,
    BsgsCandidate*        __restrict__ candidate)
{
    if (plan->radix != 58u || plan->giant_unknown == 0) return;

    const uint64_t range_begin = giant_offset;
    uint64_t range_end = giant_offset + giant_limit;
    if (range_end < range_begin || range_end > plan->giant_count)
        range_end = plan->giant_count;
    if (range_begin >= range_end) return;

    const uint64_t first_tile = range_begin / (uint64_t)BSGS_RADIX58_TILE;
    const uint64_t row_worker = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t first_worker_tile = first_tile + row_worker * (uint64_t)BSGS_RADIX58_ROWS_PER_THREAD;
    uint64_t tile_base_idx = first_worker_tile * (uint64_t)BSGS_RADIX58_TILE;
    if (tile_base_idx >= range_end) return;

    ECPointA base;
    bsgs_build_acc_from_radix_index(plan, true, tile_base_idx, base);

    uint8_t base_digits[BSGS_MAX_UNKN];
    uint64_t base_low_sum = bsgs_init_radix_digits(plan, true, tile_base_idx, base_digits);
    uint64_t base_top_sum = base_low_sum >> WIF_SHIFT;
    if (base_top_sum != 0)
        bsgs_add_wif_cross_carry(base, base_top_sum, true);

    const uint32_t low_group = plan->giant_unknown - 1u;
    const BsgsUnknown& low_unknown = plan->giant[low_group];
    const uint64_t low_mask = (WIF_SHIFT == 64u) ? ~0ULL : ((1ULL << WIF_SHIFT) - 1ULL);

    #pragma unroll 1
    for (uint32_t row = 0; row < BSGS_RADIX58_ROWS_PER_THREAD; row++) {
        if (tile_base_idx >= range_end || tile_base_idx >= plan->giant_count) break;

        if (tile_base_idx >= range_begin) {
            if constexpr (BLOOM_ONLY) {
                bsgs_emit_bloom_hit_with_carry_wif<WIF_SHIFT>(
                    plan, base, bloom, bloom_m_bits,
                    hits, hit_count, hit_capacity, tile_base_idx, candidate);
            } else {
                bsgs_try_bucketed_match_with_carry_wif<WIF_SHIFT>(
                    plan, base, baby_entries, bucket_offsets, bucket_mask,
                    bloom, bloom_m_bits,
                    hits, hit_count, hit_capacity, tile_base_idx, candidate, base_low_sum);
            }
        }

        uint64_t bx[4], by[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            bx[i] = base.X[i];
            by[i] = base.Y[i];
        }
        fieldNormalize(bx);
        fieldNormalize(by);

        int      valid_idx[BSGS_RADIX58_TILE - 1u];
        uint64_t acc_prod [BSGS_RADIX58_TILE - 1u][4];
        int      nv = 0;

        #pragma unroll 1
        for (uint32_t k = 1; k < BSGS_RADIX58_TILE; k++) {
            const uint64_t idx = tile_base_idx + (uint64_t)k;
            if (idx < range_begin || idx >= range_end || idx >= plan->giant_count) continue;

            const BsgsPoint& low = plan->giant_contrib[low_group][k];
            if (low.flags & BSGS_POINT_INFINITY) {
                ECPointA candidate_point = base;
                uint64_t low_sum = base_low_sum + (((uint64_t)k * low_unknown.wif_low_weight) & low_mask);
                const uint64_t top_sum = low_sum >> WIF_SHIFT;
                bsgs_adjust_wif_cross_carry(candidate_point, (int64_t)top_sum - (int64_t)base_top_sum, true);
                if constexpr (BLOOM_ONLY) {
                    bsgs_emit_bloom_hit_with_carry_wif<WIF_SHIFT>(
                        plan, candidate_point, bloom, bloom_m_bits,
                        hits, hit_count, hit_capacity, idx, candidate);
                } else {
                    bsgs_try_bucketed_match_with_carry_wif<WIF_SHIFT>(
                        plan, candidate_point, baby_entries, bucket_offsets, bucket_mask,
                        bloom, bloom_m_bits, hits, hit_count, hit_capacity,
                        idx, candidate, low_sum);
                }
                continue;
            }

            uint64_t dx[4];
            fieldSub(low.x, bx, dx);
            if (bsgs_limbs_is_zero4(dx)) {
                ECPointA candidate_point = base;
                bsgs_point_add_or_sub(candidate_point, low, true);
                uint64_t low_sum = base_low_sum + (((uint64_t)k * low_unknown.wif_low_weight) & low_mask);
                const uint64_t top_sum = low_sum >> WIF_SHIFT;
                bsgs_adjust_wif_cross_carry(candidate_point, (int64_t)top_sum - (int64_t)base_top_sum, true);
                if constexpr (BLOOM_ONLY) {
                    bsgs_emit_bloom_hit_with_carry_wif<WIF_SHIFT>(
                        plan, candidate_point, bloom, bloom_m_bits,
                        hits, hit_count, hit_capacity, idx, candidate);
                } else {
                    bsgs_try_bucketed_match_with_carry_wif<WIF_SHIFT>(
                        plan, candidate_point, baby_entries, bucket_offsets, bucket_mask,
                        bloom, bloom_m_bits, hits, hit_count, hit_capacity,
                        idx, candidate, low_sum);
                }
                continue;
            }

            valid_idx[nv] = (int)k;
            if (nv == 0) bsgs_limbs_copy4(dx, acc_prod[0]);
            else fieldMul(acc_prod[nv - 1], dx, acc_prod[nv]);
            nv++;
        }

        if (nv > 0) {
            uint64_t running_inv[4];
            fieldNormalize(acc_prod[nv - 1]);
            fieldInv(acc_prod[nv - 1], running_inv);

            for (int pos = nv - 1; pos >= 0; pos--) {
                const uint32_t k = (uint32_t)valid_idx[pos];
                const BsgsPoint& low = plan->giant_contrib[low_group][k];

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

                const uint64_t idx = tile_base_idx + (uint64_t)k;
                if constexpr (BLOOM_ONLY) {
                    bsgs_resolve_radix58_candidate_wif<WIF_SHIFT, true>(
                        plan, bx, by, dx_inv, low,
                        baby_entries, bucket_offsets, bucket_mask,
                        bloom, bloom_m_bits,
                        hits, hit_count, hit_capacity, idx, candidate,
                        base_low_sum, base_top_sum,
                        low_unknown.wif_low_weight, low_mask, k);
                } else {
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

                    ECPointA candidate_point;
                    candidate_point.infinity = false;
                    #pragma unroll
                    for (int limb = 0; limb < 4; limb++) {
                        candidate_point.X[limb] = x3[limb];
                        candidate_point.Y[limb] = y3[limb];
                    }

                    uint64_t low_sum = base_low_sum + (((uint64_t)k * low_unknown.wif_low_weight) & low_mask);
                    const uint64_t top_sum = low_sum >> WIF_SHIFT;
                    bsgs_adjust_wif_cross_carry(candidate_point, (int64_t)top_sum - (int64_t)base_top_sum, true);

                    bsgs_try_bucketed_match_with_carry_wif<WIF_SHIFT>(
                        plan, candidate_point, baby_entries, bucket_offsets, bucket_mask,
                        bloom, bloom_m_bits, hits, hit_count, hit_capacity,
                        idx, candidate, low_sum);
                }
            }
        }

        const uint64_t next_tile_base = tile_base_idx + (uint64_t)BSGS_RADIX58_TILE;
        if (row + 1u >= BSGS_RADIX58_ROWS_PER_THREAD ||
            next_tile_base >= range_end ||
            next_tile_base >= plan->giant_count)
            break;

        bsgs_step_radix58_tile_base(plan, true, base_digits, base_low_sum, base_top_sum, base);
        tile_base_idx = next_tile_base;
    }
}


// ============================================================================
// WIF / RADIX HOST LAUNCHERS
// ============================================================================

static inline cudaError_t bsgs_launch_giant_bloom_bucketed(
    const BsgsGpuBuffers& buffers,
    uint64_t giant_offset,
    uint64_t giant_count,
    int threads = 256)
{
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_bucket_offsets || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    if (buffers.hit_capacity > UINT32_MAX) return cudaErrorInvalidValue;
    const int blocks = (int)((giant_count + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_giant_bloom_bucketed_kernel<<<blocks, threads>>>(
        buffers.d_plan, buffers.d_bucketed_entries, buffers.d_bucket_offsets,
        buffers.bucket_count - 1u,
        buffers.d_bsgs_bloom, buffers.bloom_m_bits,
        buffers.d_hits, buffers.d_hit_count, (uint32_t)buffers.hit_capacity,
        giant_offset, giant_count,
        buffers.d_candidate);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_giant_bloom_bucketed_radix(
    const BsgsGpuBuffers& buffers,
    uint64_t giant_offset,
    uint64_t giant_count,
    int threads = 256)
{
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_bucket_offsets || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    if (buffers.hit_capacity > UINT32_MAX) return cudaErrorInvalidValue;
    const uint64_t workers =
        (giant_count + (uint64_t)BSGS_RADIX_ITEMS_PER_THREAD - 1ULL) /
        (uint64_t)BSGS_RADIX_ITEMS_PER_THREAD;
    const int blocks = (int)((workers + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_giant_bloom_bucketed_radix_kernel<<<blocks, threads>>>(
        buffers.d_plan, buffers.d_bucketed_entries, buffers.d_bucket_offsets,
        buffers.bucket_count - 1u,
        buffers.d_bsgs_bloom, buffers.bloom_m_bits,
        buffers.d_hits, buffers.d_hit_count, (uint32_t)buffers.hit_capacity,
        giant_offset, giant_count,
        buffers.d_candidate);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_giant_bloom_bucketed_radix58_tiled(
    const BsgsGpuBuffers& buffers,
    uint64_t giant_offset,
    uint64_t giant_count,
    uint32_t wif_shift = 0,
    int threads = 128)
{
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_bucket_offsets || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    if (buffers.hit_capacity > UINT32_MAX) return cudaErrorInvalidValue;
    if (giant_count == 0) return cudaSuccess;

    const uint64_t first_tile = giant_offset / (uint64_t)BSGS_RADIX58_TILE;
    const uint64_t last_idx = giant_offset + giant_count - 1ULL;
    const uint64_t last_tile = last_idx / (uint64_t)BSGS_RADIX58_TILE;
    const uint64_t tile_count = last_tile - first_tile + 1ULL;
    const uint64_t worker_count = (tile_count + (uint64_t)BSGS_RADIX58_ROWS_PER_THREAD - 1ULL) /
                                  (uint64_t)BSGS_RADIX58_ROWS_PER_THREAD;
    const int blocks = (int)((worker_count + (uint64_t)threads - 1) / (uint64_t)threads);
    if (wif_shift == 40u) {
        bsgs_giant_bloom_bucketed_radix58_tiled_wif_kernel<40, false><<<blocks, threads>>>(
            buffers.d_plan, buffers.d_bucketed_entries, buffers.d_bucket_offsets,
            buffers.bucket_count - 1u,
            buffers.d_bsgs_bloom, buffers.bloom_m_bits,
            buffers.d_hits, buffers.d_hit_count, (uint32_t)buffers.hit_capacity,
            giant_offset, giant_count,
            buffers.d_candidate);
    } else if (wif_shift == 32u) {
        bsgs_giant_bloom_bucketed_radix58_tiled_wif_kernel<32, false><<<blocks, threads>>>(
            buffers.d_plan, buffers.d_bucketed_entries, buffers.d_bucket_offsets,
            buffers.bucket_count - 1u,
            buffers.d_bsgs_bloom, buffers.bloom_m_bits,
            buffers.d_hits, buffers.d_hit_count, (uint32_t)buffers.hit_capacity,
            giant_offset, giant_count,
            buffers.d_candidate);
    } else {
        return cudaErrorInvalidValue;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_giant_bloom_only_radix58_tiled(
    const BsgsGpuBuffers& buffers,
    uint64_t giant_offset,
    uint64_t giant_count,
    uint32_t wif_shift,
    int threads = 128)
{
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (buffers.hit_capacity > UINT32_MAX) return cudaErrorInvalidValue;
    if (giant_count == 0) return cudaSuccess;

    const uint64_t first_tile = giant_offset / (uint64_t)BSGS_RADIX58_TILE;
    const uint64_t last_idx = giant_offset + giant_count - 1ULL;
    const uint64_t last_tile = last_idx / (uint64_t)BSGS_RADIX58_TILE;
    const uint64_t tile_count = last_tile - first_tile + 1ULL;
    const uint64_t worker_count = (tile_count + (uint64_t)BSGS_RADIX58_ROWS_PER_THREAD - 1ULL) /
                                  (uint64_t)BSGS_RADIX58_ROWS_PER_THREAD;
    const int blocks = (int)((worker_count + (uint64_t)threads - 1) / (uint64_t)threads);
    if (wif_shift == 40u) {
        bsgs_giant_bloom_bucketed_radix58_tiled_wif_kernel<40, true><<<blocks, threads>>>(
            buffers.d_plan, nullptr, nullptr, 0u,
            buffers.d_bsgs_bloom, buffers.bloom_m_bits,
            buffers.d_hits, buffers.d_hit_count, (uint32_t)buffers.hit_capacity,
            giant_offset, giant_count,
            buffers.d_candidate);
    } else if (wif_shift == 32u) {
        bsgs_giant_bloom_bucketed_radix58_tiled_wif_kernel<32, true><<<blocks, threads>>>(
            buffers.d_plan, nullptr, nullptr, 0u,
            buffers.d_bsgs_bloom, buffers.bloom_m_bits,
            buffers.d_hits, buffers.d_hit_count, (uint32_t)buffers.hit_capacity,
            giant_offset, giant_count,
            buffers.d_candidate);
    } else {
        return cudaErrorInvalidValue;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}
