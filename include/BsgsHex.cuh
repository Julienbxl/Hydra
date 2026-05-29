#pragma once
#include "BsgsCommon.cuh"

__device__ __forceinline__ bool bsgs_emit_bloom_hit_hex(
    const ECPointA& point,
    const uint64_t* __restrict__ bloom,
    uint64_t bloom_m_bits,
    BsgsHit* __restrict__ hits,
    uint32_t* __restrict__ hit_count,
    uint32_t hit_capacity,
    uint64_t idx_b)
{
    uint64_t x[4];
    uint8_t y_parity = 0;
    uint8_t flags = 0;
    bsgs_normalize_point_key(point, x, y_parity, flags);

    if (!bsgs_bloom_check_key(x, y_parity, flags, bloom, bloom_m_bits))
        return false;

    const uint32_t hit_slot = atomicAdd(hit_count, 1u);
    if (hit_slot < hit_capacity) {
        hits[hit_slot].idx_a = 0;
        hits[hit_slot].idx_b = idx_b;
        #pragma unroll
        for (int limb = 0; limb < 4; limb++) hits[hit_slot].lookup_x[limb] = x[limb];
        hits[hit_slot].lookup_y_parity = y_parity;
        hits[hit_slot].carry = 0;
    }
    return true;
}

template<typename BabyEntryT>
__global__ void bsgs_giant_bloom_bucketed_gray_kernel_t(
    const BsgsPlan*       __restrict__ plan,
    const BabyEntryT*     __restrict__ baby_entries,
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
    const uint64_t local_start = worker * (uint64_t)BSGS_GRAY_ITEMS_PER_THREAD;
    if (local_start >= giant_limit) return;

    uint64_t local_end = local_start + (uint64_t)BSGS_GRAY_ITEMS_PER_THREAD;
    if (local_end > giant_limit) local_end = giant_limit;

    const uint64_t start_idx = giant_offset + local_start;
    if (start_idx >= plan->giant_count) return;

    ECPointA acc;
    bsgs_build_acc_from_binary_gray(plan, true, start_idx, acc);

    for (uint64_t local = local_start; local < local_end; local++) {
        const uint64_t tid = giant_offset + local;
        if (tid >= plan->giant_count) return;
        if (atomicAdd(&candidate->found, 0) != 0) return;

        const uint64_t value_idx = tid ^ (tid >> 1);
        if (bsgs_try_bucketed_match(
                acc, baby_entries, bucket_offsets, bucket_mask, bloom, bloom_m_bits,
                hits, hit_count, hit_capacity, value_idx, 0, candidate, true))
            return;

        if (local + 1ULL < local_end)
            bsgs_step_binary_gray(plan, true, plan->giant_unknown, tid, acc);
    }
}

__global__ void bsgs_giant_bloom_only_gray_kernel(
    const BsgsPlan*       __restrict__ plan,
    const uint64_t*       __restrict__ bloom,
    uint64_t                          bloom_m_bits,
    BsgsHit*              __restrict__ hits,
    uint32_t*             __restrict__ hit_count,
    uint32_t                          hit_capacity,
    uint64_t                          giant_offset,
    uint64_t                          giant_limit)
{
    const uint64_t worker = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t local_start = worker * (uint64_t)BSGS_GRAY_ITEMS_PER_THREAD;
    if (local_start >= giant_limit) return;

    uint64_t local_end = local_start + (uint64_t)BSGS_GRAY_ITEMS_PER_THREAD;
    if (local_end > giant_limit) local_end = giant_limit;

    const uint64_t start_idx = giant_offset + local_start;
    if (start_idx >= plan->giant_count) return;

    ECPointA acc;
    bsgs_build_acc_from_binary_gray(plan, true, start_idx, acc);

    for (uint64_t local = local_start; local < local_end; local++) {
        const uint64_t tid = giant_offset + local;
        if (tid >= plan->giant_count) return;

        const uint64_t value_idx = tid ^ (tid >> 1);
        bsgs_emit_bloom_hit_hex(
            acc, bloom, bloom_m_bits, hits, hit_count, hit_capacity, value_idx);

        if (local + 1ULL < local_end)
            bsgs_step_binary_gray(plan, true, plan->giant_unknown, tid, acc);
    }
}

template<typename BabyEntryT>
__global__ void bsgs_giant_bloom_bucketed_tiled_kernel_t(
    const BsgsPlan*       __restrict__ plan,
    const BabyEntryT*     __restrict__ baby_entries,
    const uint32_t*       __restrict__ bucket_offsets,
    uint32_t                          bucket_mask,
    const uint64_t*       __restrict__ bloom,
    uint64_t                          bloom_m_bits,
    BsgsHit*              __restrict__ hits,
    uint32_t*             __restrict__ hit_count,
    uint32_t                          hit_capacity,
    uint64_t                          high_offset,
    uint64_t                          high_limit,
    BsgsCandidate*        __restrict__ candidate)
{
    const uint64_t high_local = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (high_local >= high_limit) return;
    const uint64_t high_idx = high_offset + high_local;
    if (high_idx >= plan->giant_high_count) return;
    if (atomicAdd(&candidate->found, 0) != 0) return;

    ECPointA base;
    base.infinity = false;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        base.X[i] = plan->p_start_x[i];
        base.Y[i] = plan->p_start_y[i];
    }

    const uint64_t high_gray = high_idx ^ (high_idx >> 1);
    for (uint32_t bit = 0; bit < plan->giant_high_bits; bit++) {
        if (((high_gray >> bit) & 1ULL) == 0) continue;
        const uint32_t abs_bit = plan->giant_low_bits + bit;
        uint32_t group = 0;
        uint32_t digit = 0;
        if (!bsgs_bit_to_group_digit(plan->giant, plan->giant_unknown, abs_bit, group, digit)) continue;
        bsgs_point_add_or_sub(base, plan->giant_contrib[group][digit], true);
    }

    const uint64_t base_idx = high_gray << plan->giant_low_bits;
    if (bsgs_try_bucketed_match(
            base, baby_entries, bucket_offsets, bucket_mask, bloom, bloom_m_bits,
            hits, hit_count, hit_capacity, base_idx, 0, candidate, true))
        return;

    if (LOW_SIZE <= 1 || atomicAdd(&candidate->found, 0) != 0) return;

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
            const BsgsPoint& low = plan->giant_low_dict[k];
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
        if (nt_tiles == 0) {
            bsgs_limbs_copy4(prod, tile_prefix_prod[0]);
        } else {
            fieldMul(tile_prefix_prod[nt_tiles - 1], prod, tile_prefix_prod[nt_tiles]);
        }
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
            const BsgsPoint& low = plan->giant_low_dict[k];
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
            const BsgsPoint& low = plan->giant_low_dict[k];

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

            ECPointA candidate_point;
            candidate_point.infinity = false;
            #pragma unroll
            for (int limb = 0; limb < 4; limb++) {
                candidate_point.X[limb] = x3[limb];
                candidate_point.Y[limb] = y3[limb];
            }

            if (bsgs_try_bucketed_match(
                    candidate_point, baby_entries, bucket_offsets, bucket_mask,
                    bloom, bloom_m_bits, hits, hit_count, hit_capacity,
                    base_idx | (uint64_t)k, 0, candidate, true))
                return;

            if (atomicAdd(&candidate->found, 0) != 0) return;
        }
    }
}

__global__ void bsgs_giant_bloom_only_tiled_kernel(
    const BsgsPlan*       __restrict__ plan,
    const uint64_t*       __restrict__ bloom,
    uint64_t                          bloom_m_bits,
    BsgsHit*              __restrict__ hits,
    uint32_t*             __restrict__ hit_count,
    uint32_t                          hit_capacity,
    uint64_t                          high_offset,
    uint64_t                          high_limit)
{
    const uint64_t high_local = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (high_local >= high_limit) return;
    const uint64_t high_idx = high_offset + high_local;
    if (high_idx >= plan->giant_high_count) return;

    ECPointA base;
    base.infinity = false;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        base.X[i] = plan->p_start_x[i];
        base.Y[i] = plan->p_start_y[i];
    }

    const uint64_t high_gray = high_idx ^ (high_idx >> 1);
    for (uint32_t bit = 0; bit < plan->giant_high_bits; bit++) {
        if (((high_gray >> bit) & 1ULL) == 0) continue;
        const uint32_t abs_bit = plan->giant_low_bits + bit;
        uint32_t group = 0;
        uint32_t digit = 0;
        if (!bsgs_bit_to_group_digit(plan->giant, plan->giant_unknown, abs_bit, group, digit)) continue;
        bsgs_point_add_or_sub(base, plan->giant_contrib[group][digit], true);
    }

    const uint64_t base_idx = high_gray << plan->giant_low_bits;
    bsgs_emit_bloom_hit_hex(
        base, bloom, bloom_m_bits, hits, hit_count, hit_capacity, base_idx);

    if (LOW_SIZE <= 1) return;

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
            const BsgsPoint& low = plan->giant_low_dict[k];
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
        if (nt_tiles == 0) {
            bsgs_limbs_copy4(prod, tile_prefix_prod[0]);
        } else {
            fieldMul(tile_prefix_prod[nt_tiles - 1], prod, tile_prefix_prod[nt_tiles]);
        }
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
            const BsgsPoint& low = plan->giant_low_dict[k];
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
            const BsgsPoint& low = plan->giant_low_dict[k];

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

            ECPointA candidate_point;
            candidate_point.infinity = false;
            #pragma unroll
            for (int limb = 0; limb < 4; limb++) {
                candidate_point.X[limb] = x3[limb];
                candidate_point.Y[limb] = y3[limb];
            }

            bsgs_emit_bloom_hit_hex(
                candidate_point, bloom, bloom_m_bits, hits, hit_count,
                hit_capacity, base_idx | (uint64_t)k);
        }
    }
}


// ============================================================================
// HEX HOST LAUNCHERS
// ============================================================================

static inline cudaError_t bsgs_launch_giant_bloom_bucketed_gray(
    const BsgsGpuBuffers& buffers,
    uint64_t giant_offset,
    uint64_t giant_count,
    int threads = 256)
{
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_bucket_offsets || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    if (buffers.hit_capacity > UINT32_MAX) return cudaErrorInvalidValue;
    const uint64_t workers =
        (giant_count + (uint64_t)BSGS_GRAY_ITEMS_PER_THREAD - 1ULL) /
        (uint64_t)BSGS_GRAY_ITEMS_PER_THREAD;
    const int blocks = (int)((workers + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_giant_bloom_bucketed_gray_kernel_t<BsgsBabyEntry><<<blocks, threads>>>(
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

static inline cudaError_t bsgs_launch_giant_bloom_bucketed_tiled(
    const BsgsGpuBuffers& buffers,
    uint64_t giant_offset,
    uint64_t giant_count,
    int threads = 256)
{
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_bucket_offsets || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    if (buffers.hit_capacity > UINT32_MAX) return cudaErrorInvalidValue;
    if ((giant_offset & (uint64_t)(LOW_SIZE - 1)) != 0) return cudaErrorInvalidValue;
    if ((giant_count & (uint64_t)(LOW_SIZE - 1)) != 0) return cudaErrorInvalidValue;

    const uint64_t high_offset = giant_offset / (uint64_t)LOW_SIZE;
    const uint64_t high_count = giant_count / (uint64_t)LOW_SIZE;
    const int blocks = (int)((high_count + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_giant_bloom_bucketed_tiled_kernel_t<BsgsBabyEntry><<<blocks, threads>>>(
        buffers.d_plan, buffers.d_bucketed_entries, buffers.d_bucket_offsets,
        buffers.bucket_count - 1u,
        buffers.d_bsgs_bloom, buffers.bloom_m_bits,
        buffers.d_hits, buffers.d_hit_count, (uint32_t)buffers.hit_capacity,
        high_offset, high_count,
        buffers.d_candidate);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_giant_hex8_gray(
    const BsgsGpuBuffers& buffers,
    uint64_t giant_offset,
    uint64_t giant_count,
    int threads = 256)
{
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_hex8_bucketed_entries) return cudaErrorInvalidValue;
    if (!buffers.d_bucket_offsets || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    if (buffers.hit_capacity > UINT32_MAX) return cudaErrorInvalidValue;
    const uint64_t workers =
        (giant_count + (uint64_t)BSGS_GRAY_ITEMS_PER_THREAD - 1ULL) /
        (uint64_t)BSGS_GRAY_ITEMS_PER_THREAD;
    const int blocks = (int)((workers + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_giant_bloom_bucketed_gray_kernel_t<BsgsHex8Entry><<<blocks, threads>>>(
        buffers.d_plan, buffers.d_hex8_bucketed_entries, buffers.d_bucket_offsets,
        buffers.bucket_count - 1u,
        buffers.d_bsgs_bloom, buffers.bloom_m_bits,
        buffers.d_hits, buffers.d_hit_count, (uint32_t)buffers.hit_capacity,
        giant_offset, giant_count,
        buffers.d_candidate);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_giant_hex8_tiled(
    const BsgsGpuBuffers& buffers,
    uint64_t giant_offset,
    uint64_t giant_count,
    int threads = 256)
{
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_hex8_bucketed_entries) return cudaErrorInvalidValue;
    if (!buffers.d_bucket_offsets || buffers.bucket_count == 0) return cudaErrorInvalidValue;
    if (buffers.hit_capacity > UINT32_MAX) return cudaErrorInvalidValue;
    if ((giant_offset & (uint64_t)(LOW_SIZE - 1)) != 0) return cudaErrorInvalidValue;
    if ((giant_count & (uint64_t)(LOW_SIZE - 1)) != 0) return cudaErrorInvalidValue;

    const uint64_t high_offset = giant_offset / (uint64_t)LOW_SIZE;
    const uint64_t high_count = giant_count / (uint64_t)LOW_SIZE;
    const int blocks = (int)((high_count + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_giant_bloom_bucketed_tiled_kernel_t<BsgsHex8Entry><<<blocks, threads>>>(
        buffers.d_plan, buffers.d_hex8_bucketed_entries, buffers.d_bucket_offsets,
        buffers.bucket_count - 1u,
        buffers.d_bsgs_bloom, buffers.bloom_m_bits,
        buffers.d_hits, buffers.d_hit_count, (uint32_t)buffers.hit_capacity,
        high_offset, high_count,
        buffers.d_candidate);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_giant_bloom_only_hex_gray(
    const BsgsGpuBuffers& buffers,
    uint64_t giant_offset,
    uint64_t giant_count,
    int threads = 256)
{
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_hits || !buffers.d_hit_count) return cudaErrorInvalidValue;
    if (buffers.hit_capacity > UINT32_MAX) return cudaErrorInvalidValue;
    const uint64_t workers =
        (giant_count + (uint64_t)BSGS_GRAY_ITEMS_PER_THREAD - 1ULL) /
        (uint64_t)BSGS_GRAY_ITEMS_PER_THREAD;
    const int blocks = (int)((workers + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_giant_bloom_only_gray_kernel<<<blocks, threads>>>(
        buffers.d_plan,
        buffers.d_bsgs_bloom, buffers.bloom_m_bits,
        buffers.d_hits, buffers.d_hit_count, (uint32_t)buffers.hit_capacity,
        giant_offset, giant_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}

static inline cudaError_t bsgs_launch_giant_bloom_only_hex_tiled(
    const BsgsGpuBuffers& buffers,
    uint64_t giant_offset,
    uint64_t giant_count,
    int threads = 256)
{
    if (!buffers.d_bsgs_bloom || buffers.bloom_m_bits == 0) return cudaErrorInvalidValue;
    if (!buffers.d_hits || !buffers.d_hit_count) return cudaErrorInvalidValue;
    if (buffers.hit_capacity > UINT32_MAX) return cudaErrorInvalidValue;
    if ((giant_offset & (uint64_t)(LOW_SIZE - 1)) != 0) return cudaErrorInvalidValue;
    if ((giant_count & (uint64_t)(LOW_SIZE - 1)) != 0) return cudaErrorInvalidValue;

    const uint64_t high_offset = giant_offset / (uint64_t)LOW_SIZE;
    const uint64_t high_count = giant_count / (uint64_t)LOW_SIZE;
    const int blocks = (int)((high_count + (uint64_t)threads - 1) / (uint64_t)threads);
    bsgs_giant_bloom_only_tiled_kernel<<<blocks, threads>>>(
        buffers.d_plan,
        buffers.d_bsgs_bloom, buffers.bloom_m_bits,
        buffers.d_hits, buffers.d_hit_count, (uint32_t)buffers.hit_capacity,
        high_offset, high_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}
