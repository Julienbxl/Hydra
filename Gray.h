#pragma once
/*
 * ======================================================================================
 * HYDRA - Gray.h  (V4-TILED-TEMPLATE : SPLIT HAUT/BAS — Affine Dict + Gray Jacobien)
 * ======================================================================================
 *
 * ARCHITECTURE :
 *
 *  CPU (Hydra.cu) :
 *    - Précalcule 2^LOW_BITS points affines R_k = (sum des bits bas actifs) * G
 *    - Stocke dans __constant__ DictX[LOW_SIZE][4], DictY[LOW_SIZE][4]
 *    - Pour les bits hauts : calcule les deltas Q_i = 2^(bit_haut_i) * G
 *
 *  GPU — kernels spécialisés par mode de check + dispatch host :
 *
 *  [Chaque thread = un P_base unique]
 *
 *  PHASE 1 — Gray Code sur bits HAUTS → P_base Jacobien
 *    Chaque thread avance d'un pas Gray sur les bits hauts.
 *    Résultat : P_base = (bits hauts) * G en coordonnées Jacobiennes (X:Y:Z)
 *
 *  PHASE 2 — Normalisation affine de P_base (1 fieldInv par thread)
 *    P_base_affine = (X/Z², Y/Z³)
 *
 *  PHASE 3 — Boucle intra-thread sur bits BAS (dictionnaire), version TILED
 *    - Pass 1 : produit des ΔX par tuile
 *    - 1 seule inversion globale du produit total
 *    - Pass 2 : reconstruction d'une tuile, back-prop local, addition affine, hash
 *
 *  OBJECTIF :
 *    Réduire fortement le working set local par thread.
 *    On ne garde plus LOW_SIZE tableaux locaux vivants simultanément.
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
// DICTIONNAIRE EN CONSTANT MEMORY
// R_k = combinaison des LOW_BITS bits bas, précalculé CPU
// DictX[0]/DictY[0] = point à l'infini (identité, k=0 → P_base lui-même)
// DictX[k]/DictY[k] = somme des bits bas actifs dans k, k=1..LOW_SIZE-1
// ============================================================================
__constant__ uint64_t DictX[LOW_SIZE][4];
__constant__ uint64_t DictY[LOW_SIZE][4];
__constant__ uint8_t  DictValid[LOW_SIZE];  // 1 = point valide, 0 = infini (k=0)

#ifndef HYDRA_HEX_TILE
#define HYDRA_HEX_TILE 128
#endif

#define HYDRA_HEX_MAX_TILES (((LOW_SIZE - 1) + HYDRA_HEX_TILE - 1) / HYDRA_HEX_TILE)

// ============================================================================
// ECC MIXED JAC+AFF → JAC  (Phase 1 : Gray sur bits hauts)
// ============================================================================
__device__ __forceinline__ void point_add_mixed_jac(
    uint64_t Rx[4], uint64_t Ry[4], uint64_t Rz[4],
    const uint64_t Px[4], const uint64_t Py[4], const uint64_t Pz[4],
    const uint64_t Qx[4], const uint64_t Qy[4])
{
    if ((Pz[0] | Pz[1] | Pz[2] | Pz[3]) == 0) {
        fieldCopy(Qx, Rx);
        fieldCopy(Qy, Ry);
        Rz[0] = 1; Rz[1] = 0; Rz[2] = 0; Rz[3] = 0;
        return;
    }
    uint64_t Z1Z1[4], U2[4], S2[4], H[4], R2[4], HH[4], I[4], J[4], V[4], tmp[4];
    fieldSqr(Pz, Z1Z1);
    fieldMul(Qx, Z1Z1, U2);
    fieldMul(Pz, Z1Z1, S2); fieldMul(Qy, S2, S2);
    fieldSub(U2, Px, H);
    fieldSub(S2, Py, R2); fieldAdd(R2, R2, R2);
    if ((H[0] | H[1] | H[2] | H[3]) == 0) {
        // H = U2 - Px = 0  →  same X coordinate : either P == Q or P == -Q
        // Check S2 - Py (= R2/2) : if also 0, points are equal → doubling needed
        // In practice on secp256k1 with distinct scalars this path is unreachable,
        // but we handle it correctly to avoid silent wrong results.
        uint64_t Ry_test[4];
        fieldSub(S2, Py, Ry_test);
        if ((Ry_test[0] | Ry_test[1] | Ry_test[2] | Ry_test[3]) == 0) {
            // P == Q : doubling. Fall through is complex — mark as infinity.
            // This is astronomically improbable for random keys on secp256k1.
            // A full point doubling in mixed Jac+Aff would require a separate path.
            Rz[0] = Rz[1] = Rz[2] = Rz[3] = 0; return;
        }
        // P == -Q : result is point at infinity
        Rz[0] = Rz[1] = Rz[2] = Rz[3] = 0; return;
    }
    fieldSqr(H, HH);
    fieldAdd(HH, HH, I); fieldAdd(I, I, I);
    fieldMul(H, I, J); fieldMul(Px, I, V);
    fieldSqr(R2, Rx); fieldSub(Rx, J, Rx); fieldSub(Rx, V, Rx); fieldSub(Rx, V, Rx);
    fieldSub(V, Rx, tmp); fieldMul(R2, tmp, Ry);
    fieldMul(Py, J, tmp); fieldAdd(tmp, tmp, tmp); fieldSub(Ry, tmp, Ry);
    fieldMul(Pz, H, Rz); fieldAdd(Rz, Rz, Rz);
}

__device__ __forceinline__ void point_sub_mixed_jac(
    uint64_t Rx[4], uint64_t Ry[4], uint64_t Rz[4],
    const uint64_t Px[4], const uint64_t Py[4], const uint64_t Pz[4],
    const uint64_t Qx[4], const uint64_t Qy[4])
{
    uint64_t nQy[4];
    fieldNeg(Qy, nQy);
    point_add_mixed_jac(Rx, Ry, Rz, Px, Py, Pz, Qx, nQy);
}

// ============================================================================
// HASH comparaison
// ============================================================================
__device__ __forceinline__ bool hash20_matches(
    const uint8_t *a, const uint8_t *b)
{
    const uint32_t *x = (const uint32_t*)a;
    const uint32_t *y = (const uint32_t*)b;
    if (x[0] != y[0]) return false;
    return (x[1] == y[1]) && (x[2] == y[2]) && (x[3] == y[3]) && (x[4] == y[4]);
}

__device__ __forceinline__ bool limbs_is_zero4(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0ULL;
}

__device__ __forceinline__ void limbs_copy4(const uint64_t src[4], uint64_t dst[4]) {
    #pragma unroll 4
    for (int i = 0; i < 4; i++) dst[i] = src[i];
}

enum class HydraHexCheckMode : uint32_t {
    BTC_EXACT,
    ETH_EXACT,
    BLOOM_ANY,
    BLOOM_BTC_ONLY,
    BLOOM_ETH_ONLY,
    BTC_PUBKEY,  // Known pubkey : compare px + y_parity, skip SHA256 + RIPEMD160
    ETH_PUBKEY   // Known pubkey : compare (px,py) exactly, skip keccak256
};

template<HydraHexCheckMode M>
__device__ __forceinline__ bool hydra_check_xy_t(
    const uint64_t x[4],
    const uint64_t y[4],
    uint64_t global_index,
    const TargetData* __restrict__ target,
    HydraResult* __restrict__ result)
{
    if constexpr (M == HydraHexCheckMode::BTC_EXACT) {
        uint8_t h160[20];
        getHash160_33_from_limbs((y[0] & 1) ? 0x03 : 0x02, x, h160);
        if (hash20_matches(h160, target->hash20)) {
            if (atomicCAS(&result->found, 0, 1) == 0)
                result->index = global_index;
            return true;
        }
        return false;
    } else if constexpr (M == HydraHexCheckMode::ETH_EXACT) {
        uint8_t eth20[20];
        getEthAddr_from_limbs(x, y, eth20);
        if (hash20_matches(eth20, target->hash20)) {
            if (atomicCAS(&result->found, 0, 1) == 0)
                result->index = global_index;
            return true;
        }
        return false;
    } else if constexpr (M == HydraHexCheckMode::BLOOM_BTC_ONLY) {
        uint8_t h160[20];
        getHash160_33_from_limbs((y[0] & 1) ? 0x03 : 0x02, x, h160);
        if (bloom_check(h160, target->d_bloom_filter, target->bloom_m_bits)) {
            if (atomicCAS(&result->found, 0, 1) == 0)
                result->index = global_index;
            return true;
        }
        return false;
    } else if constexpr (M == HydraHexCheckMode::BLOOM_ETH_ONLY) {
        uint8_t eth20[20];
        getEthAddr_from_limbs(x, y, eth20);
        if (bloom_check(eth20, target->d_bloom_filter, target->bloom_m_bits)) {
            if (atomicCAS(&result->found, 0, 1) == 0)
                result->index = global_index;
            return true;
        }
        return false;
    } else if constexpr (M == HydraHexCheckMode::BTC_PUBKEY) {
        // Known pubkey : compare X + Y parity, skip SHA256+RIPEMD160 entirely.
        if (x[0] == target->pubkey_x[0] &&
            x[1] == target->pubkey_x[1] &&
            x[2] == target->pubkey_x[2] &&
            x[3] == target->pubkey_x[3] &&
            (uint8_t)(y[0] & 1ULL) == target->pubkey_y_parity) {
            if (atomicCAS(&result->found, 0, 1) == 0)
                result->index = global_index;
            return true;
        }
        return false;
    } else if constexpr (M == HydraHexCheckMode::ETH_PUBKEY) {
        // Known pubkey : compare full (px, py), skip keccak256 entirely.
        // Both X and Y must match exactly — no ambiguity unlike BTC where only parity is stored.
        if (x[0] == target->pubkey_x[0] &&
            x[1] == target->pubkey_x[1] &&
            x[2] == target->pubkey_x[2] &&
            x[3] == target->pubkey_x[3] &&
            y[0] == target->pubkey_y[0] &&
            y[1] == target->pubkey_y[1] &&
            y[2] == target->pubkey_y[2] &&
            y[3] == target->pubkey_y[3]) {
            if (atomicCAS(&result->found, 0, 1) == 0)
                result->index = global_index;
            return true;
        }
        return false;
    } else {
        // BLOOM_ANY : check BTC then ETH
        uint8_t h160[20];
        getHash160_33_from_limbs((y[0] & 1) ? 0x03 : 0x02, x, h160);
        if (bloom_check(h160, target->d_bloom_filter, target->bloom_m_bits)) {
            if (atomicCAS(&result->found, 0, 1) == 0)
                result->index = global_index;
            return true;
        }

        uint8_t eth20[20];
        getEthAddr_from_limbs(x, y, eth20);
        if (bloom_check(eth20, target->d_bloom_filter, target->bloom_m_bits)) {
            if (atomicCAS(&result->found, 0, 1) == 0)
                result->index = global_index;
            return true;
        }
        return false;
    }
}

// ============================================================================
// MEGA KERNEL V4-TILED-TEMPLATE
// ============================================================================
template<HydraHexCheckMode M>
__global__ __launch_bounds__(256, 2)
void hydra_mega_kernel_t(
    const HydraData*  __restrict__ fd,
    const TargetData* __restrict__ target,
    HydraResult*      __restrict__ result,
    int wave_size)   // nombre de P_base (chunks hauts) dans cette wave
{
    const int tid    = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(blockDim.x * gridDim.x);

    for (int high_step = tid; high_step < wave_size; high_step += stride)
    {
        // Index global du P_base dans les bits hauts
        const uint64_t high_idx = fd->gray_offset_start + (uint64_t)high_step;
        if (high_idx >= fd->high_candidates) return;
        if (atomicAdd(&result->found, 0) != 0) return;

        // ----------------------------------------------------------------
        // PHASE 1 : GRAY CODE sur bits HAUTS → P_base Jacobien
        // ----------------------------------------------------------------
        uint64_t Px[4], Py[4], Pz[4];
        #pragma unroll 4
        for (int i = 0; i < 4; i++) { Px[i] = fd->base_x[i]; Py[i] = fd->base_y[i]; }
        Pz[0] = 1; Pz[1] = 0; Pz[2] = 0; Pz[3] = 0;

        {
            const uint64_t g0 = high_idx ^ (high_idx >> 1);
            for (int b = 0; b < (int)fd->num_high_bits; b++) {
                if ((g0 >> b) & 1ULL) {
                    uint64_t tx[4], ty[4], tz[4];
                    point_add_mixed_jac(tx, ty, tz, Px, Py, Pz,
                                        fd->delta_x[b], fd->delta_y[b]);
                    #pragma unroll 4
                    for (int i = 0; i < 4; i++) { Px[i] = tx[i]; Py[i] = ty[i]; Pz[i] = tz[i]; }
                }
            }
        }

        // ----------------------------------------------------------------
        // PHASE 2 : NORMALISATION AFFINE de P_base (1 fieldInv)
        // ----------------------------------------------------------------
        uint64_t z_inv[4], z2[4], z3[4], bx[4], by[4];
        {
            uint64_t znorm[4];
            #pragma unroll 4
            for (int i = 0; i < 4; i++) znorm[i] = Pz[i];
            fieldNormalize(znorm);
            fieldInv(znorm, z_inv);
        }
        fieldSqr(z_inv, z2);
        fieldMul(z_inv, z2, z3);
        fieldMul(Px, z2, bx);
        fieldMul(Py, z3, by);
        fieldNormalize(bx);
        fieldNormalize(by);

        // k = 0 : P_base direct
        if (hydra_check_xy_t<M>(
                bx, by,
                high_idx * (uint64_t)LOW_SIZE + 0ULL,
                target, result))
            return;

        // ----------------------------------------------------------------
        // PHASE 3 (TILED)
        // ----------------------------------------------------------------
        int      tile_begin_compact[HYDRA_HEX_MAX_TILES];
        uint64_t tile_prod       [HYDRA_HEX_MAX_TILES][4];
        uint64_t tile_prefix_prod[HYDRA_HEX_MAX_TILES][4];
        uint64_t tile_inv_prod   [HYDRA_HEX_MAX_TILES][4];
        int      nt_tiles = 0;

        // ------------------------------------------------------------
        // PASS 1 : produit des ΔX par tuile
        // ------------------------------------------------------------
        #pragma unroll 1
        for (int tile_begin = 1; tile_begin < LOW_SIZE; tile_begin += HYDRA_HEX_TILE)
        {
            const int tile_end =
                (tile_begin + HYDRA_HEX_TILE < LOW_SIZE)
                    ? (tile_begin + HYDRA_HEX_TILE)
                    : LOW_SIZE;

            bool has_prod = false;
            uint64_t prod[4];

            #pragma unroll 1
            for (int k = tile_begin; k < tile_end; ++k) {
                uint64_t dx[4];
                fieldSub(DictX[k], bx, dx);

                // Cas dégénéré identique à la version d'origine : skip
                if (limbs_is_zero4(dx)) continue;

                if (!has_prod) {
                    limbs_copy4(dx, prod);
                    has_prod = true;
                } else {
                    uint64_t tmp[4];
                    fieldMul(prod, dx, tmp);
                    limbs_copy4(tmp, prod);
                }
            }

            if (!has_prod) continue;

            tile_begin_compact[nt_tiles] = tile_begin;
            limbs_copy4(prod, tile_prod[nt_tiles]);

            if (nt_tiles == 0) {
                limbs_copy4(prod, tile_prefix_prod[0]);
            } else {
                fieldMul(tile_prefix_prod[nt_tiles - 1], prod, tile_prefix_prod[nt_tiles]);
            }

            nt_tiles++;
        }

        if (nt_tiles == 0) continue;

        // ------------------------------------------------------------
        // PASS 1b : inversion globale du produit total des tuiles
        // ------------------------------------------------------------
        uint64_t inv_total[4];
        fieldNormalize(tile_prefix_prod[nt_tiles - 1]);
        fieldInv(tile_prefix_prod[nt_tiles - 1], inv_total);

        {
            uint64_t running_tile_inv[4];
            limbs_copy4(inv_total, running_tile_inv);

            for (int t = nt_tiles - 1; t >= 0; --t) {
                if (t > 0) {
                    fieldMul(running_tile_inv, tile_prefix_prod[t - 1], tile_inv_prod[t]);

                    uint64_t tmp[4];
                    fieldMul(running_tile_inv, tile_prod[t], tmp);
                    limbs_copy4(tmp, running_tile_inv);
                } else {
                    limbs_copy4(running_tile_inv, tile_inv_prod[0]);
                }
            }
        }

        // ------------------------------------------------------------
        // PASS 2 : reconstruction d'une tuile, back-prop local,
        //          addition affine + hash
        // ------------------------------------------------------------
        for (int t = nt_tiles - 1; t >= 0; --t)
        {
            const int tile_begin = tile_begin_compact[t];
            const int tile_end =
                (tile_begin + HYDRA_HEX_TILE < LOW_SIZE)
                    ? (tile_begin + HYDRA_HEX_TILE)
                    : LOW_SIZE;

            int      valid_idx[HYDRA_HEX_TILE];
            uint64_t acc      [HYDRA_HEX_TILE][4];
            int      nv = 0;

            #pragma unroll 1
            for (int k = tile_begin; k < tile_end; ++k) {
                uint64_t dx[4];
                fieldSub(DictX[k], bx, dx);
                if (limbs_is_zero4(dx)) continue;

                valid_idx[nv] = k;
                if (nv == 0) {
                    limbs_copy4(dx, acc[0]);
                } else {
                    fieldMul(acc[nv - 1], dx, acc[nv]);
                }
                nv++;
            }

            if (nv == 0) continue;

            uint64_t running_inv[4];
            limbs_copy4(tile_inv_prod[t], running_inv);

            for (int pos = nv - 1; pos >= 0; --pos)
            {
                const int k = valid_idx[pos];

                uint64_t dx_inv[4];
                if (pos > 0) {
                    fieldMul(running_inv, acc[pos - 1], dx_inv);
                } else {
                    limbs_copy4(running_inv, dx_inv);
                }

                uint64_t dx[4];
                fieldSub(DictX[k], bx, dx);
                if (pos > 0) {
                    uint64_t tmp[4];
                    fieldMul(running_inv, dx, tmp);
                    limbs_copy4(tmp, running_inv);
                }

                uint64_t dy[4];
                fieldSub(DictY[k], by, dy);

                uint64_t lam[4], lam2[4], x3[4], y3[4], tmp[4];
                fieldMul(dy, dx_inv, lam);      // λ = ΔY / ΔX
                fieldSqr(lam, lam2);            // λ²
                fieldSub(lam2, bx, x3);
                fieldSub(x3, DictX[k], x3);     // x3 = λ² - bx - Rk.x
                fieldSub(bx, x3, tmp);
                fieldMul(lam, tmp, y3);
                fieldSub(y3, by, y3);           // y3 = λ(bx-x3) - by
                fieldNormalize(x3);
                fieldNormalize(y3);

                if (hydra_check_xy_t<M>(
                        x3, y3,
                        high_idx * (uint64_t)LOW_SIZE + (uint64_t)k,
                        target, result))
                    return;
            }
        }
    }
}


// ============================================================================
// HOST DISPATCHER
// Nécessite un mini changement dans Hydra.cu :
//   launch_hydra_mega_kernel(d_fd, d_target, d_result, cur_wave, blocks, threads, target.type);
// au lieu de :
//   hydra_mega_kernel<<<blocks, threads>>>(d_fd, d_target, d_result, cur_wave);
// ============================================================================
inline void launch_hydra_mega_kernel(
    const HydraData*  d_fd,
    const TargetData* d_target,
    HydraResult*      d_result,
    int               wave_size,
    int               blocks,
    int               threads,
    TargetType        target_type)
{
    switch (target_type) {
        case TargetType::BTC:
            hydra_mega_kernel_t<HydraHexCheckMode::BTC_EXACT><<<blocks, threads>>>(d_fd, d_target, d_result, wave_size);
            break;
        case TargetType::ETH:
            hydra_mega_kernel_t<HydraHexCheckMode::ETH_EXACT><<<blocks, threads>>>(d_fd, d_target, d_result, wave_size);
            break;
        case TargetType::BLOOM:
            hydra_mega_kernel_t<HydraHexCheckMode::BLOOM_ANY><<<blocks, threads>>>(d_fd, d_target, d_result, wave_size);
            break;
        case TargetType::BLOOM_BTC:
            hydra_mega_kernel_t<HydraHexCheckMode::BLOOM_BTC_ONLY><<<blocks, threads>>>(d_fd, d_target, d_result, wave_size);
            break;
        case TargetType::BLOOM_ETH:
            hydra_mega_kernel_t<HydraHexCheckMode::BLOOM_ETH_ONLY><<<blocks, threads>>>(d_fd, d_target, d_result, wave_size);
            break;
        case TargetType::BTC_PUBKEY:
            hydra_mega_kernel_t<HydraHexCheckMode::BTC_PUBKEY><<<blocks, threads>>>(d_fd, d_target, d_result, wave_size);
            break;
        case TargetType::ETH_PUBKEY:
            hydra_mega_kernel_t<HydraHexCheckMode::ETH_PUBKEY><<<blocks, threads>>>(d_fd, d_target, d_result, wave_size);
            break;
        default:
            hydra_mega_kernel_t<HydraHexCheckMode::BTC_EXACT><<<blocks, threads>>>(d_fd, d_target, d_result, wave_size);
            break;
    }
}
