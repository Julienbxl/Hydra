#pragma once
/*
 * Brainwallet.cuh — SHA256(passphrase) → privkey → wCOMB ECC → hash/compare
 *
 * Pipeline (4 kernels, portés depuis Barracuda/ECC.h + Hash.h) :
 *   kernel_bw_ecc           — SHA256 + wCOMB → Jacobian buffer (0 smem, haute occupancy)
 *   kernel_bw_local_prod    — prefix/suffix scan des Z par bloc (smem = 2×BLK×32 B)
 *   kernel_bw_invert_blocks — 1 fieldInv par bloc via bw_batch_invert_128
 *   kernel_bw_finalize      — z_inv = local_except × block_inv → affine → hash/bloom
 */

#include "Rules.cuh"

#define MAX_BRAIN_LEN    128
#define BW_THREADS_ECC   256   // kernel 1 : pas de smem → occupancy libre
#define BW_THREADS_INV   256   // kernels 2, 4 : prefix/suffix + hash

// ─── Type de base : élément de corps (compatible avec Barracuda uint256_t) ───
struct bw_u256 { uint64_t v[4]; };
__constant__ uint64_t d_Cx[4];
__constant__ uint64_t d_Cy[4];

__device__ __forceinline__ bool bw_is_zero(const bw_u256& x) {
    return (x.v[0]|x.v[1]|x.v[2]|x.v[3]) == 0;
}
__device__ __forceinline__ bw_u256 bw_fp_one() {
    bw_u256 r; r.v[0]=1; r.v[1]=0; r.v[2]=0; r.v[3]=0; return r;
}

// ─── Shuffles 64-bit et u256 (verbatim Barracuda/ECC.h) ──────────────────────
__device__ __forceinline__ uint64_t bw_shfl_up_64(uint64_t x, int off, unsigned m) {
    uint32_t lo = __shfl_up_sync(m,(uint32_t)(x),off);
    uint32_t hi = __shfl_up_sync(m,(uint32_t)(x>>32),off);
    return (uint64_t)hi<<32|lo;
}
__device__ __forceinline__ uint64_t bw_shfl_down_64(uint64_t x, int off, unsigned m) {
    uint32_t lo = __shfl_down_sync(m,(uint32_t)(x),off);
    uint32_t hi = __shfl_down_sync(m,(uint32_t)(x>>32),off);
    return (uint64_t)hi<<32|lo;
}
__device__ __forceinline__ bw_u256 bw_shfl_up_u256(bw_u256 a, int off, unsigned m) {
    bw_u256 r;
    r.v[0]=bw_shfl_up_64(a.v[0],off,m); r.v[1]=bw_shfl_up_64(a.v[1],off,m);
    r.v[2]=bw_shfl_up_64(a.v[2],off,m); r.v[3]=bw_shfl_up_64(a.v[3],off,m);
    return r;
}
__device__ __forceinline__ bw_u256 bw_shfl_down_u256(bw_u256 a, int off, unsigned m) {
    bw_u256 r;
    r.v[0]=bw_shfl_down_64(a.v[0],off,m); r.v[1]=bw_shfl_down_64(a.v[1],off,m);
    r.v[2]=bw_shfl_down_64(a.v[2],off,m); r.v[3]=bw_shfl_down_64(a.v[3],off,m);
    return r;
}

// ─── Batch inversion 128 threads (Barracuda scan, normalized between muls) ───
// s_mem : (128+4) × sizeof(bw_u256). In-place safe (z_in == z_out).
__device__ __forceinline__
void bw_batch_invert_128(const bw_u256* z_in, bw_u256* z_out, bw_u256* s_mem) {
    bw_u256* s_scan = s_mem;
    bw_u256* s_warp = s_mem + 128;
    const int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    bw_u256 x = *z_in;
    const bool is_zero = bw_is_zero(x);
    if (is_zero) x = bw_fp_one();

    bw_u256 prefix = x;
    #pragma unroll
    for (int d=1;d<32;d<<=1) {
        bw_u256 tmp=bw_shfl_up_u256(prefix,d,0xFFFFFFFF); __syncwarp();
        if (lane>=d) { bw_u256 np; fieldMul(tmp.v,prefix.v,np.v); fieldNormalize(np.v); prefix=np; }
        __syncwarp();
    }
    if (lane==31) s_warp[warp]=prefix;
    __syncthreads();
    if (tid==0) {
        bw_u256 w0=s_warp[0],w1=s_warp[1],w2=s_warp[2],w3=s_warp[3],t1,t2,t3;
        fieldMul(w0.v,w1.v,t1.v); fieldNormalize(t1.v);
        fieldMul(t1.v,w2.v,t2.v); fieldNormalize(t2.v);
        fieldMul(t2.v,w3.v,t3.v); fieldNormalize(t3.v);
        s_warp[0]=w0; s_warp[1]=t1; s_warp[2]=t2; s_warp[3]=t3;
    }
    __syncthreads();
    bw_u256 warp_base=(warp>0)?s_warp[warp-1]:bw_fp_one();
    if (warp>0) { bw_u256 np; fieldMul(prefix.v,warp_base.v,np.v); fieldNormalize(np.v); prefix=np; }
    s_scan[tid]=prefix; __syncthreads();
    bw_u256 prev_prefix=(tid>0)?s_scan[tid-1]:bw_fp_one();
    if (tid==127) {
        bw_u256 inv; fieldNormalize(prefix.v); fieldInv(prefix.v,inv.v); s_scan[127]=inv;
    }
    __syncthreads();
    bw_u256 total_inv=s_scan[127]; __syncthreads();

    bw_u256 suffix=x;
    #pragma unroll
    for (int d=1;d<32;d<<=1) {
        bw_u256 tmp=bw_shfl_down_u256(suffix,d,0xFFFFFFFF); __syncwarp();
        if (lane+d<32) { bw_u256 ns; fieldMul(suffix.v,tmp.v,ns.v); fieldNormalize(ns.v); suffix=ns; }
        __syncwarp();
    }
    if (lane==0) s_warp[warp]=suffix; __syncthreads();
    if (tid==0) {
        bw_u256 p1=s_warp[1],p2=s_warp[2],p3=s_warp[3],a23,a123;
        fieldMul(p2.v,p3.v,a23.v); fieldNormalize(a23.v);
        fieldMul(p1.v,a23.v,a123.v); fieldNormalize(a123.v);
        s_warp[0]=a123; s_warp[1]=a23; s_warp[2]=p3; s_warp[3]=bw_fp_one();
    }
    __syncthreads();
    bw_u256 warp_suf=s_warp[warp];
    { bw_u256 ns; fieldMul(suffix.v,warp_suf.v,ns.v); fieldNormalize(ns.v); suffix=ns; }
    s_scan[tid]=suffix; __syncthreads();
    bw_u256 s_next=(tid==127)?bw_fp_one():s_scan[tid+1];
    bw_u256 res,t0;
    fieldMul(prev_prefix.v,total_inv.v,t0.v); fieldNormalize(t0.v);
    fieldMul(t0.v,s_next.v,res.v); fieldNormalize(res.v);
    if (is_zero) { res.v[0]=0; res.v[1]=0; res.v[2]=0; res.v[3]=0; }
    *z_out=res;
}

// ─── Jacobian point (accumulateur wCOMB) ─────────────────────────────────────
struct ECPointJ_bw { uint64_t X[4],Y[4],Z[4]; };
__device__ __forceinline__ void bw_set_inf(ECPointJ_bw& P) {
    for(int i=0;i<4;i++){P.X[i]=0;P.Y[i]=0;P.Z[i]=0;}
}
__device__ __forceinline__ bool bw_is_inf(const ECPointJ_bw& P) {
    return (P.Z[0]|P.Z[1]|P.Z[2]|P.Z[3])==0;
}

__device__ __forceinline__ void bw_dbl(ECPointJ_bw& P) {
    if (bw_is_inf(P)) return;
    uint64_t T1[4],T2[4],T3[4],M[4],S2[4],SX3[4],Xo[4],Yo[4],Zo[4];
    fieldSqr(P.Y,T1); fieldMul(P.X,T1,T2);
    fieldAdd(T2,T2,T2); fieldAdd(T2,T2,T2);          // S=4XY²
    fieldSqr(P.X,T3); fieldAdd(T3,T3,M); fieldAdd(M,T3,T3); // M=3X²
    fieldSqr(T3,Xo); fieldAdd(T2,T2,S2); fieldSub(Xo,S2,Xo);
    fieldMul(P.Y,P.Z,Zo); fieldAdd(Zo,Zo,Zo);
    fieldSub(T2,Xo,SX3); fieldMul(T3,SX3,Yo);
    fieldSqr(T1,T1); fieldAdd(T1,T1,T1); fieldAdd(T1,T1,T1); fieldAdd(T1,T1,T1);
    fieldSub(Yo,T1,Yo);
    for(int i=0;i<4;i++){P.X[i]=Xo[i];P.Y[i]=Yo[i];P.Z[i]=Zo[i];}
}

__device__ __forceinline__ void bw_madd(ECPointJ_bw& P,
    const uint64_t* __restrict__ Qx, const uint64_t* __restrict__ Qy)
{
    if (bw_is_inf(P)) {
        for(int i=0;i<4;i++){P.X[i]=Qx[i];P.Y[i]=Qy[i];P.Z[i]=0;} P.Z[0]=1; return;
    }
    uint64_t Z2[4],U[4],Z3p[4],V[4],H[4],r[4],H2[4],H3[4],X1H2[4],nX[4],nY[4],nZ[4];
    fieldSqr(P.Z,Z2); fieldMul(Qx,Z2,U); fieldMul(P.Z,Z2,Z3p); fieldMul(Qy,Z3p,V);
    fieldSub(U,P.X,H); fieldSub(V,P.Y,r);
    if ((H[0]|H[1]|H[2]|H[3])==0) {
        if ((r[0]|r[1]|r[2]|r[3])==0) bw_dbl(P); else bw_set_inf(P); return;
    }
    fieldMul(P.Z,H,nZ); fieldSqr(H,H2); fieldMul(H,H2,H3); fieldMul(P.X,H2,X1H2);
    fieldSqr(r,nX); fieldSub(nX,H3,nX); fieldSub(nX,X1H2,nX); fieldSub(nX,X1H2,nX);
    fieldSub(X1H2,nX,X1H2); fieldMul(r,X1H2,nY); fieldMul(P.Y,H3,H3); fieldSub(nY,H3,nY);
    for(int i=0;i<4;i++){P.X[i]=nX[i];P.Y[i]=nY[i];P.Z[i]=nZ[i];}
}

__device__ __forceinline__ uint32_t bw_get_window(const uint64_t k[4],int j,int w) {
    const int bp=j*w; if(bp>=256) return 0;
    const int wi=bp>>6, bo=bp&63;
    const uint64_t lo=(wi<4)?k[wi]:0ULL, hi=(wi<3)?k[wi+1]:0ULL;
    const uint64_t v=(bo==0)?lo:((lo>>bo)|(hi<<(64-bo)));
    const int ew=(256-bp>=w)?w:(256-bp);
    const uint64_t mask=(ew>0&&ew<64)?((1ULL<<ew)-1ULL):0ULL;
    return (uint32_t)(v&mask);
}

template<int W>
__device__ __forceinline__ uint32_t bw_get_window_t(const uint64_t k[4], int j) {
    constexpr int bp_static_max = 256 + W;
    (void)bp_static_max;
    const int bp = j * W;
    if (bp >= 256) return 0;
    const int wi = bp >> 6;
    const int bo = bp & 63;
    const uint64_t lo = (wi < 4) ? k[wi] : 0ULL;
    const uint64_t hi = (wi < 3) ? k[wi + 1] : 0ULL;
    const uint64_t v = (bo == 0) ? lo : ((lo >> bo) | (hi << (64 - bo)));
    const int ew = (256 - bp >= W) ? W : (256 - bp);
    const uint64_t mask = (ew > 0 && ew < 64) ? ((1ULL << ew) - 1ULL) : 0ULL;
    return (uint32_t)(v & mask);
}

__device__ __noinline__ void scalar_mul_comb_bw(
    const uint64_t k[4], const uint64_t* __restrict__ pX, const uint64_t* __restrict__ pY,
    int w,int cols,int stride, uint64_t oX[4],uint64_t oY[4],uint64_t oZ[4])
{
    const int HALF=stride, FULL=stride*2;
    int32_t sd[17]={0};
    { int carry=0;
      for(int j=0;j<cols;j++) {
          int d=(int)bw_get_window(k,j,w)+carry; carry=0;
          if(d>HALF){sd[j]=d-FULL;carry=1;} else sd[j]=d;
      }
    }
    ECPointJ_bw R; bw_set_inf(R);
    uint64_t Qx[4],Qy[4];
    for(int j=cols-1;j>=0;j--) {
        const int32_t d=sd[j]; if(!d) continue;
        const int ad=(d>0)?d:-d;
        const size_t ti=(size_t)j*stride+(ad-1);
        const uint64_t* px=pX+ti*4; const uint64_t* py=pY+ti*4;
        for(int i=0;i<4;i++){Qx[i]=__ldg(px+i);Qy[i]=__ldg(py+i);}
        if(d<0){uint64_t ny[4]; fieldNeg(Qy,ny); bw_madd(R,Qx,ny);}
        else bw_madd(R,Qx,Qy);
    }
    if(!bw_is_inf(R)){for(int i=0;i<4;i++){oX[i]=R.X[i];oY[i]=R.Y[i];oZ[i]=R.Z[i];}}
    else              {for(int i=0;i<4;i++){oX[i]=0;oY[i]=0;oZ[i]=0;}}
}

template<int W>
__device__ __noinline__ void scalar_mul_comb_bw_t(
    const uint64_t k[4], const uint64_t* __restrict__ pX, const uint64_t* __restrict__ pY,
    uint64_t oX[4], uint64_t oY[4], uint64_t oZ[4])
{
    constexpr int COLS = (256 + W - 1) / W;
    constexpr int STRIDE = 1 << (W - 1);
    constexpr int HALF = STRIDE;
    constexpr int FULL = STRIDE * 2;
    int32_t sd[COLS] = {};
    int carry = 0;
    for (int j = 0; j < COLS; j++) {
        int d = (int)bw_get_window_t<W>(k, j) + carry;
        carry = 0;
        if (d > HALF) { sd[j] = d - FULL; carry = 1; }
        else sd[j] = d;
    }

    ECPointJ_bw R; bw_set_inf(R);
    uint64_t Qx[4], Qy[4];
    for (int j = COLS - 1; j >= 0; j--) {
        const int32_t d = sd[j];
        if (!d) continue;
        const int ad = (d > 0) ? d : -d;
        const size_t ti = (size_t)j * STRIDE + (ad - 1);
        const uint64_t* px = pX + ti * 4;
        const uint64_t* py = pY + ti * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) { Qx[i] = __ldg(px + i); Qy[i] = __ldg(py + i); }
        if (d < 0) { uint64_t ny[4]; fieldNeg(Qy, ny); bw_madd(R, Qx, ny); }
        else bw_madd(R, Qx, Qy);
    }
    if (!bw_is_inf(R)) {
        #pragma unroll
        for (int i = 0; i < 4; i++) { oX[i] = R.X[i]; oY[i] = R.Y[i]; oZ[i] = R.Z[i]; }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) { oX[i] = 0; oY[i] = 0; oZ[i] = 0; }
    }
}

__device__ __forceinline__ void bw_sha256(const uint8_t* data,uint32_t len,uint8_t out[32]) {
    uint32_t state[8]; SHA256Initialize(state);
    const uint8_t* src=data; uint32_t rem=len;
    while(rem>=64){
        uint32_t M[16];
        for(int i=0;i<16;i++){M[i]=((uint32_t)src[0]<<24)|((uint32_t)src[1]<<16)|((uint32_t)src[2]<<8)|(uint32_t)src[3];src+=4;}
        SHA256Transform(state,M); rem-=64;
    }
    uint32_t M[16]; for(int i=0;i<16;i++) M[i]=0;
    for(uint32_t i=0;i<rem;i++) M[i>>2]|=(uint32_t)src[i]<<(24-((i&3)<<3));
    M[rem>>2]|=0x80u<<(24-((rem&3)<<3));
    if(rem>=56){SHA256Transform(state,M);for(int i=0;i<16;i++)M[i]=0;}
    const uint64_t bl=(uint64_t)len<<3; M[14]=(uint32_t)(bl>>32); M[15]=(uint32_t)bl;
    SHA256Transform(state,M);
    for(int i=0;i<8;i++){out[i*4]=(uint8_t)(state[i]>>24);out[i*4+1]=(uint8_t)(state[i]>>16);out[i*4+2]=(uint8_t)(state[i]>>8);out[i*4+3]=(uint8_t)state[i];}
}

// ─── Buffer de résultat Jacobien par passphrase ───────────────────────────────
struct BrainJacobian {
    bw_u256  jx, jy, jz;
    uint32_t isValid;
    uint32_t _pad;
    uint64_t batch_idx;   // index dans le batch courant → retrouve la passphrase
};

__global__ __launch_bounds__(256)
void hydra_mark_rejected_jac(
    const uint8_t* __restrict__ d_rejected,
    BrainJacobian* __restrict__ d_jac,
    int N)
{
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= N) return;
    if (d_rejected[tid]) {
        d_jac[tid].isValid = 0;
        d_jac[tid].jz.v[0] = 0;
        d_jac[tid].jz.v[1] = 0;
        d_jac[tid].jz.v[2] = 0;
        d_jac[tid].jz.v[3] = 0;
    }
}

struct EccAffinePoint {
    bw_u256  x, y;
    uint32_t isValid;
    uint32_t _pad;
};

struct EccDiagResult {
    int mismatch;
    int idx;
    int kind;
    EccAffinePoint local;
    EccAffinePoint batch;
    BrainJacobian jac;
    bw_u256 local_except;
    bw_u256 block_prod;
    bw_u256 block_inv;
    bw_u256 direct_z_inv;
    bw_u256 batch_z_inv;
    bw_u256 block_check;
    bw_u256 z_check;
    uint8_t z_zero;
    uint8_t _pad[7];
};

__device__ __forceinline__ uint64_t eccdiag_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__global__ void kernel_eccdiag_make_nodes(uint8_t* __restrict__ d_nodes, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    uint8_t* node = d_nodes + (size_t)idx * 64;
    #pragma unroll
    for (int i = 0; i < 64; i++) node[i] = 0;

    // Deterministic non-zero 255-bit scalars, encoded as BIP32-style big-endian privkeys.
    uint64_t limbs[4];
    limbs[0] = eccdiag_splitmix64((uint64_t)idx * 4 + 1);
    limbs[1] = eccdiag_splitmix64((uint64_t)idx * 4 + 2);
    limbs[2] = eccdiag_splitmix64((uint64_t)idx * 4 + 3);
    limbs[3] = eccdiag_splitmix64((uint64_t)idx * 4 + 4) & 0x7FFFFFFFFFFFFFFFULL;
    bool zero = true;
    #pragma unroll
    for (int i = 0; i < 4; i++) zero = zero && (limbs[i] == 0);
    if (zero) limbs[0] = 1;

    #pragma unroll
    for (int li = 0; li < 4; li++) {
        uint64_t v = limbs[3 - li];
        node[li*8+0] = (uint8_t)(v >> 56);
        node[li*8+1] = (uint8_t)(v >> 48);
        node[li*8+2] = (uint8_t)(v >> 40);
        node[li*8+3] = (uint8_t)(v >> 32);
        node[li*8+4] = (uint8_t)(v >> 24);
        node[li*8+5] = (uint8_t)(v >> 16);
        node[li*8+6] = (uint8_t)(v >>  8);
        node[li*8+7] = (uint8_t)(v);
    }
}

__global__ void kernel_eccdiag_compare_affine(
    const EccAffinePoint* __restrict__ d_local,
    const EccAffinePoint* __restrict__ d_batch,
    const BrainJacobian* __restrict__ d_jac,
    const bw_u256* __restrict__ d_local_except,
    const bw_u256* __restrict__ d_block_prods,
    const bw_u256* __restrict__ d_block_inv,
    const uint8_t* __restrict__ d_z_zero,
    int N,
    EccDiagResult* __restrict__ d_result)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (atomicAdd(&d_result->mismatch, 0) != 0) return;

    const EccAffinePoint a = d_local[idx];
    const EccAffinePoint b = d_batch[idx];
    int kind = 0;
    if (a.isValid != b.isValid) {
        kind = 1;
    } else if (a.isValid) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (a.x.v[i] != b.x.v[i]) kind = kind ? kind : 2;
            if (a.y.v[i] != b.y.v[i]) kind = kind ? kind : 3;
        }
    }
    if (!kind) return;

    if (atomicCAS(&d_result->mismatch, 0, 1) == 0) {
        d_result->idx = idx;
        d_result->kind = kind;
        d_result->local = a;
        d_result->batch = b;
        d_result->jac = d_jac[idx];
        d_result->local_except = d_local_except[idx];
        const int block = idx / BW_THREADS_INV;
        d_result->block_prod = d_block_prods[block];
        d_result->block_inv = d_block_inv[block];
        bw_u256 z = d_jac[idx].jz;
        fieldNormalize(z.v);
        if (!bw_is_zero(z)) fieldInv(z.v, d_result->direct_z_inv.v);
        fieldMul(d_local_except[idx].v, d_block_inv[block].v, d_result->batch_z_inv.v);
        fieldNormalize(d_result->batch_z_inv.v);
        fieldMul(d_block_prods[block].v, d_block_inv[block].v, d_result->block_check.v);
        fieldNormalize(d_result->block_check.v);
        fieldMul(z.v, d_result->batch_z_inv.v, d_result->z_check.v);
        fieldNormalize(d_result->z_check.v);
        d_result->z_zero = d_z_zero[idx];
    }
}

__device__ __forceinline__ void ecc_bytes32_be_to_limbs_le(const uint8_t in[32], uint64_t k[4]) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint8_t* b = in + j * 8;
        k[3-j] = ((uint64_t)b[0] << 56) | ((uint64_t)b[1] << 48)
               | ((uint64_t)b[2] << 40) | ((uint64_t)b[3] << 32)
               | ((uint64_t)b[4] << 24) | ((uint64_t)b[5] << 16)
               | ((uint64_t)b[6] <<  8) |  (uint64_t)b[7];
    }
}

template<int W>
__global__ __launch_bounds__(BW_THREADS_ECC)
void kernel_ecc_nodes_to_jac_t(
    const uint8_t* __restrict__ d_nodes,  // [N][64] = priv[32] || chain[32]
    int N,
    const uint64_t* __restrict__ d_combX,
    const uint64_t* __restrict__ d_combY,
    BrainJacobian* __restrict__ d_jac)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    BrainJacobian J; J.batch_idx = (uint64_t)idx; J._pad = 0;
    if (idx >= N) {
        return;
    }

    uint64_t k[4];
    ecc_bytes32_be_to_limbs_le(d_nodes + (size_t)idx * 64, k);
    uint64_t Jx[4], Jy[4], Jz[4];
    scalar_mul_comb_bw_t<W>(k, d_combX, d_combY, Jx, Jy, Jz);
    J.isValid = ((Jz[0] | Jz[1] | Jz[2] | Jz[3]) != 0) ? 1u : 0u;
    #pragma unroll
    for (int i = 0; i < 4; i++) { J.jx.v[i] = Jx[i]; J.jy.v[i] = Jy[i]; J.jz.v[i] = Jz[i]; }
    d_jac[idx] = J;
}

template<int W>
__global__ __launch_bounds__(BW_THREADS_ECC)
void kernel_ecc_nodes_to_affine_local_t(
    const uint8_t* __restrict__ d_nodes,
    int N,
    const uint64_t* __restrict__ d_combX,
    const uint64_t* __restrict__ d_combY,
    EccAffinePoint* __restrict__ d_affine)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    EccAffinePoint out; out._pad = 0;
    uint64_t k[4], Jx[4], Jy[4], Jz[4];
    ecc_bytes32_be_to_limbs_le(d_nodes + (size_t)idx * 64, k);
    scalar_mul_comb_bw_t<W>(k, d_combX, d_combY, Jx, Jy, Jz);
    if ((Jz[0] | Jz[1] | Jz[2] | Jz[3]) == 0) {
        out.isValid = 0; out.x = out.y = bw_u256{0,0,0,0};
        d_affine[idx] = out; return;
    }
    bw_u256 z, z_inv, zi2, zi3;
    #pragma unroll
    for (int i = 0; i < 4; i++) z.v[i] = Jz[i];
    fieldNormalize(z.v);
    fieldInv(z.v, z_inv.v);
    fieldSqr(z_inv.v, zi2.v);
    fieldMul(zi2.v, z_inv.v, zi3.v);
    fieldMul(Jx, zi2.v, out.x.v);
    fieldMul(Jy, zi3.v, out.y.v);
    fieldNormalize(out.x.v);
    fieldNormalize(out.y.v);
    out.isValid = 1;
    d_affine[idx] = out;
}

__global__ __launch_bounds__(BW_THREADS_INV)
void kernel_ecc_affine_local_from_jac(
    const BrainJacobian* __restrict__ d_jac,
    int N,
    EccAffinePoint* __restrict__ d_affine)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    EccAffinePoint out; out._pad = 0;
    const BrainJacobian& J = d_jac[gid];
    if (!J.isValid || bw_is_zero(J.jz)) {
        out.isValid = 0; out.x = out.y = bw_u256{0,0,0,0};
        d_affine[gid] = out; return;
    }

    bw_u256 z, z_inv, zi2, zi3;
    z = J.jz;
    fieldNormalize(z.v);
    fieldInv(z.v, z_inv.v);
    fieldSqr(z_inv.v, zi2.v);
    fieldMul(zi2.v, z_inv.v, zi3.v);
    fieldMul(J.jx.v, zi2.v, out.x.v);
    fieldMul(J.jy.v, zi3.v, out.y.v);
    fieldNormalize(out.x.v);
    fieldNormalize(out.y.v);
    out.isValid = 1;
    d_affine[gid] = out;
}

__global__ __launch_bounds__(BW_THREADS_INV)
void kernel_ecc_affine_from_jac(
    const BrainJacobian* __restrict__ d_jac,
    const bw_u256*       __restrict__ d_local_except,
    const bw_u256*       __restrict__ d_block_inv,
    const uint8_t*       __restrict__ d_z_zero,
    int N,
    EccAffinePoint* __restrict__ d_affine)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    EccAffinePoint out; out._pad = 0;
    if (d_z_zero[gid] || !d_jac[gid].isValid) {
        out.isValid = 0; out.x = out.y = bw_u256{0,0,0,0};
        d_affine[gid] = out; return;
    }

    bw_u256 z_inv;
    fieldMul(d_local_except[gid].v, d_block_inv[blockIdx.x].v, z_inv.v);
    fieldNormalize(z_inv.v);

    bw_u256 zi2, zi3;
    fieldSqr(z_inv.v, zi2.v);
    fieldMul(zi2.v, z_inv.v, zi3.v);
    fieldMul(d_jac[gid].jx.v, zi2.v, out.x.v);
    fieldMul(d_jac[gid].jy.v, zi3.v, out.y.v);
    fieldNormalize(out.x.v);
    fieldNormalize(out.y.v);
    out.isValid = 1;
    d_affine[gid] = out;
}

template<int W>
__global__ __launch_bounds__(BW_THREADS_ECC)
void kernel_ecc_final_local_t(
    const uint8_t* __restrict__ d_nodes,
    const uint64_t* __restrict__ d_indices,
    int N,
    const uint64_t* __restrict__ d_combX,
    const uint64_t* __restrict__ d_combY,
    const TargetData* __restrict__ d_target,
    HydraResult* __restrict__ d_result)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (atomicAdd(&d_result->found, 0) != 0) return;

    EccAffinePoint p;
    uint64_t k[4], Jx[4], Jy[4], Jz[4];
    ecc_bytes32_be_to_limbs_le(d_nodes + (size_t)idx * 64, k);
    scalar_mul_comb_bw_t<W>(k, d_combX, d_combY, Jx, Jy, Jz);
    if ((Jz[0] | Jz[1] | Jz[2] | Jz[3]) == 0) return;
    bw_u256 z, z_inv, zi2, zi3;
    #pragma unroll
    for (int i = 0; i < 4; i++) z.v[i] = Jz[i];
    fieldNormalize(z.v);
    fieldInv(z.v, z_inv.v);
    fieldSqr(z_inv.v, zi2.v);
    fieldMul(zi2.v, z_inv.v, zi3.v);
    fieldMul(Jx, zi2.v, p.x.v);
    fieldMul(Jy, zi3.v, p.y.v);
    fieldNormalize(p.x.v);
    fieldNormalize(p.y.v);

    uint8_t computed[20];
    bool hit = false;
    if (is_any_bloom(d_target)) {
        if (bloom_want_btc(d_target)) {
            uint8_t h160[20];
            getHash160_33_from_limbs((p.y.v[0] & 1) ? 0x03 : 0x02, p.x.v, h160);
            hit = bloom_check(h160, d_target->d_bloom_filter, d_target->bloom_m_bits);
        }
        if (!hit && bloom_want_eth(d_target)) {
            uint8_t eth20[20];
            getEthAddr_from_limbs(p.x.v, p.y.v, eth20);
            hit = bloom_check(eth20, d_target->d_bloom_filter, d_target->bloom_m_bits);
        }
    } else {
        if (d_target->type == TargetType::BTC) {
            getHash160_33_from_limbs((p.y.v[0] & 1) ? 0x03 : 0x02, p.x.v, computed);
            hit = hash20_matches(computed, d_target->hash20);
        } else if (d_target->type == TargetType::ETH) {
            getEthAddr_from_limbs(p.x.v, p.y.v, computed);
            hit = hash20_matches(computed, d_target->hash20);
        } else if (d_target->type == TargetType::BTC_PUBKEY) {
            bool ok = true;
            for(int i = 0; i < 4; i++) if (p.x.v[i] != d_target->pubkey_x[i]) { ok = false; break; }
            if (ok && (uint8_t)(p.y.v[0] & 1) == d_target->pubkey_y_parity) hit = true;
        } else if (d_target->type == TargetType::ETH_PUBKEY) {
            bool ok = true;
            for(int i = 0; i < 4; i++) if (p.x.v[i] != d_target->pubkey_x[i] || p.y.v[i] != d_target->pubkey_y[i]) { ok = false; break; }
            hit = ok;
        }
    }
    if (hit) {
        if (atomicCAS(&d_result->found, 0, 1) == 0)
            d_result->index = d_indices ? d_indices[idx] : (uint64_t)idx;
    }
}

__global__ __launch_bounds__(256, 2)
void kernel_bip32_derive_normal_from_affine(
    const uint8_t* __restrict__ d_parent_nodes, // [N][64] = priv[32] || chain[32]
    const EccAffinePoint* __restrict__ d_parent_pub,
    uint32_t child_index,
    int N,
    uint8_t* __restrict__ d_child_nodes)        // [N][64] = child priv || child chain
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride) {
        uint8_t* dst = d_child_nodes + (size_t)i * 64;
        const EccAffinePoint& pub = d_parent_pub[i];
        if (!pub.isValid) {
            #pragma unroll
            for (int b = 0; b < 64; b++) dst[b] = 0;
            continue;
        }

        const uint8_t* parent = d_parent_nodes + (size_t)i * 64;
        const uint8_t* parent_priv = parent;
        const uint8_t* parent_chain = parent + 32;

        uint8_t data[37];
        data[0] = (pub.y.v[0] & 1ULL) ? 0x03 : 0x02;
        #pragma unroll
        for (int li = 0; li < 4; li++) {
            uint64_t w = pub.x.v[3-li];
            data[1+li*8+0] = (uint8_t)(w >> 56);
            data[1+li*8+1] = (uint8_t)(w >> 48);
            data[1+li*8+2] = (uint8_t)(w >> 40);
            data[1+li*8+3] = (uint8_t)(w >> 32);
            data[1+li*8+4] = (uint8_t)(w >> 24);
            data[1+li*8+5] = (uint8_t)(w >> 16);
            data[1+li*8+6] = (uint8_t)(w >>  8);
            data[1+li*8+7] = (uint8_t)(w);
        }
        data[33] = (uint8_t)(child_index >> 24);
        data[34] = (uint8_t)(child_index >> 16);
        data[35] = (uint8_t)(child_index >>  8);
        data[36] = (uint8_t)(child_index);

        uint8_t out[64];
        hmac_sha512_bip32(parent_chain, data, out);
        add_mod_n(out, parent_priv, dst);
        #pragma unroll
        for (int b = 0; b < 32; b++) dst[32+b] = out[32+b];
    }
}

__global__ __launch_bounds__(BW_THREADS_INV)
void kernel_ecc_finalize_from_jac_index(
    const BrainJacobian* __restrict__ d_jac,
    const bw_u256*       __restrict__ d_local_except,
    const bw_u256*       __restrict__ d_block_inv,
    const uint8_t*       __restrict__ d_z_zero,
    const uint64_t*      __restrict__ d_indices,
    int N,
    const TargetData* __restrict__ d_target,
    HydraResult*      __restrict__ d_result)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    if (d_z_zero[gid] || !d_jac[gid].isValid) return;
    if (atomicAdd(&d_result->found, 0) != 0) return;

    bw_u256 z_inv;
    fieldMul(d_local_except[gid].v, d_block_inv[blockIdx.x].v, z_inv.v);
    fieldNormalize(z_inv.v);

    bw_u256 zi2, zi3, ax, ay;
    fieldSqr(z_inv.v, zi2.v);
    fieldMul(zi2.v, z_inv.v, zi3.v);
    fieldMul(d_jac[gid].jx.v, zi2.v, ax.v);
    fieldMul(d_jac[gid].jy.v, zi3.v, ay.v);
    fieldNormalize(ax.v);
    fieldNormalize(ay.v);

    uint8_t computed[20];
    bool hit = false;
    if (is_any_bloom(d_target)) {
        if (bloom_want_btc(d_target)) {
            uint8_t h160[20];
            getHash160_33_from_limbs((ay.v[0] & 1) ? 0x03 : 0x02, ax.v, h160);
            hit = bloom_check(h160, d_target->d_bloom_filter, d_target->bloom_m_bits);
        }
        if (!hit && bloom_want_eth(d_target)) {
            uint8_t eth20[20];
            getEthAddr_from_limbs(ax.v, ay.v, eth20);
            hit = bloom_check(eth20, d_target->d_bloom_filter, d_target->bloom_m_bits);
        }
    } else {
        if (d_target->type == TargetType::BTC) {
            getHash160_33_from_limbs((ay.v[0] & 1) ? 0x03 : 0x02, ax.v, computed);
            hit = hash20_matches(computed, d_target->hash20);
        } else if (d_target->type == TargetType::ETH) {
            getEthAddr_from_limbs(ax.v, ay.v, computed);
            hit = hash20_matches(computed, d_target->hash20);
        } else if (d_target->type == TargetType::BTC_PUBKEY) {
            bool ok = true;
            for(int i = 0; i < 4; i++) if (ax.v[i] != d_target->pubkey_x[i]) { ok = false; break; }
            if (ok && (uint8_t)(ay.v[0] & 1) == d_target->pubkey_y_parity) hit = true;
        } else if (d_target->type == TargetType::ETH_PUBKEY) {
            bool ok = true;
            for(int i = 0; i < 4; i++) if (ax.v[i] != d_target->pubkey_x[i] || ay.v[i] != d_target->pubkey_y[i]) { ok = false; break; }
            hit = ok;
        }
    }
    if (hit) {
        if (atomicCAS(&d_result->found, 0, 1) == 0)
            d_result->index = d_indices ? d_indices[gid] : (uint64_t)gid;
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Kernel 1 : SHA256 + wCOMB → Jacobians   (0 smem — occupancy maximale)
// ═════════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void pointDoubleJ_ip(BrainJacobian &P) {
    if ((P.jz.v[0]|P.jz.v[1]|P.jz.v[2]|P.jz.v[3]) == 0ULL || ((P.jy.v[0]|P.jy.v[1]|P.jy.v[2]|P.jy.v[3]) == 0ULL)) {
        #pragma unroll
        for (int i=0; i<4; i++) { P.jx.v[i]=0; P.jy.v[i]=0; P.jz.v[i]=0; }
        return;
    }
    uint64_t T1[4], T2[4], T3[4], RX[4], RY[4], RZ[4];
    fieldSqr(P.jy.v, T1);
    fieldMul(P.jx.v, T1, T2);
    fieldAdd(T2, T2, T2); fieldAdd(T2, T2, T2);   // S = 4*X*Y^2
    fieldSqr(P.jx.v, T3);
    fieldAdd(T3, T3, RX); fieldAdd(RX, T3, T3); // M = 3*X^2
    fieldSqr(T3, RX);
    fieldAdd(T2, T2, RY); fieldSub(RX, RY, RX); // X3 = M^2 - 2S
    fieldMul(P.jy.v, P.jz.v, RZ); fieldAdd(RZ, RZ, RZ); // Z3 = 2*Y*Z
    fieldSub(T2, RX, T2);
    fieldMul(T3, T2, RY);
    fieldSqr(T1, T1);
    fieldAdd(T1, T1, T1); fieldAdd(T1, T1, T1); fieldAdd(T1, T1, T1);
    fieldSub(RY, T1, RY); // Y3 = M*(S-X3) - 8*Y^4
    #pragma unroll
    for (int i=0; i<4; i++) { P.jx.v[i]=RX[i]; P.jy.v[i]=RY[i]; P.jz.v[i]=RZ[i]; }
}

__device__ __forceinline__ void point_add_mixed_light(
    BrainJacobian &P,
    const uint64_t* __restrict__ Qx,
    const uint64_t* __restrict__ Qy)
{
    if ((P.jz.v[0]|P.jz.v[1]|P.jz.v[2]|P.jz.v[3]) == 0ULL) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) { P.jx.v[i]=Qx[i]; P.jy.v[i]=Qy[i]; P.jz.v[i]=0; }
        P.jz.v[0] = 1; return;
    }
    uint64_t t[4], u[4], v[4];
    fieldSqr(P.jz.v, t);
    fieldMul(Qx, t, u);
    fieldMul(P.jz.v, t, t);
    fieldMul(Qy, t, v);
    fieldSub(u, P.jx.v, u);   // H
    fieldSub(v, P.jy.v, v);   // r
    if ((u[0]|u[1]|u[2]|u[3]) == 0ULL) {
        if ((v[0]|v[1]|v[2]|v[3]) == 0ULL) pointDoubleJ_ip(P);
        else {
            #pragma unroll
            for(int i=0; i<4; ++i) { P.jx.v[i]=0; P.jy.v[i]=0; P.jz.v[i]=0; }
        }
        return;
    }
    fieldMul(P.jz.v, u, P.jz.v);   // Z3
    fieldSqr(u, t);           // H^2
    fieldMul(u, t, u);        // H^3
    fieldMul(P.jx.v, t, t);      // X1*H^2
    fieldSqr(v, P.jx.v);         // r^2
    fieldSub(P.jx.v, u, P.jx.v);
    fieldSub(P.jx.v, t, P.jx.v);
    fieldSub(P.jx.v, t, P.jx.v);    // X3
    fieldSub(t, P.jx.v, t);
    fieldMul(v, t, t);
    fieldMul(P.jy.v, u, P.jy.v);
    fieldSub(t, P.jy.v, P.jy.v);    // Y3
}

__device__ __forceinline__ void ecc_add_constant_point(
    BrainJacobian* res,
    const uint64_t* cx,
    const uint64_t* cy)
{
    if (res->isValid) {
        point_add_mixed_light(*res, cx, cy);
    }
}

__device__ __forceinline__
void derive_C_minus_P(const BrainJacobian* base,
                      BrainJacobian*       out)
{
    *out = *base;           // copie complète (jx, jy, jz, isValid)
    // Negate Jy : out = -base  (en Jacobien, -P = (X, -Y, Z))
    fieldNormalize(out->jy.v);
    fieldNeg(out->jy.v, out->jy.v);    // out = -P
    // out = C + (-P) = C - P  (addition mixte Jacobien + affine)
    ecc_add_constant_point(out, d_Cx, d_Cy);
    if ((out->jz.v[0]|out->jz.v[1]|out->jz.v[2]|out->jz.v[3]) == 0) out->isValid = 0;
}

template<bool USE_CP = false>
__global__ __launch_bounds__(BW_THREADS_ECC)
void kernel_bw_ecc(
    const uint8_t* __restrict__ d_chunk, const uint32_t* __restrict__ d_offsets,
    int N, int batch_count,
    const uint64_t* __restrict__ d_combX, const uint64_t* __restrict__ d_combY,
    int comb_w, int comb_cols, int comb_stride,
    BrainJacobian* __restrict__ d_jac,
    const GpuRule* __restrict__ d_rules, int num_rules)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    BrainJacobian J; J.batch_idx=(uint64_t)idx; J._pad=0;
    if (idx >= N) {
        return;
    }
    
    int word_idx = num_rules > 0 ? (idx % batch_count) : idx;
    int rule_idx = num_rules > 0 ? (idx / batch_count) : 0;

    uint32_t offset = d_offsets[word_idx];
    uint32_t pass_len = (uint32_t)d_chunk[offset];
    const uint8_t* pass = d_chunk + offset + 1;

    uint8_t local_pass[MAX_BRAIN_LEN];
    uint32_t out_len = pass_len;
    bool valid = true;
    if (num_rules > 0) {
        valid = apply_rule(pass, pass_len, local_pass, out_len, &d_rules[rule_idx], MAX_BRAIN_LEN);
    } else {
        #pragma unroll 4
        for (uint32_t i = 0; i < pass_len; ++i) local_pass[i] = pass[i];
    }
    
    if (!valid) {
        J.isValid = 0;
        J.jx = bw_u256{{0,0,0,0}};
        J.jy = bw_u256{{0,0,0,0}};
        J.jz = bw_u256{{0,0,0,0}};
        if (USE_CP) {
            int base_N = batch_count * (num_rules > 0 ? num_rules : 1);
            d_jac[idx] = J;
            J.batch_idx = base_N + idx;
            d_jac[base_N + idx] = J;
        } else {
            d_jac[idx] = J;
        }
        return;
    }

    uint8_t sha[32];
    bw_sha256(local_pass, out_len, sha);
    uint64_t k[4];
    #pragma unroll
    for(int j=0;j<4;j++){
        const uint8_t* b=sha+j*8;
        k[3-j]=((uint64_t)b[0]<<56)|((uint64_t)b[1]<<48)|((uint64_t)b[2]<<40)|((uint64_t)b[3]<<32)
              |((uint64_t)b[4]<<24)|((uint64_t)b[5]<<16)|((uint64_t)b[6]<<8)|(uint64_t)b[7];
    }
    uint64_t Jx[4],Jy[4],Jz[4];
    scalar_mul_comb_bw(k, d_combX, d_combY, comb_w, comb_cols, comb_stride, Jx, Jy, Jz);
    J.isValid=((Jz[0]|Jz[1]|Jz[2]|Jz[3])!=0)?1:0;
    for(int i=0;i<4;i++){J.jx.v[i]=Jx[i];J.jy.v[i]=Jy[i];J.jz.v[i]=Jz[i];}

    if (USE_CP) {
        int base_N = batch_count * (num_rules > 0 ? num_rules : 1);
        d_jac[idx] = J;
        
        BrainJacobian J_cp;
        derive_C_minus_P(&J, &J_cp);
        J_cp.batch_idx = base_N + idx;
        J_cp._pad = 0;
        d_jac[base_N + idx] = J_cp;
    } else {
        d_jac[idx] = J;
    }
}

template<int W, bool USE_CP = false>
__global__ __launch_bounds__(BW_THREADS_ECC)
void kernel_bw_ecc_t(
    const uint8_t* __restrict__ d_chunk, const uint32_t* __restrict__ d_offsets,
    int N, int batch_count,
    const uint64_t* __restrict__ d_combX, const uint64_t* __restrict__ d_combY,
    BrainJacobian* __restrict__ d_jac,
    const GpuRule* __restrict__ d_rules, int num_rules)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    BrainJacobian J; J.batch_idx = (uint64_t)idx; J._pad = 0;
    if (idx >= N) {
        return;
    }
    
    int word_idx = num_rules > 0 ? (idx % batch_count) : idx;
    int rule_idx = num_rules > 0 ? (idx / batch_count) : 0;

    uint32_t offset = d_offsets[word_idx];
    uint32_t pass_len = (uint32_t)d_chunk[offset];
    const uint8_t* pass = d_chunk + offset + 1;

    uint8_t local_pass[MAX_BRAIN_LEN];
    uint32_t out_len = pass_len;
    bool valid = true;
    if (num_rules > 0) {
        valid = apply_rule(pass, pass_len, local_pass, out_len, &d_rules[rule_idx], MAX_BRAIN_LEN);
    } else {
        #pragma unroll 4
        for (uint32_t i = 0; i < pass_len; ++i) local_pass[i] = pass[i];
    }

    if (!valid) {
        J.isValid = 0;
        J.jx = bw_u256{{0,0,0,0}};
        J.jy = bw_u256{{0,0,0,0}};
        J.jz = bw_u256{{0,0,0,0}};
        if (USE_CP) {
            int base_N = batch_count * (num_rules > 0 ? num_rules : 1);
            d_jac[idx] = J;
            J.batch_idx = base_N + idx;
            d_jac[base_N + idx] = J;
        } else {
            d_jac[idx] = J;
        }
        return;
    }

    uint8_t sha[32];
    bw_sha256(local_pass, out_len, sha);
    uint64_t k[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint8_t* b = sha + j * 8;
        k[3-j] = ((uint64_t)b[0]<<56) | ((uint64_t)b[1]<<48) | ((uint64_t)b[2]<<40) | ((uint64_t)b[3]<<32)
               | ((uint64_t)b[4]<<24) | ((uint64_t)b[5]<<16) | ((uint64_t)b[6]<<8)  | (uint64_t)b[7];
    }
    uint64_t Jx[4], Jy[4], Jz[4];
    scalar_mul_comb_bw_t<W>(k, d_combX, d_combY, Jx, Jy, Jz);
    J.isValid = ((Jz[0]|Jz[1]|Jz[2]|Jz[3]) != 0) ? 1 : 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) { J.jx.v[i] = Jx[i]; J.jy.v[i] = Jy[i]; J.jz.v[i] = Jz[i]; }
    
    if (USE_CP) {
        int base_N = batch_count * (num_rules > 0 ? num_rules : 1);
        d_jac[idx] = J;
        
        BrainJacobian J_cp;
        derive_C_minus_P(&J, &J_cp);
        J_cp.batch_idx = base_N + idx;
        J_cp._pad = 0;
        d_jac[base_N + idx] = J_cp;
    } else {
        d_jac[idx] = J;
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Kernel 2 : Prefix/suffix scan des Z par bloc → local_except + block_products
// (portage de kernel_compute_local_except_and_block_prod, Barracuda/ECC.h)
// smem = 2 × blockDim.x × sizeof(bw_u256)
// ═════════════════════════════════════════════════════════════════════════════
__global__ __launch_bounds__(BW_THREADS_INV)
void kernel_bw_local_prod(
    const BrainJacobian* __restrict__ d_jac, int N,
    bw_u256* __restrict__ d_local_except,
    bw_u256* __restrict__ d_block_prods,
    uint8_t* __restrict__ d_z_zero)
{
    extern __shared__ bw_u256 s_mem[];
    bw_u256* s_z   = s_mem;
    bw_u256* s_tmp = s_mem + blockDim.x;
    const int tid=threadIdx.x;
    const int gid=blockIdx.x*blockDim.x+tid;

    bw_u256 z=bw_fp_one(); bool is_zero=false;
    if (gid<N) {
        const BrainJacobian& J=d_jac[gid];
        if (J.isValid && !bw_is_zero(J.jz)) {
            z=J.jz;
            fieldNormalize(z.v);
        } else {
            is_zero=true;
        }
        d_z_zero[gid]=(uint8_t)is_zero;
    }
    s_z[tid]=z;
    s_tmp[tid]=z;
    __syncthreads();

    // Shared-memory scans are slightly less clever than the original warp
    // transpose, but they handle partial blocks predictably and normalize every
    // product. Invalid lanes contribute one, so the final block product remains
    // valid for short tail blocks.
    for(int off=1;off<blockDim.x;off<<=1){
        bw_u256 prev=bw_fp_one();
        if(tid>=off) prev=s_tmp[tid-off];
        __syncthreads();
        if(tid>=off) {
            bw_u256 np;
            fieldMul(prev.v,s_tmp[tid].v,np.v);
            fieldNormalize(np.v);
            s_tmp[tid]=np;
        }
        __syncthreads();
    }
    bw_u256 P_i=bw_fp_one();
    if(tid>0) P_i=s_tmp[tid-1];
    if(tid==blockDim.x-1) d_block_prods[blockIdx.x]=s_tmp[tid];
    __syncthreads();

    s_tmp[tid]=s_z[tid];
    __syncthreads();
    for(int off=1;off<blockDim.x;off<<=1){
        bw_u256 next=bw_fp_one();
        if(tid+off<blockDim.x) next=s_tmp[tid+off];
        __syncthreads();
        if(tid+off<blockDim.x) {
            bw_u256 ns;
            fieldMul(s_tmp[tid].v,next.v,ns.v);
            fieldNormalize(ns.v);
            s_tmp[tid]=ns;
        }
        __syncthreads();
    }
    bw_u256 S_i=bw_fp_one();
    if(tid+1<blockDim.x) S_i=s_tmp[tid+1];
    if(gid<N){
        bw_u256 local=P_i;
        bw_u256 nl;
        fieldMul(local.v,S_i.v,nl.v);
        fieldNormalize(nl.v);
        local=nl;
        d_local_except[gid]=local;
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Kernel 3 : Inversion des produits de blocs (1 fieldInv par 128 éléments)
// (portage de kernel_batch_invert_block_products, Barracuda/ECC.h)
// smem = 132 × sizeof(bw_u256)
// ═════════════════════════════════════════════════════════════════════════════
__global__ __launch_bounds__(128)
void kernel_bw_invert_blocks(
    const bw_u256* __restrict__ d_block_prods,
    bw_u256* __restrict__ d_block_inv,
    int num_blocks)
{
    extern __shared__ bw_u256 s_mem[];
    const int gid=blockIdx.x*128+threadIdx.x;
    bw_u256 val=bw_fp_one();
    if(gid<num_blocks){
        val=d_block_prods[gid];
        if(bw_is_zero(val)) val=bw_fp_one();
    }
    bw_u256 result;
    bw_batch_invert_128(&val,&result,s_mem);
    if(gid<num_blocks) d_block_inv[gid]=result;
}


// ═════════════════════════════════════════════════════════════════════════════
// Kernel 4 : z_inv = local_except × block_inv → affine → hash/compare
// (portage de kernel_finalize_and_hash, Barracuda/Hash.h)
// 0 smem
// ═════════════════════════════════════════════════════════════════════════════
__global__ __launch_bounds__(BW_THREADS_INV)
void kernel_bw_finalize(
    const BrainJacobian* __restrict__ d_jac,
    const bw_u256*       __restrict__ d_local_except,
    const bw_u256*       __restrict__ d_block_inv,
    const uint8_t*       __restrict__ d_z_zero,
    int N,
    const TargetData* __restrict__ d_target,
    HydraResult*      __restrict__ d_result)
{
    const int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid>=N) return;
    if(d_z_zero[gid]) return;
    if(atomicAdd(&d_result->found,0)!=0) return;
    const BrainJacobian* J=&d_jac[gid];
    if(!J->isValid) return;

    // z_inv = local_except[gid] × block_inv[blockIdx.x]
    bw_u256 z_inv;
    fieldMul(d_local_except[gid].v, d_block_inv[blockIdx.x].v, z_inv.v);
    fieldNormalize(z_inv.v);

    // Jacobien → affine
    bw_u256 zi2,zi3,ax,ay;
    fieldSqr(z_inv.v,zi2.v); fieldMul(zi2.v,z_inv.v,zi3.v);
    fieldMul(J->jx.v,zi2.v,ax.v); fieldMul(J->jy.v,zi3.v,ay.v);
    fieldNormalize(ax.v); fieldNormalize(ay.v);

    bool hit=false;
    if(is_any_bloom(d_target)){
        if(bloom_want_btc(d_target)){
            uint8_t h160[20];
            getHash160_33_from_limbs((ay.v[0]&1)?0x03:0x02,ax.v,h160);
            hit=bloom_check(h160,d_target->d_bloom_filter,d_target->bloom_m_bits);
        }
        if(!hit&&bloom_want_eth(d_target)){
            uint8_t eth20[20];
            getEthAddr_from_limbs(ax.v,ay.v,eth20);
            hit=bloom_check(eth20,d_target->d_bloom_filter,d_target->bloom_m_bits);
        }
    } else if(d_target->type==TargetType::BTC){
        uint8_t h160[20];
        getHash160_33_from_limbs((ay.v[0]&1)?0x03:0x02,ax.v,h160);
        hit=true;
        for(int b=0;b<20;b++) if(h160[b]!=d_target->hash20[b]){hit=false;break;}
    } else if(d_target->type==TargetType::ETH){
        uint8_t eth20[20];
        getEthAddr_from_limbs(ax.v,ay.v,eth20);
        hit=true;
        for(int b=0;b<20;b++) if(eth20[b]!=d_target->hash20[b]){hit=false;break;}
    } else if(d_target->type==TargetType::BTC_PUBKEY){
        bool ok=true;
        for(int i=0;i<4;i++) if(ax.v[i]!=d_target->pubkey_x[i]){ok=false;break;}
        if(ok&&(uint8_t)(ay.v[0]&1)==d_target->pubkey_y_parity) hit=true;
    } else if(d_target->type==TargetType::ETH_PUBKEY){
        bool ok=true;
        for(int i=0;i<4;i++) if(ax.v[i]!=d_target->pubkey_x[i]||ay.v[i]!=d_target->pubkey_y[i]){ok=false;break;}
        hit=ok;
    }

    if(hit)
        if(atomicCAS(&d_result->found,0,1)==0)
            d_result->index=J->batch_idx;
}
