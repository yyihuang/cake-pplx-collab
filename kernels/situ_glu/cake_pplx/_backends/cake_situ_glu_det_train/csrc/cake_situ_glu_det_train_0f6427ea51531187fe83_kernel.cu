/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_exp2_noftz(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_situ_glu_det_train_0f6427ea51531187fe83(__nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ up, __nv_bfloat16* __restrict__ dh, __nv_bfloat16* __restrict__ d_gate, __nv_bfloat16* __restrict__ d_up, __nv_bfloat16* __restrict__ h, int rows, int width, long long gate_stride, long long up_stride, long long dh_stride, long long d_gate_stride, long long d_up_stride, long long h_stride, float beta, float lb)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int part = warp_1 & 3;
    int sub = warp_1 >> 2;
    int nseg = width >> 8;
    float k1 = -2.88539f / beta;
    float k2 = -2.88539f / lb;
    float bl = beta * lb;
    float cg = k1 * 0.6931472f;
    float cu = k2 * 0.6931472f;
    float nb = -beta;
    float nl = -lb;
    float sdb = beta * 0.99999905f;
    float sdl = lb * 0.99999905f;
    float nb_0 = -beta;
    float nl_0 = -lb;
    float ub = beta * 1.0000005f;
    float ul = lb * 1.0000005f;
    #pragma unroll 1
    for (int first = bid; first < rows; first += num_bids) {
        int row = first + sub;
        int limit = ((row < rows) ? nseg : 0);
        int _min_0 = ((row) < (rows - 1) ? (row) : (rows - 1));
        long long at = (long long)_min_0;
        long long goff = at * gate_stride;
        long long uoff = at * up_stride;
        long long doff = at * dh_stride;
        long long dgoff = at * d_gate_stride;
        long long duoff = at * d_up_stride;
        long long hoff = at * h_stride;
        #pragma unroll 1
        for (int s0 = part; s0 < nseg; s0 += 4) {
            unsigned int gw[4];
            unsigned int uw[4];
            unsigned int dw[4];
            int seg_0 = ((s0) < (nseg - 1) ? (s0) : (nseg - 1));
            long long col_0 = (long long)(seg_0 * 256 + lane_1 * 8);
            {
                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(gate + (goff + col_0) + 0);
                gw[0 + 0] = _uv4_1.x;
                gw[0 + 1] = _uv4_1.y;
                gw[0 + 2] = _uv4_1.z;
                gw[0 + 3] = _uv4_1.w;
            }
            {
                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(up + (uoff + col_0) + 0);
                uw[0 + 0] = _uv4_2.x;
                uw[0 + 1] = _uv4_2.y;
                uw[0 + 2] = _uv4_2.z;
                uw[0 + 3] = _uv4_2.w;
            }
            {
                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(dh + (doff + col_0) + 0);
                dw[0 + 0] = _uv4_3.x;
                dw[0 + 1] = _uv4_3.y;
                dw[0 + 2] = _uv4_3.z;
                dw[0 + 3] = _uv4_3.w;
            }
            if (beta == 4.0f) {
                float2 x_0 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[0]));
                float2 v_0 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[0]));
                float2 d_0 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[0]));
                unsigned int sign_0 = (gw[0] ^ uw[0]) & 2147516416u;
                float _abs_4;
                asm("abs.f32 %0, %1;" : "=f"(_abs_4) : "f"(x_0.x));
                float _abs_5;
                asm("abs.f32 %0, %1;" : "=f"(_abs_5) : "f"(x_0.y));
                float2 ax = make_float2(_abs_4, _abs_5);
                float _abs_6;
                asm("abs.f32 %0, %1;" : "=f"(_abs_6) : "f"(v_0.x));
                float _abs_7;
                asm("abs.f32 %0, %1;" : "=f"(_abs_7) : "f"(v_0.y));
                float2 av = make_float2(_abs_6, _abs_7);
                float2 _f32x2_mul_b_8 = make_float2(k1, k1);
                float2 te;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te) : "l"(*(const unsigned long long*)&ax), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
                float2 _f32x2_mul_b_9 = make_float2(k2, k2);
                float2 tw;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw) : "l"(*(const unsigned long long*)&av), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
                float _exp2_10 = approx_exp2(te.x);
                float _exp2_11 = approx_exp2(te.y);
                float2 e = make_float2(_exp2_10, _exp2_11);
                float _exp2_12 = approx_exp2(tw.x);
                float _exp2_13 = approx_exp2(tw.y);
                float2 w = make_float2(_exp2_12, _exp2_13);
                float2 p;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p) : "l"(*(const unsigned long long*)&e), "l"(*(const unsigned long long*)&e));
                float2 d1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                float2 d2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p);
                float2 dw_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                float2 ng = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                float2 nu = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                float _abs_14;
                asm("abs.f32 %0, %1;" : "=f"(_abs_14) : "f"(x_0.x));
                float _abs_15;
                asm("abs.f32 %0, %1;" : "=f"(_abs_15) : "f"(x_0.y));
                float2 ax_0 = make_float2(_abs_14, _abs_15);
                float _abs_16;
                asm("abs.f32 %0, %1;" : "=f"(_abs_16) : "f"(v_0.x));
                float _abs_17;
                asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(v_0.y));
                float2 av_0 = make_float2(_abs_16, _abs_17);
                float2 d12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12) : "l"(*(const unsigned long long*)&d1), "l"(*(const unsigned long long*)&d2));
                float both = approx_rcp(mul_f32x2_noftz(d12, dw_0).x * mul_f32x2_noftz(d12, dw_0).y);
                float2 _f32x2_mul_a_18 = make_float2(mul_f32x2_noftz(d12, dw_0).y, mul_f32x2_noftz(d12, dw_0).x);
                float2 _f32x2_mul_b_19 = make_float2(both, both);
                float2 r_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
                float2 r = r_0;
                float2 _f32x2_mul_b_20 = mul_f32x2_noftz(r, r);
                float2 m1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1) : "l"(*(const unsigned long long*)&d_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
                float2 _f32x2_mul_b_21 = fma_f32x2_rn_noftz(ax_0, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near) : "l"(*(const unsigned long long*)&ax_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
                float2 far = fma_f32x2_rn_noftz(e, make_float2(nb, nb), make_float2(sdb, sdb));
                float2 sbng = make_float2(max_noftz(near.x, far.x), max_noftz(near.y, far.y));
                float2 _f32x2_mul_b_22 = fma_f32x2_rn_noftz(av_0, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_0) : "l"(*(const unsigned long long*)&av_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
                float2 far_0 = fma_f32x2_rn_noftz(w, make_float2(nl, nl), make_float2(sdl, sdl));
                float2 slnu = make_float2(max_noftz(near_0.x, far_0.x), max_noftz(near_0.y, far_0.y));
                unsigned int lo = __as_u32(sbng.x) | __as_u32(x_0.x) & 2147483648u;
                unsigned int hi = __as_u32(sbng.y) | __as_u32(x_0.y) & 2147483648u;
                float2 bng = make_float2(__uint_as_float(lo), __uint_as_float(hi));
                unsigned int lo_0 = __as_u32(slnu.x) | __as_u32(v_0.x) & 2147483648u;
                unsigned int hi_0 = __as_u32(slnu.y) | __as_u32(v_0.y) & 2147483648u;
                float2 lnu = make_float2(__uint_as_float(lo_0), __uint_as_float(hi_0));
                float2 q = make_float2(((x_0.x < 0.0f) ? 1.0f : p.x), ((x_0.y < 0.0f) ? 1.0f : p.y));
                float2 post = make_float2(((x_0.x < 0.0f) ? p.x : 1.0f), ((x_0.y < 0.0f) ? p.y : 1.0f));
                float2 inner = fma_f32x2_rn_noftz(bng, mul_f32x2_noftz(d1, q), mul_f32x2_noftz(mul_f32x2_noftz(e, make_float2(4.0f, 4.0f)), d2));
                float2 _f32x2_mul_a_23 = mul_f32x2_noftz(mul_f32x2_noftz(m1, lnu), mul_f32x2_noftz(dw_0, inner));
                float2 dg;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&post));
                float2 _f32x2_mul_a_24 = mul_f32x2_noftz(mul_f32x2_noftz(m1, bng), mul_f32x2_noftz(mul_f32x2_noftz(w, make_float2(4.0f, 4.0f)), d12));
                float2 du;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&post));
                float _rcp_25 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p, d1, d1), dw_0).x);
                float _rcp_26 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p, d1, d1), dw_0).y);
                float2 r_1 = make_float2(_rcp_25, _rcp_26);
                float2 rh = r_1;
                float2 near_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_1) : "l"(*(const unsigned long long*)&ax_0), "l"(*(const unsigned long long*)&d1));
                float2 far_1 = fma_f32x2_rn_noftz(e, make_float2(nb, nb), make_float2(ub, ub));
                float _min_27;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_27) : "f"(near_1.x), "f"(far_1.x));
                float _min_28;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_28) : "f"(near_1.y), "f"(far_1.y));
                float2 hb = make_float2(_min_27, _min_28);
                float2 near_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_2) : "l"(*(const unsigned long long*)&av_0), "l"(*(const unsigned long long*)&dw_0));
                float2 far_2 = fma_f32x2_rn_noftz(w, make_float2(nl, nl), make_float2(ul, ul));
                float _min_29;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_29) : "f"(near_2.x), "f"(far_2.x));
                float _min_30;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_30) : "f"(near_2.y), "f"(far_2.y));
                float2 hl = make_float2(_min_29, _min_30);
                float2 _f32x2_mul_a_31 = mul_f32x2_noftz(mul_f32x2_noftz(hb, hl), rh);
                float2 habs;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&habs) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&post));
                gw[0] = __as_u32(__float22bfloat162_rn(dg));
                uw[0] = __as_u32(__float22bfloat162_rn(du));
                dw[0] = __as_u32(__float22bfloat162_rn(habs)) | sign_0;
                float2 x_1 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[1]));
                float2 v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[1]));
                float2 d_1 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[1]));
                unsigned int sign_1 = (gw[1] ^ uw[1]) & 2147516416u;
                float _abs_32;
                asm("abs.f32 %0, %1;" : "=f"(_abs_32) : "f"(x_1.x));
                float _abs_33;
                asm("abs.f32 %0, %1;" : "=f"(_abs_33) : "f"(x_1.y));
                float2 ax_1 = make_float2(_abs_32, _abs_33);
                float _abs_34;
                asm("abs.f32 %0, %1;" : "=f"(_abs_34) : "f"(v_1.x));
                float _abs_35;
                asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(v_1.y));
                float2 av_1 = make_float2(_abs_34, _abs_35);
                float2 _f32x2_mul_b_36 = make_float2(k1, k1);
                float2 te_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_0) : "l"(*(const unsigned long long*)&ax_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
                float2 _f32x2_mul_b_37 = make_float2(k2, k2);
                float2 tw_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_0) : "l"(*(const unsigned long long*)&av_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_37));
                float _exp2_38 = approx_exp2(te_0.x);
                float _exp2_39 = approx_exp2(te_0.y);
                float2 e_0 = make_float2(_exp2_38, _exp2_39);
                float _exp2_40 = approx_exp2(tw_0.x);
                float _exp2_41 = approx_exp2(tw_0.y);
                float2 w_0 = make_float2(_exp2_40, _exp2_41);
                float2 p_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_0) : "l"(*(const unsigned long long*)&e_0), "l"(*(const unsigned long long*)&e_0));
                float2 d1_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                float2 d2_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_0);
                float2 dw_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                float2 ng_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                float2 nu_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                float _abs_42;
                asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(x_1.x));
                float _abs_43;
                asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(x_1.y));
                float2 ax_2 = make_float2(_abs_42, _abs_43);
                float _abs_44;
                asm("abs.f32 %0, %1;" : "=f"(_abs_44) : "f"(v_1.x));
                float _abs_45;
                asm("abs.f32 %0, %1;" : "=f"(_abs_45) : "f"(v_1.y));
                float2 av_2 = make_float2(_abs_44, _abs_45);
                float2 d12_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_0) : "l"(*(const unsigned long long*)&d1_0), "l"(*(const unsigned long long*)&d2_0));
                float both_0 = approx_rcp(mul_f32x2_noftz(d12_0, dw_1).x * mul_f32x2_noftz(d12_0, dw_1).y);
                float2 _f32x2_mul_a_46 = make_float2(mul_f32x2_noftz(d12_0, dw_1).y, mul_f32x2_noftz(d12_0, dw_1).x);
                float2 _f32x2_mul_b_47 = make_float2(both_0, both_0);
                float2 r_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
                float2 r_2 = r_3;
                float2 _f32x2_mul_b_48 = mul_f32x2_noftz(r_2, r_2);
                float2 m1_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_0) : "l"(*(const unsigned long long*)&d_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_48));
                float2 _f32x2_mul_b_49 = fma_f32x2_rn_noftz(ax_2, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_3) : "l"(*(const unsigned long long*)&ax_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
                float2 far_3 = fma_f32x2_rn_noftz(e_0, make_float2(nb, nb), make_float2(sdb, sdb));
                float2 sbng_0 = make_float2(max_noftz(near_3.x, far_3.x), max_noftz(near_3.y, far_3.y));
                float2 _f32x2_mul_b_50 = fma_f32x2_rn_noftz(av_2, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_4) : "l"(*(const unsigned long long*)&av_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_50));
                float2 far_4 = fma_f32x2_rn_noftz(w_0, make_float2(nl, nl), make_float2(sdl, sdl));
                float2 slnu_0 = make_float2(max_noftz(near_4.x, far_4.x), max_noftz(near_4.y, far_4.y));
                unsigned int lo_1 = __as_u32(sbng_0.x) | __as_u32(x_1.x) & 2147483648u;
                unsigned int hi_1 = __as_u32(sbng_0.y) | __as_u32(x_1.y) & 2147483648u;
                float2 bng_0 = make_float2(__uint_as_float(lo_1), __uint_as_float(hi_1));
                unsigned int lo_2 = __as_u32(slnu_0.x) | __as_u32(v_1.x) & 2147483648u;
                unsigned int hi_2 = __as_u32(slnu_0.y) | __as_u32(v_1.y) & 2147483648u;
                float2 lnu_0 = make_float2(__uint_as_float(lo_2), __uint_as_float(hi_2));
                float2 q_0 = make_float2(((x_1.x < 0.0f) ? 1.0f : p_0.x), ((x_1.y < 0.0f) ? 1.0f : p_0.y));
                float2 post_0 = make_float2(((x_1.x < 0.0f) ? p_0.x : 1.0f), ((x_1.y < 0.0f) ? p_0.y : 1.0f));
                float2 inner_0 = fma_f32x2_rn_noftz(bng_0, mul_f32x2_noftz(d1_0, q_0), mul_f32x2_noftz(mul_f32x2_noftz(e_0, make_float2(4.0f, 4.0f)), d2_0));
                float2 _f32x2_mul_a_51 = mul_f32x2_noftz(mul_f32x2_noftz(m1_0, lnu_0), mul_f32x2_noftz(dw_1, inner_0));
                float2 dg_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&post_0));
                float2 _f32x2_mul_a_52 = mul_f32x2_noftz(mul_f32x2_noftz(m1_0, bng_0), mul_f32x2_noftz(mul_f32x2_noftz(w_0, make_float2(4.0f, 4.0f)), d12_0));
                float2 du_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&post_0));
                float _rcp_53 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_0, d1_0, d1_0), dw_1).x);
                float _rcp_54 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_0, d1_0, d1_0), dw_1).y);
                float2 r_4 = make_float2(_rcp_53, _rcp_54);
                float2 rh_0 = r_4;
                float2 near_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_5) : "l"(*(const unsigned long long*)&ax_2), "l"(*(const unsigned long long*)&d1_0));
                float2 far_5 = fma_f32x2_rn_noftz(e_0, make_float2(nb, nb), make_float2(ub, ub));
                float _min_55;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_55) : "f"(near_5.x), "f"(far_5.x));
                float _min_56;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_56) : "f"(near_5.y), "f"(far_5.y));
                float2 hb_0 = make_float2(_min_55, _min_56);
                float2 near_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_6) : "l"(*(const unsigned long long*)&av_2), "l"(*(const unsigned long long*)&dw_1));
                float2 far_6 = fma_f32x2_rn_noftz(w_0, make_float2(nl, nl), make_float2(ul, ul));
                float _min_57;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_57) : "f"(near_6.x), "f"(far_6.x));
                float _min_58;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_58) : "f"(near_6.y), "f"(far_6.y));
                float2 hl_0 = make_float2(_min_57, _min_58);
                float2 _f32x2_mul_a_59 = mul_f32x2_noftz(mul_f32x2_noftz(hb_0, hl_0), rh_0);
                float2 habs_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&habs_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&post_0));
                gw[1] = __as_u32(__float22bfloat162_rn(dg_0));
                uw[1] = __as_u32(__float22bfloat162_rn(du_0));
                dw[1] = __as_u32(__float22bfloat162_rn(habs_0)) | sign_1;
                float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[2]));
                float2 v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[2]));
                float2 d_2 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[2]));
                unsigned int sign_2 = (gw[2] ^ uw[2]) & 2147516416u;
                float _abs_60;
                asm("abs.f32 %0, %1;" : "=f"(_abs_60) : "f"(x_2.x));
                float _abs_61;
                asm("abs.f32 %0, %1;" : "=f"(_abs_61) : "f"(x_2.y));
                float2 ax_3 = make_float2(_abs_60, _abs_61);
                float _abs_62;
                asm("abs.f32 %0, %1;" : "=f"(_abs_62) : "f"(v_2.x));
                float _abs_63;
                asm("abs.f32 %0, %1;" : "=f"(_abs_63) : "f"(v_2.y));
                float2 av_3 = make_float2(_abs_62, _abs_63);
                float2 _f32x2_mul_b_64 = make_float2(k1, k1);
                float2 te_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_1) : "l"(*(const unsigned long long*)&ax_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
                float2 _f32x2_mul_b_65 = make_float2(k2, k2);
                float2 tw_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_1) : "l"(*(const unsigned long long*)&av_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
                float _exp2_66 = approx_exp2(te_1.x);
                float _exp2_67 = approx_exp2(te_1.y);
                float2 e_1 = make_float2(_exp2_66, _exp2_67);
                float _exp2_68 = approx_exp2(tw_1.x);
                float _exp2_69 = approx_exp2(tw_1.y);
                float2 w_1 = make_float2(_exp2_68, _exp2_69);
                float2 p_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_1) : "l"(*(const unsigned long long*)&e_1), "l"(*(const unsigned long long*)&e_1));
                float2 d1_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                float2 d2_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_1);
                float2 dw_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                float2 ng_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                float2 nu_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                float _abs_70;
                asm("abs.f32 %0, %1;" : "=f"(_abs_70) : "f"(x_2.x));
                float _abs_71;
                asm("abs.f32 %0, %1;" : "=f"(_abs_71) : "f"(x_2.y));
                float2 ax_4 = make_float2(_abs_70, _abs_71);
                float _abs_72;
                asm("abs.f32 %0, %1;" : "=f"(_abs_72) : "f"(v_2.x));
                float _abs_73;
                asm("abs.f32 %0, %1;" : "=f"(_abs_73) : "f"(v_2.y));
                float2 av_4 = make_float2(_abs_72, _abs_73);
                float2 d12_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_1) : "l"(*(const unsigned long long*)&d1_1), "l"(*(const unsigned long long*)&d2_1));
                float both_1 = approx_rcp(mul_f32x2_noftz(d12_1, dw_2).x * mul_f32x2_noftz(d12_1, dw_2).y);
                float2 _f32x2_mul_a_74 = make_float2(mul_f32x2_noftz(d12_1, dw_2).y, mul_f32x2_noftz(d12_1, dw_2).x);
                float2 _f32x2_mul_b_75 = make_float2(both_1, both_1);
                float2 r_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
                float2 r_5 = r_6;
                float2 _f32x2_mul_b_76 = mul_f32x2_noftz(r_5, r_5);
                float2 m1_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_1) : "l"(*(const unsigned long long*)&d_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_76));
                float2 _f32x2_mul_b_77 = fma_f32x2_rn_noftz(ax_4, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_7) : "l"(*(const unsigned long long*)&ax_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
                float2 far_7 = fma_f32x2_rn_noftz(e_1, make_float2(nb, nb), make_float2(sdb, sdb));
                float2 sbng_1 = make_float2(max_noftz(near_7.x, far_7.x), max_noftz(near_7.y, far_7.y));
                float2 _f32x2_mul_b_78 = fma_f32x2_rn_noftz(av_4, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_8) : "l"(*(const unsigned long long*)&av_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_78));
                float2 far_8 = fma_f32x2_rn_noftz(w_1, make_float2(nl, nl), make_float2(sdl, sdl));
                float2 slnu_1 = make_float2(max_noftz(near_8.x, far_8.x), max_noftz(near_8.y, far_8.y));
                unsigned int lo_3 = __as_u32(sbng_1.x) | __as_u32(x_2.x) & 2147483648u;
                unsigned int hi_3 = __as_u32(sbng_1.y) | __as_u32(x_2.y) & 2147483648u;
                float2 bng_1 = make_float2(__uint_as_float(lo_3), __uint_as_float(hi_3));
                unsigned int lo_4 = __as_u32(slnu_1.x) | __as_u32(v_2.x) & 2147483648u;
                unsigned int hi_4 = __as_u32(slnu_1.y) | __as_u32(v_2.y) & 2147483648u;
                float2 lnu_1 = make_float2(__uint_as_float(lo_4), __uint_as_float(hi_4));
                float2 q_1 = make_float2(((x_2.x < 0.0f) ? 1.0f : p_1.x), ((x_2.y < 0.0f) ? 1.0f : p_1.y));
                float2 post_1 = make_float2(((x_2.x < 0.0f) ? p_1.x : 1.0f), ((x_2.y < 0.0f) ? p_1.y : 1.0f));
                float2 inner_1 = fma_f32x2_rn_noftz(bng_1, mul_f32x2_noftz(d1_1, q_1), mul_f32x2_noftz(mul_f32x2_noftz(e_1, make_float2(4.0f, 4.0f)), d2_1));
                float2 _f32x2_mul_a_79 = mul_f32x2_noftz(mul_f32x2_noftz(m1_1, lnu_1), mul_f32x2_noftz(dw_2, inner_1));
                float2 dg_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&post_1));
                float2 _f32x2_mul_a_80 = mul_f32x2_noftz(mul_f32x2_noftz(m1_1, bng_1), mul_f32x2_noftz(mul_f32x2_noftz(w_1, make_float2(4.0f, 4.0f)), d12_1));
                float2 du_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&post_1));
                float _rcp_81 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_1, d1_1, d1_1), dw_2).x);
                float _rcp_82 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_1, d1_1, d1_1), dw_2).y);
                float2 r_7 = make_float2(_rcp_81, _rcp_82);
                float2 rh_1 = r_7;
                float2 near_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_9) : "l"(*(const unsigned long long*)&ax_4), "l"(*(const unsigned long long*)&d1_1));
                float2 far_9 = fma_f32x2_rn_noftz(e_1, make_float2(nb, nb), make_float2(ub, ub));
                float _min_83;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_83) : "f"(near_9.x), "f"(far_9.x));
                float _min_84;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_84) : "f"(near_9.y), "f"(far_9.y));
                float2 hb_1 = make_float2(_min_83, _min_84);
                float2 near_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_10) : "l"(*(const unsigned long long*)&av_4), "l"(*(const unsigned long long*)&dw_2));
                float2 far_10 = fma_f32x2_rn_noftz(w_1, make_float2(nl, nl), make_float2(ul, ul));
                float _min_85;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_85) : "f"(near_10.x), "f"(far_10.x));
                float _min_86;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_86) : "f"(near_10.y), "f"(far_10.y));
                float2 hl_1 = make_float2(_min_85, _min_86);
                float2 _f32x2_mul_a_87 = mul_f32x2_noftz(mul_f32x2_noftz(hb_1, hl_1), rh_1);
                float2 habs_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&habs_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&post_1));
                gw[2] = __as_u32(__float22bfloat162_rn(dg_1));
                uw[2] = __as_u32(__float22bfloat162_rn(du_1));
                dw[2] = __as_u32(__float22bfloat162_rn(habs_1)) | sign_2;
                float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[3]));
                float2 v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[3]));
                float2 d_3 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[3]));
                unsigned int sign_3 = (gw[3] ^ uw[3]) & 2147516416u;
                float _abs_88;
                asm("abs.f32 %0, %1;" : "=f"(_abs_88) : "f"(x_3.x));
                float _abs_89;
                asm("abs.f32 %0, %1;" : "=f"(_abs_89) : "f"(x_3.y));
                float2 ax_5 = make_float2(_abs_88, _abs_89);
                float _abs_90;
                asm("abs.f32 %0, %1;" : "=f"(_abs_90) : "f"(v_3.x));
                float _abs_91;
                asm("abs.f32 %0, %1;" : "=f"(_abs_91) : "f"(v_3.y));
                float2 av_5 = make_float2(_abs_90, _abs_91);
                float2 _f32x2_mul_b_92 = make_float2(k1, k1);
                float2 te_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_2) : "l"(*(const unsigned long long*)&ax_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_92));
                float2 _f32x2_mul_b_93 = make_float2(k2, k2);
                float2 tw_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_2) : "l"(*(const unsigned long long*)&av_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_93));
                float _exp2_94 = approx_exp2(te_2.x);
                float _exp2_95 = approx_exp2(te_2.y);
                float2 e_2 = make_float2(_exp2_94, _exp2_95);
                float _exp2_96 = approx_exp2(tw_2.x);
                float _exp2_97 = approx_exp2(tw_2.y);
                float2 w_2 = make_float2(_exp2_96, _exp2_97);
                float2 p_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_2) : "l"(*(const unsigned long long*)&e_2), "l"(*(const unsigned long long*)&e_2));
                float2 d1_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                float2 d2_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_2);
                float2 dw_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                float2 ng_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                float2 nu_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                float _abs_98;
                asm("abs.f32 %0, %1;" : "=f"(_abs_98) : "f"(x_3.x));
                float _abs_99;
                asm("abs.f32 %0, %1;" : "=f"(_abs_99) : "f"(x_3.y));
                float2 ax_6 = make_float2(_abs_98, _abs_99);
                float _abs_100;
                asm("abs.f32 %0, %1;" : "=f"(_abs_100) : "f"(v_3.x));
                float _abs_101;
                asm("abs.f32 %0, %1;" : "=f"(_abs_101) : "f"(v_3.y));
                float2 av_6 = make_float2(_abs_100, _abs_101);
                float2 d12_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_2) : "l"(*(const unsigned long long*)&d1_2), "l"(*(const unsigned long long*)&d2_2));
                float both_2 = approx_rcp(mul_f32x2_noftz(d12_2, dw_3).x * mul_f32x2_noftz(d12_2, dw_3).y);
                float2 _f32x2_mul_a_102 = make_float2(mul_f32x2_noftz(d12_2, dw_3).y, mul_f32x2_noftz(d12_2, dw_3).x);
                float2 _f32x2_mul_b_103 = make_float2(both_2, both_2);
                float2 r_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&_f32x2_mul_b_103));
                float2 r_8 = r_9;
                float2 _f32x2_mul_b_104 = mul_f32x2_noftz(r_8, r_8);
                float2 m1_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_2) : "l"(*(const unsigned long long*)&d_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_104));
                float2 _f32x2_mul_b_105 = fma_f32x2_rn_noftz(ax_6, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_11) : "l"(*(const unsigned long long*)&ax_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_105));
                float2 far_11 = fma_f32x2_rn_noftz(e_2, make_float2(nb, nb), make_float2(sdb, sdb));
                float2 sbng_2 = make_float2(max_noftz(near_11.x, far_11.x), max_noftz(near_11.y, far_11.y));
                float2 _f32x2_mul_b_106 = fma_f32x2_rn_noftz(av_6, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_12) : "l"(*(const unsigned long long*)&av_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
                float2 far_12 = fma_f32x2_rn_noftz(w_2, make_float2(nl, nl), make_float2(sdl, sdl));
                float2 slnu_2 = make_float2(max_noftz(near_12.x, far_12.x), max_noftz(near_12.y, far_12.y));
                unsigned int lo_5 = __as_u32(sbng_2.x) | __as_u32(x_3.x) & 2147483648u;
                unsigned int hi_5 = __as_u32(sbng_2.y) | __as_u32(x_3.y) & 2147483648u;
                float2 bng_2 = make_float2(__uint_as_float(lo_5), __uint_as_float(hi_5));
                unsigned int lo_6 = __as_u32(slnu_2.x) | __as_u32(v_3.x) & 2147483648u;
                unsigned int hi_6 = __as_u32(slnu_2.y) | __as_u32(v_3.y) & 2147483648u;
                float2 lnu_2 = make_float2(__uint_as_float(lo_6), __uint_as_float(hi_6));
                float2 q_2 = make_float2(((x_3.x < 0.0f) ? 1.0f : p_2.x), ((x_3.y < 0.0f) ? 1.0f : p_2.y));
                float2 post_2 = make_float2(((x_3.x < 0.0f) ? p_2.x : 1.0f), ((x_3.y < 0.0f) ? p_2.y : 1.0f));
                float2 inner_2 = fma_f32x2_rn_noftz(bng_2, mul_f32x2_noftz(d1_2, q_2), mul_f32x2_noftz(mul_f32x2_noftz(e_2, make_float2(4.0f, 4.0f)), d2_2));
                float2 _f32x2_mul_a_107 = mul_f32x2_noftz(mul_f32x2_noftz(m1_2, lnu_2), mul_f32x2_noftz(dw_3, inner_2));
                float2 dg_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&post_2));
                float2 _f32x2_mul_a_108 = mul_f32x2_noftz(mul_f32x2_noftz(m1_2, bng_2), mul_f32x2_noftz(mul_f32x2_noftz(w_2, make_float2(4.0f, 4.0f)), d12_2));
                float2 du_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&post_2));
                float _rcp_109 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_2, d1_2, d1_2), dw_3).x);
                float _rcp_110 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_2, d1_2, d1_2), dw_3).y);
                float2 r_10 = make_float2(_rcp_109, _rcp_110);
                float2 rh_2 = r_10;
                float2 near_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_13) : "l"(*(const unsigned long long*)&ax_6), "l"(*(const unsigned long long*)&d1_2));
                float2 far_13 = fma_f32x2_rn_noftz(e_2, make_float2(nb, nb), make_float2(ub, ub));
                float _min_111;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_111) : "f"(near_13.x), "f"(far_13.x));
                float _min_112;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_112) : "f"(near_13.y), "f"(far_13.y));
                float2 hb_2 = make_float2(_min_111, _min_112);
                float2 near_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_14) : "l"(*(const unsigned long long*)&av_6), "l"(*(const unsigned long long*)&dw_3));
                float2 far_14 = fma_f32x2_rn_noftz(w_2, make_float2(nl, nl), make_float2(ul, ul));
                float _min_113;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_113) : "f"(near_14.x), "f"(far_14.x));
                float _min_114;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_114) : "f"(near_14.y), "f"(far_14.y));
                float2 hl_2 = make_float2(_min_113, _min_114);
                float2 _f32x2_mul_a_115 = mul_f32x2_noftz(mul_f32x2_noftz(hb_2, hl_2), rh_2);
                float2 habs_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&habs_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&post_2));
                gw[3] = __as_u32(__float22bfloat162_rn(dg_2));
                uw[3] = __as_u32(__float22bfloat162_rn(du_2));
                dw[3] = __as_u32(__float22bfloat162_rn(habs_2)) | sign_3;
            } else {
                float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[0]));
                float2 v_4 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[0]));
                float2 d_4 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[0]));
                unsigned int sign_4 = (gw[0] ^ uw[0]) & 2147516416u;
                float _abs_116;
                asm("abs.f32 %0, %1;" : "=f"(_abs_116) : "f"(x_4.x));
                float _abs_117;
                asm("abs.f32 %0, %1;" : "=f"(_abs_117) : "f"(x_4.y));
                float2 ax_7 = make_float2(_abs_116, _abs_117);
                float _abs_118;
                asm("abs.f32 %0, %1;" : "=f"(_abs_118) : "f"(v_4.x));
                float _abs_119;
                asm("abs.f32 %0, %1;" : "=f"(_abs_119) : "f"(v_4.y));
                float2 av_7 = make_float2(_abs_118, _abs_119);
                float2 _f32x2_mul_b_120 = make_float2(k1, k1);
                float2 te_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_3) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_120));
                float2 _f32x2_mul_b_121 = make_float2(k2, k2);
                float2 tw_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_3) : "l"(*(const unsigned long long*)&av_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_121));
                float _exp2_122 = approx_exp2(te_3.x);
                float _exp2_123 = approx_exp2(te_3.y);
                float2 e_3 = make_float2(_exp2_122, _exp2_123);
                float _exp2_124 = approx_exp2(tw_3.x);
                float _exp2_125 = approx_exp2(tw_3.y);
                float2 w_3 = make_float2(_exp2_124, _exp2_125);
                float2 _f32x2_mul_b_126 = make_float2(-1.442695f, -1.442695f);
                float2 tp;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_126));
                float _exp2_127 = approx_exp2_noftz(tp.x);
                float _exp2_128 = approx_exp2_noftz(tp.y);
                float2 p_3 = make_float2(_exp2_127, _exp2_128);
                float2 d1_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                float2 d2_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_3);
                float2 dw_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                float2 ng_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                float2 nu_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                float _abs_129;
                asm("abs.f32 %0, %1;" : "=f"(_abs_129) : "f"(x_4.x));
                float _abs_130;
                asm("abs.f32 %0, %1;" : "=f"(_abs_130) : "f"(x_4.y));
                float2 ax_7_1 = make_float2(_abs_129, _abs_130);
                float _abs_131;
                asm("abs.f32 %0, %1;" : "=f"(_abs_131) : "f"(v_4.x));
                float _abs_132;
                asm("abs.f32 %0, %1;" : "=f"(_abs_132) : "f"(v_4.y));
                float2 av_7_1 = make_float2(_abs_131, _abs_132);
                float2 d12_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_3) : "l"(*(const unsigned long long*)&d1_3), "l"(*(const unsigned long long*)&d2_3));
                float both_3 = approx_rcp(mul_f32x2_noftz(d12_3, dw_4).x * mul_f32x2_noftz(d12_3, dw_4).y);
                float2 _f32x2_mul_a_133 = make_float2(mul_f32x2_noftz(d12_3, dw_4).y, mul_f32x2_noftz(d12_3, dw_4).x);
                float2 _f32x2_mul_b_134 = make_float2(both_3, both_3);
                float2 r_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&_f32x2_mul_b_134));
                float2 r_12 = r_11;
                float2 _f32x2_mul_b_135 = mul_f32x2_noftz(r_12, r_12);
                float2 m1_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_3) : "l"(*(const unsigned long long*)&d_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_135));
                float2 _f32x2_mul_b_136 = fma_f32x2_rn_noftz(ax_7_1, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_15) : "l"(*(const unsigned long long*)&ax_7_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_136));
                float2 far_15 = fma_f32x2_rn_noftz(e_3, make_float2(nb, nb), make_float2(sdb, sdb));
                float2 sbng_3 = make_float2(max_noftz(near_15.x, far_15.x), max_noftz(near_15.y, far_15.y));
                float2 _f32x2_mul_b_137 = fma_f32x2_rn_noftz(av_7_1, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_15_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_15_1) : "l"(*(const unsigned long long*)&av_7_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_137));
                float2 far_15_1 = fma_f32x2_rn_noftz(w_3, make_float2(nl, nl), make_float2(sdl, sdl));
                float2 slnu_3 = make_float2(max_noftz(near_15_1.x, far_15_1.x), max_noftz(near_15_1.y, far_15_1.y));
                unsigned int lo_7 = __as_u32(sbng_3.x) | __as_u32(x_4.x) & 2147483648u;
                unsigned int hi_7 = __as_u32(sbng_3.y) | __as_u32(x_4.y) & 2147483648u;
                float2 bng_3 = make_float2(__uint_as_float(lo_7), __uint_as_float(hi_7));
                unsigned int lo_7_1 = __as_u32(slnu_3.x) | __as_u32(v_4.x) & 2147483648u;
                unsigned int hi_7_1 = __as_u32(slnu_3.y) | __as_u32(v_4.y) & 2147483648u;
                float2 lnu_3 = make_float2(__uint_as_float(lo_7_1), __uint_as_float(hi_7_1));
                float2 q_3 = make_float2(((x_4.x < 0.0f) ? 1.0f : p_3.x), ((x_4.y < 0.0f) ? 1.0f : p_3.y));
                float2 post_3 = make_float2(((x_4.x < 0.0f) ? p_3.x : 1.0f), ((x_4.y < 0.0f) ? p_3.y : 1.0f));
                float2 inner_3 = fma_f32x2_rn_noftz(bng_3, mul_f32x2_noftz(d1_3, q_3), mul_f32x2_noftz(mul_f32x2_noftz(e_3, make_float2(4.0f, 4.0f)), d2_3));
                float2 _f32x2_mul_a_138 = mul_f32x2_noftz(mul_f32x2_noftz(m1_3, lnu_3), mul_f32x2_noftz(dw_4, inner_3));
                float2 dg_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&post_3));
                float2 _f32x2_mul_a_139 = mul_f32x2_noftz(mul_f32x2_noftz(m1_3, bng_3), mul_f32x2_noftz(mul_f32x2_noftz(w_3, make_float2(4.0f, 4.0f)), d12_3));
                float2 du_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&post_3));
                float _rcp_140 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_3, d1_3, d1_3), dw_4).x);
                float _rcp_141 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_3, d1_3, d1_3), dw_4).y);
                float2 r_12_1 = make_float2(_rcp_140, _rcp_141);
                float2 rh_3 = r_12_1;
                float2 near_16;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_16) : "l"(*(const unsigned long long*)&ax_7_1), "l"(*(const unsigned long long*)&d1_3));
                float2 far_16 = fma_f32x2_rn_noftz(e_3, make_float2(nb, nb), make_float2(ub, ub));
                float _min_142;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_142) : "f"(near_16.x), "f"(far_16.x));
                float _min_143;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_143) : "f"(near_16.y), "f"(far_16.y));
                float2 hb_3 = make_float2(_min_142, _min_143);
                float2 near_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_17) : "l"(*(const unsigned long long*)&av_7_1), "l"(*(const unsigned long long*)&dw_4));
                float2 far_17 = fma_f32x2_rn_noftz(w_3, make_float2(nl, nl), make_float2(ul, ul));
                float _min_144;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_144) : "f"(near_17.x), "f"(far_17.x));
                float _min_145;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_145) : "f"(near_17.y), "f"(far_17.y));
                float2 hl_3 = make_float2(_min_144, _min_145);
                float2 _f32x2_mul_a_146 = mul_f32x2_noftz(mul_f32x2_noftz(hb_3, hl_3), rh_3);
                float2 habs_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&habs_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_146), "l"(*(const unsigned long long*)&post_3));
                gw[0] = __as_u32(__float22bfloat162_rn(dg_3));
                uw[0] = __as_u32(__float22bfloat162_rn(du_3));
                dw[0] = __as_u32(__float22bfloat162_rn(habs_3)) | sign_4;
                float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[1]));
                float2 v_5 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[1]));
                float2 d_5 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[1]));
                unsigned int sign_5 = (gw[1] ^ uw[1]) & 2147516416u;
                float _abs_147;
                asm("abs.f32 %0, %1;" : "=f"(_abs_147) : "f"(x_5.x));
                float _abs_148;
                asm("abs.f32 %0, %1;" : "=f"(_abs_148) : "f"(x_5.y));
                float2 ax_8 = make_float2(_abs_147, _abs_148);
                float _abs_149;
                asm("abs.f32 %0, %1;" : "=f"(_abs_149) : "f"(v_5.x));
                float _abs_150;
                asm("abs.f32 %0, %1;" : "=f"(_abs_150) : "f"(v_5.y));
                float2 av_8 = make_float2(_abs_149, _abs_150);
                float2 _f32x2_mul_b_151 = make_float2(k1, k1);
                float2 te_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_3_1) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_151));
                float2 _f32x2_mul_b_152 = make_float2(k2, k2);
                float2 tw_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_3_1) : "l"(*(const unsigned long long*)&av_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_152));
                float _exp2_153 = approx_exp2(te_3_1.x);
                float _exp2_154 = approx_exp2(te_3_1.y);
                float2 e_3_1 = make_float2(_exp2_153, _exp2_154);
                float _exp2_155 = approx_exp2(tw_3_1.x);
                float _exp2_156 = approx_exp2(tw_3_1.y);
                float2 w_3_1 = make_float2(_exp2_155, _exp2_156);
                float2 _f32x2_mul_b_157 = make_float2(-1.442695f, -1.442695f);
                float2 tp_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_0) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_157));
                float _exp2_158 = approx_exp2_noftz(tp_0.x);
                float _exp2_159 = approx_exp2_noftz(tp_0.y);
                float2 p_3_1 = make_float2(_exp2_158, _exp2_159);
                float2 d1_3_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_3_1);
                float2 d2_3_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_3_1);
                float2 dw_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_3_1);
                float2 ng_3_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_3_1);
                float2 nu_3_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_3_1);
                float _abs_160;
                asm("abs.f32 %0, %1;" : "=f"(_abs_160) : "f"(x_5.x));
                float _abs_161;
                asm("abs.f32 %0, %1;" : "=f"(_abs_161) : "f"(x_5.y));
                float2 ax_9 = make_float2(_abs_160, _abs_161);
                float _abs_162;
                asm("abs.f32 %0, %1;" : "=f"(_abs_162) : "f"(v_5.x));
                float _abs_163;
                asm("abs.f32 %0, %1;" : "=f"(_abs_163) : "f"(v_5.y));
                float2 av_9 = make_float2(_abs_162, _abs_163);
                float2 d12_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_3_1) : "l"(*(const unsigned long long*)&d1_3_1), "l"(*(const unsigned long long*)&d2_3_1));
                float both_3_1 = approx_rcp(mul_f32x2_noftz(d12_3_1, dw_5).x * mul_f32x2_noftz(d12_3_1, dw_5).y);
                float2 _f32x2_mul_a_164 = make_float2(mul_f32x2_noftz(d12_3_1, dw_5).y, mul_f32x2_noftz(d12_3_1, dw_5).x);
                float2 _f32x2_mul_b_165 = make_float2(both_3_1, both_3_1);
                float2 r_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
                float2 r_13 = r_14;
                float2 _f32x2_mul_b_166 = mul_f32x2_noftz(r_13, r_13);
                float2 m1_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_3_1) : "l"(*(const unsigned long long*)&d_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_166));
                float2 _f32x2_mul_b_167 = fma_f32x2_rn_noftz(ax_9, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_18;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_18) : "l"(*(const unsigned long long*)&ax_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_167));
                float2 far_18 = fma_f32x2_rn_noftz(e_3_1, make_float2(nb, nb), make_float2(sdb, sdb));
                float2 sbng_3_1 = make_float2(max_noftz(near_18.x, far_18.x), max_noftz(near_18.y, far_18.y));
                float2 _f32x2_mul_b_168 = fma_f32x2_rn_noftz(av_9, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_19;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_19) : "l"(*(const unsigned long long*)&av_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_168));
                float2 far_19 = fma_f32x2_rn_noftz(w_3_1, make_float2(nl, nl), make_float2(sdl, sdl));
                float2 slnu_3_1 = make_float2(max_noftz(near_19.x, far_19.x), max_noftz(near_19.y, far_19.y));
                unsigned int lo_8 = __as_u32(sbng_3_1.x) | __as_u32(x_5.x) & 2147483648u;
                unsigned int hi_8 = __as_u32(sbng_3_1.y) | __as_u32(x_5.y) & 2147483648u;
                float2 bng_3_1 = make_float2(__uint_as_float(lo_8), __uint_as_float(hi_8));
                unsigned int lo_9 = __as_u32(slnu_3_1.x) | __as_u32(v_5.x) & 2147483648u;
                unsigned int hi_9 = __as_u32(slnu_3_1.y) | __as_u32(v_5.y) & 2147483648u;
                float2 lnu_3_1 = make_float2(__uint_as_float(lo_9), __uint_as_float(hi_9));
                float2 q_3_1 = make_float2(((x_5.x < 0.0f) ? 1.0f : p_3_1.x), ((x_5.y < 0.0f) ? 1.0f : p_3_1.y));
                float2 post_3_1 = make_float2(((x_5.x < 0.0f) ? p_3_1.x : 1.0f), ((x_5.y < 0.0f) ? p_3_1.y : 1.0f));
                float2 inner_3_1 = fma_f32x2_rn_noftz(bng_3_1, mul_f32x2_noftz(d1_3_1, q_3_1), mul_f32x2_noftz(mul_f32x2_noftz(e_3_1, make_float2(4.0f, 4.0f)), d2_3_1));
                float2 _f32x2_mul_a_169 = mul_f32x2_noftz(mul_f32x2_noftz(m1_3_1, lnu_3_1), mul_f32x2_noftz(dw_5, inner_3_1));
                float2 dg_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&post_3_1));
                float2 _f32x2_mul_a_170 = mul_f32x2_noftz(mul_f32x2_noftz(m1_3_1, bng_3_1), mul_f32x2_noftz(mul_f32x2_noftz(w_3_1, make_float2(4.0f, 4.0f)), d12_3_1));
                float2 du_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_170), "l"(*(const unsigned long long*)&post_3_1));
                float _rcp_171 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_3_1, d1_3_1, d1_3_1), dw_5).x);
                float _rcp_172 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_3_1, d1_3_1, d1_3_1), dw_5).y);
                float2 r_15 = make_float2(_rcp_171, _rcp_172);
                float2 rh_3_1 = r_15;
                float2 near_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_20) : "l"(*(const unsigned long long*)&ax_9), "l"(*(const unsigned long long*)&d1_3_1));
                float2 far_20 = fma_f32x2_rn_noftz(e_3_1, make_float2(nb, nb), make_float2(ub, ub));
                float _min_173;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_173) : "f"(near_20.x), "f"(far_20.x));
                float _min_174;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_174) : "f"(near_20.y), "f"(far_20.y));
                float2 hb_3_1 = make_float2(_min_173, _min_174);
                float2 near_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_21) : "l"(*(const unsigned long long*)&av_9), "l"(*(const unsigned long long*)&dw_5));
                float2 far_21 = fma_f32x2_rn_noftz(w_3_1, make_float2(nl, nl), make_float2(ul, ul));
                float _min_175;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_175) : "f"(near_21.x), "f"(far_21.x));
                float _min_176;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_176) : "f"(near_21.y), "f"(far_21.y));
                float2 hl_3_1 = make_float2(_min_175, _min_176);
                float2 _f32x2_mul_a_177 = mul_f32x2_noftz(mul_f32x2_noftz(hb_3_1, hl_3_1), rh_3_1);
                float2 habs_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&habs_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_177), "l"(*(const unsigned long long*)&post_3_1));
                gw[1] = __as_u32(__float22bfloat162_rn(dg_3_1));
                uw[1] = __as_u32(__float22bfloat162_rn(du_3_1));
                dw[1] = __as_u32(__float22bfloat162_rn(habs_3_1)) | sign_5;
                float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[2]));
                float2 v_6 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[2]));
                float2 d_6 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[2]));
                unsigned int sign_6 = (gw[2] ^ uw[2]) & 2147516416u;
                float _abs_178;
                asm("abs.f32 %0, %1;" : "=f"(_abs_178) : "f"(x_6.x));
                float _abs_179;
                asm("abs.f32 %0, %1;" : "=f"(_abs_179) : "f"(x_6.y));
                float2 ax_10 = make_float2(_abs_178, _abs_179);
                float _abs_180;
                asm("abs.f32 %0, %1;" : "=f"(_abs_180) : "f"(v_6.x));
                float _abs_181;
                asm("abs.f32 %0, %1;" : "=f"(_abs_181) : "f"(v_6.y));
                float2 av_10 = make_float2(_abs_180, _abs_181);
                float2 _f32x2_mul_b_182 = make_float2(k1, k1);
                float2 te_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_4) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_182));
                float2 _f32x2_mul_b_183 = make_float2(k2, k2);
                float2 tw_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_4) : "l"(*(const unsigned long long*)&av_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_183));
                float _exp2_184 = approx_exp2(te_4.x);
                float _exp2_185 = approx_exp2(te_4.y);
                float2 e_4 = make_float2(_exp2_184, _exp2_185);
                float _exp2_186 = approx_exp2(tw_4.x);
                float _exp2_187 = approx_exp2(tw_4.y);
                float2 w_4 = make_float2(_exp2_186, _exp2_187);
                float2 _f32x2_mul_b_188 = make_float2(-1.442695f, -1.442695f);
                float2 tp_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_1) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_188));
                float _exp2_189 = approx_exp2_noftz(tp_1.x);
                float _exp2_190 = approx_exp2_noftz(tp_1.y);
                float2 p_4 = make_float2(_exp2_189, _exp2_190);
                float2 d1_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                float2 d2_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_4);
                float2 dw_6 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                float2 ng_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                float2 nu_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                float _abs_191;
                asm("abs.f32 %0, %1;" : "=f"(_abs_191) : "f"(x_6.x));
                float _abs_192;
                asm("abs.f32 %0, %1;" : "=f"(_abs_192) : "f"(x_6.y));
                float2 ax_11 = make_float2(_abs_191, _abs_192);
                float _abs_193;
                asm("abs.f32 %0, %1;" : "=f"(_abs_193) : "f"(v_6.x));
                float _abs_194;
                asm("abs.f32 %0, %1;" : "=f"(_abs_194) : "f"(v_6.y));
                float2 av_11 = make_float2(_abs_193, _abs_194);
                float2 d12_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_4) : "l"(*(const unsigned long long*)&d1_4), "l"(*(const unsigned long long*)&d2_4));
                float both_4 = approx_rcp(mul_f32x2_noftz(d12_4, dw_6).x * mul_f32x2_noftz(d12_4, dw_6).y);
                float2 _f32x2_mul_a_195 = make_float2(mul_f32x2_noftz(d12_4, dw_6).y, mul_f32x2_noftz(d12_4, dw_6).x);
                float2 _f32x2_mul_b_196 = make_float2(both_4, both_4);
                float2 r_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_195), "l"(*(const unsigned long long*)&_f32x2_mul_b_196));
                float2 r_16 = r_17;
                float2 _f32x2_mul_b_197 = mul_f32x2_noftz(r_16, r_16);
                float2 m1_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_4) : "l"(*(const unsigned long long*)&d_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_197));
                float2 _f32x2_mul_b_198 = fma_f32x2_rn_noftz(ax_11, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_22) : "l"(*(const unsigned long long*)&ax_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_198));
                float2 far_22 = fma_f32x2_rn_noftz(e_4, make_float2(nb, nb), make_float2(sdb, sdb));
                float2 sbng_4 = make_float2(max_noftz(near_22.x, far_22.x), max_noftz(near_22.y, far_22.y));
                float2 _f32x2_mul_b_199 = fma_f32x2_rn_noftz(av_11, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_23) : "l"(*(const unsigned long long*)&av_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_199));
                float2 far_23 = fma_f32x2_rn_noftz(w_4, make_float2(nl, nl), make_float2(sdl, sdl));
                float2 slnu_4 = make_float2(max_noftz(near_23.x, far_23.x), max_noftz(near_23.y, far_23.y));
                unsigned int lo_10 = __as_u32(sbng_4.x) | __as_u32(x_6.x) & 2147483648u;
                unsigned int hi_10 = __as_u32(sbng_4.y) | __as_u32(x_6.y) & 2147483648u;
                float2 bng_4 = make_float2(__uint_as_float(lo_10), __uint_as_float(hi_10));
                unsigned int lo_11 = __as_u32(slnu_4.x) | __as_u32(v_6.x) & 2147483648u;
                unsigned int hi_11 = __as_u32(slnu_4.y) | __as_u32(v_6.y) & 2147483648u;
                float2 lnu_4 = make_float2(__uint_as_float(lo_11), __uint_as_float(hi_11));
                float2 q_4 = make_float2(((x_6.x < 0.0f) ? 1.0f : p_4.x), ((x_6.y < 0.0f) ? 1.0f : p_4.y));
                float2 post_4 = make_float2(((x_6.x < 0.0f) ? p_4.x : 1.0f), ((x_6.y < 0.0f) ? p_4.y : 1.0f));
                float2 inner_4 = fma_f32x2_rn_noftz(bng_4, mul_f32x2_noftz(d1_4, q_4), mul_f32x2_noftz(mul_f32x2_noftz(e_4, make_float2(4.0f, 4.0f)), d2_4));
                float2 _f32x2_mul_a_200 = mul_f32x2_noftz(mul_f32x2_noftz(m1_4, lnu_4), mul_f32x2_noftz(dw_6, inner_4));
                float2 dg_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_200), "l"(*(const unsigned long long*)&post_4));
                float2 _f32x2_mul_a_201 = mul_f32x2_noftz(mul_f32x2_noftz(m1_4, bng_4), mul_f32x2_noftz(mul_f32x2_noftz(w_4, make_float2(4.0f, 4.0f)), d12_4));
                float2 du_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_201), "l"(*(const unsigned long long*)&post_4));
                float _rcp_202 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_4, d1_4, d1_4), dw_6).x);
                float _rcp_203 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_4, d1_4, d1_4), dw_6).y);
                float2 r_18 = make_float2(_rcp_202, _rcp_203);
                float2 rh_4 = r_18;
                float2 near_24;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_24) : "l"(*(const unsigned long long*)&ax_11), "l"(*(const unsigned long long*)&d1_4));
                float2 far_24 = fma_f32x2_rn_noftz(e_4, make_float2(nb, nb), make_float2(ub, ub));
                float _min_204;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_204) : "f"(near_24.x), "f"(far_24.x));
                float _min_205;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_205) : "f"(near_24.y), "f"(far_24.y));
                float2 hb_4 = make_float2(_min_204, _min_205);
                float2 near_25;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_25) : "l"(*(const unsigned long long*)&av_11), "l"(*(const unsigned long long*)&dw_6));
                float2 far_25 = fma_f32x2_rn_noftz(w_4, make_float2(nl, nl), make_float2(ul, ul));
                float _min_206;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_206) : "f"(near_25.x), "f"(far_25.x));
                float _min_207;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_207) : "f"(near_25.y), "f"(far_25.y));
                float2 hl_4 = make_float2(_min_206, _min_207);
                float2 _f32x2_mul_a_208 = mul_f32x2_noftz(mul_f32x2_noftz(hb_4, hl_4), rh_4);
                float2 habs_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&habs_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_208), "l"(*(const unsigned long long*)&post_4));
                gw[2] = __as_u32(__float22bfloat162_rn(dg_4));
                uw[2] = __as_u32(__float22bfloat162_rn(du_4));
                dw[2] = __as_u32(__float22bfloat162_rn(habs_4)) | sign_6;
                float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[3]));
                float2 v_7 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[3]));
                float2 d_7 = cake_bf16x2_to_f32x2(__as_bf16x2(dw[3]));
                unsigned int sign_7 = (gw[3] ^ uw[3]) & 2147516416u;
                float _abs_209;
                asm("abs.f32 %0, %1;" : "=f"(_abs_209) : "f"(x_7.x));
                float _abs_210;
                asm("abs.f32 %0, %1;" : "=f"(_abs_210) : "f"(x_7.y));
                float2 ax_12 = make_float2(_abs_209, _abs_210);
                float _abs_211;
                asm("abs.f32 %0, %1;" : "=f"(_abs_211) : "f"(v_7.x));
                float _abs_212;
                asm("abs.f32 %0, %1;" : "=f"(_abs_212) : "f"(v_7.y));
                float2 av_12 = make_float2(_abs_211, _abs_212);
                float2 _f32x2_mul_b_213 = make_float2(k1, k1);
                float2 te_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_5) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_213));
                float2 _f32x2_mul_b_214 = make_float2(k2, k2);
                float2 tw_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_5) : "l"(*(const unsigned long long*)&av_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_214));
                float _exp2_215 = approx_exp2(te_5.x);
                float _exp2_216 = approx_exp2(te_5.y);
                float2 e_5 = make_float2(_exp2_215, _exp2_216);
                float _exp2_217 = approx_exp2(tw_5.x);
                float _exp2_218 = approx_exp2(tw_5.y);
                float2 w_5 = make_float2(_exp2_217, _exp2_218);
                float2 _f32x2_mul_b_219 = make_float2(-1.442695f, -1.442695f);
                float2 tp_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_2) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_219));
                float _exp2_220 = approx_exp2_noftz(tp_2.x);
                float _exp2_221 = approx_exp2_noftz(tp_2.y);
                float2 p_5 = make_float2(_exp2_220, _exp2_221);
                float2 d1_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                float2 d2_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_5);
                float2 dw_7 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                float2 ng_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                float2 nu_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                float _abs_222;
                asm("abs.f32 %0, %1;" : "=f"(_abs_222) : "f"(x_7.x));
                float _abs_223;
                asm("abs.f32 %0, %1;" : "=f"(_abs_223) : "f"(x_7.y));
                float2 ax_13 = make_float2(_abs_222, _abs_223);
                float _abs_224;
                asm("abs.f32 %0, %1;" : "=f"(_abs_224) : "f"(v_7.x));
                float _abs_225;
                asm("abs.f32 %0, %1;" : "=f"(_abs_225) : "f"(v_7.y));
                float2 av_13 = make_float2(_abs_224, _abs_225);
                float2 d12_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d12_5) : "l"(*(const unsigned long long*)&d1_5), "l"(*(const unsigned long long*)&d2_5));
                float both_5 = approx_rcp(mul_f32x2_noftz(d12_5, dw_7).x * mul_f32x2_noftz(d12_5, dw_7).y);
                float2 _f32x2_mul_a_226 = make_float2(mul_f32x2_noftz(d12_5, dw_7).y, mul_f32x2_noftz(d12_5, dw_7).x);
                float2 _f32x2_mul_b_227 = make_float2(both_5, both_5);
                float2 r_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_226), "l"(*(const unsigned long long*)&_f32x2_mul_b_227));
                float2 r_19 = r_20;
                float2 _f32x2_mul_b_228 = mul_f32x2_noftz(r_19, r_19);
                float2 m1_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m1_5) : "l"(*(const unsigned long long*)&d_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_228));
                float2 _f32x2_mul_b_229 = fma_f32x2_rn_noftz(ax_13, make_float2(cg, cg), make_float2(2.0f, 2.0f));
                float2 near_26;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_26) : "l"(*(const unsigned long long*)&ax_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_229));
                float2 far_26 = fma_f32x2_rn_noftz(e_5, make_float2(nb, nb), make_float2(sdb, sdb));
                float2 sbng_5 = make_float2(max_noftz(near_26.x, far_26.x), max_noftz(near_26.y, far_26.y));
                float2 _f32x2_mul_b_230 = fma_f32x2_rn_noftz(av_13, make_float2(cu, cu), make_float2(2.0f, 2.0f));
                float2 near_27;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_27) : "l"(*(const unsigned long long*)&av_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_230));
                float2 far_27 = fma_f32x2_rn_noftz(w_5, make_float2(nl, nl), make_float2(sdl, sdl));
                float2 slnu_5 = make_float2(max_noftz(near_27.x, far_27.x), max_noftz(near_27.y, far_27.y));
                unsigned int lo_12 = __as_u32(sbng_5.x) | __as_u32(x_7.x) & 2147483648u;
                unsigned int hi_12 = __as_u32(sbng_5.y) | __as_u32(x_7.y) & 2147483648u;
                float2 bng_5 = make_float2(__uint_as_float(lo_12), __uint_as_float(hi_12));
                unsigned int lo_13 = __as_u32(slnu_5.x) | __as_u32(v_7.x) & 2147483648u;
                unsigned int hi_13 = __as_u32(slnu_5.y) | __as_u32(v_7.y) & 2147483648u;
                float2 lnu_5 = make_float2(__uint_as_float(lo_13), __uint_as_float(hi_13));
                float2 q_5 = make_float2(((x_7.x < 0.0f) ? 1.0f : p_5.x), ((x_7.y < 0.0f) ? 1.0f : p_5.y));
                float2 post_5 = make_float2(((x_7.x < 0.0f) ? p_5.x : 1.0f), ((x_7.y < 0.0f) ? p_5.y : 1.0f));
                float2 inner_5 = fma_f32x2_rn_noftz(bng_5, mul_f32x2_noftz(d1_5, q_5), mul_f32x2_noftz(mul_f32x2_noftz(e_5, make_float2(4.0f, 4.0f)), d2_5));
                float2 _f32x2_mul_a_231 = mul_f32x2_noftz(mul_f32x2_noftz(m1_5, lnu_5), mul_f32x2_noftz(dw_7, inner_5));
                float2 dg_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dg_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_231), "l"(*(const unsigned long long*)&post_5));
                float2 _f32x2_mul_a_232 = mul_f32x2_noftz(mul_f32x2_noftz(m1_5, bng_5), mul_f32x2_noftz(mul_f32x2_noftz(w_5, make_float2(4.0f, 4.0f)), d12_5));
                float2 du_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&du_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_232), "l"(*(const unsigned long long*)&post_5));
                float _rcp_233 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_5, d1_5, d1_5), dw_7).x);
                float _rcp_234 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_5, d1_5, d1_5), dw_7).y);
                float2 r_21 = make_float2(_rcp_233, _rcp_234);
                float2 rh_5 = r_21;
                float2 near_28;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_28) : "l"(*(const unsigned long long*)&ax_13), "l"(*(const unsigned long long*)&d1_5));
                float2 far_28 = fma_f32x2_rn_noftz(e_5, make_float2(nb, nb), make_float2(ub, ub));
                float _min_235;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_235) : "f"(near_28.x), "f"(far_28.x));
                float _min_236;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_236) : "f"(near_28.y), "f"(far_28.y));
                float2 hb_5 = make_float2(_min_235, _min_236);
                float2 near_29;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_29) : "l"(*(const unsigned long long*)&av_13), "l"(*(const unsigned long long*)&dw_7));
                float2 far_29 = fma_f32x2_rn_noftz(w_5, make_float2(nl, nl), make_float2(ul, ul));
                float _min_237;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_237) : "f"(near_29.x), "f"(far_29.x));
                float _min_238;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_238) : "f"(near_29.y), "f"(far_29.y));
                float2 hl_5 = make_float2(_min_237, _min_238);
                float2 _f32x2_mul_a_239 = mul_f32x2_noftz(mul_f32x2_noftz(hb_5, hl_5), rh_5);
                float2 habs_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&habs_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_239), "l"(*(const unsigned long long*)&post_5));
                gw[3] = __as_u32(__float22bfloat162_rn(dg_5));
                uw[3] = __as_u32(__float22bfloat162_rn(du_5));
                dw[3] = __as_u32(__float22bfloat162_rn(habs_5)) | sign_7;
            }
            if (s0 < limit) {
                long long col_1 = (long long)(s0 * 256 + lane_1 * 8);
                reinterpret_cast<int4*>(d_gate + (dgoff + col_1))[0] = reinterpret_cast<int4*>(gw)[0];
                reinterpret_cast<int4*>(d_up + (duoff + col_1))[0] = reinterpret_cast<int4*>(uw)[0];
                reinterpret_cast<int4*>(h + (hoff + col_1))[0] = reinterpret_cast<int4*>(dw)[0];
            }
        }
    }
}

} // extern "C"
