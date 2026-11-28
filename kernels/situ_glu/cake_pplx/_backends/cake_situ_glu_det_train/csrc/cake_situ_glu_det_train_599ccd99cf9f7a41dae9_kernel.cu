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
kernel_cake_situ_glu_det_train_599ccd99cf9f7a41dae9(__nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ up, __nv_bfloat16* __restrict__ h, int rows, int width, long long gate_stride, long long up_stride, long long h_stride, float beta, float lb)
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
    float nb = -beta;
    float nl = -lb;
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
        long long hoff = at * h_stride;
        int nxt = row + num_bids;
        if (nxt < rows) {
            long long ngoff = (long long)nxt * gate_stride;
            long long nuoff = (long long)nxt * up_stride;
            int nlines = (nseg - part + 3 >> 2) * 4;
            #pragma unroll 1
            for (int l0 = 0; l0 < nlines; l0 += 32) {
                int li = l0 + lane_1;
                long long pcol = (long long)((part + (li >> 2) * 4) * 256 + (li & 3) * 64);
                if (li < nlines) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(gate + (ngoff + pcol)))); }
                if (li < nlines) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(up + (nuoff + pcol)))); }
            }
        }
        if (beta == 4.0f) {
            #pragma unroll 1
            for (int s0 = part; s0 < nseg; s0 += 4) {
                unsigned int gw[4];
                unsigned int uw[4];
                int seg_0 = ((s0) < (nseg - 1) ? (s0) : (nseg - 1));
                long long col_0 = (long long)(seg_0 * 256 + lane_1 * 8);
                {
                    uint32_t _uv4_1_0;
                    uint32_t _uv4_1_1;
                    uint32_t _uv4_1_2;
                    uint32_t _uv4_1_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(gate + (goff + col_0) + (0))));
                    gw[0 + 0] = _uv4_1_0;
                    gw[0 + 1] = _uv4_1_1;
                    gw[0 + 2] = _uv4_1_2;
                    gw[0 + 3] = _uv4_1_3;
                }
                {
                    uint32_t _uv4_2_0;
                    uint32_t _uv4_2_1;
                    uint32_t _uv4_2_2;
                    uint32_t _uv4_2_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3) : "l"((const void*)(up + (uoff + col_0) + (0))));
                    uw[0 + 0] = _uv4_2_0;
                    uw[0 + 1] = _uv4_2_1;
                    uw[0 + 2] = _uv4_2_2;
                    uw[0 + 3] = _uv4_2_3;
                }
                float2 x_0 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[0]));
                float2 v_0 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[0]));
                unsigned int sign_0 = (gw[0] ^ uw[0]) & 2147516416u;
                float _abs_3;
                asm("abs.f32 %0, %1;" : "=f"(_abs_3) : "f"(x_0.x));
                float _abs_4;
                asm("abs.f32 %0, %1;" : "=f"(_abs_4) : "f"(x_0.y));
                float2 ax = make_float2(_abs_3, _abs_4);
                float _abs_5;
                asm("abs.f32 %0, %1;" : "=f"(_abs_5) : "f"(v_0.x));
                float _abs_6;
                asm("abs.f32 %0, %1;" : "=f"(_abs_6) : "f"(v_0.y));
                float2 av = make_float2(_abs_5, _abs_6);
                float2 _f32x2_mul_b_7 = make_float2(k1, k1);
                float2 te;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te) : "l"(*(const unsigned long long*)&ax), "l"(*(const unsigned long long*)&_f32x2_mul_b_7));
                float2 _f32x2_mul_b_8 = make_float2(k2, k2);
                float2 tw;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw) : "l"(*(const unsigned long long*)&av), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
                float _exp2_9 = approx_exp2(te.x);
                float _exp2_10 = approx_exp2(te.y);
                float2 e = make_float2(_exp2_9, _exp2_10);
                float _exp2_11 = approx_exp2(tw.x);
                float _exp2_12 = approx_exp2(tw.y);
                float2 w = make_float2(_exp2_11, _exp2_12);
                float2 p;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p) : "l"(*(const unsigned long long*)&e), "l"(*(const unsigned long long*)&e));
                float2 d1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                float2 d2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p);
                float2 dw = add_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                float2 ng = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                float2 nu = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                float _abs_13;
                asm("abs.f32 %0, %1;" : "=f"(_abs_13) : "f"(x_0.x));
                float _abs_14;
                asm("abs.f32 %0, %1;" : "=f"(_abs_14) : "f"(x_0.y));
                float2 ax_0 = make_float2(_abs_13, _abs_14);
                float _abs_15;
                asm("abs.f32 %0, %1;" : "=f"(_abs_15) : "f"(v_0.x));
                float _abs_16;
                asm("abs.f32 %0, %1;" : "=f"(_abs_16) : "f"(v_0.y));
                float2 av_0 = make_float2(_abs_15, _abs_16);
                float _rcp_17 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p, d1, d1), dw).x);
                float _rcp_18 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p, d1, d1), dw).y);
                float2 r_0 = make_float2(_rcp_17, _rcp_18);
                float2 r = r_0;
                float2 near;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near) : "l"(*(const unsigned long long*)&ax_0), "l"(*(const unsigned long long*)&d1));
                float2 far = fma_f32x2_rn_noftz(e, make_float2(nb, nb), make_float2(ub, ub));
                float _min_19;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_19) : "f"(near.x), "f"(far.x));
                float _min_20;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_20) : "f"(near.y), "f"(far.y));
                float2 bng = make_float2(_min_19, _min_20);
                float2 near_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_0) : "l"(*(const unsigned long long*)&av_0), "l"(*(const unsigned long long*)&dw));
                float2 far_0 = fma_f32x2_rn_noftz(w, make_float2(nl, nl), make_float2(ul, ul));
                float _min_21;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_21) : "f"(near_0.x), "f"(far_0.x));
                float _min_22;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_22) : "f"(near_0.y), "f"(far_0.y));
                float2 lnu = make_float2(_min_21, _min_22);
                float2 _f32x2_mul_a_23 = mul_f32x2_noftz(bng, lnu);
                float2 m;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&r));
                float2 ns = make_float2(((x_0.x < 0.0f) ? p.x : 1.0f), ((x_0.y < 0.0f) ? p.y : 1.0f));
                gw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m, ns))) | sign_0;
                float2 x_1 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[1]));
                float2 v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[1]));
                unsigned int sign_1 = (gw[1] ^ uw[1]) & 2147516416u;
                float _abs_24;
                asm("abs.f32 %0, %1;" : "=f"(_abs_24) : "f"(x_1.x));
                float _abs_25;
                asm("abs.f32 %0, %1;" : "=f"(_abs_25) : "f"(x_1.y));
                float2 ax_1 = make_float2(_abs_24, _abs_25);
                float _abs_26;
                asm("abs.f32 %0, %1;" : "=f"(_abs_26) : "f"(v_1.x));
                float _abs_27;
                asm("abs.f32 %0, %1;" : "=f"(_abs_27) : "f"(v_1.y));
                float2 av_1 = make_float2(_abs_26, _abs_27);
                float2 _f32x2_mul_b_28 = make_float2(k1, k1);
                float2 te_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_0) : "l"(*(const unsigned long long*)&ax_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
                float2 _f32x2_mul_b_29 = make_float2(k2, k2);
                float2 tw_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_0) : "l"(*(const unsigned long long*)&av_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
                float _exp2_30 = approx_exp2(te_0.x);
                float _exp2_31 = approx_exp2(te_0.y);
                float2 e_0 = make_float2(_exp2_30, _exp2_31);
                float _exp2_32 = approx_exp2(tw_0.x);
                float _exp2_33 = approx_exp2(tw_0.y);
                float2 w_0 = make_float2(_exp2_32, _exp2_33);
                float2 p_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_0) : "l"(*(const unsigned long long*)&e_0), "l"(*(const unsigned long long*)&e_0));
                float2 d1_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                float2 d2_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_0);
                float2 dw_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                float2 ng_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                float2 nu_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                float _abs_34;
                asm("abs.f32 %0, %1;" : "=f"(_abs_34) : "f"(x_1.x));
                float _abs_35;
                asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(x_1.y));
                float2 ax_2 = make_float2(_abs_34, _abs_35);
                float _abs_36;
                asm("abs.f32 %0, %1;" : "=f"(_abs_36) : "f"(v_1.x));
                float _abs_37;
                asm("abs.f32 %0, %1;" : "=f"(_abs_37) : "f"(v_1.y));
                float2 av_2 = make_float2(_abs_36, _abs_37);
                float _rcp_38 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_0, d1_0, d1_0), dw_0).x);
                float _rcp_39 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_0, d1_0, d1_0), dw_0).y);
                float2 r_2 = make_float2(_rcp_38, _rcp_39);
                float2 r_1 = r_2;
                float2 near_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_1) : "l"(*(const unsigned long long*)&ax_2), "l"(*(const unsigned long long*)&d1_0));
                float2 far_1 = fma_f32x2_rn_noftz(e_0, make_float2(nb, nb), make_float2(ub, ub));
                float _min_40;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_40) : "f"(near_1.x), "f"(far_1.x));
                float _min_41;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_41) : "f"(near_1.y), "f"(far_1.y));
                float2 bng_0 = make_float2(_min_40, _min_41);
                float2 near_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_2) : "l"(*(const unsigned long long*)&av_2), "l"(*(const unsigned long long*)&dw_0));
                float2 far_2 = fma_f32x2_rn_noftz(w_0, make_float2(nl, nl), make_float2(ul, ul));
                float _min_42;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_42) : "f"(near_2.x), "f"(far_2.x));
                float _min_43;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_43) : "f"(near_2.y), "f"(far_2.y));
                float2 lnu_0 = make_float2(_min_42, _min_43);
                float2 _f32x2_mul_a_44 = mul_f32x2_noftz(bng_0, lnu_0);
                float2 m_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&r_1));
                float2 ns_0 = make_float2(((x_1.x < 0.0f) ? p_0.x : 1.0f), ((x_1.y < 0.0f) ? p_0.y : 1.0f));
                gw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_0, ns_0))) | sign_1;
                float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[2]));
                float2 v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[2]));
                unsigned int sign_2 = (gw[2] ^ uw[2]) & 2147516416u;
                float _abs_45;
                asm("abs.f32 %0, %1;" : "=f"(_abs_45) : "f"(x_2.x));
                float _abs_46;
                asm("abs.f32 %0, %1;" : "=f"(_abs_46) : "f"(x_2.y));
                float2 ax_3 = make_float2(_abs_45, _abs_46);
                float _abs_47;
                asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(v_2.x));
                float _abs_48;
                asm("abs.f32 %0, %1;" : "=f"(_abs_48) : "f"(v_2.y));
                float2 av_3 = make_float2(_abs_47, _abs_48);
                float2 _f32x2_mul_b_49 = make_float2(k1, k1);
                float2 te_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_1) : "l"(*(const unsigned long long*)&ax_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
                float2 _f32x2_mul_b_50 = make_float2(k2, k2);
                float2 tw_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_1) : "l"(*(const unsigned long long*)&av_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_50));
                float _exp2_51 = approx_exp2(te_1.x);
                float _exp2_52 = approx_exp2(te_1.y);
                float2 e_1 = make_float2(_exp2_51, _exp2_52);
                float _exp2_53 = approx_exp2(tw_1.x);
                float _exp2_54 = approx_exp2(tw_1.y);
                float2 w_1 = make_float2(_exp2_53, _exp2_54);
                float2 p_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_1) : "l"(*(const unsigned long long*)&e_1), "l"(*(const unsigned long long*)&e_1));
                float2 d1_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                float2 d2_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_1);
                float2 dw_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                float2 ng_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                float2 nu_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                float _abs_55;
                asm("abs.f32 %0, %1;" : "=f"(_abs_55) : "f"(x_2.x));
                float _abs_56;
                asm("abs.f32 %0, %1;" : "=f"(_abs_56) : "f"(x_2.y));
                float2 ax_4 = make_float2(_abs_55, _abs_56);
                float _abs_57;
                asm("abs.f32 %0, %1;" : "=f"(_abs_57) : "f"(v_2.x));
                float _abs_58;
                asm("abs.f32 %0, %1;" : "=f"(_abs_58) : "f"(v_2.y));
                float2 av_4 = make_float2(_abs_57, _abs_58);
                float _rcp_59 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_1, d1_1, d1_1), dw_1).x);
                float _rcp_60 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_1, d1_1, d1_1), dw_1).y);
                float2 r_4 = make_float2(_rcp_59, _rcp_60);
                float2 r_3 = r_4;
                float2 near_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_3) : "l"(*(const unsigned long long*)&ax_4), "l"(*(const unsigned long long*)&d1_1));
                float2 far_3 = fma_f32x2_rn_noftz(e_1, make_float2(nb, nb), make_float2(ub, ub));
                float _min_61;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_61) : "f"(near_3.x), "f"(far_3.x));
                float _min_62;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_62) : "f"(near_3.y), "f"(far_3.y));
                float2 bng_1 = make_float2(_min_61, _min_62);
                float2 near_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_4) : "l"(*(const unsigned long long*)&av_4), "l"(*(const unsigned long long*)&dw_1));
                float2 far_4 = fma_f32x2_rn_noftz(w_1, make_float2(nl, nl), make_float2(ul, ul));
                float _min_63;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_63) : "f"(near_4.x), "f"(far_4.x));
                float _min_64;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_64) : "f"(near_4.y), "f"(far_4.y));
                float2 lnu_1 = make_float2(_min_63, _min_64);
                float2 _f32x2_mul_a_65 = mul_f32x2_noftz(bng_1, lnu_1);
                float2 m_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&r_3));
                float2 ns_1 = make_float2(((x_2.x < 0.0f) ? p_1.x : 1.0f), ((x_2.y < 0.0f) ? p_1.y : 1.0f));
                gw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_1, ns_1))) | sign_2;
                float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(gw[3]));
                float2 v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(uw[3]));
                unsigned int sign_3 = (gw[3] ^ uw[3]) & 2147516416u;
                float _abs_66;
                asm("abs.f32 %0, %1;" : "=f"(_abs_66) : "f"(x_3.x));
                float _abs_67;
                asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(x_3.y));
                float2 ax_5 = make_float2(_abs_66, _abs_67);
                float _abs_68;
                asm("abs.f32 %0, %1;" : "=f"(_abs_68) : "f"(v_3.x));
                float _abs_69;
                asm("abs.f32 %0, %1;" : "=f"(_abs_69) : "f"(v_3.y));
                float2 av_5 = make_float2(_abs_68, _abs_69);
                float2 _f32x2_mul_b_70 = make_float2(k1, k1);
                float2 te_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_2) : "l"(*(const unsigned long long*)&ax_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
                float2 _f32x2_mul_b_71 = make_float2(k2, k2);
                float2 tw_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_2) : "l"(*(const unsigned long long*)&av_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
                float _exp2_72 = approx_exp2(te_2.x);
                float _exp2_73 = approx_exp2(te_2.y);
                float2 e_2 = make_float2(_exp2_72, _exp2_73);
                float _exp2_74 = approx_exp2(tw_2.x);
                float _exp2_75 = approx_exp2(tw_2.y);
                float2 w_2 = make_float2(_exp2_74, _exp2_75);
                float2 p_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_2) : "l"(*(const unsigned long long*)&e_2), "l"(*(const unsigned long long*)&e_2));
                float2 d1_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                float2 d2_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_2);
                float2 dw_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                float2 ng_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                float2 nu_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                float _abs_76;
                asm("abs.f32 %0, %1;" : "=f"(_abs_76) : "f"(x_3.x));
                float _abs_77;
                asm("abs.f32 %0, %1;" : "=f"(_abs_77) : "f"(x_3.y));
                float2 ax_6 = make_float2(_abs_76, _abs_77);
                float _abs_78;
                asm("abs.f32 %0, %1;" : "=f"(_abs_78) : "f"(v_3.x));
                float _abs_79;
                asm("abs.f32 %0, %1;" : "=f"(_abs_79) : "f"(v_3.y));
                float2 av_6 = make_float2(_abs_78, _abs_79);
                float _rcp_80 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_2, d1_2, d1_2), dw_2).x);
                float _rcp_81 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_2, d1_2, d1_2), dw_2).y);
                float2 r_6 = make_float2(_rcp_80, _rcp_81);
                float2 r_5 = r_6;
                float2 near_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_5) : "l"(*(const unsigned long long*)&ax_6), "l"(*(const unsigned long long*)&d1_2));
                float2 far_5 = fma_f32x2_rn_noftz(e_2, make_float2(nb, nb), make_float2(ub, ub));
                float _min_82;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_82) : "f"(near_5.x), "f"(far_5.x));
                float _min_83;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_83) : "f"(near_5.y), "f"(far_5.y));
                float2 bng_2 = make_float2(_min_82, _min_83);
                float2 near_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_6) : "l"(*(const unsigned long long*)&av_6), "l"(*(const unsigned long long*)&dw_2));
                float2 far_6 = fma_f32x2_rn_noftz(w_2, make_float2(nl, nl), make_float2(ul, ul));
                float _min_84;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_84) : "f"(near_6.x), "f"(far_6.x));
                float _min_85;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_85) : "f"(near_6.y), "f"(far_6.y));
                float2 lnu_2 = make_float2(_min_84, _min_85);
                float2 _f32x2_mul_a_86 = mul_f32x2_noftz(bng_2, lnu_2);
                float2 m_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&r_5));
                float2 ns_2 = make_float2(((x_3.x < 0.0f) ? p_2.x : 1.0f), ((x_3.y < 0.0f) ? p_2.y : 1.0f));
                gw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_2, ns_2))) | sign_3;
                if (s0 < limit) {
                    long long col_1 = (long long)(s0 * 256 + lane_1 * 8);
                    reinterpret_cast<int4*>(h + (hoff + col_1))[0] = reinterpret_cast<int4*>(gw)[0];
                }
            }
        } else {
            #pragma unroll 1
            for (int s0_1 = part; s0_1 < nseg; s0_1 += 4) {
                unsigned int gw_1[4];
                unsigned int uw_1[4];
                int seg_1 = ((s0_1) < (nseg - 1) ? (s0_1) : (nseg - 1));
                long long col_2 = (long long)(seg_1 * 256 + lane_1 * 8);
                {
                    uint32_t _uv4_87_0;
                    uint32_t _uv4_87_1;
                    uint32_t _uv4_87_2;
                    uint32_t _uv4_87_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_87_0), "=r"(_uv4_87_1), "=r"(_uv4_87_2), "=r"(_uv4_87_3) : "l"((const void*)(gate + (goff + col_2) + (0))));
                    gw_1[0 + 0] = _uv4_87_0;
                    gw_1[0 + 1] = _uv4_87_1;
                    gw_1[0 + 2] = _uv4_87_2;
                    gw_1[0 + 3] = _uv4_87_3;
                }
                {
                    uint32_t _uv4_88_0;
                    uint32_t _uv4_88_1;
                    uint32_t _uv4_88_2;
                    uint32_t _uv4_88_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_88_0), "=r"(_uv4_88_1), "=r"(_uv4_88_2), "=r"(_uv4_88_3) : "l"((const void*)(up + (uoff + col_2) + (0))));
                    uw_1[0 + 0] = _uv4_88_0;
                    uw_1[0 + 1] = _uv4_88_1;
                    uw_1[0 + 2] = _uv4_88_2;
                    uw_1[0 + 3] = _uv4_88_3;
                }
                float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gw_1[0]));
                float2 v_4 = cake_bf16x2_to_f32x2(__as_bf16x2(uw_1[0]));
                unsigned int sign_4 = (gw_1[0] ^ uw_1[0]) & 2147516416u;
                float _abs_89;
                asm("abs.f32 %0, %1;" : "=f"(_abs_89) : "f"(x_4.x));
                float _abs_90;
                asm("abs.f32 %0, %1;" : "=f"(_abs_90) : "f"(x_4.y));
                float2 ax_7 = make_float2(_abs_89, _abs_90);
                float _abs_91;
                asm("abs.f32 %0, %1;" : "=f"(_abs_91) : "f"(v_4.x));
                float _abs_92;
                asm("abs.f32 %0, %1;" : "=f"(_abs_92) : "f"(v_4.y));
                float2 av_7 = make_float2(_abs_91, _abs_92);
                float2 _f32x2_mul_b_93 = make_float2(k1, k1);
                float2 te_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_3) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_93));
                float2 _f32x2_mul_b_94 = make_float2(k2, k2);
                float2 tw_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_3) : "l"(*(const unsigned long long*)&av_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
                float _exp2_95 = approx_exp2(te_3.x);
                float _exp2_96 = approx_exp2(te_3.y);
                float2 e_3 = make_float2(_exp2_95, _exp2_96);
                float _exp2_97 = approx_exp2(tw_3.x);
                float _exp2_98 = approx_exp2(tw_3.y);
                float2 w_3 = make_float2(_exp2_97, _exp2_98);
                float2 _f32x2_mul_b_99 = make_float2(-1.442695f, -1.442695f);
                float2 tp;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_99));
                float _exp2_100 = approx_exp2_noftz(tp.x);
                float _exp2_101 = approx_exp2_noftz(tp.y);
                float2 p_3 = make_float2(_exp2_100, _exp2_101);
                float2 d1_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                float2 d2_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_3);
                float2 dw_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                float2 ng_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                float2 nu_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                float _abs_102;
                asm("abs.f32 %0, %1;" : "=f"(_abs_102) : "f"(x_4.x));
                float _abs_103;
                asm("abs.f32 %0, %1;" : "=f"(_abs_103) : "f"(x_4.y));
                float2 ax_7_1 = make_float2(_abs_102, _abs_103);
                float _abs_104;
                asm("abs.f32 %0, %1;" : "=f"(_abs_104) : "f"(v_4.x));
                float _abs_105;
                asm("abs.f32 %0, %1;" : "=f"(_abs_105) : "f"(v_4.y));
                float2 av_7_1 = make_float2(_abs_104, _abs_105);
                float _rcp_106 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_3, d1_3, d1_3), dw_3).x);
                float _rcp_107 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_3, d1_3, d1_3), dw_3).y);
                float2 r_7 = make_float2(_rcp_106, _rcp_107);
                float2 r_8 = r_7;
                float2 near_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_7) : "l"(*(const unsigned long long*)&ax_7_1), "l"(*(const unsigned long long*)&d1_3));
                float2 far_7 = fma_f32x2_rn_noftz(e_3, make_float2(nb, nb), make_float2(ub, ub));
                float _min_108;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_108) : "f"(near_7.x), "f"(far_7.x));
                float _min_109;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_109) : "f"(near_7.y), "f"(far_7.y));
                float2 bng_3 = make_float2(_min_108, _min_109);
                float2 near_7_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_7_1) : "l"(*(const unsigned long long*)&av_7_1), "l"(*(const unsigned long long*)&dw_3));
                float2 far_7_1 = fma_f32x2_rn_noftz(w_3, make_float2(nl, nl), make_float2(ul, ul));
                float _min_110;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_110) : "f"(near_7_1.x), "f"(far_7_1.x));
                float _min_111;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_111) : "f"(near_7_1.y), "f"(far_7_1.y));
                float2 lnu_3 = make_float2(_min_110, _min_111);
                float2 _f32x2_mul_a_112 = mul_f32x2_noftz(bng_3, lnu_3);
                float2 m_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&r_8));
                float2 ns_3 = make_float2(((x_4.x < 0.0f) ? p_3.x : 1.0f), ((x_4.y < 0.0f) ? p_3.y : 1.0f));
                gw_1[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_3, ns_3))) | sign_4;
                float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(gw_1[1]));
                float2 v_5 = cake_bf16x2_to_f32x2(__as_bf16x2(uw_1[1]));
                unsigned int sign_5 = (gw_1[1] ^ uw_1[1]) & 2147516416u;
                float _abs_113;
                asm("abs.f32 %0, %1;" : "=f"(_abs_113) : "f"(x_5.x));
                float _abs_114;
                asm("abs.f32 %0, %1;" : "=f"(_abs_114) : "f"(x_5.y));
                float2 ax_8 = make_float2(_abs_113, _abs_114);
                float _abs_115;
                asm("abs.f32 %0, %1;" : "=f"(_abs_115) : "f"(v_5.x));
                float _abs_116;
                asm("abs.f32 %0, %1;" : "=f"(_abs_116) : "f"(v_5.y));
                float2 av_8 = make_float2(_abs_115, _abs_116);
                float2 _f32x2_mul_b_117 = make_float2(k1, k1);
                float2 te_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_3_1) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_117));
                float2 _f32x2_mul_b_118 = make_float2(k2, k2);
                float2 tw_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_3_1) : "l"(*(const unsigned long long*)&av_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_118));
                float _exp2_119 = approx_exp2(te_3_1.x);
                float _exp2_120 = approx_exp2(te_3_1.y);
                float2 e_3_1 = make_float2(_exp2_119, _exp2_120);
                float _exp2_121 = approx_exp2(tw_3_1.x);
                float _exp2_122 = approx_exp2(tw_3_1.y);
                float2 w_3_1 = make_float2(_exp2_121, _exp2_122);
                float2 _f32x2_mul_b_123 = make_float2(-1.442695f, -1.442695f);
                float2 tp_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_0) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_123));
                float _exp2_124 = approx_exp2_noftz(tp_0.x);
                float _exp2_125 = approx_exp2_noftz(tp_0.y);
                float2 p_3_1 = make_float2(_exp2_124, _exp2_125);
                float2 d1_3_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_3_1);
                float2 d2_3_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_3_1);
                float2 dw_3_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_3_1);
                float2 ng_3_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_3_1);
                float2 nu_3_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_3_1);
                float _abs_126;
                asm("abs.f32 %0, %1;" : "=f"(_abs_126) : "f"(x_5.x));
                float _abs_127;
                asm("abs.f32 %0, %1;" : "=f"(_abs_127) : "f"(x_5.y));
                float2 ax_9 = make_float2(_abs_126, _abs_127);
                float _abs_128;
                asm("abs.f32 %0, %1;" : "=f"(_abs_128) : "f"(v_5.x));
                float _abs_129;
                asm("abs.f32 %0, %1;" : "=f"(_abs_129) : "f"(v_5.y));
                float2 av_9 = make_float2(_abs_128, _abs_129);
                float _rcp_130 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_3_1, d1_3_1, d1_3_1), dw_3_1).x);
                float _rcp_131 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_3_1, d1_3_1, d1_3_1), dw_3_1).y);
                float2 r_9 = make_float2(_rcp_130, _rcp_131);
                float2 r_8_1 = r_9;
                float2 near_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_8) : "l"(*(const unsigned long long*)&ax_9), "l"(*(const unsigned long long*)&d1_3_1));
                float2 far_8 = fma_f32x2_rn_noftz(e_3_1, make_float2(nb, nb), make_float2(ub, ub));
                float _min_132;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_132) : "f"(near_8.x), "f"(far_8.x));
                float _min_133;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_133) : "f"(near_8.y), "f"(far_8.y));
                float2 bng_3_1 = make_float2(_min_132, _min_133);
                float2 near_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_9) : "l"(*(const unsigned long long*)&av_9), "l"(*(const unsigned long long*)&dw_3_1));
                float2 far_9 = fma_f32x2_rn_noftz(w_3_1, make_float2(nl, nl), make_float2(ul, ul));
                float _min_134;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_134) : "f"(near_9.x), "f"(far_9.x));
                float _min_135;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_135) : "f"(near_9.y), "f"(far_9.y));
                float2 lnu_3_1 = make_float2(_min_134, _min_135);
                float2 _f32x2_mul_a_136 = mul_f32x2_noftz(bng_3_1, lnu_3_1);
                float2 m_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&r_8_1));
                float2 ns_3_1 = make_float2(((x_5.x < 0.0f) ? p_3_1.x : 1.0f), ((x_5.y < 0.0f) ? p_3_1.y : 1.0f));
                gw_1[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_3_1, ns_3_1))) | sign_5;
                float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gw_1[2]));
                float2 v_6 = cake_bf16x2_to_f32x2(__as_bf16x2(uw_1[2]));
                unsigned int sign_6 = (gw_1[2] ^ uw_1[2]) & 2147516416u;
                float _abs_137;
                asm("abs.f32 %0, %1;" : "=f"(_abs_137) : "f"(x_6.x));
                float _abs_138;
                asm("abs.f32 %0, %1;" : "=f"(_abs_138) : "f"(x_6.y));
                float2 ax_10 = make_float2(_abs_137, _abs_138);
                float _abs_139;
                asm("abs.f32 %0, %1;" : "=f"(_abs_139) : "f"(v_6.x));
                float _abs_140;
                asm("abs.f32 %0, %1;" : "=f"(_abs_140) : "f"(v_6.y));
                float2 av_10 = make_float2(_abs_139, _abs_140);
                float2 _f32x2_mul_b_141 = make_float2(k1, k1);
                float2 te_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_4) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_141));
                float2 _f32x2_mul_b_142 = make_float2(k2, k2);
                float2 tw_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_4) : "l"(*(const unsigned long long*)&av_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_142));
                float _exp2_143 = approx_exp2(te_4.x);
                float _exp2_144 = approx_exp2(te_4.y);
                float2 e_4 = make_float2(_exp2_143, _exp2_144);
                float _exp2_145 = approx_exp2(tw_4.x);
                float _exp2_146 = approx_exp2(tw_4.y);
                float2 w_4 = make_float2(_exp2_145, _exp2_146);
                float2 _f32x2_mul_b_147 = make_float2(-1.442695f, -1.442695f);
                float2 tp_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_1) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_147));
                float _exp2_148 = approx_exp2_noftz(tp_1.x);
                float _exp2_149 = approx_exp2_noftz(tp_1.y);
                float2 p_4 = make_float2(_exp2_148, _exp2_149);
                float2 d1_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                float2 d2_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_4);
                float2 dw_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                float2 ng_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                float2 nu_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                float _abs_150;
                asm("abs.f32 %0, %1;" : "=f"(_abs_150) : "f"(x_6.x));
                float _abs_151;
                asm("abs.f32 %0, %1;" : "=f"(_abs_151) : "f"(x_6.y));
                float2 ax_11 = make_float2(_abs_150, _abs_151);
                float _abs_152;
                asm("abs.f32 %0, %1;" : "=f"(_abs_152) : "f"(v_6.x));
                float _abs_153;
                asm("abs.f32 %0, %1;" : "=f"(_abs_153) : "f"(v_6.y));
                float2 av_11 = make_float2(_abs_152, _abs_153);
                float _rcp_154 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_4, d1_4, d1_4), dw_4).x);
                float _rcp_155 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_4, d1_4, d1_4), dw_4).y);
                float2 r_11 = make_float2(_rcp_154, _rcp_155);
                float2 r_10 = r_11;
                float2 near_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_10) : "l"(*(const unsigned long long*)&ax_11), "l"(*(const unsigned long long*)&d1_4));
                float2 far_10 = fma_f32x2_rn_noftz(e_4, make_float2(nb, nb), make_float2(ub, ub));
                float _min_156;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_156) : "f"(near_10.x), "f"(far_10.x));
                float _min_157;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_157) : "f"(near_10.y), "f"(far_10.y));
                float2 bng_4 = make_float2(_min_156, _min_157);
                float2 near_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_11) : "l"(*(const unsigned long long*)&av_11), "l"(*(const unsigned long long*)&dw_4));
                float2 far_11 = fma_f32x2_rn_noftz(w_4, make_float2(nl, nl), make_float2(ul, ul));
                float _min_158;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_158) : "f"(near_11.x), "f"(far_11.x));
                float _min_159;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_159) : "f"(near_11.y), "f"(far_11.y));
                float2 lnu_4 = make_float2(_min_158, _min_159);
                float2 _f32x2_mul_a_160 = mul_f32x2_noftz(bng_4, lnu_4);
                float2 m_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&r_10));
                float2 ns_4 = make_float2(((x_6.x < 0.0f) ? p_4.x : 1.0f), ((x_6.y < 0.0f) ? p_4.y : 1.0f));
                gw_1[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_4, ns_4))) | sign_6;
                float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(gw_1[3]));
                float2 v_7 = cake_bf16x2_to_f32x2(__as_bf16x2(uw_1[3]));
                unsigned int sign_7 = (gw_1[3] ^ uw_1[3]) & 2147516416u;
                float _abs_161;
                asm("abs.f32 %0, %1;" : "=f"(_abs_161) : "f"(x_7.x));
                float _abs_162;
                asm("abs.f32 %0, %1;" : "=f"(_abs_162) : "f"(x_7.y));
                float2 ax_12 = make_float2(_abs_161, _abs_162);
                float _abs_163;
                asm("abs.f32 %0, %1;" : "=f"(_abs_163) : "f"(v_7.x));
                float _abs_164;
                asm("abs.f32 %0, %1;" : "=f"(_abs_164) : "f"(v_7.y));
                float2 av_12 = make_float2(_abs_163, _abs_164);
                float2 _f32x2_mul_b_165 = make_float2(k1, k1);
                float2 te_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_5) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
                float2 _f32x2_mul_b_166 = make_float2(k2, k2);
                float2 tw_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_5) : "l"(*(const unsigned long long*)&av_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_166));
                float _exp2_167 = approx_exp2(te_5.x);
                float _exp2_168 = approx_exp2(te_5.y);
                float2 e_5 = make_float2(_exp2_167, _exp2_168);
                float _exp2_169 = approx_exp2(tw_5.x);
                float _exp2_170 = approx_exp2(tw_5.y);
                float2 w_5 = make_float2(_exp2_169, _exp2_170);
                float2 _f32x2_mul_b_171 = make_float2(-1.442695f, -1.442695f);
                float2 tp_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_2) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_171));
                float _exp2_172 = approx_exp2_noftz(tp_2.x);
                float _exp2_173 = approx_exp2_noftz(tp_2.y);
                float2 p_5 = make_float2(_exp2_172, _exp2_173);
                float2 d1_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                float2 d2_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_5);
                float2 dw_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                float2 ng_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                float2 nu_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                float _abs_174;
                asm("abs.f32 %0, %1;" : "=f"(_abs_174) : "f"(x_7.x));
                float _abs_175;
                asm("abs.f32 %0, %1;" : "=f"(_abs_175) : "f"(x_7.y));
                float2 ax_13 = make_float2(_abs_174, _abs_175);
                float _abs_176;
                asm("abs.f32 %0, %1;" : "=f"(_abs_176) : "f"(v_7.x));
                float _abs_177;
                asm("abs.f32 %0, %1;" : "=f"(_abs_177) : "f"(v_7.y));
                float2 av_13 = make_float2(_abs_176, _abs_177);
                float _rcp_178 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_5, d1_5, d1_5), dw_5).x);
                float _rcp_179 = approx_rcp(mul_f32x2_noftz(fma_f32x2_rn_noftz(p_5, d1_5, d1_5), dw_5).y);
                float2 r_13 = make_float2(_rcp_178, _rcp_179);
                float2 r_12 = r_13;
                float2 near_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_12) : "l"(*(const unsigned long long*)&ax_13), "l"(*(const unsigned long long*)&d1_5));
                float2 far_12 = fma_f32x2_rn_noftz(e_5, make_float2(nb, nb), make_float2(ub, ub));
                float _min_180;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_180) : "f"(near_12.x), "f"(far_12.x));
                float _min_181;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_181) : "f"(near_12.y), "f"(far_12.y));
                float2 bng_5 = make_float2(_min_180, _min_181);
                float2 near_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&near_13) : "l"(*(const unsigned long long*)&av_13), "l"(*(const unsigned long long*)&dw_5));
                float2 far_13 = fma_f32x2_rn_noftz(w_5, make_float2(nl, nl), make_float2(ul, ul));
                float _min_182;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_182) : "f"(near_13.x), "f"(far_13.x));
                float _min_183;
                asm("min.f32 %0, %1, %2;" : "=f"(_min_183) : "f"(near_13.y), "f"(far_13.y));
                float2 lnu_5 = make_float2(_min_182, _min_183);
                float2 _f32x2_mul_a_184 = mul_f32x2_noftz(bng_5, lnu_5);
                float2 m_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&r_12));
                float2 ns_5 = make_float2(((x_7.x < 0.0f) ? p_5.x : 1.0f), ((x_7.y < 0.0f) ? p_5.y : 1.0f));
                gw_1[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_5, ns_5))) | sign_7;
                if (s0_1 < limit) {
                    long long col_3 = (long long)(s0_1 * 256 + lane_1 * 8);
                    reinterpret_cast<int4*>(h + (hoff + col_3))[0] = reinterpret_cast<int4*>(gw_1)[0];
                }
            }
        }
    }
}

} // extern "C"
