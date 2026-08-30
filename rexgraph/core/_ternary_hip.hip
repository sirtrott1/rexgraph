// rexgraph.core._ternary_hip: the packed ternary product as a native device kernel.
//
// The portable device lane goes through torch, where the product is a chain of
// elementwise expressions. Fused by the compiler it reaches about 70% of what the
// memory system can stream; unfused it is a ninth of that, because each expression
// round-trips the whole operator. Neither reaches the popcount instruction the
// hardware already has: torch exposes none, so the emulation costs either a byte-table
// gather or a dozen SWAR arithmetic steps per word.
//
// This is the same arithmetic with none of that. One pass over the planes, one
// __popcll per word, and the +-1 vector held in LDS because every row reads all of it.
//
//     out[i] = k[i] - 2 * popcount( P[i] & (S[i] ^ X) )
//
// The identity is what halves the work: the agreeing and disagreeing counts sum to the
// row's arity, which is a property of the operator and known before any product runs.
//
// Build (optional, only where hipcc exists):
//   hipcc --offload-arch=<gfx> -O3 -shared -fPIC -o lib_ternary_hip.so _ternary_hip.hip

#include <hip/hip_runtime.h>
#include <stdint.h>

// One block per row. Threads stride the row so consecutive lanes read consecutive
// words, which is the access pattern the memory system wants; the partial counts then
// reduce through LDS.
extern "C" __global__ void __launch_bounds__(256)
ternary_pm1_kernel(const uint64_t* __restrict__ P,
                   const uint64_t* __restrict__ S,
                   const uint64_t* __restrict__ X,
                   const int64_t*  __restrict__ K,
                   int64_t* __restrict__ out,
                   int nw, int lds_words)
{
    extern __shared__ uint64_t smem[];
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const size_t row = blockIdx.x;

    // The vector goes in LDS when it fits, since every row reads all of it. When it
    // does not, it is read from global instead: still the same answer, and still
    // cached, just without the guarantee. `lds_words` is 0 in that case.
    uint64_t* sX  = smem;
    int64_t*  red = (int64_t*)(smem + lds_words);
    if (lds_words) {
        for (int w = tid; w < nw; w += nth) sX[w] = X[w];
    }
    __syncthreads();

    const uint64_t* p = P + row * (size_t)nw;
    const uint64_t* s = S + row * (size_t)nw;
    const uint64_t* xv = lds_words ? sX : X;

    int64_t acc = 0;
    for (int w = tid; w < nw; w += nth)
        acc += __popcll(p[w] & (s[w] ^ xv[w]));

    red[tid] = acc;
    __syncthreads();
    for (int half = nth >> 1; half > 0; half >>= 1) {
        if (tid < half) red[tid] += red[tid + half];
        __syncthreads();
    }
    if (tid == 0) out[row] = K[row] - 2 * red[0];
}

extern "C" int ternary_pm1_launch(const void* P, const void* S, const void* X,
                                  const void* K, void* out, int nrows, int nw,
                                  int threads)
{
    // 256 is the launch bound the kernel is compiled for, and it is also the fastest
    // block measured here, so a larger request is clamped rather than refused.
    if (threads <= 0 || threads > 256) threads = 256;

    // LDS is finite. Hold the vector there when it fits inside a conservative 48 KiB
    // of the workgroup's share, and read it from global when it does not.
    const size_t LDS_BUDGET = 48u * 1024u;
    size_t red_bytes = (size_t)threads * sizeof(int64_t);
    int lds_words = 0;
    if ((size_t)nw * sizeof(uint64_t) + red_bytes <= LDS_BUDGET) lds_words = nw;
    size_t shared = (size_t)lds_words * sizeof(uint64_t) + red_bytes;

    hipLaunchKernelGGL(ternary_pm1_kernel, dim3(nrows), dim3(threads), shared, 0,
                       (const uint64_t*)P, (const uint64_t*)S, (const uint64_t*)X,
                       (const int64_t*)K, (int64_t*)out, nw, lds_words);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return (int)hipDeviceSynchronize();
}

// Device-side allocation and copy, so a caller can hold the planes resident without
// depending on any particular tensor library to own the memory.
extern "C" int ternary_alloc(void** ptr, size_t nbytes) {
    return (int)hipMalloc(ptr, nbytes);
}
extern "C" int ternary_upload(void* dst, const void* src, size_t nbytes) {
    return (int)hipMemcpy(dst, src, nbytes, hipMemcpyHostToDevice);
}
extern "C" int ternary_download(void* dst, const void* src, size_t nbytes) {
    return (int)hipMemcpy(dst, src, nbytes, hipMemcpyDeviceToHost);
}
extern "C" int ternary_free(void* ptr) { return (int)hipFree(ptr); }

// The float path. Unlike the +-1 product this is not bound by the planes: every row
// reads all of v, so v traffic exceeds plane traffic by the word width. Rows are
// therefore blocked, four to a workgroup, so one read of v serves four accumulators.
// v stays in global memory because it is generally too large for LDS, but it is read
// by every block and lands in cache.
//
// BLK is how many rows share one read of v. Four is measured, not assumed: at 4 the
// v traffic is 239 GB/s on a device that streams 218, so v is the ceiling, and the
// obvious inference is that 8 would halve it and go faster. It does not. At 8 the
// traffic falls to 106 GB/s and throughput falls with it, 119.5 to 106.0 Gentry/s,
// because eight accumulators and a 16 KB reduction cost more occupancy than the
// halved traffic buys. The constraint moves from bandwidth to residency in between.
#define TERNARY_F64_BLK 4

extern "C" __global__ void __launch_bounds__(256)
ternary_f64_kernel(const uint64_t* __restrict__ P,
                   const uint64_t* __restrict__ S,
                   const double*   __restrict__ V,
                   double* __restrict__ out,
                   int nrows, int nw)
{
    extern __shared__ double red[];              // blockDim * BLK partial sums
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const size_t r0 = (size_t)blockIdx.x * TERNARY_F64_BLK;
    const int nrow = (int)min((size_t)TERNARY_F64_BLK, (size_t)nrows - r0);

    double acc[TERNARY_F64_BLK];
    #pragma unroll
    for (int r = 0; r < TERNARY_F64_BLK; ++r) acc[r] = 0.0;

    for (int w = tid; w < nw; w += nth) {
        const double* base = V + ((size_t)w << 6);
        for (int r = 0; r < nrow; ++r) {
            const size_t off = (r0 + r) * (size_t)nw + w;
            uint64_t pres = P[off];
            if (!pres) continue;
            uint64_t sgn = S[off];
            while (pres) {
                uint64_t bit = pres & (~pres + 1ULL);
                int b = __ffsll((unsigned long long)bit) - 1;
                acc[r] += ((sgn >> b) & 1ULL) ? -base[b] : base[b];
                pres ^= bit;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < TERNARY_F64_BLK; ++r) red[r * nth + tid] = acc[r];
    __syncthreads();
    for (int half = nth >> 1; half > 0; half >>= 1) {
        if (tid < half)
            for (int r = 0; r < TERNARY_F64_BLK; ++r)
                red[r*nth + tid] += red[r*nth + tid + half];
        __syncthreads();
    }
    if (tid < nrow) out[r0 + tid] = red[tid * nth];
}

extern "C" int ternary_f64_launch(const void* P, const void* S, const void* V,
                                  void* out, int nrows, int nw, int threads)
{
    if (threads <= 0 || threads > 256) threads = 256;
    int nblocks = (nrows + TERNARY_F64_BLK - 1) / TERNARY_F64_BLK;
    size_t shared = (size_t)threads * TERNARY_F64_BLK * sizeof(double);
    hipLaunchKernelGGL(ternary_f64_kernel, dim3(nblocks), dim3(threads), shared, 0,
                       (const uint64_t*)P, (const uint64_t*)S, (const double*)V,
                       (double*)out, nrows, nw);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return (int)hipDeviceSynchronize();
}

// The channel tower.
// The four diagonals at any arity. Same shape as the CPU kernel and for the same
// reason: accumulating straight into the vertex arrays has every relation writing
// wherever its support points, which on a device means an atomic per nonzero. With the
// incidence transposed the accumulation is a loop OVER VERTICES, one thread per vertex,
// each owning what it writes, so no atomic appears anywhere.
//
// C is unweighted and F is not, so the mass is carried twice. A witness is (+1) and
// joins the POSITIVE mass rather than taking the head rule.

extern "C" __global__ void tower_vertex_mass(
    const int*     __restrict__ bp,      // relation offsets, length nE+1
    const int*     __restrict__ owner,   // relation owning each transposed entry
    const unsigned char* __restrict__ is_head,
    const double*  __restrict__ w,       // per-relation weight
    const long*    __restrict__ vptr,    // vertex offsets into the transpose
    double* __restrict__ negw, double* __restrict__ posw,
    double* __restrict__ negu, double* __restrict__ posu,
    int nV)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nV) return;
    double nw = 0.0, pw = 0.0, nu = 0.0, pu = 0.0;
    for (long q = vptr[v]; q < vptr[v + 1]; ++q) {
        int f = owner[q];
        int kf = bp[f + 1] - bp[f];
        if (kf == 1) { pw += w[f]; pu += 1.0; }            // witness is (+1)
        else if (is_head[q]) { nw += w[f]; nu += 1.0; }    // head, magnitude 1
        else { double mg = 1.0 / (double)(kf - 1); pw += w[f] * mg; pu += mg; }
    }
    negw[v] = nw; posw[v] = pw; negu[v] = nu; posu[v] = pu;
}

extern "C" __global__ void tower_read(
    const int*    __restrict__ bp,
    const int*    __restrict__ bi,
    const double* __restrict__ w,
    const double* __restrict__ negw, const double* __restrict__ posw,
    const double* __restrict__ negu, const double* __restrict__ posu,
    double* __restrict__ T, double* __restrict__ G,
    double* __restrict__ F, double* __restrict__ C,
    int nE)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nE) return;
    int s = bp[e], t = bp[e + 1], k = t - s;
    if (k == 0) { T[e] = G[e] = F[e] = C[e] = 0.0; return; }
    double we = w[e], c = 0.0, f = 0.0;
    if (k == 1) {
        T[e] = G[e] = we * we;
        int v = bi[s];
        C[e] = 1.0 * (negu[v] + posu[v] - 1.0);            // C is unweighted
        F[e] = 2.0 * we * negw[v];
        return;
    }
    double share = 1.0 / (double)(k - 1);
    T[e] = G[e] = we * we * (1.0 + share);
    int v = bi[s];
    c += 1.0 * (negu[v] + posu[v] - 1.0);
    f += we * posw[v];
    for (int p = s + 1; p < t; ++p) {
        v = bi[p];
        c += share * (negu[v] + posu[v] - share);
        f += we * share * negw[v];
    }
    C[e] = c; F[e] = 2.0 * f;
}

extern "C" int tower_launch(const void* bp, const void* bi, const void* owner,
                            const void* is_head, const void* w, const void* vptr,
                            void* negw, void* posw, void* negu, void* posu,
                            void* T, void* G, void* F, void* C,
                            int nV, int nE, int threads)
{
    if (threads <= 0 || threads > 256) threads = 256;
    hipLaunchKernelGGL(tower_vertex_mass, dim3((nV + threads - 1) / threads),
                       dim3(threads), 0, 0,
                       (const int*)bp, (const int*)owner,
                       (const unsigned char*)is_head, (const double*)w,
                       (const long*)vptr, (double*)negw, (double*)posw,
                       (double*)negu, (double*)posu, nV);
    hipError_t e1 = hipGetLastError();
    if (e1 != hipSuccess) return (int)e1;
    hipLaunchKernelGGL(tower_read, dim3((nE + threads - 1) / threads), dim3(threads),
                       0, 0, (const int*)bp, (const int*)bi, (const double*)w,
                       (const double*)negw, (const double*)posw,
                       (const double*)negu, (const double*)posu,
                       (double*)T, (double*)G, (double*)F, (double*)C, nE);
    hipError_t e2 = hipGetLastError();
    if (e2 != hipSuccess) return (int)e2;
    return (int)hipDeviceSynchronize();
}
