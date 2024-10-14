#include <iostream>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <hip/hip_bfloat16.h>

#define M 16
#define N 16
#define K 16

// GEMM configuration.

#define M_TILES 512
#define N_TILES 512
#define K_TILES 512

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)


using FP32 = rocwmma::float32_t;
using SP32 = rocwmma::float32_t;

using FP64 = rocwmma::float64_t;
using FP16 = rocwmma::float16_t;
using BF16 = rocwmma::bfloat16_t;
using DeviceProp = hipDeviceProp_t;

hipError_t (*deviceMalloc)(void**, size_t) = hipMalloc;


void getDeviceProperties(DeviceProp* prop, int device) {
    hipGetDeviceProperties(prop, device);
}

#define CHECK_hip_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           int const line)
{
    if (err != hipSuccess)
    {
        std::cerr << "hip Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << hipGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_hip_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
    hipError_t err{hipGetLastError()};
    if (err != hipSuccess)
    {
        std::cerr << "hip Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << hipGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_DEVICE_ERROR(val) CHECK_hip_ERROR(val)
#define CHECK_LAST_DEVICE_ERROR() CHECK_LAST_hip_ERROR()
#define __float2bfloat16(val) hip_bfloat16(val)
#define __bfloat162float(val) float(val)
#define wmma_init_run_32(v1, v2, v3, v4, v5, v6, v7) wmma_init_run(v1, v2, v3, v4, v5, v6, v7)
#define wmma_init_run_64(v1, v2, v3, v4, v5, v6, v7) wmma_init_run(v1, v2, v3, v4, v5, v6, v7)

// template <typename inputtype, typename returntype>
// void host_reset(inputtype *a, inputtype *b, returntype *c) {
//   memset(a, 0, 16*16*sizeof(inputtype));
//   memset(b, 0, 16*16*sizeof(inputtype));
//   memset(c, 0, 16*16*sizeof(returntype));
// }

/* Compute C += A*B, where A, B, and C are 16x16x16 matrices.
   The matrix C is initialized to 0 when `init` is true. */
template <typename inputtype, typename returntype>
__global__ void wmma_ker(inputtype *a, inputtype *b, returntype *c, bool init) {

  // Declare fragments.
  rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, inputtype, rocwmma::row_major> a_fragment;
  rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, inputtype, rocwmma::col_major> b_fragment;
  rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, returntype> c_fragment;

  // Load input matrices and initialize output (if required).
  rocwmma::load_matrix_sync(a_fragment, a, 16);
  rocwmma::load_matrix_sync(b_fragment, b, 16);
  if (init)
    rocwmma::fill_fragment(c_fragment, returntype(0.0));
  else
    rocwmma::load_matrix_sync(c_fragment, c, 16, rocwmma::mem_col_major);

  // Multiply
  rocwmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

  // Store the output
  rocwmma::store_matrix_sync(c, c_fragment, 16, rocwmma::mem_col_major);
}

/* Copy data from host to device, perform the operation, and copy result back to
   host. */
template <typename inputtype, typename returntype>
void wmma_init_run (inputtype *h_a, inputtype *h_b, returntype *h_c,
                    inputtype *d_a, inputtype *d_b, returntype *d_c,
                    bool init) {

  // Copy input from host to device.
  hipMemcpy(d_a, h_a, 16*16*sizeof(inputtype), hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b, 16*16*sizeof(inputtype), hipMemcpyHostToDevice);
  hipMemcpy(d_c, h_c, 16*16*sizeof(returntype), hipMemcpyHostToDevice);

  // Perform matrix multiplication.
  wmma_ker<<<1,64>>>(d_a, d_b, d_c, init);

  // Copy result from device to host.
  hipMemcpy(h_c, d_c, 16*16*sizeof(returntype), hipMemcpyDeviceToHost);
}

template <typename inputtype, typename returntype>
__global__ void simple_wmma_16(inputtype *a, inputtype *b, returntype *c, returntype *d, int m_ld, int n_ld, int k_ld, float alpha, float beta)
{
   // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   rocwmma::fragment<rocwmma::matrix_a, M, N, K, inputtype, rocwmma::row_major> a_frag;
   rocwmma::fragment<rocwmma::matrix_b, M, N, K, inputtype, rocwmma::col_major> b_frag;
   rocwmma::fragment<rocwmma::accumulator, M, N, K, returntype> acc_frag;
   rocwmma::fragment<rocwmma::accumulator, M, N, K, returntype> c_frag;

   rocwmma::fill_fragment(acc_frag, returntype(0.0));

   // Loop over k
   for (int i = 0; i < k_ld; i += K) {
      int aCol = i; 
      int aRow = warpM * M;

      int bCol = warpN * N;
      int bRow = i;

      // int cCol = warpN * N;
      // int cRow = warpM * M;
      // Bounds checking
      if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
         // Load the inputs
         rocwmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
         rocwmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
         // rocwmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, rocwmma::mem_row_major);
 
         // Perform the matrix multiplication
         rocwmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
         // rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
         // rocwmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, rocwmma::mem_row_major);
      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cCol = warpN * N;
   int cRow = warpM * M;

   if (cRow < m_ld && cCol < n_ld) {
      rocwmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, rocwmma::mem_row_major);

      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      rocwmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, rocwmma::mem_row_major);
   }
}

template <typename inputtype, typename returntype>
void simple_wmma_run (inputtype *h_a, inputtype *h_b, returntype *h_c,returntype *h_d,
                    inputtype *d_a, inputtype *d_b, returntype *d_c,returntype *d_d,
                    float alpha, float beta) {
  dim3 gridDim;
  dim3 blockDim;

  // blockDim.x must be a multple of warpSize
  // 128x4 means we have 16 warps and a block computes a 64x64 output tile
  blockDim.x = 256;
  blockDim.y = 4;

  gridDim.x = (M_GLOBAL + (M * blockDim.x / 64 - 1)) / (M * blockDim.x / 64);
  gridDim.y = (N_GLOBAL + N * blockDim.y - 1) / (N * blockDim.y);

  hipMemcpy(d_a, h_a, M_GLOBAL*K_GLOBAL*sizeof(inputtype), hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b, N_GLOBAL*K_GLOBAL*sizeof(inputtype), hipMemcpyHostToDevice);
  hipMemcpy(d_c, h_c, M_GLOBAL*N_GLOBAL*sizeof(returntype), hipMemcpyHostToDevice);
  hipMemcpy(d_d, h_d, M_GLOBAL*N_GLOBAL*sizeof(returntype), hipMemcpyHostToDevice);


  printf("Computing... using simple_wmma_gemm kernel\n");
  simple_wmma_16<<<gridDim, blockDim>>>(d_a, d_b, d_c, d_d, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);

  // Copy result from device to host.
  hipMemcpy(h_d, d_d, M_GLOBAL*N_GLOBAL*sizeof(returntype), hipMemcpyDeviceToHost);

}