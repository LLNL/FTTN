#include <mma.h>
using namespace nvcuda;
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


// MMA matrix tile dimensions.

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


// Implementation constants.



using FP32 = float;
// using SP32 = wmma::precision::tf32;
using FP64 = double;
using FP16 = half;
using BF16 = nv_bfloat16;
using DeviceProp = cudaDeviceProp;

cudaError_t (*deviceMalloc)(void**, size_t) = cudaMalloc;

void getDeviceProperties(DeviceProp* prop, int device) {
    cudaGetDeviceProperties(prop, device);
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_DEVICE_ERROR(val) CHECK_CUDA_ERROR(val)
#define CHECK_LAST_DEVICE_ERROR() CHECK_LAST_CUDA_ERROR()

template <typename inputtype, typename returntype>
__global__ void wmma_ker(inputtype *a, inputtype *b, returntype *c, bool init) {

  // Declare fragments.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, inputtype, wmma::row_major> a_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, inputtype, wmma::col_major> b_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, returntype> c_fragment;

  // Load input matrices and initialize output (if required).
  wmma::load_matrix_sync(a_fragment, a, 16);
  wmma::load_matrix_sync(b_fragment, b, 16);
  if (init)
    wmma::fill_fragment(c_fragment, 0.0f);
  else
    wmma::load_matrix_sync(c_fragment, c, 16, wmma::mem_col_major);

  // Multiply
  wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

  // Store the output
  wmma::store_matrix_sync(c, c_fragment, 16, wmma::mem_col_major);
}

/* Copy data from host to device, perform the operation, and copy result back to
   host. */
template <typename inputtype, typename returntype>
void wmma_init_run (inputtype *h_a, inputtype *h_b, returntype *h_c,
                    inputtype *d_a, inputtype *d_b, returntype *d_c,
                    bool init) {

  // Copy input from host to device.
  cudaMemcpy(d_a, h_a, 16*16*sizeof(inputtype), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, 16*16*sizeof(inputtype), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, 16*16*sizeof(returntype), cudaMemcpyHostToDevice);

  // Perform matrix multiplication.
  wmma_ker<<<1,32>>>(d_a, d_b, d_c, init);

  // Copy result from device to host.
  cudaMemcpy(h_c, d_c, 16*16*sizeof(returntype), cudaMemcpyDeviceToHost);
}

__global__ void wmma_ker_32(FP32 *a, FP32 *b, FP32 *c, bool init) {

  // Declare fragments.

  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 8, FP32> c_fragment;
  

  for(int iter = 0; iter<2; iter++){

    // Load input matrices and initialize output (if required).
    wmma::load_matrix_sync(a_fragment, a+8*iter, 16);
    wmma::load_matrix_sync(b_fragment, b+8*iter, 16);
    if (init)
      wmma::fill_fragment(c_fragment, 0.0f);
    else
      wmma::load_matrix_sync(c_fragment, c, 16, wmma::mem_col_major);

    // Multiply
    wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

    // Store the output
    wmma::store_matrix_sync(c, c_fragment, 16, wmma::mem_col_major);
  }
}

/* Copy data from host to device, perform the operation, and copy result back to
   host. */
template <typename inputtype, typename returntype>
void wmma_init_run_32 (inputtype *h_a, inputtype *h_b, returntype *h_c,
                    inputtype *d_a, inputtype *d_b, returntype *d_c,
                    bool init) {

  // Copy input from host to device.
  cudaMemcpy(d_a, h_a, 16*16*sizeof(inputtype), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, 16*16*sizeof(inputtype), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, 16*16*sizeof(returntype), cudaMemcpyHostToDevice);

  // Perform matrix multiplication.
  wmma_ker_32<<<1,32>>>(d_a, d_b, d_c, init);

  // Copy result from device to host.
  cudaMemcpy(h_c, d_c, 16*16*sizeof(returntype), cudaMemcpyDeviceToHost);
}

__global__ void wmma_ker_64(double *a, double *b, double *c, bool init) {

  // Declare fragments.
  wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_fragment;
  wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> b_fragment;
  wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_fragment;

  int warpM = threadIdx.x /32;
  int warpN = threadIdx.y;
  // printf("WarpM = %d, ", warpM);
  // printf("WarpN = %d\n", warpN);
  

  // Load input matrices and initialize output (if required).
  for (int iter = 0; iter <4; iter++){

  
    wmma::load_matrix_sync(a_fragment, a + warpN*16*8 + iter *4 , 16);
    wmma::load_matrix_sync(b_fragment, b + iter*4 + warpM *16*8 , 16);
    if (init)
      wmma::fill_fragment(c_fragment, 0.0f);
    else
      wmma::load_matrix_sync(c_fragment, c+warpM*8+warpN*16*8, 16, wmma::mem_row_major);

    // Multiply
    wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

    // Store the output
    wmma::store_matrix_sync(c+warpM*8+warpN*16*8, c_fragment, 16, wmma::mem_row_major);
  }
}

/* Copy data from host to device, perform the operation, and copy result back to
   host. */
void wmma_init_run_64 (double *h_a, double *h_b, double *h_c,
                    double *d_a, double *d_b, double *d_c,
                    bool init) {

  // Copy input from host to device.
  cudaMemcpy(d_a, h_a, 16*16*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, 16*16*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, 16*16*sizeof(double), cudaMemcpyHostToDevice);

  dim3 blockDim;
  blockDim.x = 64;
  blockDim.y = 2;
  // Perform matrix multiplication.
  wmma_ker_64<<<1,blockDim>>>(d_a, d_b, d_c, init);

  // Copy result from device to host.
  cudaMemcpy(h_c, d_c, 16*16*sizeof(double), cudaMemcpyDeviceToHost);
}


template <typename inputtype, typename returntype>
__global__ void simple_wmma_16(inputtype *a, inputtype *b, returntype *c, returntype *d, int m_ld, int n_ld, int k_ld, float alpha, float beta)
{
   // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

   // Tile using a 2D grid
  //  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, M, N, K, inputtype, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, M, N, K, inputtype, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, M, N, K, returntype> acc_frag;
   wmma::fragment<wmma::accumulator, M, N, K, returntype> c_frag;

   wmma::fill_fragment(acc_frag, returntype(0.0));

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
         wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
        //  wmma::load_matrix_sync(,c_frag c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        //  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        // wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cCol = warpN * N;
   int cRow = warpM * M;

   if (cRow < m_ld && cCol < n_ld) {
      wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
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
  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (M_GLOBAL + (M * blockDim.x / 32 - 1)) / (M * blockDim.x / 32);
  gridDim.y = (N_GLOBAL + N * blockDim.y - 1) / (N * blockDim.y);

  cudaMemcpy(d_a, h_a, M_GLOBAL*K_GLOBAL*sizeof(inputtype), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N_GLOBAL*K_GLOBAL*sizeof(inputtype), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, M_GLOBAL*N_GLOBAL*sizeof(returntype), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, h_d, M_GLOBAL*N_GLOBAL*sizeof(returntype), cudaMemcpyHostToDevice);


  printf("Computing... using simple_wmma_gemm kernel\n");
  simple_wmma_16<<<gridDim, blockDim>>>(d_a, d_b, d_c, d_d, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);

  // Copy result from device to host.
  cudaMemcpy(h_d, d_d, M_GLOBAL*N_GLOBAL*sizeof(returntype), cudaMemcpyDeviceToHost);
}


       
