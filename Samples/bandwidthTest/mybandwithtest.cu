
#include <cuda_runtime.h>

// includes
#include <helper_cuda.h> // helper functions for CUDA error checking and initialization
#include <helper_functions.h> // helper for shared functions common to CUDA Samples

#include <cuda.h>

#include <cassert>
#include <iostream>
#include <memory>
using namespace std;

#define SHMOO_LIMIT_32MB (32 * 1e6) // 32 MB
#define MEMCOPY_ITERATIONS 1000

enum memoryMode { PINNED, PAGEABLE };
float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode,
                               bool wc);

int main() {
  float bd = testDeviceToHostTransfer(SHMOO_LIMIT_32MB, PINNED, false);
  cout << bd << '\n';
}

float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode,
                               bool wc) {
  StopWatchInterface *timer = NULL;
  float elapsedTimeInMs = 0.0f;
  float bandwidthInGBs = 0.0f;
  unsigned char *h_idata = NULL;
  unsigned char *h_odata = NULL;
  cudaEvent_t start, stop;
  const int deviceCount = 1;

  sdkCreateTimer(&timer);
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // allocate host memory
  // pinned memory mode - use special function to get OS-pinned memory
  checkCudaErrors(cudaHostAlloc((void **)&h_idata, memSize,
                                (wc) ? cudaHostAllocWriteCombined : 0));
  checkCudaErrors(cudaHostAlloc((void **)&h_odata, memSize,
                                (wc) ? cudaHostAllocWriteCombined : 0));

  // initialize the memory
  for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
    h_idata[i] = (unsigned char)(i & 0xff);
  }

  // allocate device memory
  unsigned char *d_idata[deviceCount];
  cudaStream_t cudaStreamHandle[deviceCount];
  for (int d = 0; d < deviceCount; d++) {
    cudaSetDevice(d);
    cudaStreamCreate(&cudaStreamHandle[d]);
    checkCudaErrors(cudaMalloc((void **)&(d_idata[d]), memSize));

    // initialize the device memory
    checkCudaErrors(
        cudaMemcpy(d_idata[d], h_idata, memSize, cudaMemcpyHostToDevice));
  }
  // copy data from GPU to Host
  checkCudaErrors(cudaEventRecord(start, cudaStreamHandle[0]));
  for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
    for (int d = 0; d < deviceCount; d++) {
      cudaSetDevice(d);
      checkCudaErrors(cudaMemcpyAsync(h_odata, d_idata[d], memSize,
                                      cudaMemcpyDeviceToHost,
                                      cudaStreamHandle[d]));
    }
  }
  checkCudaErrors(cudaEventRecord(stop, cudaStreamHandle[0]));
  for (int d = 0; d < deviceCount; d++) {
    cudaSetDevice(d);
    checkCudaErrors(cudaStreamSynchronize(cudaStreamHandle[d]));
  }
  checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

  // calculate bandwidth in GB/s
  double time_s = elapsedTimeInMs / 1e3;
  bandwidthInGBs = (memSize * (float)MEMCOPY_ITERATIONS) / (double)1e9;
  bandwidthInGBs = bandwidthInGBs / time_s;
  // clean up memory
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaEventDestroy(start));
  sdkDeleteTimer(&timer);

  checkCudaErrors(cudaFreeHost(h_idata));
  checkCudaErrors(cudaFreeHost(h_odata));

  for (int d = 0; d < deviceCount; d++) {
    cudaSetDevice(d);
    checkCudaErrors(cudaFree(d_idata[d]));
  }

  return bandwidthInGBs;
}
