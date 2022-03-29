// Optimize the reduction in GPU
// Yuting Xie
// 2022.3.28

#include <stdio.h>
#include <stdlib.h>

#define BlockSize (128)
// For implementation convenience, just keep NumBlock <= BlockSize
#define NumBlock (64)
#define N ((BlockSize) * (NumBlock))

// The most naive implementation: Tree-based reduction
__global__ void reduce0(int *nums, int *res) {
	__shared__ int my_part[BlockSize];
	int bid = blockIdx.x, tid = threadIdx.x;

	// Step1: copy data from GPU global memory to block-shared memory
	my_part[tid] = nums[bid * blockDim.x + tid];
	__syncthreads(); // Wait for every thread finish copying

	// Step2: do tree-based reduction
	for (int interval = 1; interval <= blockDim.x / 2; interval *= 2) {
		if (tid % (interval * 2) == 0) {
			my_part[tid] += my_part[tid + interval];
		}
		__syncthreads();
	}

	if (tid == 0) {
		res[bid] = my_part[0];
	}
}

// Optimized with stride index to solve warp divergence
__global__ void reduce1(int *nums, int *res) {
	__shared__ int my_part[BlockSize];
	int bid = blockIdx.x, tid = threadIdx.x;

	// Step1: copy data from GPU global memory to block-shared memory
	my_part[tid] = nums[bid * blockDim.x + tid];
	__syncthreads(); // Wait for every thread finish copying

	// Step2: do stride-index reduction, solve the Warp Divergence problem!
	for (int stride = 1; stride <= blockDim.x / 2; stride <<= 1) {
		int idx = 2 * stride * tid; // idx is the pos which thread_idx is responsible for
		if (idx < blockDim.x) {
			my_part[idx] += my_part[idx + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		res[bid] = my_part[0];
	}
}

// Optimized with inverse stride index to solve bank conflict
__global__ void reduce2(int *nums, int *res) {
	__shared__ int my_part[BlockSize];
	int bid = blockIdx.x, tid = threadIdx.x;

	// Step1: copy data from GPU global memory to block-shared memory
	my_part[tid] = nums[bid * blockDim.x + tid];
	__syncthreads(); // Wait for every thread finish copying

	// Step2: do stride-index reduction from large to small, solve bank conflict!
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid + stride < blockDim.x) {
			my_part[tid] += my_part[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		res[bid] = my_part[0];
	}
}

// Optimized with reducing idle threads to squeeze GPU labor!
__global__ void reduce3(int *nums, int *res) {
	__shared__ int my_part[BlockSize];
	int bid = blockIdx.x, tid = threadIdx.x;

	// Step1: copy data from GPU global memory to block-shared memory and DO ONE SUM UP!
	int idx = bid * blockDim.x + tid;
	my_part[tid] = nums[idx] + nums[idx + blockDim.x * gridDim.x];
	__syncthreads(); // Wait for every thread finish copying

	// Step2: do stride-index reduction from large to small, solve bank conflict!
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid + stride < blockDim.x) {
			my_part[tid] += my_part[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		res[bid] = my_part[0];
	}
}

// When there are only 32 (exactly one warp) threads left to work, no need to sync cuz they are inherently synced.
__device__ void last_warp_reduce(volatile int *my_part, int tid) {
	// Need volatile to avoid instruction rearrangement!
	my_part[tid] += my_part[tid + 32];
	my_part[tid] += my_part[tid + 16];
	my_part[tid] += my_part[tid + 8];
	my_part[tid] += my_part[tid + 4];
	my_part[tid] += my_part[tid + 2];
	my_part[tid] += my_part[tid + 1];
}
// Optimized by unfolding the last loops!
__global__ void reduce4(int *nums, int *res) {
	__shared__ int my_part[BlockSize];
	int bid = blockIdx.x, tid = threadIdx.x;

	// Step1: copy data from GPU global memory to block-shared memory and DO ONE SUM UP!
	int idx = bid * blockDim.x + tid;
	my_part[tid] = nums[idx] + nums[idx + blockDim.x * gridDim.x];
	__syncthreads(); // Wait for every thread finish copying

	// Step2: do stride-index reduction from large to small until there are <= one warp of threads!
	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
		if (tid + stride < blockDim.x) {
			my_part[tid] += my_part[tid + stride];
		}
		__syncthreads();
	}

	// Step3: unfold the last loops w/i a warp
	last_warp_reduce(my_part, tid);
	if (tid == 0) {
		res[bid] = my_part[0];
	}
}

int main() {
	int nums[N];
	for (int i = 0; i < N; i++) {
		nums[i] = 1;
	}

	int res[NumBlock];
	int *nums_d, *res_d;
	cudaMalloc((void**)&nums_d, sizeof(int) * N);
	cudaMalloc((void**)&res_d, sizeof(int) * NumBlock);
	cudaMemcpy(nums_d, nums, sizeof(int) * N, cudaMemcpyHostToDevice);

	// For reduce1 and reduce2
//	reduce2<<<NumBlock, BlockSize>>>(nums_d, res_d);
//	cudaDeviceSynchronize(); // Wait the first reduction to finish
//	reduce2<<<1, NumBlock>>>(res_d, res_d);

	// For reduce3 and reduce4
	reduce4<<<NumBlock, BlockSize / 2>>>(nums_d, res_d);
	cudaDeviceSynchronize();
	reduce4<<<1, NumBlock / 2>>>(res_d, res_d);

	cudaMemcpy(&res[0], &res_d[0], sizeof(int), cudaMemcpyDeviceToHost);

	printf("Sum of array is: %d\n", res[0]);

	cudaFree(nums_d);
	cudaFree(res_d);
	return 0;
}


