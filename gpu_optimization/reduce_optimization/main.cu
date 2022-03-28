// Optimize the reduction in GPU
// Yuting Xie
// 2022.3.28

#include <stdio.h>
#include <stdlib.h>

#define BlockSize (64)
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
		printf("%d, %d writing %d\n", blockIdx.x, threadIdx.x, my_part[0]);
	}
}

// Optimized warp divergence with stride index
__global__ void reduce1(int *nums, int *res) {
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
		printf("%d, %d writing %d\n", blockIdx.x, threadIdx.x, my_part[0]);
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

	reduce0<<<NumBlock, BlockSize>>>(nums_d, res_d);
	cudaDeviceSynchronize(); // Wait the first reduction to finish
	reduce0<<<1, NumBlock>>>(res_d, res_d);

	cudaMemcpy(&res[0], &res_d[0], sizeof(int), cudaMemcpyDeviceToHost);

	printf("Sum of array is: %d\n", res[0]);

	cudaFree(nums_d);
	cudaFree(res_d);
	return 0;
}


