// CUDA streams for vector addition
// Author: Yuting Xie
// 2022.2.22

#include <stdio.h>

#define N (15000)

__global__ void vector_add(int *v1, int *v2, int *res) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N) {
		res[tid] = v1[tid] + v2[tid];
	}
}

int main(void) {
	int v1[2 * N], v2[2 * N], res[2 * N];
	for (int i = 0; i < 2 * N; ++i) {
		v1[i] = v2[i] = i;
	}

	int *d_v1, *d_v2, *d_res;
	cudaMalloc((void**)&d_v1, 2 * N * sizeof(int));
	cudaMalloc((void**)&d_v2, 2 * N * sizeof(int));
	cudaMalloc((void**)&d_res, 2 * N * sizeof(int));

	// Create CUDA streams
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	// Copy memory with streams use async version
	cudaMemcpyAsync(d_v1, v1, N * sizeof(int), cudaMemcpyHostToDevice, stream0); // This async operation assigned to stream0
	cudaMemcpyAsync(d_v2, v2, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(d_v1 + N, v1 + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1); // This async operation assigned to stream1
	cudaMemcpyAsync(d_v2 + N, v2 + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);

	// Complete kernel call form <<<Block, Threads, Shm_size, Stream>>>
	vector_add<<<(N + 127) / 128, 128, 0, stream0>>>(d_v1, d_v2, d_res);
	vector_add<<<(N + 127) / 128, 128, 0, stream1>>>(d_v1 + N, d_v2 + N, d_res + N);

	// Collect results
	cudaMemcpyAsync(res, d_res, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);
	cudaMemcpyAsync(res + N, d_res + N, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);

	// Wait all streams to finish their CE and KE jobs
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	for (int i = 0; i < 20; ++i) {
		printf("%d, ", res[i]);
	}

	return 0;
}
