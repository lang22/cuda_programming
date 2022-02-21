#include <stdio.h>

#define N (10)

__global__ void kernel_non_atomic_add(int *data) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	data[idx % N]++;
}

__global__ void kernel_atomic_add(int *data) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd(data + (idx % N), 1); // Atomic addition
}

int main(void) {
	int data[N];
	for (int i = 0; i < N; ++i) {
		data[i] = 0;
	}

	int *d_data;
	cudaMalloc((void**)&d_data, N * sizeof(int));
	cudaMemcpy(d_data, data, N * sizeof(int), cudaMemcpyHostToDevice);

	kernel_atomic_add<<<100, 100 * N>>>(d_data); // Each data[i] should be increased by 100 * 100 = 10000
	// kernel_non_atomic_add<<<100, 100 * N>>>(d_data); // Each data[i] should be increased by 100 * 100 = 10000

	cudaMemcpy(data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_data);

	for (int i = 0; i < N; ++i) {
		printf("data[%d] has been increased by %d times\n", i, data[i]);
	}

	return 0;
}

