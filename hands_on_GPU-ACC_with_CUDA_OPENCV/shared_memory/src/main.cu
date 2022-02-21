// Use shared memoy w/i a block to calculate accumulative average
// Author: Yuting Xie
// 2022.2.21

#include <stdio.h>

#define N 200

__global__ void kernel_acc_avg(float *data) {
	int i;
	int idx = threadIdx.x;

	float sum = 0.f;

	// Shared memory
	__shared__ float shm_array[N];

	// Copy data from global memory to shared memory
	shm_array[idx] = data[idx];

	// Sync threads to ensure all shm writes have completed
	__syncthreads();

	for (i = 0; i <= idx; i++) {
		sum += shm_array[i];
	}

	// Writes to global memory, this wont cause contention
	data[idx] = sum / (idx + 1);
	// printf("In %d, writes %f\n", idx, data[idx]);
}

int main(void) {
	float data[N];
	for (int i = 0; i < N; ++i) {
		data[i] = i + 1.f;
	}

	float *d_data;
	cudaMalloc((void**)&d_data, N * sizeof(float));
	cudaMemcpy(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice);

	kernel_acc_avg<<<1, N>>>(d_data);

	cudaMemcpy(data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_data);
	for (int i = 0; i < N; ++i) {
		printf("%.2f, ", data[i]);
	}
	return 0;
}



