// Use contant memory which is GPU-wise
// Author: Yuting Xie
// 2022.2.21

#include <stdio.h>

#define N (5)

// This two lives in constant memory on GPU
__constant__ int cn1;
__constant__ int cn2;
__constant__ int c_array[2] = {3, 7};

__global__ void kernel_constant_memory(int *data) {
	int idx = threadIdx.x;
	data[idx] = cn1 * data[idx] + cn2 + c_array[0] * data[idx] + c_array[1];
}

int main(void) {
	int data[N];
	for (int i = 0; i < N; ++i) {
		data[i] = i;
	}

	int *d_data;
	cudaMalloc((void**)&d_data, N * sizeof(int));
	cudaMemcpy(d_data, data, N * sizeof(int), cudaMemcpyHostToDevice);

	// Set constant memory
	int n1 = 3, n2 = 7, array[2] = {4, 5};
	cudaMemcpyToSymbol(cn1, &n1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cn2, &n2, sizeof(int), 0, cudaMemcpyHostToDevice);
	// cudaMemcpy(c_array, array, 2 * sizeof(int), cudaMemcpyHostToDevice); // This line not work

	kernel_constant_memory<<<1, N>>>(d_data);

	cudaMemcpy(data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_data);

	for (int i = 0; i < N; ++i) {
		printf("%d, ", data[i]);
	}

	return 0;
}
