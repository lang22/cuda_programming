// Begin follow tutorial on http://cuda-programming.blogspot.com/
// A first cuda program
// Author: Yuting Xie
// 2022.2.21

#include <stdio.h>
#include <cmath>
#include <algorithm>

#define N (1 << 20)

__global__
void saxpy(float a, float *x, float *y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		y[idx] = a * x[idx] + y[idx];
	}
}

int main(void) {
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N * sizeof(float));
	y = (float*)malloc(N * sizeof(float));
	for (int i = 0; i < N; ++i) {
		x[i] = 1.f;
		y[i] = 2.f;
	}

	cudaMalloc((void**)&d_x, N * sizeof(float));
	cudaMalloc((void**)&d_y, N * sizeof(float));
	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

	saxpy<<<(N + 1023) / 1024, 1024>>>(2.f, d_x, d_y);

	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	float maxErr = 0.f;
	for (int i = 0; i < N; ++i) {
		maxErr = std::max(maxErr, std::fabs(4.f - y[i]));
	}
	printf("Max error is %.6f\n", maxErr);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);

	return 0;
}
