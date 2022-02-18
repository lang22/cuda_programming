// Profile and compare large vector addition on GPU and CPU
// Author: Yuting Xie
// 2022.2.18

#include <stdio.h>
#include <chrono>

#define N 500000

std::chrono::time_point<std::chrono::system_clock> t1;
std::chrono::time_point<std::chrono::system_clock> t2;

inline void time_checkpoint(bool start = false) {
	if (start) {
		t1 = std::chrono::system_clock::now();
		return;
	}
	t2 = std::chrono::system_clock::now();
	auto duration = t2 - t1;
	printf("%ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
	t1 = std::chrono::system_clock::now();
}

__global__ void kernel(int *v1, int *v2, int *res) {
	int idx = threadIdx.x;
	while (idx < N) {
		res[idx] = v1[idx] + v2[idx];
		idx += blockDim.x * gridDim.x;
	}
}

int main(void) {
	int v1[N], v2[N], res[N];
	memset(v1, 1, N * sizeof(int));
	memset(v2, 2, N * sizeof(int));

	int *v1_d, *v2_d, *res_d;
	cudaMalloc((void**)&v1_d, N * sizeof(int));
	cudaMalloc((void**)&v2_d, N * sizeof(int));
	cudaMalloc((void**)&res_d, N * sizeof(int));

	time_checkpoint(true);

	cudaMemcpy(v1_d, v1, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(v2_d, v2, N * sizeof(int), cudaMemcpyHostToDevice);
	kernel<<<(N + 511) / 512, 512>>>(v1_d, v2_d, res_d);
	cudaMemcpy(res, res_d, N * sizeof(int), cudaMemcpyDeviceToHost);

	time_checkpoint();

	for (int i = 0; i < 10; ++i) {
		printf("%d, ", res[i]);
	}
	printf("\n");

	// Do it on CPU and compare
	time_checkpoint(true);
	for (int i = 0; i < N; i++) {
		res[i] = v1[i] + v2[i];
	}
	time_checkpoint();

	return 0;
}
