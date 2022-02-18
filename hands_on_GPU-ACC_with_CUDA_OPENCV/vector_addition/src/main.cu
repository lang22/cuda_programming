// Vector addition on GPU
// Author: Yuting Xie
// 2022.2.18

#include <stdio.h>
#include <cstring>

#define NB 4
#define TPB 500
#define N (NB * TPB)

__global__ void kernel_vec_add(int *v1, int *v2, int *res) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = bid * TPB + tid;

	res[idx] = v1[idx] + v2[idx];
}

int main(void) {
	// Allocate operands on host
	int v1[N], v2[N], res[N];
	memset(v1, 1, N * sizeof(int));
	memset(v2, 2, N * sizeof(int));

	// Allocate operands on device
	int *v1_d, *v2_d, *res_d;
	cudaMalloc((void**)&v1_d, N * sizeof(int));
	cudaMalloc((void**)&v2_d, N * sizeof(int));
	cudaMalloc((void**)&res_d, N * sizeof(int));

	// Copy operands to device
	cudaMemcpy(v1_d, v1, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(v2_d, v2, N * sizeof(int), cudaMemcpyHostToDevice);

	// Execute kernel
	kernel_vec_add<<<NB, TPB>>>(v1_d, v2_d, res_d);

	// Colloct result from device
	cudaMemcpy(res, res_d, N * sizeof(int), cudaMemcpyDeviceToHost);

	// Display result
	for (int i = 0; i < 10; ++i) {
		printf("%d, ", res[i]);
	}
	printf("\n");

	return 0;
}

/* Expected output for memset v1 to 1s and v2 to 2s:
 * 50529027, 50529027, 50529027, 50529027, 50529027, 50529027, 50529027, 50529027, 50529027, 50529027,
 */
