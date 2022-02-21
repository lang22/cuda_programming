// Use local memory on GPU within a thread
// Author: Yuting Xie
// 2022.2.18

#include <stdio.h>

__global__ void kernel_local_memory(int n) {
	// t_local is exclusive w/i the thread, in register.
	int t_local;
	t_local = n * threadIdx.x;
	printf("Local variable: %d\n", t_local);
}

int main(void) {
	kernel_local_memory<<<1, 10>>>(2);
	cudaDeviceSynchronize();
	return 0;
}
