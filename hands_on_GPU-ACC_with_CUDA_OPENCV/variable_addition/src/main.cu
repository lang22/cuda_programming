// CUDA adds two variable with kernel function
// Author: Yuting Xie
// 2022.2.9

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


using std::cin;
using std::cout;

// this is a kernel function so d_res is a device pointer
__global__ void addTwoVariable(int a, int b, int *d_res) {
	*d_res = a + b;
}

__global__ void addTwoVariableByPtr(int *d_a, int *d_b, int *d_res) {
	*d_res = *d_a + *d_b;
}

int main(void) {
	int a, b, res;
	int *d_res;
	cin >> a >> b;

	/* Add by value */
	// Allocate memory for result, had to in-place modify the value of d_res, so pass as void**
	cudaMalloc((void**)&d_res, sizeof(int));
	// Kernel call
	addTwoVariable<<<1, 1>>>(a, b, d_res);
	// Copy the result out from device
	cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_res);
	cout << res << "\n";

	/* Add by ptr */
	// Allocate memory for operands and result
	int *d_a, *d_b;
	cudaMalloc((void**)&d_a, sizeof(int));
	cudaMalloc((void**)&d_b, sizeof(int));
	cudaMalloc((void**)&d_res, sizeof(int));
	// Copy operand values
	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
	// Kernel call
	addTwoVariableByPtr<<<1, 1>>>(d_a, d_b, d_res);
	// Copy result from device to host
	cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);

	cout << res << "\n";


	return 0;

}

