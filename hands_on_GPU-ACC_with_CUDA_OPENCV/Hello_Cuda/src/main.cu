// CUDA hello world
// Author: Yuting Xie
// 2022.2.9

#include <stdio.h>

__global__ void my_kernel(void) {

}

int main(void) {
	// 1 block with 1 thread per block
	my_kernel <<<1, 1>>> ();
	printf("Hello, CUDA\n");
	return 0;
}
