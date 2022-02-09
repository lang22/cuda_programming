// Examine the order of blocks and threads and a simple synchronization
// Author: Yuting Xie
// 2022.2.9

#include <stdio.h>

__global__ void kernel(void) {
	printf("This is from the thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main(void) {
	// 16 blocks, 3 thread per block
	kernel<<<16, 3>>>();
	// Wait for all kernels to finish!
	cudaDeviceSynchronize();
	printf("All threads finished\n");
	return 0;
}

/* An example of output (blocks are OOO but threads in a block is ordered):
This is from the thread 0 in block 15
This is from the thread 1 in block 15
This is from the thread 2 in block 15
This is from the thread 0 in block 8
This is from the thread 1 in block 8
This is from the thread 2 in block 8
This is from the thread 0 in block 1
This is from the thread 1 in block 1
This is from the thread 2 in block 1
This is from the thread 0 in block 9
This is from the thread 1 in block 9
This is from the thread 2 in block 9
This is from the thread 0 in block 2
This is from the thread 1 in block 2
This is from the thread 2 in block 2
This is from the thread 0 in block 10
This is from the thread 1 in block 10
This is from the thread 2 in block 10
This is from the thread 0 in block 3
This is from the thread 1 in block 3
This is from the thread 2 in block 3
This is from the thread 0 in block 14
This is from the thread 1 in block 14
This is from the thread 2 in block 14
This is from the thread 0 in block 7
This is from the thread 1 in block 7
This is from the thread 2 in block 7
This is from the thread 0 in block 13
This is from the thread 1 in block 13
This is from the thread 2 in block 13
This is from the thread 0 in block 6
This is from the thread 1 in block 6
This is from the thread 2 in block 6
This is from the thread 0 in block 11
This is from the thread 1 in block 11
This is from the thread 2 in block 11
This is from the thread 0 in block 4
This is from the thread 1 in block 4
This is from the thread 2 in block 4
This is from the thread 0 in block 12
This is from the thread 1 in block 12
This is from the thread 2 in block 12
This is from the thread 0 in block 5
This is from the thread 1 in block 5
This is from the thread 2 in block 5
This is from the thread 0 in block 0
This is from the thread 1 in block 0
This is from the thread 2 in block 0
All threads finished
*/
