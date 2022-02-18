// Profile the async-running of GPU and CPU codes
// Author: Yuting Xie
// 2022.2.18

#include <stdio.h>
#include <cuda.h>
#include <chrono>

__global__ void kernel_print(void) {
	printf("Hello from %d, %d\n", blockIdx.x, threadIdx.x);
}

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


int main(void) {
	time_checkpoint(true);
	kernel_print<<<16, 4>>>();
	time_checkpoint();
	for(int i = 0; i < 10000; ++i) {
		if (i % 100 == 0) {
			printf("\t\tHello in main thread\n");
		}
	}
	time_checkpoint();
	cudaDeviceSynchronize(); // w/o this line, nothing from kernel_print would be printed!
	time_checkpoint();
	printf("All finished\n");
	return 0;
}
