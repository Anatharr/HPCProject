#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void cuda_hello(int id){
	printf("Hello World du GPU %d\n", id);
}

int main(void) {
	printf("Hello World du CPU\n");
    for (int i=0;i<10;i++) {
	    cuda_hello<<<2,5>>>(i);
    }
    cudaDeviceSynchronize();
	return EXIT_SUCCESS;
}