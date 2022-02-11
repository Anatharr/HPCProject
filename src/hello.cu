#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

__device__ char *dev_plain;

__global__ void cuda_hello(){
	int length = 5;
	dev_plain = new char[length];
	dev_plain[0] = 'z';
	dev_plain[1] = 'g';
	dev_plain[2] = 'e';
	dev_plain[3] = 'g';
	dev_plain[4] = '\0';
	printf("GPU %s\n", dev_plain);
	printf("Hello World du GPU\n");
}

int main(int argc, char **argv) {

	printf("Hello World du CPU\n");

	char *d_plain = NULL;
	char plain[10];
	
	cuda_hello<<<1,1>>>();

    cudaDeviceSynchronize();

	printf("%p %p %p\n", plain, d_plain, &d_plain);
	cudaMemcpyFromSymbol(&d_plain, "dev_plain", sizeof(d_plain), 0, cudaMemcpyDeviceToHost);
	printf("%p %p %p\n", plain, d_plain, &d_plain);
	cudaMemcpy(plain, d_plain, 10*sizeof(char), cudaMemcpyDeviceToHost);

	// printf("ERROR %s : %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
	printf("plain %x %x %x %s\n", plain[0], plain[1], plain[2], plain);

	return EXIT_SUCCESS;
}