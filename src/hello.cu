#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

// __global__ void cuda_hello(int id){
// 	printf("Hello World du GPU %d\n", id);
// }

int main(int argc, char **argv) {
	if (argc<2) {
		printf("Usage: %s <file>", argv[0]);
		exit(EXIT_FAILURE);
	}

	FILE *shadow_fd = fopen(argv[1], "r");
	char shadow_db[5000][50];

	if (shadow_fd == NULL)
		exit(EXIT_FAILURE);

	int line = 0;

	while ((fgets(shadow_db[line], 50, shadow_fd)) != NULL)
	{
		shadow_db[line][strlen(shadow_db[line]) - 1] = '\0';
		printf("address:%p -> %s\n", shadow_db[line], shadow_db[line]);
		line++;
	}

	printf("[DEBUG] Shadow content - head : \n");
	for (int i = 0; i < 10; i++)
	{
		printf("[%i] : %s\n", i, shadow_db[i]);
	}

	// printf("Hello World du CPU\n");
    // for (int i=0;i<10;i++) {
	//     cuda_hello<<<2,5>>>(i);
    // }
    // cudaDeviceSynchronize();
	return EXIT_SUCCESS;
}