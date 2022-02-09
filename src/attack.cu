#include <stdio.h>
#define N 10000000

__global__ void vector_add(float *out, float *a, float *b, int n)
{   
    for (int i = 0; i < n; i++)
    {
        out[i] = a[i] + b[i];
    }
}

int main()
{
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    // Allocate memory
    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);
    c = (float *)malloc(sizeof(float) * N);

    // Initialize array
    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory for arrays
    cudaMalloc((void **)&d_a, sizeof(float) * N);
    cudaMalloc((void **)&d_b, sizeof(float) * N);
    cudaMalloc((void **)&d_c, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, sizeof(float) * N, cudaMemcpyHostToDevice);

    printf("%f\n", d_a[0]);
    // Main function
    vector_add<<<1, 1>>>(d_c, d_a, d_b, N);
    printf("\n[x] %f + %f = %f\n", d_a[0], d_b[0], d_c[0]);

    // Cleanup after kernel execution
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return EXIT_SUCCESS;
}
