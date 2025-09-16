#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__
void kernel(int *flag, int *a, size_t b)
{
    *flag = 0x1;

}


int main()
{
    
    int *input = (int *)malloc(4096);

    kernel<<<1,1>>>(input, NULL, 0);
    cudaDeviceSynchronize();

    free(input);
    
    return 0;
} 

