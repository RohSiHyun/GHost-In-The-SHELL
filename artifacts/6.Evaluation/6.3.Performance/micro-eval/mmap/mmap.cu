#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
// declare the kernel from kernels.cu

#define MAP_HMM (0x1000000000000000)

__global__
void kernel(int *flag, int *a, size_t b)
{
    *flag = 0x1;

}


int main()
{
    // measure begin time
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    int *flag = (int *)mmap(NULL, 4096* sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    clock_gettime(CLOCK_MONOTONIC, &end);    

    int a[4096];

    for(int i = 0; i < 1024; i++){
        flag[i] = 0;
    }

    kernel<<<1, 1>>>(flag, a, sizeof(a));
    cudaDeviceSynchronize();

    if(flag[0] == 0x1){
        printf("kernel success\n");
    } else {
        printf("kernel fail\n");
    }
    printf("time taken: %ld ns\n", (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec));

    return 0;
} 

