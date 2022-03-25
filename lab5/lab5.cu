#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>

using namespace std;

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

__global__ void kernel_copy_data(int* gpu_data, int* gpu_histogramm, int* gpu_out_data, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetX = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += offsetX) {
        int index = atomicAdd(gpu_data[i] + gpu_histogramm, -1) - 1;
        gpu_out_data[index] = gpu_data[i];
    }
}

__global__ void kernel_count_histogramm(int* gpu_data, int* gpu_histogramm, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetX = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += offsetX) {
        atomicAdd(gpu_histogramm + gpu_data[idx], 1);
    }
}

int main() {
    int size;
    fread(&size, sizeof(int), 1, stdin);

    if (size == 0) {
        return 0;
    }

    int* data = (int*) malloc(sizeof(int) * size);
    fread(data, sizeof(int), size, stdin);

    int max_value = 0;

    for (int i = 0; i < size; i++) {
        if (data[i] > max_value) {
            max_value = data[i];
        }
    }

    int* gpu_data = NULL;
    CSC(cudaMalloc(&gpu_data, size * sizeof(int)));
    CSC(cudaMemcpy(gpu_data, data, size * sizeof(int), cudaMemcpyHostToDevice));

    int* gpu_out_data = NULL;
    CSC(cudaMalloc(&gpu_out_data, size * sizeof(int)));

    int* gpu_histogramm = NULL;
    CSC(cudaMalloc(&gpu_histogramm, sizeof(int) * (max_value + 1)));
    CSC(cudaMemset(gpu_histogramm, 0, sizeof(int) * (max_value + 1)));

    kernel_count_histogramm<<<32, 128>>>(gpu_histogramm, gpu_data, size);
    CSC(cudaGetLastError());

    thrust::device_ptr<int> ptr = thrust::device_pointer_cast(gpu_histogramm);
    thrust::inclusive_scan(ptr, ptr + max_value + 1, ptr);

    kernel_copy_data<<<32, 128>>>(gpu_data, gpu_histogramm, gpu_out_data, size);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, gpu_out_data, sizeof(int) * size, cudaMemcpyDeviceToHost));

    fwrite(data, sizeof(int), size, stdout);

    CSC(cudaFree(gpu_data));
    CSC(cudaFree(gpu_histogramm));
    CSC(cudaFree(gpu_out_data));
    free(data);

    return 0;
}