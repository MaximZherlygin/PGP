#include <iostream>
#include <stdio.h>
#include <limits>
#include <cfloat>
#include <math.h>
#include <time.h>
#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

__global__ void kernel_calculate_histogramm(float* gpu_data, int n, int* result_data, float min, float max, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += offsetX) {
        int val = (int) ((count - 1) * (gpu_data[i] - min) / (max - min));
        atomicAdd(&(result_data[val]), 1);
    }
}

__global__ void kernel_split_histogramm(float* gpu_data, int n, float* split_data,
                                        int* first, unsigned int* size, float min,
                                        float max, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += offsetX) {
        int val = first[((count - 1) * (gpu_data[i] - min) / (max - min))] +
                atomicAdd(&(size[((count - 1) * (gpu_data[i] - min) / (max - min) * )]), 1)
        split_data[val] = gpu_data[i];
    }
}

__global__ void kernel_scan(int* data, int n, int* next, int* res) {
    __shared__ int shared_data[((2 * 32) >> 5) + 2 * 32];


    int idx = threadIdx.x * 2;
    int first_idx = threadIdx.x;
    int second_idx = threadIdx.x + (n / 2);
    int first_val = first_idx >> 5;
    int second_val = second_idx >> 5;

    shared_data[first_val + first_idx] = data[32 * blockIdx.x * 2 + first_idx];
    shared_data[second_val + second_idx] = data[32 * blockIdx.x * 2 + second_idx];
    int temp = 1;
    for (int i = n >> 1; i > 0; i >>= 1) {

        __syncthreads();

        if (i > threadIdx.x) {
            shared_data[(idx + 2) * temp - 1 + (((idx + 2) * temp - 1) >> 5)] +=
                    shared_data[(idx + 1) * temp - 1 + (((idx + 1) * temp - 1) >> 5)];
        }

        temp <<= 1;
    }

    if (threadIdx.x == 0) {
        next[blockIdx.x] = shared_data[n + ((n - 1) >> 5) - 1];
        shared_data[n + ((n - 1) >> 5) - 1] = 0;
    }

    for (int i = 1; i < n; i <<= 1) {
        temp >>= 1;

        __syncthreads();

        if (i > threadIdx.x) {
            int temp_val = shared_data[(idx + 1) * temp_val - 1 + (((idx + 1) * temp_val - 1) >> 5)];
            shared_data[(idx + 1) * temp_val - 1 + (((idx + 1) * temp_val - 1) >> 5)] =
                    shared_data[(idx + 2) * temp_val - 1 + (((idx + 2) * temp_val - 1) >> 5)];
            shared_data[(idx + 2) * temp_val - 1 + (((idx + 2) * temp_val - 1) >> 5)] += temp_val;
        }
    }

    __syncthreads();

    res[second_idx + 32 * blockIdx.x * 2] = shared_data[second_idx + second_val];
    res[first_idx + 32 * blockIdx.x * 2] = shared_data[first_idx + first_val];
}

__global__ void kernel_dis_scan(int* gpu_data, int* values) {
    int id = blockIdx.x * 32 * 2 + threadIdx.x;
    gpu_data[id] += values[blockIdx.x];
}

__host__ void r_scan(int* gpu_data, int size, int* out_data) {
    int* arr1;
    int* arr2;
    cudaMalloc((void **) &arr1, size / (2 * 32) + 1 * sizeof(int));
    cudaMalloc((void **) &arr2, size / (2 * 32) + 1 * sizeof(int));

    kernel_scan<<<dim3(size / (2 * 32) + 1, 1, 1), dim3(32, 1, 1)>>>(gpu_data, 2 * 32, arr1, out_data);
    if (size < 2 * 32) {
        cudaMemcpy(arr2, arr1, size / (2 * 32) + 1 * sizeof(int), cudaMemcpyDeviceToDevice);
    } else {
        r_scan(arr1, size / (2 * 32) + 1, arr2);
    }

    if (size / (2 * 32) + 1 > 1) {
        kernel_dis_scan<<<dim3(size / (2 * 32), 1, 1), dim3(2 * 32, 1, 1) >>>(out_data + (2 * 32), arr2 + 1);
    }
}

__global__ void kernel_sort23(float* pockets, int size, int* pos, int* sizes) {
    __shared__ float pocket_buff[2048];
    int pocket_size = sizes[blockIdx.x];

    if (pocket_size == -1) {
        return;
    }
    
    int idx = threadIdx.x * 2;

    pocket_buff[idx] = FLT_MAX;
    pocket_buff[idx + 1] = FLT_MAX;

    __syncthreads();

    if (pocket_size > idx) {
        pocket_buff[idx] = pockets[pos[blockIdx.x] + idx];
    }

    if (pocket_size > idx + 1) {
        pocket_buff[idx + 1] = pockets[pos[blockIdx.x] + idx + 1];
    }

    __syncthreads();

    for (int i = 0; i < blockDim.x; i++) {
        if (idx + 1 < 2047) {
            if (pocket_buff[idx + 2] < pocket_buff[idx + 1]) {
                float temp = pocket_buff[idx + 1];
                pocket_buff[idx + 1] = pocket_buff[idx + 2];
                pocket_buff[idx + 2] = temp;
            }
        }

        __syncthreads();

        if (threadIdx.x < 2048) {
            if (pocket_buff[idx + 1] < pocket_buff[idx]) {
                float temp = pocket_buff[idx];
                pocket_buff[idx] = pocket_buff[idx + 1];
                pocket_buff[idx + 1] = temp;
            }
        }
        
        __syncthreads();
    }

    if (pocket_size > idx + 1) {
        pockets[pos[blockIdx.x] + idx + 1] = pocket_buff[idx + 1];
    }
    
    if (pocket_size > idx) {
        pockets[pos[blockIdx.x] + idx] = pocket_buff[idx];
    }
}

__host__ void pocket_sort(float *gpu_data, int size) {
    int count = size / 550 + 1;
    int* s_size;
    int* spl;
    unsigned int* spl_size;
    float* gpu_split;

    thrust::device_ptr<const float> data_device_ptr = thrust::device_pointer_cast(gpu_data);
    auto minmax_elem = thrust::minmax_element(thrust::device, data_device_ptr, data_device_ptr + size);

    float maximum = *minmax_elem.second;
    float minimum = *minmax_elem.first;
    // printf("%f %f",min,max);

    if (fabs(minimum - maximum) < 1e-9) {
        return;
    }

    cudaMalloc((void **) &s_size, sizeof(int) * count);
    cudaMemset(s_size, 0, sizeof(int) * count);

    kernel_calculate_histogramm <<<512, 512>>>(gpu_data, size, s_size, minimum, maximum, count);

    cudaMalloc((void **) &spl, sizeof(int) * count);
    r_scan(s_size, count, spl);

    cudaMalloc((void **) &spl_size, sizeof(unsigned int) * count);
    cudaMemset(spl_size, 0, sizeof(unsigned int) * count);

    cudaMalloc((void **) &gpu_split, sizeof(float) * size);
    kernel_split_histogramm <<<512, 512 >>>(gpu_data, size, gpu_split, spl, spl_size, minimum, maximum, count);

    int count_b = count;

    int* pocket_size = (int*) malloc(sizeof(int) * count_b);
    memset(pocket_size, 0, sizeof(int) * count_b);

    int* pocket = (int*) malloc(sizeof(int) * count_b);

    int id_b = 0;
    for (int id_s = 0; id_s < count; id_s++) {
        int split_1 = 0;
        cudaMemcpy(&split_1, &(spl[id_s]), sizeof(int), cudaMemcpyDeviceToHost);

        int split_1_size = 0;
        cudaMemcpy(&split_1_size, &(s_size[id_s]), sizeof(int), cudaMemcpyDeviceToHost);

        if (split_1_size <= 2048) {
            int buckcur = 2048 - pocket_size[id_b];
            if (split_1_size <= buckcur) {
                if (buckcur == 2048)
                    pocket[id_b] = split_1;
                pocket_size[id_b] += split_1_size;
            } else {
                id_b++;
                pocket[id_b] = split_1;
                pocket_size[id_b] = split_1_size;
            }

        } else {
            id_b++;
            float* temp = &(gpu_split[split_1]);
            pocket_sort(temp, split_1_size);
            pocket_size[id_b] = -1;
            pocket[id_b] = split_1;
            id_b++;
        }
    }
    if (pocket_size[id_b] != 0) {
        count_b = id_b + 1;
    } else {
        count_b = id_b;
    }

    int* gpu_val;
    cudaMalloc((void**) &gpu_val, sizeof(int) * count_b);
    cudaMemcpy(gpu_val, pocket, sizeof(int) * count_b, cudaMemcpyHostToDevice);

    int* val_size;
    cudaMalloc((void**) &val_size, sizeof(int) * count_b);
    cudaMemcpy(val_size, pocket_size, sizeof(int) * count_b, cudaMemcpyHostToDevice);

    kernel_sort23 <<<dim3(count_b, 1, 1), dim3(2048 / 2, 1, 1)>>>(gpu_split, size, gpu_val, val_size);

    cudaMemcpy(gpu_data, gpu_split, size * sizeof(float), cudaMemcpyDeviceToDevice);
}


int main() {

    int size = 0;
    fread(&size, sizeof(int), 1, stdin);

    if (size == 0) {
        return 0;
    }
    float *values = new float[size];
    float *gpu_values;

    fread(values, sizeof(float), size, stdin);

    cudaMalloc((void **)&gpu_values, size * sizeof(float));
    cudaMemcpy(gpu_values, values, size * sizeof(float), cudaMemcpyHostToDevice);

    pocket_sort(gpu_values, size);

    cudaMemcpy(values, gpu_values, size * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(values, sizeof(float), size, stdout);
    return 0;
}

