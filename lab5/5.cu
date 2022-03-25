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

__global__ void kernel_dis_scan(int* gpu_data, int* values) {
    int id = threadIdx.x + blockIdx.x * 64;
    gpu_data[id] += values[blockIdx.x];
}

__global__ void kernel_scan(int *values, int *split, int *out_values int size,) {
    int ix = threadIdx.x * 2;
    int first_index = threadIdx.x;
    int first_val = (first_index >> 5);
    
    int second_index = (size / 2) + threadIdx.x;
    int second_val = (second_index >> 5);

    __shared__ int shared_data[64 + (64 >> 5)];
    shared_data[first_index + first_val] = values[first_index + 64 * blockIdx.x];
    shared_data[second_index + second_val] = values[second_index + 64 * blockIdx.x];

    int temp = 1;
    for (int idx = size >> 1; idx > 0; idx >>= 1) {
        __syncthreads();
        if (idx > threadIdx.x) {
            int i1 = temp * (ix + 1) - 1 + ((temp * (ix + 1) - 1) >> 5);
            int i2 = temp * (ix + 2) - 1 + ((temp * (ix + 2) - 1) >> 5);
            shared_data[i2] += shared_data[i1];
        }
        temp <<= 1;
    }

    if (first_index == 0) {
        int index = ((size - 1) >> 5) + size - 1;
        split[blockIdx.x] = shared_data[index];
        shared_data[index] = 0;
    }

    for (int idx = 1; idx < size; idx <<= 1) {
        temp >>= 1;
        
        __syncthreads();
        
        if (idx > threadIdx.x) {
            int index1 = ((temp * (ix + 1) - 1) >> 5) + temp * (ix + 1) - 1;
            int index2 = ((temp * (ix + 2) - 1) >> 5) + temp * (ix + 2) - 1;
            int temp = shared_data[index1];
            shared_data[index1] = shared_data[index2];
            shared_data[index2] += temp;
        }
    }
    
    __syncthreads();

    out_values[first_index + blockIdx.x * 64] = shared_data[first_index + first_val];
    out_values[second_index + blockIdx.x * 64] = shared_data[second_index + second_val];
}

__host__ void r_scan(int* gpu_data, int* out_data, int size) {
    int *first_values = NULL;
    int *second_values = NULL;

    cudaMalloc((void **)&first_values, (size / 64 + 1) * sizeof(int));
    cudaMalloc((void **)&second_values, (size / 64 + 1) * sizeof(int));

    kernel_scan<<<dim3(32, 1, 1), dim3(size / 64 + 1, 1, 1)>>>(gpu_data, first_values, out_data, 64);

    if (size < 64) {
        cudaMemcpy(second_values, first_values, size / 64 + 1 * sizeof(int), cudaMemcpyDeviceToDevice);
    } else {
        r_scan(first_values, second_values, size / 64 + 1);
    }

    if (size / 64 + 1 > 1)
        kernel_dis_scan<<<dim3(size / 64, 1, 1), dim3(64, 1, 1)>>>(out_data + 64, second_values + 1);
}

__global__ void kernel_sort23(float* pockets, int* pockets_pos, int* pocket_sizes, int size) {
    __shared__ float pocket_shared_buff[2048];
    int ix = threadIdx.x * 2;

    pocket_shared_buff[ix] = FLT_MAX;
    pocket_shared_buff[ix + 1] = FLT_MAX;

    int current_size = pocket_sizes[blockIdx.x];
    if (current_size == -1)
        return;

    __syncthreads();

    if (current_size > ix) {
        pocket_shared_buff[ix] = pockets[pockets_pos[blockIdx.x] + ix];
    }

    if (current_size > ix + 1) {
        pocket_shared_buff[ix + 1] = pockets[ix + 1 + pockets_pos[blockIdx.x]];
    }

    __syncthreads();

    for (int i = 0; i < blockDim.x; i++) {
        if (ix + 1 < 2047) {
            if (pocket_shared_buff[ix + 2] < pocket_shared_buff[ix + 1]) {
                float temp_value = pocket_shared_buff[ix + 1];
                pocket_shared_buff[ix + 1] = pocket_shared_buff[ix + 2];
                pocket_shared_buff[ix + 2] = temp_value;
            }
        }

        __syncthreads();

        if (threadIdx.x < 2048) {
            if (pocket_shared_buff[ix] > pocket_shared_buff[ix + 1]) {
                float temp_value = pocket_shared_buff[ix];
                pocket_shared_buff[ix] = pocket_shared_buff[ix + 1];
                pocket_shared_buff[ix + 1] = temp_value;
            }
        }

        __syncthreads();
    }

    if (ix < current_size) {
        pockets[ix + pockets_pos[blockIdx.x]] = pocket_shared_buff[ix];
    }

    if (ix + 1 < current_size) {
        pockets[ix + 1 + pockets_pos[blockIdx.x]] = pocket_shared_buff[ix + 1];
    }
}

__global__ void kernel_calculate_histogramm(float *gpu_data, int *result_data, float minimum,
                                            float maximum, int size, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    for (int i = idx; i < size; i += offsetX) {
        atomicAdd(&(result_data[(int) ((gpu_data[i] - minimum) / (maximum - minimum) * (count - 1))]), 1);
    }
}

__global__ void kernel_split_histogramm(float* gpu_data, float* split_data,
                                        float minimum, float maximum,
                                        int* first, unsigned int *sizes,
                                        int size, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += offsetX) {
        int id = ((gpu_data[i] - minimum) / (maximum - minimum) * (count - 1));
        split_data[first[id] + atomicAdd(&(sizes[id]), 1)] = gpu_data[i];
    }
}

__host__ void pocket_sort(int size, float* gpu_data) {
    int count = size / 550 + 1;
    thrust::device_ptr<const float> ptr = thrust::device_pointer_cast(gpu_data);
    auto minmax_values = thrust::minmax_element(thrust::device, ptr, ptr + size);

    float minimum = *minmax_values.first;
    float maximum = *minmax_values.second;

    if (fabs(maximum - minimum) < 1e-9) {
        return;
    }

    int* s_size = NULL;
    float* gpu_val = NULL;
    unsigned int* s_sizes = NULL;
    int* spl_values = NULL;
    int* gpu_v = NULL;
    int* v_size = NULL;

    cudaMalloc((void **) &s_size, count * sizeof(int));
    cudaMemset(s_size, 0, count * sizeof(int));

    kernel_calculate_histogramm <<<512, 512 >>>(gpu_data, s_size, minimum, maximum, size, count);

    cudaMalloc((void **) &spl_values, count * sizeof(int));

    r_scan(s_size, spl_values, count);

    cudaMalloc((void **) &s_sizes, count * sizeof(unsigned int));
    cudaMemset(s_sizes, 0, count * sizeof(unsigned int));

    cudaMalloc((void **) &gpu_val, size * sizeof(float));
    kernel_split_histogramm <<<512, 512 >>>(gpu_data, gpu_val, minimum, maximum, spl_values, s_sizes, size, count);

    int pockets_count = count;
    int* pockets_sizes = (int *)malloc(pockets_count * sizeof(int));
    memset(pockets_sizes, 0, pockets_count * sizeof(int));

    int* pocket = (int *) malloc(pockets_count * sizeof(int));
    int pocket_id = 0;

    for (int spl_id = 0; spl_id < count; spl_id++) {
        int spl_1 = 0;
        cudaMemcpy(&spl_1, &(spl_values[spl_id]), sizeof(int), cudaMemcpyDeviceToHost);

        int spl_size = 0;
        cudaMemcpy(&spl_size, &(s_size[spl_id]), sizeof(int), cudaMemcpyDeviceToHost);

        if (spl_size > 2048) {
            pocket_id++;

            pocket_sort(spl_size, &(gpu_val[spl_1]));
            pockets_sizes[pocket_id] = -1;
            pocket[pocket_id] = spl_1;

            pocket_id++;
        } else {
            int current_calue = 2048 - pockets_sizes[pocket_id];
            if (spl_size <= current_calue) {
                if (current_calue == 2048) {
                    pocket[pocket_id] = spl_1;
                }

                pockets_sizes[pocket_id] += spl_size;
            } else {
                pocket_id++;

                pocket[pocket_id] = spl_1;
                pockets_sizes[pocket_id] = spl_size;
            }
        }
    }

    if (pockets_sizes[pocket_id] == 0) {
        pockets_count = pocket_id;
    } else {
        pockets_count = pocket_id + 1;
    }


    cudaMalloc((void **) &gpu_v, pockets_count * sizeof(int));
    cudaMemcpy(gpu_v, pocket, pockets_count * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &v_size, pockets_count * sizeof(int));
    cudaMemcpy(v_size, pockets_sizes, pockets_count * sizeof(int), cudaMemcpyHostToDevice);

    kernel_sort23<<<dim3(pockets_count, 1, 1), dim3(1024, 1, 1)>>>(gpu_val, gpu_v, v_size, size);

    cudaMemcpy(gpu_data, gpu_val, size * sizeof(float), cudaMemcpyDeviceToDevice);
}


int main() {
    int size;
    fread(&size, sizeof(int), 1, stdin);

    if (size == 0) {
        return 0;
    }

    float* values = new float[size];
    float *gpu_values = NULL;

    fread(values, sizeof(float), size, stdin);

    cudaMalloc((void **) &gpu_values, size * sizeof(float));
    cudaMemcpy(gpu_values, values, size * sizeof(float), cudaMemcpyHostToDevice);


    pocket_sort(size, gpu_values);

    cudaMemcpy(values, gpu_values, size * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(values, sizeof(float), size, stdout);

    return 0;
}

