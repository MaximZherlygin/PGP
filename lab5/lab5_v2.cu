#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <map>

#define CSC(call) { \
    cudaError_t res = call;\
    if (res != cudaSuccess) {\
        fprintf(stderr, "ERROR %u IN %s:%d. Message: %s\n", res, __FILE__, __LINE__, cudaGetErrorString(res));\
        exit(0);\
    }\
}

#define BASE 1024

__global__ void prevKernel(const float* gpu_data, const unsigned int start,
                           const unsigned int n, const unsigned int max,
                           const unsigned int min, unsigned int* keys,
                           unsigned int* hist, const unsigned int bs) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int offsetX = blockDim.x * gridDim.x;

    float value = 0.999 / (gpu_data[max] - gpu_data[min]) * bs;

    for (unsigned int i = idx; i < n; i += offsetX) {
        keys[i] = unsigned int((gpu_data[i + start] - gpu_data[min]) * value);
        atomicAdd(hist + keys[i], 1);
    }
}

__global__ void groupKernel(
    const unsigned int* keys,  unsigned int* new_akeys, const unsigned int n,
    unsigned int* scan,
    const float * gpu_data, float* new_gpu_data, const unsigned int start)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int shift = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += shift) {
        unsigned int prev = atomicSub(scan + keys[i], 1);
        prev--;
        new_gpu_data[prev] = gpu_data[i + start];
        new_keys[prev] = keys[i];
    }
}

__global__ void scanKernel(unsigned int* scan, const unsigned int n) {
    __shared__ unsigned int small_arr[BASE];
    for (unsigned int i = blockIdx.x; i < ((n % BASE == 0) ? (n / BASE) : (n / BASE + 1)); i += gridDim.x) {
        for (unsigned int j = threadIdx.x; j < BASE; ) {
            unsigned int tmp = i * BASE + j;
            small_arr[j] = (tmp < n ? scan[tmp] : 0);
            break;
        }
        __syncthreads();

        for (unsigned int k = 1; k < BASE; k *= 2) {
            for (unsigned int j = threadIdx.x; k - 1 + 2 * k * j + k < BASE; ) {
                small_arr[k - 1 + 2 * k * j + k] += small_arr[k - 1 + 2 * k * j];
                break;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
            small_arr[BASE - 1] = 0;
        __syncthreads();

        for (unsigned int k = BASE / 2; k >= 1; k /= 2) {
            for (unsigned int j = threadIdx.x; k - 1 + 2 * k * j + k < BASE; ) {
                unsigned int tmp = small_arr[k - 1 + 2 * k * j + k];
                small_arr[k - 1 + 2 * k * j + k] += small_arr[k - 1 + 2 * k * j];
                small_arr[k - 1 + 2 * k * j] = tmp;
                break;
            }
            __syncthreads();
        }

        for (unsigned int j = threadIdx.x; j + 1 < BASE; ) {
            unsigned int tmp = i * BASE + j;
            if (tmp + 1 < n)
                scan[tmp] = small_arr[j + 1];
            break;
        }
        if (threadIdx.x == 0)
            if ((i + 1) * BASE - 1 < n)
                scan[(i + 1) * BASE - 1] += small_arr[BASE - 1];
            else
                scan[n - 1] = small_arr[BASE - 1];
    }
}

__global__ void fillKernel(const unsigned int* scan_prev, unsigned int* scan, const unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int shift = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += shift) {
        scan[i] = scan_prev[(i + 1) * BASE - 1 < (n - 1) * BASE ?
            (i + 1) * BASE - 1 : n * BASE - 1];
    }
}

__global__ void addKernel(unsigned int* scan_prev, const unsigned int* scan, const unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int shift = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i + BASE < n; i += shift)
        scan_prev[i + BASE] += scan[i / BASE];
}

const unsigned char split_size = 32;
#define POCKET_SIZE 512

__global__ void sort23Kernel(float* gpu_data, unsigned int* left_bord, unsigned int* right_bord, const unsigned int n) {
    unsigned int i;
    unsigned int bord;
    unsigned int tmp1;
    float tmp;
    __shared__ float vector[POCKET_SIZE];

    for (unsigned int b = blockIdx.x; b < n; b += gridDim.x) {
        for (i = threadIdx.x; i < right_bord[b] - left_bord[b]; i += blockDim.x)
            vector[i] = gpu_data[i + left_bord[b]];
        __syncthreads();

        for (unsigned int k = 0; k < right_bord[b] - left_bord[b]; k += 2) {
            bord = (right_bord[b] - left_bord[b]) / 2;
            for (i = threadIdx.x; i < bord; i += blockDim.x) {
                tmp1 = 2 * i;
                if (vector[tmp1] > vector[tmp1 + 1]) {
                    tmp = vector[tmp1 + 1];
                    vector[tmp1 + 1] = vector[tmp1];
                    vector[tmp1] = tmp;
                }
            }
            __syncthreads();
            bord = (right_bord[b] - left_bord[b] - 1) / 2;
            for (i = threadIdx.x; i < bord; i += blockDim.x) {
                tmp1 = 2 * i + 1;
                if (vector[tmp1] > vector[tmp1 + 1]) {
                    tmp = vector[tmp1 + 1];
                    vector[tmp1 + 1] = vector[tmp1];
                    vector[tmp1] = tmp;
                }
            }
            __syncthreads();
        }

        for (i = threadIdx.x; i < right_bord[b] - left_bord[b]; i += blockDim.x)
            gpu_data[i + left_bord[b]] = vector[i];
    }
}

void scan(unsigned int* arr_scan, const unsigned int n) {
    scanKernel << <1024, BASE >> > (arr_scan, n);
    CSC(cudaGetLastError());
    if (n <= BASE)
        return;

    unsigned int* next_scan;
    CSC(cudaMalloc(&next_scan, sizeof(unsigned int) * (n / BASE + 1)));
    fillKernel << <256, 256 >> > (arr_scan, next_scan, (n / BASE + 1));
    CSC(cudaGetLastError());

    scan(next_scan, (n / BASE + 1));

    addKernel << <256, 256>> > (arr_scan, next_scan, n);
    CSC(cudaGetLastError());
    CSC(cudaFree(next_scan));

}

unsigned int* keys;
unsigned int* hist;
unsigned int* new_keys;
float* new_gpu_data;
unsigned int* gpu_left_bord;
unsigned int* gpu_right_bord;

void sort(float* gpu_data, unsigned int n, const unsigned int start = 0) {
    static thrust::device_ptr<float> p_arr = thrust::device_pointer_cast(gpu_data);
    static thrust::device_ptr<float> max, min;

    max = thrust::max_element(p_arr + start, p_arr + n);
    min = thrust::min_element(p_arr + start, p_arr + n);
    if (max == min)
        return;

    n -= start;
    if (n == 1)
        return;

    unsigned int n_split = (n - 1) / split_size + 1;
    CSC(cudaMemset(hist, 0, sizeof(unsigned int) * n_split));

    prevKernel<<<2048, 1024>>>(gpu_data, start, n, unsigned int(max - p_arr), unsigned int(min - p_arr),
        keys, hist, n_split);
    CSC(cudaGetLastError());

    scan(hist, n_split);

    groupKernel << <2048, 1024 >> > (keys, new_keys, n, hist, gpu_data, new_gpu_data, start);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(gpu_data + start, new_gpu_data, sizeof(float) * n, cudaMemcpyDeviceToDevice));
    static std::vector<unsigned int> cpu_keys(n);
    CSC(cudaMemcpy(&cpu_keys[0], new_keys, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost));

    static std::vector <unsigned int> left_bord((2 * n / (POCKET_SIZE + 1) + 1));
    static std::vector <unsigned int> right_bord((2 * n / (POCKET_SIZE + 1) + 1));
    std::vector <std::pair<unsigned int, unsigned int>> rec_bord;
    unsigned int index_bord = 0;
    unsigned int count = 0;
    unsigned int pocket;
    unsigned int prev;
    unsigned int from = 0;
    for (unsigned int i = 0; i < n; ) {
        pocket = cpu_keys[i];
        prev = i;
        while (cpu_keys[i] == pocket) {
            count++;
            i++;
            if (i == n) {
                left_bord[index_bord] = from + start;
                right_bord[index_bord] = i + start;
                index_bord++;
                break;
            }
            if (count == POCKET_SIZE) {
                count = 0;
                if (pocket == cpu_keys[i]) {
                    if (from == prev) {
                        while (i < n && cpu_keys[i] == pocket)
                            i++;
                        rec_bord.push_back({ from + start , i + start});
                        from = i;
                        break;
                    }
                    left_bord[index_bord] = from + start;
                    right_bord[index_bord] = prev + start;
                    index_bord++;
                    i = prev;
                    from = prev;
                    break;
                }
                left_bord[index_bord] = from + start;
                right_bord[index_bord] = i + start;
                index_bord++;
                from = i;
                break;
            }
        }
    }

    CSC(cudaMemcpy(gpu_left_bord, &left_bord[0], sizeof(unsigned int) * index_bord, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(gpu_right_bord, &right_bord[0], sizeof(unsigned int) * index_bord, cudaMemcpyHostToDevice));
    sort23Kernel << <2048, POCKET_SIZE / 2 >> > (gpu_data, gpu_left_bord, gpu_right_bord, index_bord);
    CSC(cudaGetLastError());

    for (unsigned int i = 0; i < rec_bord.size(); i++)
        sort(gpu_data, rec_bord[i].second, rec_bord[i].first);

}

void cpu_sort23(std::vector <float>& vector, const unsigned int start, const unsigned int end) {
    for (size_t i = start; i < end; i++) {
        for (size_t j = start + ((i % 2) ? 0 : 1); j + 1 < end; j += 2) {
            if (vector[j] > vector[j + 1]) {
                std::swap(vector[j], vector[j + 1]);
            }
        }
    }
}

const unsigned int max_buckets = 1000000;

void cpu_sort(std::vector <float>& vector) {
    float max = vector[0];
    float min = vector[0];
    for (unsigned int i = 1; i < vector.size(); i++) {
        if (vector[i] > max)
            max = vector[i];
        if (vector[i] < min)
            min = vector[i];
    }
    std::vector<unsigned int> keys(vector.size());
    std::vector<unsigned int> hist(max_buckets, 0);
    float k = max_buckets * 0.999 / (max - min);
    for (unsigned int i = 0; i < keys.size(); i++) {
        keys[i] = unsigned int((vector[i] - min) * k);
        hist[keys[i]]++;
    }
    for (unsigned int i = 1; i < max_buckets; i++)
        hist[i] += hist[i - 1];
    std::vector <float> new_vector(vector.size());
    for (unsigned int i = 0; i < vector.size(); i++) {
        new_vector[--hist[keys[i]]] = vector[i];
    }
    for (unsigned int i = 0; i + 1 < hist.size(); i++)
        cpu_sort23(new_vector, hist[i], hist[i + 1]);
    cpu_sort23(new_vector, hist[hist.size() - 1], vector.size());
    vector.swap(new_vector);
}

float norm_num(const unsigned int k = 20) {
    float sum = 0;
    for (unsigned int i = 0; i < k; i++)
        sum += (rand() % 2 ? 1.f : -1.f) * float(rand()) / k;
    return sum;
}

int main()
{
    unsigned int n;
    fread(&n, sizeof(unsigned int), 1, stdin);
    if (n == 0)
        return 0;

    CSC(cudaMalloc(&keys, sizeof(unsigned int) * n));
    CSC(cudaMalloc(&hist, sizeof(unsigned int) * ((n - 1) / split_size + 1)));
    CSC(cudaMalloc(&new_keys, sizeof(unsigned int) * n));
    CSC(cudaMalloc(&new_gpu_data, sizeof(float) * n));
    CSC(cudaMalloc(&gpu_left_bord, sizeof(unsigned int) * (n / (POCKET_SIZE / 2 + 1) + 1)));
    CSC(cudaMalloc(&gpu_right_bord, sizeof(unsigned int) * (n / (POCKET_SIZE / 2 + 1) + 1)));

    std::vector<float> cpu_vector(n);
    fread(&cpu_vector[0], sizeof(float), n, stdin);

    float* gpu_data;
    CSC(cudaMalloc(&gpu_data, sizeof(float) * n));
    CSC(cudaMemcpy(gpu_data, &cpu_vector[0], sizeof(float) * n, cudaMemcpyHostToDevice));

    sort(gpu_data, n);
    CSC(cudaFree(new_keys));
    CSC(cudaFree(hist));
    CSC(cudaFree(keys));
    CSC(cudaFree(new_gpu_data));
    CSC(cudaFree(gpu_left_bord));
    CSC(cudaFree(gpu_right_bord));

    CSC(cudaMemcpy(&cpu_vector[0], gpu_data, sizeof(float) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(gpu_data));

    fwrite(&cpu_vector[0], sizeof(float), n, stdout);
    return 0;
}
