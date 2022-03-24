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

#define CE(call) { \
    cudaError_t res = call;\
    if (res != cudaSuccess) {\
        fprintf(stderr, "ERROR %u IN %s:%d. Message: %s\n", res, __FILE__, __LINE__, cudaGetErrorString(res));\
        exit(0);\
    }\
}

#define START_CLOCK(NAME, NAME1) \
static dword NAME = 0; \
dword NAME1 = clock();

#define END_CLOCK(NAME, NAME1) \
NAME += clock() - NAME1;

#define GET_CLOCK(NAME) NAME

typedef unsigned char byte;
typedef unsigned short int hword;
typedef unsigned int dword;

__global__ void prevKernel(
    const float const* gpu_vector, const dword start, const dword n, const dword max, const dword min,
    dword* keys,
    dword* hist, const dword bs)
{
    dword idx = threadIdx.x + blockDim.x * blockIdx.x;
    dword shift = blockDim.x * gridDim.x;
    float tmp = bs * 0.999 / (gpu_vector[max] - gpu_vector[min]);
    for (dword i = idx; i < n; i += shift) {
        keys[i] = dword((gpu_vector[i + start] - gpu_vector[min]) * tmp);
        atomicAdd(hist + keys[i], 1);
    }
}

__global__ void groupKernel(
    const dword const* keys,  dword* new_keys, const dword n, 
    dword* scan,
    const float const* gpu_vector, float* new_gpu_vector, const dword start)
{
    dword idx = threadIdx.x + blockDim.x * blockIdx.x;
    dword shift = blockDim.x * gridDim.x;
    for (dword i = idx; i < n; i += shift) {
        dword prev = atomicSub(scan + keys[i], 1);
        prev--;
        new_gpu_vector[prev] = gpu_vector[i + start];
        new_keys[prev] = keys[i];
    }
}

#define SCAN_BASE 1024

__global__ void scanKernel(dword* scan, const dword n) {
    __shared__ dword small_arr[SCAN_BASE];
    for (dword i = blockIdx.x; i < ((n % SCAN_BASE == 0) ? (n / SCAN_BASE) : (n / SCAN_BASE + 1)); i += gridDim.x) {
        for (dword j = threadIdx.x; j < SCAN_BASE; ) {
            dword tmp = i * SCAN_BASE + j;
            small_arr[j] = (tmp < n ? scan[tmp] : 0);
            break;
        }
        __syncthreads();

        for (dword k = 1; k < SCAN_BASE; k *= 2) {
            for (dword j = threadIdx.x; k - 1 + 2 * k * j + k < SCAN_BASE; ) {
                small_arr[k - 1 + 2 * k * j + k] += small_arr[k - 1 + 2 * k * j];
                break;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
            small_arr[SCAN_BASE - 1] = 0;
        __syncthreads();

        for (dword k = SCAN_BASE / 2; k >= 1; k /= 2) {
            for (dword j = threadIdx.x; k - 1 + 2 * k * j + k < SCAN_BASE; ) {
                dword tmp = small_arr[k - 1 + 2 * k * j + k];
                small_arr[k - 1 + 2 * k * j + k] += small_arr[k - 1 + 2 * k * j];
                small_arr[k - 1 + 2 * k * j] = tmp;
                break;
            }
            __syncthreads();
        }

        for (dword j = threadIdx.x; j + 1 < SCAN_BASE; ) {
            dword tmp = i * SCAN_BASE + j;
            if (tmp + 1 < n)
                scan[tmp] = small_arr[j + 1];
            break;
        }
        if (threadIdx.x == 0)
            if ((i + 1) * SCAN_BASE - 1 < n)
                scan[(i + 1) * SCAN_BASE - 1] += small_arr[SCAN_BASE - 1];
            else
                scan[n - 1] = small_arr[SCAN_BASE - 1];
    }
}

__global__ void fillKernel(const dword const* scan_prev, dword* scan, const dword n) {
    dword idx = threadIdx.x + blockDim.x * blockIdx.x;
    dword shift = blockDim.x * gridDim.x;
    for (dword i = idx; i < n; i += shift) {
        scan[i] = scan_prev[(i + 1) * SCAN_BASE - 1 < (n - 1) * SCAN_BASE ?
            (i + 1) * SCAN_BASE - 1 : n * SCAN_BASE - 1];
    }
}

__global__ void addKernel(dword* scan_prev, const dword const* scan, const dword n) {
    dword idx = threadIdx.x + blockDim.x * blockIdx.x;
    dword shift = blockDim.x * gridDim.x;
    for (dword i = idx; i + SCAN_BASE < n; i += shift)
        scan_prev[i + SCAN_BASE] += scan[i / SCAN_BASE];
}

const byte split_size = 32;
#define POCKET_SIZE 512

__global__ void sort23Kernel(float* gpu_vector, dword* left_bord, dword* right_bord, const dword n) {
    dword i;
    dword bord;
    dword tmp1;
    float tmp;
    __shared__ float vector[POCKET_SIZE];

    for (dword b = blockIdx.x; b < n; b += gridDim.x) {
        for (i = threadIdx.x; i < right_bord[b] - left_bord[b]; i += blockDim.x)
            vector[i] = gpu_vector[i + left_bord[b]];
        __syncthreads();

        for (dword k = 0; k < right_bord[b] - left_bord[b]; k += 2) {
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
            gpu_vector[i + left_bord[b]] = vector[i];
    }
}

void scan(dword* arr_scan, const dword n) {
    scanKernel << <1024, SCAN_BASE >> > (arr_scan, n);
    CE(cudaGetLastError());
    if (n <= SCAN_BASE)
        return;

    dword* next_scan;
    CE(cudaMalloc(&next_scan, sizeof(dword) * (n / SCAN_BASE + 1)));
    fillKernel << <256, 256 >> > (arr_scan, next_scan, (n / SCAN_BASE + 1));
    CE(cudaGetLastError());

    scan(next_scan, (n / SCAN_BASE + 1));

    addKernel << <256, 256>> > (arr_scan, next_scan, n);
    CE(cudaGetLastError());
    CE(cudaFree(next_scan));

}

dword* keys;
dword* hist;
dword* new_keys;
float* new_gpu_vector;
dword* gpu_left_bord;
dword* gpu_right_bord;

void sort(float* gpu_vector, dword n, const dword start = 0) {
    static thrust::device_ptr<float> p_arr = thrust::device_pointer_cast(gpu_vector);
    static thrust::device_ptr<float> max, min;

    max = thrust::max_element(p_arr + start, p_arr + n);
    min = thrust::min_element(p_arr + start, p_arr + n);
    if (max == min)
        return;

    n -= start;
    if (n == 1)
        return;

    dword n_split = (n - 1) / split_size + 1;
    CE(cudaMemset(hist, 0, sizeof(dword) * n_split));

    prevKernel<<<2048, 1024>>>(gpu_vector, start, n, dword(max - p_arr), dword(min - p_arr),
        keys, hist, n_split);
    CE(cudaGetLastError());

    scan(hist, n_split);

    groupKernel << <2048, 1024 >> > (keys, new_keys, n, hist, gpu_vector, new_gpu_vector, start);
    CE(cudaGetLastError());
    CE(cudaMemcpy(gpu_vector + start, new_gpu_vector, sizeof(float) * n, cudaMemcpyDeviceToDevice));
    static std::vector<dword> cpu_keys(n);
    CE(cudaMemcpy(&cpu_keys[0], new_keys, sizeof(dword) * n, cudaMemcpyDeviceToHost));

    static std::vector <dword> left_bord((2 * n / (POCKET_SIZE + 1) + 1));
    static std::vector <dword> right_bord((2 * n / (POCKET_SIZE + 1) + 1));
    std::vector <std::pair<dword, dword>> rec_bord;
    dword index_bord = 0;
    dword count = 0;
    dword pocket;
    dword prev;
    dword from = 0;
    for (dword i = 0; i < n; ) {
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

    CE(cudaMemcpy(gpu_left_bord, &left_bord[0], sizeof(dword) * index_bord, cudaMemcpyHostToDevice));
    CE(cudaMemcpy(gpu_right_bord, &right_bord[0], sizeof(dword) * index_bord, cudaMemcpyHostToDevice));
    sort23Kernel << <2048, POCKET_SIZE / 2 >> > (gpu_vector, gpu_left_bord, gpu_right_bord, index_bord);
    CE(cudaGetLastError());

    for (dword i = 0; i < rec_bord.size(); i++)
        sort(gpu_vector, rec_bord[i].second, rec_bord[i].first);

}

void cpu_sort23(std::vector <float>& vector, const dword start, const dword end) {
    for (size_t i = start; i < end; i++) {
        for (size_t j = start + ((i % 2) ? 0 : 1); j + 1 < end; j += 2) {
            if (vector[j] > vector[j + 1]) {
                std::swap(vector[j], vector[j + 1]);
            }
        }
    }
}

const dword max_buckets = 1000000;

void cpu_sort(std::vector <float>& vector) {
    float max = vector[0];
    float min = vector[0];
    for (dword i = 1; i < vector.size(); i++) {
        if (vector[i] > max)
            max = vector[i];
        if (vector[i] < min)
            min = vector[i];
    }
    std::vector<dword> keys(vector.size());
    std::vector<dword> hist(max_buckets, 0);
    float k = max_buckets * 0.999 / (max - min);
    for (dword i = 0; i < keys.size(); i++) {
        keys[i] = dword((vector[i] - min) * k);
        hist[keys[i]]++;
    }
    for (dword i = 1; i < max_buckets; i++)
        hist[i] += hist[i - 1];
    std::vector <float> new_vector(vector.size());
    for (dword i = 0; i < vector.size(); i++) {
        new_vector[--hist[keys[i]]] = vector[i];
    }
    for (dword i = 0; i + 1 < hist.size(); i++)
        cpu_sort23(new_vector, hist[i], hist[i + 1]);
    cpu_sort23(new_vector, hist[hist.size() - 1], vector.size());
    vector.swap(new_vector);
}

float norm_num(const dword k = 20) {
    float sum = 0;
    for (dword i = 0; i < k; i++)
        sum += (rand() % 2 ? 1.f : -1.f) * float(rand()) / k;
    return sum;
}

int main()
{
    dword n;
    fread(&n, sizeof(dword), 1, stdin);
    if (n == 0)
        return 0;

    CE(cudaMalloc(&keys, sizeof(dword) * n));
    CE(cudaMalloc(&hist, sizeof(dword) * ((n - 1) / split_size + 1)));
    CE(cudaMalloc(&new_keys, sizeof(dword) * n));
    CE(cudaMalloc(&new_gpu_vector, sizeof(float) * n));
    CE(cudaMalloc(&gpu_left_bord, sizeof(dword) * (n / (POCKET_SIZE / 2 + 1) + 1)));
    CE(cudaMalloc(&gpu_right_bord, sizeof(dword) * (n / (POCKET_SIZE / 2 + 1) + 1)));

    std::vector<float> cpu_vector(n);
    fread(&cpu_vector[0], sizeof(float), n, stdin);

    float* gpu_vector;
    CE(cudaMalloc(&gpu_vector, sizeof(float) * n));
    CE(cudaMemcpy(gpu_vector, &cpu_vector[0], sizeof(float) * n, cudaMemcpyHostToDevice));

    sort(gpu_vector, n);
    CE(cudaFree(new_keys));
    CE(cudaFree(hist));
    CE(cudaFree(keys));
    CE(cudaFree(new_gpu_vector));
    CE(cudaFree(gpu_left_bord));
    CE(cudaFree(gpu_right_bord));

    CE(cudaMemcpy(&cpu_vector[0], gpu_vector, sizeof(float) * n, cudaMemcpyDeviceToHost));
    CE(cudaFree(gpu_vector));

    fwrite(&cpu_vector[0], sizeof(float), n, stdout);
    return 0;
}
