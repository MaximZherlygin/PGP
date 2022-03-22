#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <time.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include "mpi.h"

using namespace std;

#define CSC(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "ERROR: CUDA failed in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return(1); \
    } \
} \

// направления
#define down 0
#define up 1
#define left 2
#define right 3
#define front 4
#define back 5

const int x_on = 0;
const int y_on = 1;
const int z_on = 2;

__constant__ int g_dimensions[3];
__constant__ int g_block[3];

int dim[3];
int block[3];
double *values;
double *next_values;
double *temporary_values;
double eps;
double l[3];
double h[3];
double u[6];
double u_start;
string out_filename;

// макросы _i, чтобы у нас можно было использовать в индексации -1 и n
#define _i(i, j, k) (((k) + 1) * (dim[y_on] + 2) * (dim[x_on] + 2) + ((j) + 1) * (dim[x_on] + 2) + (i) + 1)
#define _ip(i, j, k) ((k) * block[y_on] * block[x_on] + (j) * block[x_on] + (i))
#define _ib(i, j) (j * max(dim[x_on], max(dim[y_on], dim[z_on])) + i)

__device__ int _ind(int i, int j, int k) {
    return (((k) + 1) * (g_dimensions[1] + 2) * (g_dimensions[0] + 2) + ((j) + 1) * (g_dimensions[0] + 2) + (i) + 1);
}

__device__ int _inp(int i, int j, int k) {
    return ((k) * g_block[1] * g_block[0] + (j) * g_block[0] + (i));
}

__device__ int _inb(int i, int j) {
    return (j * max(g_dimensions[0], max(g_dimensions[1], g_dimensions[2])) + i);
}

__global__ void copy_edge_xy(double* send_edge_xy, double* recv_edge_xy, double* values, int k, int k2 = 0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;

    if (send_edge_xy && !recv_edge_xy) {
        for (int j = idy; j < g_dimensions[1]; j += offsetY) {
            for (int i = idx; i < g_dimensions[0]; i += offsetX) {
                send_edge_xy[_inb(i, j)] = values[_ind(i, j, k)];
            }
        }
    } else if (!send_edge_xy && recv_edge_xy) {
        for (int j = idy; j < g_dimensions[1]; j += offsetY) {
            for (int i = idx; i < g_dimensions[0]; i += offsetX) {
                values[_ind(i, j, k)] = recv_edge_xy[_inb(i, j)];
            }
        }
    } else {
        for (int j = idy; j < g_dimensions[1]; j += offsetY) {
            for (int i = 0; i < g_dimensions[0]; i += offsetX) {
                values[_ind(i, j, k)] = recv_edge_xy[_inb(i, j)];
                send_edge_xy[_inb(i, j)] = values[_ind(i, j, k2)];
            }
        }
    }
}

__global__ void copy_edge_xz(double* send_edge_xz, double* recv_edge_xz, double* values, int j, int j2 = 0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;

    if (send_edge_xz && !recv_edge_xz) {
        for (int k = idy; k < g_dimensions[2]; k += offsetY) {
            for (int i = idx; i < g_dimensions[1]; i += offsetX) {
                send_edge_xz[_inb(i, j)] = values[_ind(i, j, k)];
            }
        }
    } else if (!send_edge_xz && recv_edge_xz) {
        for (int k = idy; k < g_dimensions[2]; k += offsetY) {
            for (int i = idx; i < g_dimensions[1]; i += offsetX) {
                values[_ind(i, j, k)] = recv_edge_xz[_inb(i, j)];
            }
        }
    } else {
        for (int k = idy; k < g_dimensions[2]; k += offsetY) {
            for (int i = 0; i < g_dimensions[1]; i += offsetX) {
                values[_ind(i, j, k)] = recv_edge_xz[_inb(i, j)];
                send_edge_xz[_inb(i, j)] = values[_ind(i, j2, k)];
            }
        }
    }
}

__global__ void copy_edge_yz(double* send_edge_yz, double* recv_edge_yz, double* values, int i, int i2 = 0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;

    if (send_edge_yz && !recv_edge_yz) {
        for (int k = idy; k < g_dimensions[2]; k += offsetY) {
            for (int j = idx; j < g_dimensions[0]; j += offsetX) {
                send_edge_yz[_inb(i, j)] = values[_ind(i, j, k)];
            }
        }
    } else if (!send_edge_yz && recv_edge_yz) {
        for (int k = idy; k < g_dimensions[2]; k += offsetY) {
            for (int j = idx; j < g_dimensions[0]; j += offsetX) {
                values[_ind(i, j, k)] = recv_edge_yz[_inb(i, j)];
            }
        }
    } else {
        for (int k = idy; k < g_dimensions[2]; k += offsetY) {
            for (int j = 0; j < g_dimensions[0]; j += offsetX) {
                values[_ind(i, j, k)] = recv_edge_yz[_inb(i, j)];
                send_edge_yz[_inb(i, j)] = values[_ind(i2, j, k)];
            }
        }
    }
}

__global__ void compute(double* values, double* next_values, double h2x, double h2y, double h2z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    int offsetZ = blockDim.z * gridDim.z;

    for (int i = idx; i < g_dimensions[0]; i += offsetX) {
        for (int j = idy; j < g_dimensions[1]; j += offsetY) {
            for (int k = idz; k < g_dimensions[2]; k += offsetZ) {
                next_values[_ind(i, j, k)] = 0.5 * ((values[_ind(i + 1, j, k)] + values[_ind(i - 1, j, k)]) / h2x +
                                                   (values[_ind(i, j + 1, k)] + values[_ind(i, j - 1, k)]) / h2y +
                                                   (values[_ind(i, j, k + 1)] + values[_ind(i, j, k - 1)]) / h2z) /
                                                   (1 / h2x + 1 / h2y + 1 / h2z);
            }
        }
    }
}

__global__ void error(double* values, double* next_values, double* diff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    int offsetZ = blockDim.z * gridDim.z;
    int i, j, k;

    for (i = idx - 1; i < g_dimensions[0] + 1; i += offsetX) {
        for (j = idy - 1; j < g_dimensions[1] + 1; j += offsetY) {
            for (k = idz - 1; k < g_dimensions[2] + 1; k += offsetZ) {
                diff[_ind(i, j, k)] = (i != -1 && j != -1 && k != -1 && i != g_dimensions[0] && j != g_dimensions[1] && k != g_dimensions[2]) * abs(next_values[_ind(i, j, k)] - values[_ind(i, j, k)]);
            }
        }
    }
}

int main(int argc, char** argv) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    MPI_Status status;
    MPI_Init(&argc, &argv);
    int id;
    int proccess_count;
    MPI_Comm_size(MPI_COMM_WORLD, &proccess_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (id == 0) {
        cin >> block[x_on] >> block[y_on] >> block[z_on];
        cin >> dim[x_on] >> dim[y_on] >> dim[z_on];
        cin >> out_filename >> eps;
        cin >> l[x_on] >> l[y_on] >> l[z_on];
        cin >> u[down] >> u[up] >> u[left];
        cin >> u[right] >> u[front] >> u[back];
        cin >> u_start;
    }

    MPI_Bcast(block, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(l, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_start, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (block[x_on] * block[y_on] * block[z_on] * dim[x_on] * dim[y_on] * dim[z_on] == 0) {
        MPI_Finalize();
        return 0;
    }

    h[x_on] = l[x_on] / (block[x_on] * dim[x_on]);
    h[y_on] = l[y_on] / (block[y_on] * dim[y_on]);
    h[z_on] = l[z_on] / (block[z_on] * dim[z_on]);

    double h2[3];
    h2[x_on] = h[x_on] * h[x_on];
    h2[y_on] = h[y_on] * h[y_on];
    h2[z_on] = h[z_on] * h[z_on];

    int i_b = ((id) % block[x_on]);
    int j_b = (((id) / block[x_on]) % block[y_on]);
    int k_b = ((id) / (block[x_on] * block[y_on]));

    int buffer_size = max(max(dim[x_on], dim[y_on]), dim[z_on]) * max(max(dim[x_on], dim[y_on]), dim[z_on]);

    double* gpu_values;
    CSC(cudaMalloc(&gpu_values, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2)))
    double* gpu_next_values;
    CSC(cudaMalloc(&gpu_next_values, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2)))
    double* gpu_send_edge_xy, *gpu_send_edge_xz, *gpu_send_edge_yz;
    CSC(cudaMalloc(&gpu_send_edge_xy, buffer_size * sizeof(double)))
    CSC(cudaMalloc(&gpu_send_edge_xz, buffer_size * sizeof(double)))
    CSC(cudaMalloc(&gpu_send_edge_yz, buffer_size * sizeof(double)))
    double* gpu_recv_edge_xy, *gpu_recv_edge_xz, *gpu_recv_edge_yz;
    CSC(cudaMalloc(&gpu_recv_edge_xy, buffer_size * sizeof(double)))
    CSC(cudaMalloc(&gpu_recv_edge_xz, buffer_size * sizeof(double)))
    CSC(cudaMalloc(&gpu_recv_edge_yz, buffer_size * sizeof(double)))

    double* send_edge_xy = (double*)malloc(buffer_size * sizeof(double));
    double* send_edge_xz = (double*)malloc(buffer_size * sizeof(double));
    double* send_edge_yz = (double*)malloc(buffer_size * sizeof(double));
    double* recv_edge_xy = (double*)malloc(buffer_size * sizeof(double));
    double* recv_edge_xz = (double*)malloc(buffer_size * sizeof(double));
    double* recv_edge_yz = (double*)malloc(buffer_size * sizeof(double));

    values = (double*)malloc((dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2) * sizeof(double));
    next_values = (double*)malloc((dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2) * sizeof(double));

    CSC(cudaMemcpy(gpu_send_edge_xy, send_edge_xy, sizeof(double) * buffer_size, cudaMemcpyHostToDevice))
    CSC(cudaMemcpy(gpu_send_edge_xz, send_edge_xz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice))
    CSC(cudaMemcpy(gpu_send_edge_yz, send_edge_yz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice))
    CSC(cudaMemcpy(gpu_recv_edge_xy, recv_edge_xy, sizeof(double) * buffer_size, cudaMemcpyHostToDevice))
    CSC(cudaMemcpy(gpu_recv_edge_xz, recv_edge_xz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice))
    CSC(cudaMemcpy(gpu_recv_edge_yz, recv_edge_yz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice))

    double* globDiff = (double*)malloc(sizeof(double) * proccess_count);

    for (int i = 0; i <= dim[x_on]; i++) {					// Инициализация блока
        for (int j = 0; j <= dim[y_on]; j++) {
            for (int k = 0; k <= dim[z_on]; k++) {
                values[_i(i, j, k)] = u_start;
                // cerr << 'values[i, j, k] = ' << values[_i(i, j, k)] << '\n';
                // cerr << 'u_start = ' << u_start;
            }
        }
    }

    if (i_b == 0) {
        for (int k = 0; k < dim[z_on]; k++)
            for (int j = 0; j < dim[y_on]; j++) {
                values[_i(-1, j, k)] = u[left];
                next_values[_i(-1, j, k)] = u[left];
            }
    }

    if (j_b == 0) {
        for (int k = 0; k < dim[z_on]; k++)
            for (int i = 0; i < dim[x_on]; i++) {
                values[_i(i, -1, k)] = u[front];
                next_values[_i(i, -1, k)] = u[front];
            }
    }

    if (k_b == 0) {
        for (int j = 0; j < dim[y_on]; j++)
            for (int i = 0; i < dim[x_on]; i++) {
                values[_i(i, j, -1)] = u[down];
                next_values[_i(i, j, -1)] = u[down];
            }
    }

    if ((i_b + 1) == block[x_on]) {
        for (int k = 0; k < dim[z_on]; k++)
            for (int j = 0; j < dim[y_on]; j++) {
                values[_i(dim[0], j, k)] = u[right];
                next_values[_i(dim[0], j, k)] = u[right];
            }
    }

    if ((j_b + 1) == block[y_on]) {
        for (int k = 0; k < dim[z_on]; k++)
            for (int i = 0; i < dim[x_on]; i++) {
                values[_i(i, dim[1], k)] = u[back];
                next_values[_i(i, dim[1], k)] = u[back];
            }
    }

    if ((k_b + 1) == block[z_on]) {
        for (int j = 0; j < dim[y_on]; j++)
            for (int i = 0; i < dim[x_on]; i++) {
                values[_i(i, j, dim[z_on])] = u[up];
                next_values[_i(i, j, dim[z_on])] = u[up];
            }
    }

    CSC(cudaMemcpy(gpu_values, values, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2), cudaMemcpyHostToDevice))
    CSC(cudaMemcpy(gpu_next_values, next_values, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2), cudaMemcpyHostToDevice))

    dim3 gblocks(32, 32);
    dim3 threads(32, 32);

    CSC(cudaMemcpyToSymbol(g_dimensions, dim, 3 * sizeof(int)));
    CSC(cudaMemcpyToSymbol(g_block, block, 3 * sizeof(int)))


    while (true) {
        if (block[x_on]) {
            if (i_b == 0) {
                copy_edge_yz<<<gblocks, threads>>>(gpu_send_edge_yz, NULL, gpu_values, dim[x_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_yz, gpu_send_edge_yz, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_yz, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             id, recv_edge_yz, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             _ip(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_yz, recv_edge_yz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_yz<<<gblocks, threads>>>(NULL, gpu_recv_edge_yz, gpu_values, dim[x_on]);
                CSC(cudaGetLastError());
            } else if (i_b + 1 == block[x_on]) {
                copy_edge_yz<<<gblocks, threads>>>(gpu_send_edge_yz, NULL, gpu_values, 0);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_yz, gpu_send_edge_yz, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_yz, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             id, recv_edge_yz, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             _ip(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_yz, recv_edge_yz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_yz<<<gblocks, threads>>>(NULL, gpu_recv_edge_yz, gpu_values, -1);
                CSC(cudaGetLastError());
            } else {
                copy_edge_yz<<<gblocks, threads>>>(gpu_send_edge_yz, NULL, gpu_values, dim[x_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_yz, gpu_send_edge_yz, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_yz, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             id, recv_edge_yz, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             _ip(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_yz, recv_edge_yz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_yz<<<gblocks, threads>>>(gpu_send_edge_yz, gpu_recv_edge_yz, gpu_values, -1, 0);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_yz, gpu_send_edge_yz, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_yz, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             id, recv_edge_yz, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             _ip(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_yz, recv_edge_yz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_yz<<<gblocks, threads>>>(NULL, gpu_recv_edge_yz, gpu_values, dim[x_on]);
                CSC(cudaGetLastError());
            }
        }

        if (block[y_on]) {
            if (j_b == 0) {
                copy_edge_xz<<<gblocks, threads>>>(gpu_send_edge_xz, NULL, gpu_values, dim[y_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_xz, gpu_send_edge_xz, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_xz, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             id, recv_edge_xz, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             _ip(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_xz, recv_edge_xz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_xz<<<gblocks, threads>>>(NULL, gpu_recv_edge_xz, gpu_values, dim[y_on]);
                CSC(cudaGetLastError());
            }
            else if (j_b + 1 == block[y_on]) {
                copy_edge_xz<<<gblocks, threads>>>(gpu_send_edge_xz, NULL, gpu_values, 0);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_xz, gpu_send_edge_xz, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_xz, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             id, recv_edge_xz, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             _ip(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_xz, recv_edge_xz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_xz<<<gblocks, threads>>>(NULL, gpu_recv_edge_xz, gpu_values, -1);
                CSC(cudaGetLastError());
            } else {
                copy_edge_xz<<<gblocks, threads>>>(gpu_send_edge_xz, NULL, gpu_values, dim[y_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_xz, gpu_send_edge_xz, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_xz, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             id, recv_edge_xz, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             _ip(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_xz, recv_edge_xz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_xz<<<gblocks, threads>>>(gpu_send_edge_xz, gpu_recv_edge_xz, gpu_values, -1, 0);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_xz, gpu_send_edge_xz, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_xz, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             id, recv_edge_xz, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             _ip(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_xz, recv_edge_xz, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_xz<<<gblocks, threads>>>(NULL, gpu_recv_edge_xz, gpu_values, dim[y_on]);
                CSC(cudaGetLastError());
            }
        }

        if (block[z_on]) {
            if (k_b == 0) {
                copy_edge_xy<<<gblocks, threads>>>(gpu_send_edge_xy, NULL, gpu_values, dim[z_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_xy, gpu_send_edge_xy, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_xy, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             id, recv_edge_xy, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             _ip(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_xy, recv_edge_xy, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_xy<<<gblocks, threads>>>(NULL, gpu_recv_edge_xy, gpu_values, dim[z_on]);
                CSC(cudaGetLastError());
            }
            else if (k_b + 1 == block[z_on]) {
                copy_edge_xy<<<gblocks, threads>>>(gpu_send_edge_xy, NULL, gpu_values, 0);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_xy, gpu_send_edge_xy, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_xy, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             id, recv_edge_xy, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             _ip(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_xy, recv_edge_xy, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_xy<<<gblocks, threads>>>(NULL, gpu_recv_edge_xy, gpu_values, -1);
                CSC(cudaGetLastError());
            }
            else {
                copy_edge_xy<<<gblocks, threads>>>(gpu_send_edge_xy, NULL, gpu_values, dim[z_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_xy, gpu_send_edge_xy, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_xy, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             id, recv_edge_xy, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             _ip(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_xy, recv_edge_xy, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_xy<<<gblocks, threads>>>(gpu_send_edge_xy, gpu_recv_edge_xy, gpu_values, -1, 0);
                CSC(cudaGetLastError());
                CSC(cudaMemcpy(send_edge_xy, gpu_send_edge_xy, sizeof(double) * buffer_size, cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_edge_xy, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             id, recv_edge_xy, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             _ip(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(gpu_recv_edge_xy, recv_edge_xy, sizeof(double) * buffer_size, cudaMemcpyHostToDevice));
                copy_edge_xy<<<gblocks, threads>>>(NULL, gpu_recv_edge_xy, gpu_values, dim[z_on]);
                CSC(cudaGetLastError());
            }
        }

        cudaThreadSynchronize();

        double* gpu_difference;
        CSC(cudaMalloc((void**)&gpu_difference, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2)));
        compute<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(gpu_values, gpu_next_values, h2[0], h2[1], h2[2]);
        CSC(cudaGetLastError());
        cudaThreadSynchronize();
        error<<<dim3(8, 8, 8), dim3(32, 4, 4)>>> (gpu_values, gpu_next_values, gpu_difference);

        thrust::device_ptr<double> pointers = thrust::device_pointer_cast(gpu_difference);
        thrust::device_ptr<double> res = thrust::max_element(pointers, pointers + (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2));

        double diff = 0.0;
        diff = *res;

        MPI_Allgather(&diff, 1, MPI_DOUBLE, globDiff, 1, MPI_DOUBLE, MPI_COMM_WORLD);

        for (int i = 0; i < proccess_count; i++) {
            diff = max(diff, globDiff[i]);
        }

        if (diff < eps) {
            break;
        }

        temporary_values = next_values;
        next_values = values;
        values = temporary_values;

        CSC(cudaFree(gpu_difference));
    }

    CSC(cudaMemcpy(values, gpu_values, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2), cudaMemcpyDeviceToHost));
    CSC(cudaFree(gpu_values));
    CSC(cudaFree(gpu_next_values));
    CSC(cudaFree(gpu_send_edge_xy));
    CSC(cudaFree(gpu_send_edge_xz));
    CSC(cudaFree(gpu_send_edge_yz));
    CSC(cudaFree(gpu_recv_edge_xy));
    CSC(cudaFree(gpu_recv_edge_xz));
    CSC(cudaFree(gpu_recv_edge_yz));

    int buff_size = (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2);
    int new_symbol_size = 14;

    // Allocate mem
    char* buff = new char[buff_size * new_symbol_size];
    memset(buff, (char)' ', buff_size * new_symbol_size * sizeof(char));

    int i, j, k;
    for (k = 0; k < dim[z_on]; k++) {
        for (j = 0; j < dim[y_on]; j++) {
            int len_new_symbol;
            for (i = 0; i < dim[x_on] - 1; i++) {
                len_new_symbol = sprintf(&buff[_i(i, j, k) * new_symbol_size], "%.6e", values[_i(i, j, k)]);
                if (len_new_symbol < new_symbol_size) {
                    buff[_i(i, j, k) * new_symbol_size + len_new_symbol] = ' ';
                }
            }
            len_new_symbol = sprintf(&buff[_i(i, j, k) * new_symbol_size], "%.6e\n", values[_i(i, j, k)]);
            if(len_new_symbol < new_symbol_size){
                buff[_i(i, j, k) * new_symbol_size + len_new_symbol] = ' ';
            }
        }
    }

    MPI_Datatype new_representation;
    MPI_Datatype memtype;
    MPI_Datatype filetype;
    int sizes[3], starts[3], f_sizes[3], f_starts[3];

    MPI_Type_contiguous(new_symbol_size, MPI_CHAR, &new_representation);
    MPI_Type_commit(&new_representation);

    // Sizes for memtype
    sizes[x_on] = dim[x_on] + 2;
    sizes[y_on] = dim[y_on] + 2;
    sizes[z_on] = dim[z_on] + 2;
    starts[x_on] = starts[y_on] = starts[z_on] = 1;

    // Sizes for filetype
    f_sizes[x_on] = dim[x_on] * block[x_on];
    f_sizes[y_on] = dim[y_on] * block[y_on];
    f_sizes[z_on] = dim[z_on] * block[z_on];

    f_starts[x_on] = dim[x_on] * i_b;
    f_starts[y_on] = dim[y_on] * j_b;
    f_starts[z_on] = dim[z_on] * k_b;

    // Writting types
    // Memtype
    MPI_Type_create_subarray(3, sizes, dim, starts, MPI_ORDER_FORTRAN, new_representation, &memtype);
    MPI_Type_commit(&memtype);
    // Filetype
    MPI_Type_create_subarray(3, f_sizes, dim, f_starts, MPI_ORDER_FORTRAN, new_representation, &filetype);
    MPI_Type_commit(&filetype);

    // Create and open file
    MPI_File fp;
    MPI_File_delete(out_filename.c_str(), MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, out_filename.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fp);

    MPI_File_set_view(fp, 0, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fp, buff, 1, memtype, MPI_STATUS_IGNORE);

    MPI_File_close(&fp);


    MPI_Finalize();
    free(buff);
    free(values);
    free(next_values);
    free(send_edge_xy);
    free(send_edge_xz);
    free(send_edge_yz);
    free(recv_edge_xy);
    free(recv_edge_xz);
    free(recv_edge_yz);
    return 0;
}
