#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

using namespace std;

#define down 0
#define up 1
#define left 2
#define right 3
#define front 4
#define back 5

const int x_on = 0;
const int y_on = 1;
const int z_on = 2;

#define _i(i, j, k) (((k) + 1) * (dim[y_on] + 2) * (dim[x_on] + 2) + ((j) + 1) * (dim[x_on] + 2) + (i) + 1)
#define _ip(i,j, k) ((k) * block[y_on] * block[x_on] + (j) * block[x_on] + (i))
#define _ic(i, j, k) (((k) + 1) * ((dimY) + 2) * ((dimX) + 2) + ((j) + 1) * ((dimX) + 2) + (i) + 1)

#define CSC(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "ERROR: CUDA failed in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return(1); \
    } \
} \

__global__ void send_xy(double* send, double* values, int dimX, int dimY, int dimZ, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    for (int j = idy; j < dimY; j += offsetY) {
        for (int i = idx; i < dimX; i += offsetX) {
            send[j * dimX + i] = values[_ic(i, j, k)];
        }
    }
}

__global__ void send_xz(double* send, double* values, int dimX, int dimY, int dimZ, int j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetZ = blockDim.y * gridDim.y;
    for (int k = idz; k < dimZ; k += offsetZ) {
        for (int i = idx; i < dimX; i += offsetX) {
            send[k * dimX + i] = values[_ic(i, j, k)];
        }
    }
}

__global__ void send_yz(double* send, double* values, int dimX, int dimY, int dimZ, int i) {
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetY = blockDim.x * gridDim.x;
    int offsetZ = blockDim.y * gridDim.y;
    for (int k = idz; k < dimZ; k += offsetZ) {
        for (int j = idy; j < dimY; j += offsetY) {
            send[k * dimY + j] = values[_ic(i, j, k)];
        }
    }
}


__global__ void recv_xy(double* recv, double* values, int dimX, int dimY, int dimZ, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    for (int j = idy; j < dimY; j += offsetY) {
        for (int i = idx; i < dimX; i += offsetX) {
            values[_ic(i, j, k)] = recv[j * dimX + i];
        }
    }
}

__global__ void recv_xz(double* recv, double* values, int dimX, int dimY, int dimZ, int j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetZ = blockDim.y * gridDim.y;
    for (int k = idz; k < dimZ; k += offsetZ) {
        for (int i = idx; i < dimX; i += offsetX) {
            values[_ic(i, j, k)] = recv[k * dimX + i];
        }
    }
}

__global__ void recv_yz(double* recv, double* values, int dimX, int dimY, int dimZ, int i) {
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetY = blockDim.x * gridDim.x;
    int offsetZ = blockDim.y * gridDim.y;
    for (int k = idz; k < dimZ; k += offsetZ) {
        for (int j = idy; j < dimY; j += offsetY) {
            values[_ic(i, j, k)] = recv[k * dimY + j];
        }
    }
}

__global__ void calculate(double* next, double* values, int dimX, int dimY, int dimZ, double hx, double hy, double hz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    int offsetZ = blockDim.z * gridDim.z;
    int i, j, k;
    for (i = idx; i < dimX; i += offsetX) {
        for (j = idy; j < dimY; j += offsetY) {
            for (k = idz; k < dimZ; k += offsetZ) {
                next[_ic(i, j, k)] = 0.5 * ((values[_ic(i + 1, j, k)] + values[_ic(i - 1, j, k)]) / (hx * hx) +
                                           (values[_ic(i, j + 1, k)] + values[_ic(i, j - 1, k)]) / (hy * hy) +
                                           (values[_ic(i, j, k + 1)] + values[_ic(i, j, k - 1)]) / (hz * hz)) /
                                           (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
            }
        }
    }
}

__global__ void error(double* next, double* values, int dimX, int dimY, int dimZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    int offsetZ = blockDim.z * gridDim.z;
    int i, j, k;
    for (i = idx - 1; i <= dimX; i += offsetX) {
        for (j = idy - 1; j <= dimY; j += offsetY) {
            for (k = idz - 1; k <= dimZ; k += offsetZ) {
                if ((i == -1) || (j == -1) || (k == -1) || (i == dimX) || (j == dimY) || (k == dimZ)) {
                    values[_ic(i, j, k)] = 0.0;
                } else {
                    values[_ic(i, j, k)] = fabs(next[_ic(i, j, k)] - values[_ic(i, j, k)]);
                }
            }
        }
    }
}

__global__ void bord_in_xy(double* values, int dimX, int dimY, int dimZ, int k, double value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;

    for (int j = idy; j < dimY; j += offsetY) {
        for (int i = idx; i < dimX; i += offsetX) {
            values[_ic(i, j, k)] = value;
        }
    }
}

__global__ void bord_in_xz(double* values, int dimX, int dimY, int dimZ, int j, double value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetZ = blockDim.y * gridDim.y;
    int i, k;

    for (k = idz; k < dimZ; k += offsetZ) {
        for (i = idx; i < dimX; i += offsetX) {
            values[_ic(i, j, k)] = value;
        }
    }
}

__global__ void bord_in_yz(double* values, int dimX, int dimY, int dimZ, int i, double value) {
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetY = blockDim.x * gridDim.x;
    int offsetZ = blockDim.y * gridDim.y;
    int j, k;

    for (k = idz; k < dimZ; k += offsetZ) {
        for (j = idy; j < dimY; j += offsetY) {
            values[_ic(i, j, k)] = value;
        }
    }
}


int main(int argc, char** argv) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
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
        cin >> u[down] >> u[up] >> u[left] >> u[right] >> u[front] >> u[back];
        cin >> u_start;
    }

    MPI_Bcast(block, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(l, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_start, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int out_filename_size = out_filename.size();
    MPI_Bcast(&out_filename_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    out_filename.resize(out_filename_size);
    MPI_Bcast((char*) out_filename.c_str(), out_filename_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    h[x_on] = l[x_on] / (block[x_on] * dim[x_on]);
    h[y_on] = l[y_on] / (block[y_on] * dim[y_on]);
    h[z_on] = l[z_on] / (block[z_on] * dim[z_on]);

    int buffer_size = max(dim[x_on], max(dim[y_on], dim[z_on])) * max(dim[x_on], max(dim[y_on], dim[z_on]));

    double* send = (double*)malloc(sizeof(double) * buffer_size);
    double* recv = (double*)malloc(sizeof(double) * buffer_size);

    int i_b = ((id) % block[x_on]);
    int j_b = (((id) / block[x_on]) % block[y_on]);
    int k_b = ((id) / (block[x_on] * block[y_on]));

    values = (double*)malloc(sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));
    next_values = (double*)malloc(sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));
    double* globDiff = (double*)malloc(sizeof(double) * proccess_count);

    double* dev_values;
    double* dev_next_values;
    double* dev_recv;
    double* dev_send;
    CSC(cudaMalloc((void**)&dev_values, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2)));
    CSC(cudaMalloc((void**)&dev_next_values, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2)));
    CSC(cudaMalloc((void**)&dev_recv, sizeof(double) * buffer_size));
    CSC(cudaMalloc((void**)&dev_send, sizeof(double) * buffer_size));
    
    for (int k = 0; k <= dim[z_on]; k++) {
        for (int j = 0; j <= dim[y_on]; j++) {
            for (int i = 0; i <= dim[x_on]; i++) {
                values[_i(i, j, k)] = u_start;
                // cerr << 'values[i, j, k] = ' << values[_i(i, j, k)] << '\n';
                // cerr << 'u_start = ' << u_start;
            }
        }
    }

    CSC(cudaMemcpy(dev_next_values, next_values, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2), cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_values, values, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2), cudaMemcpyHostToDevice));

    while (true) {
        if (i_b == 0) {
            bord_in_yz<<<dim3(32, 32), dim3(32, 32)>>>(dev_values, dim[x_on], dim[y_on], dim[z_on], -1, u[left]);
            CSC(cudaDeviceSynchronize());
        }
        if (j_b == 0) {
            bord_in_xz<<<dim3(32, 32), dim3(32, 32)>>>(dev_values, dim[x_on], dim[y_on], dim[z_on], -1, u[front]);
            CSC(cudaDeviceSynchronize());
        }
        if (k_b == 0) {
            bord_in_xy<<<dim3(32, 32), dim3(32, 32)>>>(dev_values, dim[x_on], dim[y_on], dim[z_on], -1, u[down]);
            CSC(cudaDeviceSynchronize());
        }
        if ((i_b + 1) == block[x_on]) {
            bord_in_yz<<<dim3(32, 32), dim3(32, 32)>>>(dev_values, dim[x_on], dim[y_on], dim[z_on], dim[x_on], u[right]);
            CSC(cudaDeviceSynchronize());
        }
        if ((j_b + 1) == block[y_on]) {
            bord_in_xz<<<dim3(32, 32), dim3(32, 32)>>>(dev_values, dim[x_on], dim[y_on], dim[z_on], dim[y_on], u[back]);
            CSC(cudaDeviceSynchronize());
        }
        if ((k_b + 1) == block[z_on]) {
            bord_in_xy<<<dim3(32, 32), dim3(32, 32)>>>(dev_values, dim[x_on], dim[y_on], dim[z_on], dim[z_on], u[up]);
            CSC(cudaDeviceSynchronize());
        }

        if (block[x_on] > 1) {
            if (i_b == 0) {
                send_yz<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[x_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[y_on] * dim[z_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             _ip(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[y_on] * dim[z_on], cudaMemcpyHostToDevice));
                recv_yz<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[x_on]);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
            } else if (i_b + 1 == block[x_on]) {
                send_yz<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], 0);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[y_on] * dim[z_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b), 
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b),
                             _ip(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[y_on] * dim[z_on], cudaMemcpyHostToDevice));
                recv_yz<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], -1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
            } else {
                send_yz<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[x_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[y_on] * dim[z_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b),
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b), 
                             _ip(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[y_on] * dim[z_on], cudaMemcpyHostToDevice));
                recv_yz<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], -1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                send_yz<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], 0);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[y_on] * dim[z_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b - 1, j_b, k_b), 
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b + 1, j_b, k_b), 
                             _ip(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[y_on] * dim[z_on], cudaMemcpyHostToDevice));
                recv_yz<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[x_on]);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
            }
        }

        if (block[y_on] > 1) {
            if (j_b == 0) {
                send_xz<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[y_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[x_on] * dim[z_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b, j_b + 1, k_b), 
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b, j_b + 1, k_b), 
                             _ip(i_b, j_b + 1, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[x_on] * dim[z_on], cudaMemcpyHostToDevice));
                recv_xz<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[y_on]);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
            } else if (j_b + 1 == block[y_on]) {
                send_xz<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], 0);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[x_on] * dim[z_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b, j_b - 1, k_b), 
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b, j_b - 1, k_b), 
                             _ip(i_b, j_b - 1, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[x_on] * dim[z_on], cudaMemcpyHostToDevice));
                recv_xz<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], -1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
            } else {
                send_xz<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[y_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());

                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[x_on] * dim[z_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b, j_b + 1, k_b), 
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b, j_b - 1, k_b), 
                             _ip(i_b, j_b - 1, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[x_on] * dim[z_on], cudaMemcpyHostToDevice));
                recv_xz<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], -1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                send_xz<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], 0);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[x_on] * dim[z_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b, j_b - 1, k_b), 
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b, j_b + 1, k_b), 
                             _ip(i_b, j_b + 1, k_b), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[x_on] * dim[z_on], cudaMemcpyHostToDevice));
                recv_xz<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[y_on]);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
            }
        }

        if (block[z_on] > 1) {
            if (k_b == 0) {
                send_xy<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[z_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[x_on] * dim[y_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b, j_b, k_b + 1), 
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b, j_b, k_b + 1),
                             _ip(i_b, j_b, k_b + 1), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[x_on] * dim[y_on], cudaMemcpyHostToDevice));
                recv_xy<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[z_on]);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
            } else if (k_b + 1 == block[z_on]) {
                send_xy<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], 0);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[x_on] * dim[y_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b, j_b, k_b - 1), 
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b, j_b, k_b - 1),
                             _ip(i_b, j_b, k_b - 1), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[x_on] * dim[y_on], cudaMemcpyHostToDevice));
                recv_xy<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], -1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
            } else {
                send_xy<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[z_on] - 1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[x_on] * dim[y_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b, j_b, k_b + 1),
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b, j_b, k_b - 1),
                             _ip(i_b, j_b, k_b - 1), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[x_on] * dim[y_on], cudaMemcpyHostToDevice));
                recv_xy<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], -1);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                send_xy<<<dim3(32, 32), dim3(32, 32)>>>(dev_send, dev_values, dim[x_on], dim[y_on], dim[z_on], 0);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
                CSC(cudaMemcpy(send, dev_send, sizeof(double) * dim[x_on] * dim[y_on], cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _ip(i_b, j_b, k_b - 1),
                             id, recv, buffer_size, MPI_DOUBLE, _ip(i_b, j_b, k_b + 1),
                             _ip(i_b, j_b, k_b + 1), MPI_COMM_WORLD, &status);
                CSC(cudaMemcpy(dev_recv, recv, sizeof(double) * dim[x_on] * dim[y_on], cudaMemcpyHostToDevice));
                recv_xy<<<dim3(32, 32), dim3(32, 32)>>>(dev_recv, dev_values, dim[x_on], dim[y_on], dim[z_on], dim[z_on]);
                CSC(cudaGetLastError());
                CSC(cudaDeviceSynchronize());
            }
        }

        calculate<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(dev_next_values, dev_values, dim[x_on], dim[y_on], dim[z_on], h[x_on], h[y_on], h[z_on]);
        CSC(cudaGetLastError());

        error<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(dev_next_values, dev_values, dim[x_on], dim[y_on], dim[z_on]);
        CSC(cudaGetLastError());

        thrust::device_ptr<double> p_values = thrust::device_pointer_cast(dev_values);
        thrust::device_ptr<double> p_max_diff = thrust::max_element(p_values, p_values + (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2));
        double diff = *p_max_diff;

        MPI_Allgather(&diff, 1, MPI_DOUBLE, globDiff, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        for (int i = 0; i < proccess_count; ++i) {
            diff = max(diff, globDiff[i]);
        }

        if (diff < eps) {
            break;
        }

        temporary_values = dev_next_values;
        dev_next_values = dev_values;
        dev_values = temporary_values;
    }

    CSC(cudaMemcpy(values, dev_values, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2), cudaMemcpyDeviceToHost));
    CSC(cudaMemcpy(next_values, dev_next_values, sizeof(double) * (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2), cudaMemcpyDeviceToHost));


    CSC(cudaFree(dev_values));
    CSC(cudaFree(dev_next_values));
    CSC(cudaFree(dev_send));
    CSC(cudaFree(dev_recv));

    int buff_size = (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2);
    int new_symbol_size = 14;

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

    sizes[x_on] = dim[x_on] + 2;
    sizes[y_on] = dim[y_on] + 2;
    sizes[z_on] = dim[z_on] + 2;
    starts[x_on] = starts[y_on] = starts[z_on] = 1;

    f_sizes[x_on] = dim[x_on] * block[x_on];
    f_sizes[y_on] = dim[y_on] * block[y_on];
    f_sizes[z_on] = dim[z_on] * block[z_on];

    f_starts[x_on] = dim[x_on] * i_b;
    f_starts[y_on] = dim[y_on] * j_b;
    f_starts[z_on] = dim[z_on] * k_b;

    MPI_Type_create_subarray(3, sizes, dim, starts, MPI_ORDER_FORTRAN, new_representation, &memtype);
    MPI_Type_commit(&memtype);
    MPI_Type_create_subarray(3, f_sizes, dim, f_starts, MPI_ORDER_FORTRAN, new_representation, &filetype);
    MPI_Type_commit(&filetype);

    MPI_File fp;
    MPI_File_delete(out_filename.c_str(), MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, out_filename.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fp);

    MPI_File_set_view(fp, 0, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fp, buff, 1, memtype, MPI_STATUS_IGNORE);

    MPI_File_close(&fp);
    MPI_Finalize();

    free(values);
    free(next_values);
    free(send);
    free(recv);
}
