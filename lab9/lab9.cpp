#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <stdlib.h>
#include <time.h>

#include "mpi.h"
#include "omp.h"

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

// макросы _i, чтобы у нас можно было использовать в индексации -1 и n
#define _i(i, j, k) (((k) + 1) * (dim[y_on] + 2) * (dim[x_on] + 2) + ((j) + 1) * (dim[x_on] + 2) + (i) + 1)

#define _idp(i,j, k) ((k) * block[y_on] * block[x_on] + (j) * block[x_on] + (i))
//#define _idb(i, j) (j * buffer_length + i)

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
    MPI_Datatype edge_xy;
    MPI_Datatype edge_yz;
    MPI_Datatype edge_xz;

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
    int out_filename_size = out_filename.size();
    MPI_Bcast(&out_filename_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    out_filename.resize(out_filename_size);
    MPI_Bcast((char*) out_filename.c_str(), out_filename_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    h[x_on] = l[x_on] / (block[x_on] * dim[x_on]);
    h[y_on] = l[y_on] / (block[y_on] * dim[y_on]);
    h[z_on] = l[z_on] / (block[z_on] * dim[z_on]);

    int i_b = ((id) % block[x_on]);
    int j_b = (((id) / block[x_on]) % block[y_on]);
    int k_b = ((id) / (block[x_on] * block[y_on]));

    values = (double*)malloc(sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));
    next_values = (double*)malloc(sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));
    double* globDiff = (double*)malloc(sizeof(double) * proccess_count);

    // вычисляем размеры блоков и смещения каждого блока
    int* dimXY = (int*)malloc(sizeof(int) * dim[y_on]); // Плоскость xy
    int* dimXZ = (int*)malloc(sizeof(int) * dim[y_on] * dim[z_on]); // Плоскость zx
    int* dimYZ = (int*)malloc(sizeof(int) * dim[z_on]); // Плоскость yz

    int* offsetXY = (int*)malloc(sizeof(int) * dim[y_on]);
    int* offsetXZ = (int*)malloc(sizeof(int) * dim[y_on] * dim[z_on]);
    int* offsetYZ = (int*)malloc(sizeof(int) * dim[z_on]);

    // Плоскость xy
    int idx = dim[x_on] + 3;
    for (int i = 0; i < dim[y_on]; ++i) {
        offsetXY[i] = idx;
        idx += dim[x_on] + 2;
        dimXY[i] = dim[x_on];
    }

    MPI_Type_indexed(dim[y_on], dimXY, offsetXY, MPI_DOUBLE, &edge_xy);
    MPI_Type_commit(&edge_xy);

    // Плоскость yz
    int idy = (dim[x_on] + 2) * (dim[y_on] + 2) + 1;
    for (int i = 0; i < dim[z_on]; ++i) {
        offsetYZ[i] = idy;
        idy += (dim[x_on] + 2) * (dim[y_on] + 2);
        dimYZ[i] = dim[x_on];
    }

    // создаём новый тип функции с помощью type_indexed
    MPI_Type_indexed(dim[z_on], dimYZ, offsetYZ, MPI_DOUBLE, &edge_yz);
    MPI_Type_commit(&edge_yz); //Регистрируем его в mpi

    // Плоскость zx
    int idz = (dim[x_on] + 2) * (dim[y_on] + 2) + dim[x_on] + 2;
    for (int k = 0; k < dim[z_on]; ++k) {
        for (int j = 0; j < dim[y_on]; ++j) {
            offsetXZ[k * dim[y_on] + j] = idz;
            idz += (dim[x_on] + 2);
            dimXZ[k * dim[y_on] + j] = 1;
        }
        idz += (dim[x_on] + 2) * 2;
    }

    // создаём новый тип функции с помощью type_indexed
    MPI_Type_indexed(dim[y_on] * dim[z_on], dimXZ, offsetXZ, MPI_DOUBLE, &edge_xz);
    MPI_Type_commit(&edge_xz); //Регистрируем его в mpi

    // очищаем память от ненужных данных

    for (int k = 0; k <= dim[z_on]; k++) {					// Инициализация блока
        for (int j = 0; j <= dim[y_on]; j++) {
            for (int i = 0; i <= dim[x_on]; i++) {
                values[_i(i, j, k)] = u_start;
            }
        }
    }

    int i;
    int j;
    int k;
    if (i_b == 0) {
#pragma omp parallel for private(i, j, k) shared(values, next_values)
        for (k = 0; k < dim[z_on]; ++k)
            for (j = 0; j < dim[y_on]; ++j) {
                values[_i(-1, j, k)] = u[lft];
                next_values[_i(-1, j, k)] = u[lft];
            }
    }

    if (j_b == 0) {
#pragma omp parallel for private(i, j, k) shared(values, next_values)
        for (k = 0; k < dim[z_on]; ++k)
            for (i = 0; i < dim[x_on]; ++i) {
                values[_i(i, -1, k)] = u[front];
                next_values[_i(i, -1, k)] = u[front];
            }
    }

    if (k_b == 0) {
#pragma omp parallel for private(i, j, k) shared(values, next_values)
        for (j = 0; j < dim[y_on]; ++j)
            for (i = 0; i < dim[x_on]; ++i) {
                values[_i(i, j, -1)] = u[down];
                next_values[_i(i, j, -1)] = u[down];
            }
    }

    if ((i_b + 1) == block[x_on]) {
#pragma omp parallel for private(i, j, k) shared(values, next_values)
        for (k = 0; k < dim[z_on]; ++k)
            for (j = 0; j < dim[y_on]; ++j) {
                values[_i(dim[0], j, k)] = u[rght];
                next_values[_i(dim[0], j, k)] = u[rght];
            }
    }

    if ((j_b + 1) == block[y_on]) {
#pragma omp parallel for private(i, j, k) shared(values, next_values)
        for (k = 0; k < dim[z_on]; ++k)
            for (i = 0; i < dim[x_on]; ++i) {
                values[_i(i, dim[1], k)] = u[back];
                next_values[_i(i, dim[1], k)] = u[back];
            }
    }

    if ((k_b + 1) == block[z_on]) {
#pragma omp parallel for private(i, j, k) shared(values, next_values)
        for (j = 0; j < dim[y_on]; ++j)
            for (i = 0; i < dim[x_on]; ++i) {
                values[_i(i, j, dim[z_on])] = u[up];
                next_values[_i(i, j, dim[z_on])] = u[up];
            }
    }

    while (true) {
        if (block[x_on] > 1) {
            if (i_b == 0) {
                MPI_Sendrecv(values + dim[x_on], 1, edge_xz, _idp(i_b + 1, j_b, k_b), id,
                    values + dim[x_on] + 1, 1, edge_xz, _idp(i_b + 1, j_b, k_b), _idp(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
            } else if (i_b + 1 == block[x_on]) {
                MPI_Sendrecv(values + 1, 1, edge_xz, _idp(i_b - 1, j_b, k_b), id,
                    values, 1, edge_xz, _idp(i_b - 1, j_b, k_b), _idp(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
            } else {
                MPI_Sendrecv(values + dim[x_on], 1, edge_xz, _idp(i_b + 1, j_b, k_b), id,
                    values, 1, edge_xz, _idp(i_b - 1, j_b, k_b), _idp(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                MPI_Sendrecv(values + 1, 1, edge_xz, _idp(i_b - 1, j_b, k_b), id,
                    values + dim[x_on] + 1, 1, edge_xz, _idp(i_b + 1, j_b, k_b), _idp(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
            }
        }


        if (block[y_on] > 1) {
            if (j_b == 0) {
                MPI_Sendrecv(values + (dim[x_on] + 2) * dim[y_on], 1, edge_yz, _idp(i_b, j_b + 1, k_b), id,
                    values + (dim[x_on] + 2) * (dim[y_on] + 1), 1, edge_yz, _idp(i_b, j_b + 1, k_b), _idp(i_b, j_b + 1, k_b), MPI_COMM_WORLD, &status);
            } else if (j_b + 1 == block[y_on]) {
                MPI_Sendrecv(values + dim[x_on] + 2, 1, edge_yz, _idp(i_b, j_b - 1, k_b), id,
                    values, 1, edge_yz, _idp(i_b, j_b - 1, k_b), _idp(i_b, j_b - 1, k_b), MPI_COMM_WORLD, &status);
            } else {
                MPI_Sendrecv(values + (dim[x_on] + 2) * dim[y_on], 1, edge_yz, _idp(i_b, j_b + 1, k_b), id,
                    values, 1, edge_yz, _idp(i_b, j_b - 1, k_b), _idp(i_b, j_b - 1, k_b), MPI_COMM_WORLD, &status);
                MPI_Sendrecv(values + dim[x_on] + 2, 1, edge_yz, _idp(i_b, j_b - 1, k_b), id,
                    values + (dim[x_on] + 2) * (dim[y_on] + 1), 1, edge_yz, _idp(i_b, j_b + 1, k_b), _idp(i_b, j_b + 1, k_b), MPI_COMM_WORLD, &status);
            }
        }

        if (block[z_on] > 1) {
            if (k_b == 0) {
                MPI_Sendrecv(values + (dim[x_on] + 2) * (dim[y_on] + 2) * dim[z_on], 1, edge_xy, _idp(i_b, j_b, k_b + 1), id,
                    values + (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 1), 1, edge_xy, _idp(i_b, j_b, k_b + 1), _idp(i_b, j_b, k_b + 1), MPI_COMM_WORLD, &status);
            } else if (k_b + 1 == block[z_on]) {
                MPI_Sendrecv(values + (dim[x_on] + 2) * (dim[y_on] + 2), 1, edge_xy, _idp(i_b, j_b, k_b - 1), id,
                    values, 1, edge_xy, _idp(i_b, j_b, k_b - 1), _idp(i_b, j_b, k_b - 1), MPI_COMM_WORLD, &status);
            } else {
                MPI_Sendrecv(values + (dim[x_on] + 2) * (dim[y_on] + 2) * dim[z_on], 1, edge_xy, _idp(i_b, j_b, k_b + 1), id,
                    values, 1, edge_xy, _idp(i_b, j_b, k_b - 1), _idp(i_b, j_b, k_b - 1), MPI_COMM_WORLD, &status);
                MPI_Sendrecv(values + (dim[x_on] + 2) * (dim[y_on] + 2), 1, edge_xy, _idp(i_b, j_b, k_b - 1), id,
                    values + (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 1), 1, edge_xy, _idp(i_b, j_b, k_b + 1), _idp(i_b, j_b, k_b + 1), MPI_COMM_WORLD, &status);
            }
        }

        double diff = 0.0;

#pragma omp parallel for private(i, j, k) shared(values, next_values) reduction(max: diff)
        for (k = 0; k < dim[z_on]; ++k) {
            for (j = 0; j < dim[y_on]; ++j) {
                for (i = 0; i < dim[x_on]; ++i) {
                    double val = 0.5 * (
                        (values[_i(i + 1, j, k)] + values[_i(i - 1, j, k)]) / (h[x_on] * h[x_on]) +
                        (values[_i(i, j + 1, k)] + values[_i(i, j - 1, k)]) / (h[y_on] * h[y_on]) +
                        (values[_i(i, j, k + 1)] + values[_i(i, j, k - 1)]) / (h[z_on] * h[z_on])
                        ) / (1 / (h[x_on] * h[x_on]) + 1 / (h[y_on] * h[y_on]) + 1 / (h[z_on] * h[z_on]));

                    next_values[_i(i, j, k)] = val;
                    diff = std::max(diff, abs(val - values[_i(i, j, k)]));
                }
            }
        }

        MPI_Allgather(&diff, 1, MPI_DOUBLE, globDiff, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        for (int i = 0; i < proccess_count; ++i) {
            diff = max(diff, globDiff[i]);
        }

        if (diff < eps) {
            break;
        }

        temporary_values = next_values;
        next_values = values;
        values = temporary_values;
    }

    int n_size = 14;

    char* buffer_char = (char*)malloc(sizeof(char) * (dim[x_on]) * (dim[y_on]) * (dim[z_on]) * n_size);
    memset(buffer_char, ' ', (dim[x_on]) * (dim[y_on]) * (dim[z_on]) * n_size);

    for (int k = 0; k < dim[z_on]; ++k) {
        for (int j = 0; j < dim[y_on]; ++j) {
            for (int i = 0; i < dim[x_on]; ++i) {
                sprintf(buffer_char + ((k)*dim[y_on] * dim[x_on] + (j)*dim[x_on] + (i)) * n_size, "%.6e", next_values[_i(i, j, k)]);
            }
        }
    }

    for (int i = 0; i < (dim[x_on]) * (dim[y_on]) * (dim[z_on]) * n_size; ++i) {
        if (buffer_char[i] == '\0') {
            buffer_char[i] = ' ';
        }
    }

    MPI_Datatype num;
    MPI_Type_contiguous(n_size, MPI_CHAR, &num);
    MPI_Type_commit(&num);

    MPI_File fp;
    MPI_Datatype filetype;

    int* sizes = (int*)malloc(sizeof(int) * dim[y_on] * dim[z_on]);
    int* offsets = (int*)malloc(sizeof(int) * dim[y_on] * dim[z_on]);
    for (int i = 0; i < dim[y_on] * dim[z_on]; ++i) {
        sizes[i] = dim[x_on];
    }

    MPI_Aint offset_write = i_b * dim[x_on] + j_b * dim[y_on] * dim[x_on] * block[x_on] + k_b * dim[z_on] * dim[x_on] * block[x_on] * dim[y_on] * block[y_on];

    for (int k = 0; k < dim[z_on]; ++k) {
        for (int j = 0; j < dim[y_on]; ++j) {
            int i = k * dim[y_on] + j;
            offsets[i] = j * dim[x_on] * block[x_on] + k * dim[x_on] * block[x_on] * dim[y_on] * block[y_on];
        }
    }

    MPI_Type_indexed(dim[y_on] * dim[z_on], sizes, offsets, num, &filetype);
    MPI_Type_commit(&filetype);

    MPI_File_delete(output_name, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, output_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);

    MPI_File_set_view(fp, offset_write * n_size, num, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fp, buffer_char, (dim[x_on]) * (dim[y_on]) * (dim[z_on]) * n_size, MPI_CHAR, &status);

    MPI_File_close(&fp);
    // закрытие файла

    // очищаем память
    free(buffer_char);
    free(sizes);
    free(offsets);
    free(next_values);
    free(values);
    free(dimXY);
    free(offsetXY);
    free(dimYZ);
    free(offsetYZ);
    free(dimXZ);
    free(offsetXZ);

    MPI_Finalize(); // завершаем MPI
}
