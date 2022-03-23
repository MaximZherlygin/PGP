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

#define _i(i, j, k) (((k) + 1) * (dim[y_on] + 2) * (dim[x_on] + 2) + ((j) + 1) * (dim[x_on] + 2) + (i) + 1)
#define _ip(i,j, k) ((k) * block[y_on] * block[x_on] + (j) * block[x_on] + (i))

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

    int dimXY[1];
    dimXY[0] = dim[y_on];
    int dimXZ[1];
    dimXZ[0] = dim[y_on] * dim[z_on];
    int dimYZ[1];
    dimYZ[0] = dim[z_on];
    int begin[1];
    begin[0] = 0;

    MPI_Datatype r_type;
    MPI_Type_contiguous(dim[y_on], MPI_DOUBLE, &r_type);
    MPI_Type_commit(&r_type);

    MPI_Type_create_subarray(1, dimXY, dimXY, begin, MPI_ORDER_FORTRAN, r_type, &edge_xy);
    MPI_Type_commit(&edge_xy);

    MPI_Type_create_subarray(1, dimYZ, dimYZ, begin, MPI_ORDER_FORTRAN, r_type, &edge_yz);
    MPI_Type_commit(&edge_yz);

    MPI_Type_create_subarray(1, dimXZ, dimXZ, begin, MPI_ORDER_FORTRAN, r_type, &edge_xz);
    MPI_Type_commit(&edge_xz);

    for (int k = 0; k <= dim[z_on]; k++) {
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
                next_values[_i(-1, j, k)] = u[left];
                values[_i(-1, j, k)] = u[left];
            }
    }
    if (j_b == 0) {
    #pragma omp parallel for private(i, j, k) shared(values, next_values)
        for (k = 0; k < dim[z_on]; ++k)
            for (i = 0; i < dim[x_on]; ++i) {
                next_values[_i(i, -1, k)] = u[front];
                values[_i(i, -1, k)] = u[front];
            }
    }
    if (k_b == 0) {
    #pragma omp parallel for private(i, j, k) shared(values, next_values)
        for (j = 0; j < dim[y_on]; ++j)
            for (i = 0; i < dim[x_on]; ++i) {
                next_values[_i(i, j, -1)] = u[down];
                values[_i(i, j, -1)] = u[down];
            }
    }
    if ((i_b + 1) == block[x_on]) {
    #pragma omp parallel for private(i, j, k) shared(values, next_values)
        for (k = 0; k < dim[z_on]; ++k)
            for (j = 0; j < dim[y_on]; ++j) {
                next_values[_i(dim[0], j, k)] = u[right];
                values[_i(dim[0], j, k)] = u[right];
            }
    }
    if ((j_b + 1) == block[y_on]) {
    #pragma omp parallel for private(i, j, k) shared(values, next_values)
        for (k = 0; k < dim[z_on]; ++k)
            for (i = 0; i < dim[x_on]; ++i) {
                next_values[_i(i, dim[1], k)] = u[back];
                values[_i(i, dim[1], k)] = u[back];
            }
    }
    if ((k_b + 1) == block[z_on]) {
    #pragma omp parallel for private(i, j, k) shared(values, next_values)
        for (j = 0; j < dim[y_on]; ++j)
            for (i = 0; i < dim[x_on]; ++i) {
                next_values[_i(i, j, dim[z_on])] = u[up];
                values[_i(i, j, dim[z_on])] = u[up];
            }
    }

    while (true) {
        if (block[x_on] > 1) {
            if (i_b == 0) {
                MPI_Sendrecv(values + dim[x_on], 1, edge_xz, _ip(i_b + 1, j_b, k_b),
                             id, values + dim[x_on] + 1, 1, edge_xz, _ip(i_b + 1, j_b, k_b),
                             _ip(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
            } else if (i_b + 1 == block[x_on]) {
                MPI_Sendrecv(values + 1, 1, edge_xz, _ip(i_b - 1, j_b, k_b),
                             id, values, 1, edge_xz, _ip(i_b - 1, j_b, k_b),
                             _ip(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
            } else {
                MPI_Sendrecv(values + dim[x_on], 1, edge_xz, _ip(i_b + 1, j_b, k_b),
                             id, values, 1, edge_xz, _ip(i_b - 1, j_b, k_b),
                             _ip(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                MPI_Sendrecv(values + 1, 1, edge_xz, _ip(i_b - 1, j_b, k_b),
                             id, values + dim[x_on] + 1, 1, edge_xz, _ip(i_b + 1, j_b, k_b),
                             _ip(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
            }
        }


        if (block[y_on] > 1) {
            if (j_b == 0) {
                MPI_Sendrecv(values + (dim[x_on] + 2) * dim[y_on], 1, edge_yz, _ip(i_b, j_b + 1, k_b),
                             id, values + (dim[x_on] + 2) * (dim[y_on] + 1), 1, edge_yz, _ip(i_b, j_b + 1, k_b),
                             _ip(i_b, j_b + 1, k_b), MPI_COMM_WORLD, &status);
            } else if (j_b + 1 == block[y_on]) {
                MPI_Sendrecv(values + dim[x_on] + 2, 1, edge_yz, _ip(i_b, j_b - 1, k_b),
                             id, values, 1, edge_yz, _ip(i_b, j_b - 1, k_b),
                             _ip(i_b, j_b - 1, k_b), MPI_COMM_WORLD, &status);
            } else {
                MPI_Sendrecv(values + (dim[x_on] + 2) * dim[y_on], 1, edge_yz, _ip(i_b, j_b + 1, k_b),
                             id, values, 1, edge_yz, _ip(i_b, j_b - 1, k_b),
                             _ip(i_b, j_b - 1, k_b), MPI_COMM_WORLD, &status);
                MPI_Sendrecv(values + dim[x_on] + 2, 1, edge_yz, _ip(i_b, j_b - 1, k_b),
                             id, values + (dim[x_on] + 2) * (dim[y_on] + 1), 1, edge_yz, _ip(i_b, j_b + 1, k_b),
                             _ip(i_b, j_b + 1, k_b), MPI_COMM_WORLD, &status);
            }
        }

        if (block[z_on] > 1) {
            if (k_b == 0) {
                MPI_Sendrecv(values + (dim[x_on] + 2) * (dim[y_on] + 2) * dim[z_on], 1, edge_xy, _ip(i_b, j_b, k_b + 1),
                             id, values + (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 1), 1, edge_xy, _ip(i_b, j_b, k_b + 1),
                             _ip(i_b, j_b, k_b + 1), MPI_COMM_WORLD, &status);
            } else if (k_b + 1 == block[z_on]) {
                MPI_Sendrecv(values + (dim[x_on] + 2) * (dim[y_on] + 2), 1, edge_xy, _ip(i_b, j_b, k_b - 1),
                             id, values, 1, edge_xy, _ip(i_b, j_b, k_b - 1),
                             _ip(i_b, j_b, k_b - 1), MPI_COMM_WORLD, &status);
            } else {
                MPI_Sendrecv(values + (dim[x_on] + 2) * (dim[y_on] + 2) * dim[z_on], 1, edge_xy, _ip(i_b, j_b, k_b + 1),
                             id, values, 1, edge_xy, _ip(i_b, j_b, k_b - 1),
                             _ip(i_b, j_b, k_b - 1), MPI_COMM_WORLD, &status);
                MPI_Sendrecv(values + (dim[x_on] + 2) * (dim[y_on] + 2), 1, edge_xy, _ip(i_b, j_b, k_b - 1),
                             id, values + (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 1), 1, edge_xy, _ip(i_b, j_b, k_b + 1),
                             _ip(i_b, j_b, k_b + 1), MPI_COMM_WORLD, &status);
            }
        }

        double diff = 0.0;

        #pragma omp parallel for private(i, j, k) shared(values, next_values) reduction(max: diff)
        for (k = 0; k < dim[z_on]; ++k) {
            for (j = 0; j < dim[y_on]; ++j) {
                for (i = 0; i < dim[x_on]; ++i) {
                    next_values[_i(i, j, k)] = 0.5 * ((values[_i(i + 1, j, k)] + values[_i(i - 1, j, k)]) / (h[x_on] * h[x_on]) +
                                                     (values[_i(i, j + 1, k)] + values[_i(i, j - 1, k)]) / (h[y_on] * h[y_on]) +
                                                     (values[_i(i, j, k + 1)] + values[_i(i, j, k - 1)]) / (h[z_on] * h[z_on])) /
                                                     (1 / (h[x_on] * h[x_on]) + 1 / (h[y_on] * h[y_on]) + 1 / (h[z_on] * h[z_on]));

                    diff = max(abs(next_values[_i(i, j, k)] - values[_i(i, j, k)]), diff);
                }
            }
        }

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
    }

    int char_len = 14;
    int buff_len = (dim[x_on] + 2) * (dim[y_on] + 2) * (dim[z_on] + 2);

    char* container = new char[char_len * buff_len];
    memset(container, (char)' ', char_len * buff_len * sizeof(char));

    for (k = 0; k < dim[z_on]; k++) {
        for (j = 0; j < dim[y_on]; j++) {
            int len_new_symbol;
            for (i = 0; i < dim[x_on] - 1; i++) {
                len_new_symbol = sprintf(&container[_i(i, j, k) * char_len], "%.6e", next_values[_i(i, j, k)]);
                if (char_len > len_new_symbol) {
                    container[_i(i, j, k) * char_len + len_new_symbol] = ' ';
                }
            }
            len_new_symbol = sprintf(&container[_i(i, j, k) * char_len], "%.6e\n", next_values[_i(i, j, k)]);
            if (char_len > len_new_symbol) {
                container[_i(i, j, k) * char_len + len_new_symbol] = ' ';
            }
        }
    }

    int size[3];
    int beg[3];
    int file_size[3];
    int file_beg[3];
    MPI_Datatype repr_type;
    MPI_Datatype memory_type;
    MPI_Datatype file_type;

    MPI_Type_contiguous(char_len, MPI_CHAR, &repr_type);
    MPI_Type_commit(&repr_type);

    beg[x_on] = 1;
    beg[y_on] = 1;
    beg[z_on] = 1;
    file_beg[x_on] = dim[x_on] * i_b;
    file_beg[y_on] = dim[y_on] * j_b;
    file_beg[z_on] = dim[z_on] * k_b;
    size[x_on] = dim[x_on] + 2;
    size[y_on] = dim[y_on] + 2;
    size[z_on] = dim[z_on] + 2;
    file_size[x_on] = dim[x_on] * block[x_on];
    file_size[y_on] = dim[y_on] * block[y_on];
    file_size[z_on] = dim[z_on] * block[z_on];

    MPI_Type_create_subarray(3, size, dim, beg, MPI_ORDER_FORTRAN, repr_type, &memory_type);
    MPI_Type_commit(&memory_type);
    MPI_Type_create_subarray(3, file_size, dim, file_beg, MPI_ORDER_FORTRAN, repr_type, &file_type);
    MPI_Type_commit(&file_type);

    MPI_File file;
    MPI_File_delete(out_filename.c_str(), MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, out_filename.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);
    MPI_File_set_view(file, 0, MPI_CHAR, file_type, "native", MPI_INFO_NULL);
    MPI_File_write_all(file, container, 1, memory_type, MPI_STATUS_IGNORE);
    MPI_File_close(&file);

    // очищаем память
    free(next_values);
    free(values);

    MPI_Finalize(); // завершаем MPI
}
