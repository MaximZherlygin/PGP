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

const int x_dim = 0;
const int y_dim = 1;
const int z_dim = 2;

using namespace std;

// �����������
enum {
    down, up, lft, rght, front, back
};

// ������� _i, ����� � ��� ����� ���� ������������ � ���������� -1 � n
#define _i(i, j, k) (((k) + 1) * (nb[y_dim] + 2) * (nb[x_dim] + 2) + ((j) + 1) * (nb[x_dim] + 2) + (i) + 1)

#define _idp(i,j, k) ((k) * n[y_dim] * n[x_dim] + (j) * n[x_dim] + (i))
//#define _idb(i, j) (j * buffer_length + i)

int main(int argc, char** argv) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    // ������� ��������� ��� MPI_Init
    int nb[3];
    int n[3];
    int i, j, k;
    //int it; //it - ������� ��������
    int id, numproc, proc_name_len; // id - ����� ��������, numproc - ���������� ���������
    char proc_name[MPI_MAX_PROCESSOR_NAME]; // � ������� �������� ����� �������� ���
    // ������, �� ������� ����� ��� �������
    char output_name[1024];  // �������� ����� �����
    double e; // �������� ���������� (��������)
    double l[3], h[3], bc[6], bc_start; // x, y, z - ������ �������
    // x, y, z - ���������� � �������
    // down up left right front back
    double* data, * temp, * next; //*buff;
    // data � next - ������� ��� ������
    // temp - � ������ �������� �� ����� ������ ������� ����� �����
    // buff - ���������� ������. ��� ����������� ������� ��� ����� ���������� �������
    // ������, � ����� ����� �������� ������� �������� (�������� ������ ����� �����)

    MPI_Status status;  // ��������� status
    MPI_Init(&argc, &argv); // ������������� MPI
    MPI_Comm_size(MPI_COMM_WORLD, &numproc); // �������� ���������� ���������
    MPI_Comm_rank(MPI_COMM_WORLD, &id); // �������� ���������� ���������
    MPI_Get_processor_name(proc_name, &proc_name_len); // �������� ��� ���������

    if (id == 0) {                                  // ������������� ���������� �������
        cin >> n[x_dim] >> n[y_dim] >> n[z_dim];    // ������ ����� �� ������ ���������
        cin >> nb[x_dim] >> nb[y_dim] >> nb[z_dim]; // ������ ����� ������ (���������) �� ������ ���������	
        cin >> output_name >> e;
        cin >> l[x_dim] >> l[y_dim] >> l[z_dim];
        cin >> bc[down] >> bc[up] >> bc[lft] >> bc[rght] >> bc[front] >> bc[back];
        cin >> bc_start;
    }

    MPI_Bcast(n, 3, MPI_INT, 0, MPI_COMM_WORLD);    // �������� ���������� ������� ���� ���������
    MPI_Bcast(nb, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(output_name, 1024, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&e, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(l, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(bc, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_start, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // ������� ������ ��� ���� "������" 
    // n+2 - ������������ ��������� ������ �� ���������
    // (�� ��������� �������)
    data = (double*)malloc(sizeof(double) * (nb[0] + 2) * (nb[1] + 2) * (nb[2] + 2));
    next = (double*)malloc(sizeof(double) * (nb[0] + 2) * (nb[1] + 2) * (nb[2] + 2));

    // ������������ ����������� ��� ������, ��� ����� ������� ���� ���� ������, ��� � ������� ����
    MPI_Datatype xy_face, yz_face, zx_face;

    // ��������� ������� ������ � �������� ������� �����
    int* sizes_xy = (int*)malloc(sizeof(int) * nb[y_dim]); // ��������� xy
    int* offsets_xy = (int*)malloc(sizeof(int) * nb[y_dim]);
    int* sizes_yz = (int*)malloc(sizeof(int) * nb[z_dim]); // ��������� yz
    int* offsets_yz = (int*)malloc(sizeof(int) * nb[z_dim]);
    int* sizes_zx = (int*)malloc(sizeof(int) * nb[y_dim] * nb[z_dim]); // ��������� zx
    int* offsets_zx = (int*)malloc(sizeof(int) * nb[y_dim] * nb[z_dim]);

    // ��������� xy
    int idx = nb[x_dim] + 3;
    for (size_t i = 0; i < nb[y_dim]; ++i) {
        sizes_xy[i] = nb[x_dim];
        offsets_xy[i] = idx;
        idx += nb[x_dim] + 2;
    }

    // ������ ����� ��� ������� � ������� type_indexed
    // ���������� ������ (x,y,z), ������ �����, ���������� ����� �������, ��� ������ � ����� ��� ������

       // � ������� ����������� �� ����� ��� � ��� ����� n ��������� ���� int
    //����������� ���� MPI_Type_indexed �������� ������������ ���� ������� �� ������������ �� ����� ������ � ������������ ��������� ������ �� ������ ���������� ��������. 
    //�������� ���������� � ��������� ������� ����.

    // ������������ ������� ������������ MPI_Type_hindexed �� indexed � ���, ��� �������� ���������� � ������.

    MPI_Type_indexed(nb[y_dim], sizes_xy, offsets_xy, MPI_DOUBLE, &xy_face);
    MPI_Type_commit(&xy_face); //������������ ��� � mpi

    // ��������� yz
    idx = (nb[x_dim] + 2) * (nb[y_dim] + 2) + 1;
    for (size_t i = 0; i < nb[z_dim]; ++i) {
        sizes_yz[i] = nb[x_dim];
        offsets_yz[i] = idx;
        idx += (nb[x_dim] + 2) * (nb[y_dim] + 2);
    }

    // ������ ����� ��� ������� � ������� type_indexed
    MPI_Type_indexed(nb[z_dim], sizes_yz, offsets_yz, MPI_DOUBLE, &yz_face);
    MPI_Type_commit(&yz_face); //������������ ��� � mpi

    // ��������� zx
    idx = (nb[x_dim] + 2) * (nb[y_dim] + 2) + nb[x_dim] + 2;
    for (size_t k = 0; k < nb[z_dim]; ++k) {
        for (size_t j = 0; j < nb[y_dim]; ++j) {
            sizes_zx[k * nb[y_dim] + j] = 1;
            offsets_zx[k * nb[y_dim] + j] = idx;
            idx += (nb[x_dim] + 2);
        }
        idx += 2 * (nb[x_dim] + 2);
    }

    // ������ ����� ��� ������� � ������� type_indexed
    MPI_Type_indexed(nb[y_dim] * nb[z_dim], sizes_zx, offsets_zx, MPI_DOUBLE, &zx_face);
    MPI_Type_commit(&zx_face); //������������ ��� � mpi

    // ������� ������ �� �������� ������
    free(sizes_xy);
    free(offsets_xy);
    free(sizes_yz);
    free(offsets_yz);
    free(sizes_zx);
    free(offsets_zx);


    // b (0-2) - ����� ���������
    // �������� ������ ����� ib jb � kb
    int ib = id % n[x_dim];
    int jb = (id / n[x_dim]) % n[y_dim];
    int kb = id / (n[x_dim] * n[y_dim]); // �� ���������� � ��������� ����������

    // ��������� hx, hy � hz.
    h[x_dim] = l[x_dim] / (n[x_dim] * nb[x_dim]);
    h[y_dim] = l[y_dim] / (n[y_dim] * nb[y_dim]);
    h[z_dim] = l[z_dim] / (n[z_dim] * nb[z_dim]);

    for (int k = 0; k <= nb[z_dim]; k++) {					// ������������� �����
        for (int j = 0; j <= nb[y_dim]; j++) {
            for (int i = 0; i <= nb[x_dim]; i++) {
                data[_i(i, j, k)] = bc_start;
            }
        }
    }

    // ���� � ��� ����� ��������� ��������
    // �� �� �������� �� ����� ��, ��� ���� �� ��������

    // � ������ � ������ �� ��� ������� �� ��������
    // ��� �������������� ��� ������� �������������� ����� � ������� ���� �������� �� �������
    if (ib == 0) {
        //private - ����� ������ ������ ������, shared - ����� ���� �������
#pragma omp parallel for private(i, j, k) shared(data, next)
        for (k = 0; k < nb[z_dim]; ++k)
            for (j = 0; j < nb[y_dim]; ++j) {
                data[_i(-1, j, k)] = bc[lft];
                next[_i(-1, j, k)] = bc[lft];
            }
    }

    if (jb == 0) {
#pragma omp parallel for private(i, j, k) shared(data, next)
        for (k = 0; k < nb[z_dim]; ++k)
            for (i = 0; i < nb[x_dim]; ++i) {
                data[_i(i, -1, k)] = bc[front];
                next[_i(i, -1, k)] = bc[front];
            }
    }

    if (kb == 0) {
#pragma omp parallel for private(i, j, k) shared(data, next)
        for (j = 0; j < nb[y_dim]; ++j)
            for (i = 0; i < nb[x_dim]; ++i) {
                data[_i(i, j, -1)] = bc[down];
                next[_i(i, j, -1)] = bc[down];
            }
    }

    if ((ib + 1) == n[x_dim]) {
#pragma omp parallel for private(i, j, k) shared(data, next)
        for (k = 0; k < nb[z_dim]; ++k)
            for (j = 0; j < nb[y_dim]; ++j) {
                data[_i(nb[0], j, k)] = bc[rght];
                next[_i(nb[0], j, k)] = bc[rght];
            }
    }

    if ((jb + 1) == n[y_dim]) {
#pragma omp parallel for private(i, j, k) shared(data, next)
        for (k = 0; k < nb[z_dim]; ++k)
            for (i = 0; i < nb[x_dim]; ++i) {
                data[_i(i, nb[1], k)] = bc[back];
                next[_i(i, nb[1], k)] = bc[back];
            }
    }

    if ((kb + 1) == n[z_dim]) {
#pragma omp parallel for private(i, j, k) shared(data, next)
        for (j = 0; j < nb[y_dim]; ++j)
            for (i = 0; i < nb[x_dim]; ++i) {
                data[_i(i, j, nb[z_dim])] = bc[up];
                next[_i(i, j, nb[z_dim])] = bc[up];
            }
    }

    double* globalMax = (double*)malloc(sizeof(double) * numproc);

    while (true) {
        // ����� �������
        // ���� ������ ������ ������ �� �������
        if (n[x_dim] > 1) {
            // �� ��������� dead lock-a �� ���������� ������ �� ����� �����������
            // ������ �������
            if (ib == 0) {
                // � � ������� Sendrecv ���������� ������ �������� �� ib+1 ������
                // ��� �������������� �������� �� ���������� id - ��� ���������� ������
                //buffer_send[_idb(j, k)] = data[_i(nb[x_dim] - 1, j, k)];
                //data[_i(nb[x_dim], j, k)] = buffer_recv[_idb(j, k)];
                MPI_Sendrecv(data + nb[x_dim], 1, zx_face, _idp(ib + 1, jb, kb), id,
                    data + nb[x_dim] + 1, 1, zx_face, _idp(ib + 1, jb, kb), _idp(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
            }
            // ��������� �������
            else if (ib + 1 == n[x_dim]) {
                //buffer_send[_idb(j, k)] = data[_i(0, j, k)];
                //data[_i(-1, j, k)] = buffer_recv[_idb(j, k)];
                MPI_Sendrecv(data + 1, 1, zx_face, _idp(ib - 1, jb, kb), id,
                    data, 1, zx_face, _idp(ib - 1, jb, kb), _idp(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
            }
            // ������������� �������
            else {
                //buffer_send[_idb(j, k)] = data[_i(nb[x_dim] - 1, j, k)];
                //data[_i(-1, j, k)] = buffer_recv[_idb(j, k)];
                MPI_Sendrecv(data + nb[x_dim], 1, zx_face, _idp(ib + 1, jb, kb), id,
                    data, 1, zx_face, _idp(ib - 1, jb, kb), _idp(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
                //buffer_send[_idb(j, k)] = data[_i(0, j, k)];
                //data[_i(nb[x_dim], j, k)] = buffer_recv[_idb(j, k)];
                MPI_Sendrecv(data + 1, 1, zx_face, _idp(ib - 1, jb, kb), id,
                    data + nb[x_dim] + 1, 1, zx_face, _idp(ib + 1, jb, kb), _idp(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
            }
        }

        // ���������� ��� ���������

        if (n[y_dim] > 1) {
            if (jb == 0) {
                MPI_Sendrecv(data + (nb[x_dim] + 2) * nb[y_dim], 1, yz_face, _idp(ib, jb + 1, kb), id,
                    data + (nb[x_dim] + 2) * (nb[y_dim] + 1), 1, yz_face, _idp(ib, jb + 1, kb), _idp(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
            }
            else if (jb + 1 == n[y_dim]) {
                MPI_Sendrecv(data + nb[x_dim] + 2, 1, yz_face, _idp(ib, jb - 1, kb), id,
                    data, 1, yz_face, _idp(ib, jb - 1, kb), _idp(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
            }
            else {
                MPI_Sendrecv(data + (nb[x_dim] + 2) * nb[y_dim], 1, yz_face, _idp(ib, jb + 1, kb), id,
                    data, 1, yz_face, _idp(ib, jb - 1, kb), _idp(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
                MPI_Sendrecv(data + nb[x_dim] + 2, 1, yz_face, _idp(ib, jb - 1, kb), id,
                    data + (nb[x_dim] + 2) * (nb[y_dim] + 1), 1, yz_face, _idp(ib, jb + 1, kb), _idp(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
            }
        }

        if (n[z_dim] > 1) {
            if (kb == 0) {
                MPI_Sendrecv(data + (nb[x_dim] + 2) * (nb[y_dim] + 2) * nb[z_dim], 1, xy_face, _idp(ib, jb, kb + 1), id,
                    data + (nb[x_dim] + 2) * (nb[y_dim] + 2) * (nb[z_dim] + 1), 1, xy_face, _idp(ib, jb, kb + 1), _idp(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
            }
            else if (kb + 1 == n[z_dim]) {
                MPI_Sendrecv(data + (nb[x_dim] + 2) * (nb[y_dim] + 2), 1, xy_face, _idp(ib, jb, kb - 1), id,
                    data, 1, xy_face, _idp(ib, jb, kb - 1), _idp(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
            }
            else {
                MPI_Sendrecv(data + (nb[x_dim] + 2) * (nb[y_dim] + 2) * nb[z_dim], 1, xy_face, _idp(ib, jb, kb + 1), id,
                    data, 1, xy_face, _idp(ib, jb, kb - 1), _idp(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
                MPI_Sendrecv(data + (nb[x_dim] + 2) * (nb[y_dim] + 2), 1, xy_face, _idp(ib, jb, kb - 1), id,
                    data + (nb[x_dim] + 2) * (nb[y_dim] + 2) * (nb[z_dim] + 1), 1, xy_face, _idp(ib, jb, kb + 1), _idp(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
            }
        }

        // ���������� �������� �������������� ����
        // ��, ���� ���� �� �������� ������� ����
        double localMax = 0.0;

        //private - ����� ������ ������ ������, shared - ����� ���� �������
        //reduction(��������(+,- � �.�.): ����������)
#pragma omp parallel for private(i, j, k) shared(data, next) reduction(max: localMax)
        for (k = 0; k < nb[z_dim]; ++k) {
            for (j = 0; j < nb[y_dim]; ++j) {
                for (i = 0; i < nb[x_dim]; ++i) {
                    double val = 0.5 * (
                        (data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (h[x_dim] * h[x_dim]) +
                        (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (h[y_dim] * h[y_dim]) +
                        (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (h[z_dim] * h[z_dim])
                        ) / (1 / (h[x_dim] * h[x_dim]) + 1 / (h[y_dim] * h[y_dim]) + 1 / (h[z_dim] * h[z_dim]));

                    next[_i(i, j, k)] = val;
                    localMax = std::max(localMax, abs(val - data[_i(i, j, k)]));
                }
            }
        }

        // ��� �������� allgather
        // �� �������� �������� �� ������ �������� � ����� �� ���� ���������
        // � ����� ������ ����������, ��� ������������� �������� �� ���� ��������� ���������, ������� ��� �����
        // � � ����� �� �������� ������ ��������������� �� ��������� ��������
        // �� ���� ������ ������� ������� - ���� ��������, ������ ������� - ������ � �.�.
        // ����� ������� �� ����� �������� �������� �������� �� ���� ��������� ��������� � ����� ���������� �� ���
        // ���� ����� ������� �������� �������� ������ epsillon = 1e-7^10 - �� ���������� ������, ��� ��� epsillon �� ���������
        // MPI_ALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm)
        // sendbuf - ��������� �� �����, sendcount - ������� ���� ������������, sendtype - ��� ������������� (int, double), recvbuf - ���� ����� ���������, recvcount - ������� ����
        // �� ��������� �� ������ ������� ��������, ������� ����(���������!), recvtype - ��� ����������� (int, double), comm - ������������.
        MPI_Allgather(&localMax, 1, MPI_DOUBLE, globalMax, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        for (int i = 0; i < numproc; ++i) {
            localMax = max(localMax, globalMax[i]);
        }

        if (localMax < e) {
            break;
        }

        temp = next;
        next = data;
        data = temp;

        if (id == 0) {
            fprintf(stderr, "localMax = %lf\n", localMax);
            fflush(stderr);
        }
    }

    // �������� ���� ��������

        // n_size - �� ������ ��������� ������������� ����� �� �������� �� 14 ��������
    int n_size = 14;

    // ��������� ������ ������� ����
    // nb[x_dim] * nb[y_dim] * nb[z_dim] * n_size;
    //  buffer_char - ��������� ������, � ������� ����� �������� ���� ������ � ��� ����� ���������� � ���� � ��������� ����
    char* buffer_char = (char*)malloc(sizeof(char) * (nb[x_dim]) * (nb[y_dim]) * (nb[z_dim]) * n_size);
    // ����������� ������ � ��������� ���
    // ������� �� �������� ��� �����
    // ������ ����� �� ��������
    memset(buffer_char, ' ', (nb[x_dim]) * (nb[y_dim]) * (nb[z_dim]) * n_size);

    // ���������� �� ���� ��������� ����� data(�����)
    for (int k = 0; k < nb[z_dim]; ++k) {
        for (int j = 0; j < nb[y_dim]; ++j) {
            for (int i = 0; i < nb[x_dim]; ++i) {
                // � ������� sprintf - ������� ������� - �� ������� ���� ������\������ � �������� ������� ������ 
                // sprintf - ��������� ������� (������ char), ���� �� ����� ������
                // (��� ����� ������ � �������, �� �� �����, �� � ����)
                // (n_size - ������ ������ �������� + ((k * nb[y_dim] + j) * nb[x_dim] + i) - �������� ����� ��������)
                // data[j * nx + i], rank) - ��� �������
                // rank - ���� ��������
                sprintf(buffer_char + ((k)*nb[y_dim] * nb[x_dim] + (j)*nb[x_dim] + (i)) * n_size, "%.6e", next[_i(i, j, k)]);
            }
        }
    }

    for (size_t i = 0; i < (nb[x_dim]) * (nb[y_dim]) * (nb[z_dim]) * n_size; ++i) {
        // �� ��� ���� ��� �������� ���, ��� ����� � ����� \0, ������� ��� �� �����
        if (buffer_char[i] == '\0') { // ������ �� ��� �������
            buffer_char[i] = ' ';
        }
    }

    MPI_Datatype num; //����� ��� ������ - num
    MPI_Type_contiguous(n_size, MPI_CHAR, &num);
    MPI_Type_commit(&num);

    MPI_File fp; //����� ��� ������ - filetype
    MPI_Datatype filetype; //����� ��� ������ - filetype

    // ��������� ������� ������ � �������� ������� �����
    int* sizes = (int*)malloc(sizeof(int) * nb[y_dim] * nb[z_dim]);
    int* offsets = (int*)malloc(sizeof(int) * nb[y_dim] * nb[z_dim]);
    for (size_t i = 0; i < nb[y_dim] * nb[z_dim]; ++i) {
        sizes[i] = nb[x_dim];
    }

    // ��������� ���������� ��������
    MPI_Aint offset_write = ib * nb[x_dim] + jb * nb[y_dim] * nb[x_dim] * n[x_dim] + kb * nb[z_dim] * nb[x_dim] * n[x_dim] * nb[y_dim] * n[y_dim];

    for (int k = 0; k < nb[z_dim]; ++k) {
        for (int j = 0; j < nb[y_dim]; ++j) {
            int i = k * nb[y_dim] + j;
            offsets[i] = j * nb[x_dim] * n[x_dim] + k * nb[x_dim] * n[x_dim] * nb[y_dim] * n[y_dim];
        }
    }

    // ������ ����� ��� ������� � ������� type_indexed
    // ���������� ������ (x,y,z), ������ �����, ���������� ����� �������, ��� ������ � ����� ��� ������

        // � ������� ����������� �� ����� ��� � ��� ����� n ��������� ���� int
    //����������� ���� MPI_Type_indexed �������� ������������ ���� ������� �� ������������ �� ����� ������ � ������������ ��������� ������ �� ������ ���������� ��������. 
    //�������� ���������� � ��������� ������� ����.

    // ������������ ������� ������������ MPI_Type_hindexed �� indexed � ���, ��� �������� ���������� � ������.

    MPI_Type_indexed(nb[y_dim] * nb[z_dim], sizes, offsets, num, &filetype);
    MPI_Type_commit(&filetype); //������������ ��� � mpi

    // �������������� ������� ����, ���� �� ��� ����, ������ ��� �� �� ������������ ��� �������� (�� ���������)
    MPI_File_delete(output_name, MPI_INFO_NULL);
    // ��������� ���� � ������ ����� ������ ��������� (data)
    // MPI_MODE_CREATE - �� ��������
    // MPI_MODE_WRONLY - ������ �� ������, �� ����� ������
    // fp - �������� ����������
    MPI_File_open(MPI_COMM_WORLD, output_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
    // ������ ��������� ������ ������ �����, ����� ����� view
    // ������ �������� � ���� ����� ((nb[x_dim]) * (nb[y_dim]) * (nb[z_dim]) * n_size)
    // MPI_CHAR - ��� ������
    // filetype - ������ ������ ������
    // native - ���� ������ ��������� � ���� � ��� ����, ������� ���� � ������������� ������
    MPI_File_set_view(fp, offset_write * n_size, num, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fp, buffer_char, (nb[x_dim]) * (nb[y_dim]) * (nb[z_dim]) * n_size, MPI_CHAR, &status); // ������ ��������� � ����
    // ������ ����� �������������� � �����, ������� �� ���������� (filetype)
    MPI_File_close(&fp);
    // �������� �����

    // ������� ������
    free(buffer_char);
    free(sizes);
    free(offsets);
    free(next);
    free(data);

    MPI_Finalize(); // ��������� MPI
}
