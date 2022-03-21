#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <time.h>
#include "mpi.h"

using namespace std;

// �����������
#define down 0
#define up 1
#define left 2
#define right 3
#define front 4
#define back 5

const int x_on = 0;
const int y_on = 1;
const int z_on = 2;

// ������� _i, ����� � ��� ����� ���� ������������ � ���������� -1 � n
#define _i(i, j, k) (((k) + 1) * (dim[y_on] + 2) * (dim[x_on] + 2) + ((j) + 1) * (dim[x_on] + 2) + (i) + 1)
#define _idp(i, j, k) ((k) * block[y_on] * block[x_on] + (j) * block[x_on] + (i))
#define _idb(i, j) (j * buffer_length + i)

int main(int argc, char** argv) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    // ������� ��������� ��� MPI_Init
    int dim[3];
    int block[3];

    int id, numproc, proc_name_len; // id - ����� ��������, numproc - ���������� ���������
    char proc_name[MPI_MAX_PROCESSOR_NAME]; // � ������� �������� ����� �������� ���
    // ������, �� ������� ����� ��� �������
    string output_name; // �������� ����� �����
    double eps; // �������� ���������� (��������)
    double l[3], h[3], u[6], u_start; // x, y, z - ������ �������
    // x, y, z - ���������� � �������
    // down up left right front back
    double *values;
    double *temporary_values;
    double *next_values;
    // data � next - ������� ��� ������
    // temp - � ������ �������� �� ����� ������ ������� ����� �����
    // ������, � ����� ����� �������� ������� �������� (�������� ������ ����� �����)

    MPI_Status status;  // ��������� status
    MPI_Init(&argc, &argv); // ������������� MPI
    MPI_Comm_size(MPI_COMM_WORLD, &numproc); // �������� ���������� ���������
    MPI_Comm_rank(MPI_COMM_WORLD, &id); // �������� ���������� ���������
    MPI_Get_processor_name(proc_name, &proc_name_len); // �������� ��� ���������

    if (id == 0) {
        cin >> block[x_on] >> block[y_on] >> block[z_on];
        cin >> dim[x_on] >> dim[y_on] >> dim[z_on];
        cin >> output_name >> eps;
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


    // ������� ������ ��� ���� "������" 
    // n+2 - ������������ ��������� ������ �� ���������
    // (�� ��������� �������)
    values = (double*)malloc(sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));
    next_values = (double*)malloc(sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));

    // ������������ �����������, ���������������� Bsend � ���������������� �������� ������
    // ��� ���� ���������� ������ ������. ��������� ������������ ������
    int buffer_length = max(dim[x_on], max(dim[y_on], dim[z_on])); // ���� ������ ���������� �������, ����� �������� ��
    int buffer_size = buffer_length * buffer_length;

    // ��������� ���-�� ��-���, ��� ��-���, ������������ � ������ ������, ������� ��� ����
    // ��������� ����� ���������������� Sendrecv

    // �������� � ������ ��������������� buffer_size
    double* buffer_recv = (double*)malloc(sizeof(double) * buffer_size);
    double* buffer_send = (double*)malloc(sizeof(double) * buffer_size);

    // b (0-2) - ����� ���������
    int ib = id % block[x_on];
    int jb = (id / block[x_on]) % block[y_on];
    int kb = id / (block[x_on] * block[y_on]);

    h[x_on] = l[x_on] / (block[x_on] * dim[x_on]);
    h[y_on] = l[y_on] / (block[y_on] * dim[y_on]);
    h[z_on] = l[z_on] / (block[z_on] * dim[z_on]);

    for (int k = 0; k <= dim[z_on]; k++) {					// ������������� �����
        for (int j = 0; j <= dim[y_on]; j++) {
            for (int i = 0; i <= dim[x_on]; i++) {
                values[_i(i, j, k)] = u_start;
            }
        }
    }

    // ���� � ��� ����� ��������� ��������
    // �� �� �������� �� ����� ��, ��� ���� �� ��������

    // � ������ � ������ �� ��� ������� �� ��������
    // ��� �������������� ��� ������� �������������� ����� � ������� ���� �������� �� �������
    if (ib == 0) {
        for (int k = 0; k < dim[z_on]; ++k)
            for (int j = 0; j < dim[y_on]; ++j) {
                values[_i(-1, j, k)] = u[left];
                next_values[_i(-1, j, k)] = u[left];
            }
    }

    if (jb == 0) {
        for (int k = 0; k < dim[z_on]; ++k)
            for (int i = 0; i < dim[x_on]; ++i) {
                values[_i(i, -1, k)] = u[front];
                next_values[_i(i, -1, k)] = u[front];
            }
    }

    if (kb == 0) {
        for (int j = 0; j < dim[y_on]; ++j)
            for (int i = 0; i < dim[x_on]; ++i) {
                values[_i(i, j, -1)] = u[down];
                next_values[_i(i, j, -1)] = u[down];
            }
    }

    if ((ib + 1) == block[x_on]) {
        for (int k = 0; k < dim[z_on]; ++k)
            for (int j = 0; j < dim[y_on]; ++j) {
                values[_i(dim[0], j, k)] = u[right];
                next_values[_i(dim[0], j, k)] = u[right];
            }
    }

    if ((jb + 1) == block[y_on]) {
        for (int k = 0; k < dim[z_on]; ++k)
            for (int i = 0; i < dim[x_on]; ++i) {
                values[_i(i, dim[1], k)] = u[back];
                next_values[_i(i, dim[1], k)] = u[back];
            }
    }

    if ((kb + 1) == block[z_on]) {
        for (int j = 0; j < dim[y_on]; ++j)
            for (int i = 0; i < dim[x_on]; ++i) {
                values[_i(i, j, dim[z_on])] = u[up];
                next_values[_i(i, j, dim[z_on])] = u[up];
            }
    }

    double* globalMax = (double*)malloc(sizeof(double) * numproc);

    while (true) {
        // ����� �������
        // ���� ������ ������ ������ �� �������
        if (block[x_on] > 1) {
            // �� ��������� dead lock-a �� ���������� ������ �� ����� �����������
            // ������ �������
            if (ib == 0) {
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int j = 0; j < dim[y_on]; ++j) {
                        // �� �������� ���� ������� ����� ������� � �����
                        buffer_send[_idb(j, k)] = values[_i(dim[x_on] - 1, j, k)];
                    }
                }
                // � � ������� Sendrecv ���������� ����� �������� �� ib+1 ������
                // ��� �������������� �������� �� ���������� id - ��� ���������� ������
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib + 1, jb, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib + 1, jb, kb), _idp(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int j = 0; j < dim[y_on]; ++j) {
                        values[_i(dim[x_on], j, k)] = buffer_recv[_idb(j, k)];
                    }
                }
            }
            // ��������� �������
            else if (ib + 1 == block[x_on]) {
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int j = 0; j < dim[y_on]; ++j) {
                        buffer_send[_idb(j, k)] = values[_i(0, j, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib - 1, jb, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib - 1, jb, kb), _idp(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int j = 0; j < dim[y_on]; ++j) {
                        values[_i(-1, j, k)] = buffer_recv[_idb(j, k)];
                    }
                }
            }
            // ������������� �������
            else {
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int j = 0; j < dim[y_on]; ++j) {
                        buffer_send[_idb(j, k)] = values[_i(dim[x_on] - 1, j, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib + 1, jb, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib - 1, jb, kb), _idp(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int j = 0; j < dim[y_on]; ++j) {
                        values[_i(-1, j, k)] = buffer_recv[_idb(j, k)];
                        buffer_send[_idb(j, k)] = values[_i(0, j, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib - 1, jb, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib + 1, jb, kb), _idp(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int j = 0; j < dim[y_on]; ++j) {
                        values[_i(dim[x_on], j, k)] = buffer_recv[_idb(j, k)];
                    }
                }
            }
        }

        // ���������� ��� ���������

        if (block[y_on] > 1) {
            if (jb == 0) {
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        buffer_send[_idb(i, k)] = values[_i(i, dim[y_on] - 1, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb + 1, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb + 1, kb), _idp(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        values[_i(i, dim[y_on], k)] = buffer_recv[_idb(i, k)];
                    }
                }
            }
            else if (jb + 1 == block[y_on]) {
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        buffer_send[_idb(i, k)] = values[_i(i, 0, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb - 1, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb - 1, kb), _idp(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        values[_i(i, -1, k)] = buffer_recv[_idb(i, k)];
                    }
                }
            }
            else {
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        buffer_send[_idb(i, k)] = values[_i(i, dim[y_on] - 1, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb + 1, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb - 1, kb), _idp(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        values[_i(i, -1, k)] = buffer_recv[_idb(i, k)];
                        buffer_send[_idb(i, k)] = values[_i(i, 0, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb - 1, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb + 1, kb), _idp(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; ++k) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        values[_i(i, dim[y_on], k)] = buffer_recv[_idb(i, k)];
                    }
                }
            }
        }

        if (block[z_on] > 1) {
            if (kb == 0) {
                for (int j = 0; j < dim[y_on]; ++j) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        buffer_send[_idb(i, j)] = values[_i(i, j, dim[z_on] - 1)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb + 1), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb + 1), _idp(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < dim[y_on]; ++j) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        values[_i(i, j, dim[z_on])] = buffer_recv[_idb(i, j)];
                    }
                }
            }
            else if (kb + 1 == block[z_on]) {
                for (int j = 0; j < dim[y_on]; ++j) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        buffer_send[_idb(i, j)] = values[_i(i, j, 0)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb - 1), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb - 1), _idp(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < dim[y_on]; ++j) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        values[_i(i, j, -1)] = buffer_recv[_idb(i, j)];
                    }
                }
            }
            else {
                for (int j = 0; j < dim[y_on]; ++j) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        buffer_send[_idb(i, j)] = values[_i(i, j, dim[z_on] - 1)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb + 1), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb - 1), _idp(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < dim[y_on]; ++j) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        values[_i(i, j, -1)] = buffer_recv[_idb(i, j)];
                        buffer_send[_idb(i, j)] = values[_i(i, j, 0)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb - 1), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb + 1), _idp(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < dim[y_on]; ++j) {
                    for (int i = 0; i < dim[x_on]; ++i) {
                        values[_i(i, j, dim[z_on])] = buffer_recv[_idb(i, j)];
                    }
                }
            }
        }

        // ���������� �������� �������������� ����
        // ��, ���� ���� �� �������� ������� ����
        double localMax = 0.0;

        for (int k = 0; k < dim[z_on]; ++k) {
            for (int j = 0; j < dim[y_on]; ++j) {
                for (int i = 0; i < dim[x_on]; ++i) {
                    double val = 0.5 * (
                        (values[_i(i + 1, j, k)] + values[_i(i - 1, j, k)]) / (h[x_on] * h[x_on]) +
                        (values[_i(i, j + 1, k)] + values[_i(i, j - 1, k)]) / (h[y_on] * h[y_on]) +
                        (values[_i(i, j, k + 1)] + values[_i(i, j, k - 1)]) / (h[z_on] * h[z_on])
                        ) / (1 / (h[x_on] * h[x_on]) + 1 / (h[y_on] * h[y_on]) + 1 / (h[z_on] * h[z_on]));

                    next_values[_i(i, j, k)] = val;
                    localMax = max(localMax, abs(val - values[_i(i, j, k)]));
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
        // sendbuf - ��������� �� �����, sendcount - ������� ���� ������������, sendtype - ��� ������������� (int, double),
        // recvbuf - ���� ����� ���������, recvcount - ������� ����
        // �� ��������� �� ������ ������� ��������, ������� ����(���������!), recvtype - ��� ����������� (int, double), comm - ������������.
        MPI_Allgather(&localMax, 1, MPI_DOUBLE, globalMax, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        for (int i = 0; i < numproc; ++i) {
            localMax = max(localMax, globalMax[i]);
        }

        if (localMax < eps) {
            break;
        }

        temporary_values = next_values;
        next_values = values;
        values = temporary_values;

        if (id == 0) {
            //fprintf(stderr, "localMax = %lf\n", localMax);
            //fflush(stderr);
        }
    }

    // �������� ���� ��������

    // ���������� ����� ������� �����

    // ���������� ����� ������� �����

    if (id != 0) {
        for (int k = 0; k < dim[z_on]; ++k) {
            for (int j = 0; j < dim[y_on]; ++j) {
                for (int i = 0; i < dim[x_on]; ++i) {
                    buffer_send[i] = next_values[_i(i, j, k)];
                }
                MPI_Send(buffer_send, buffer_length, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
            }
        }
    }
    else {
        std::fstream output(output_name, ios::out);
        output << scientific << setprecision(6);
        // ����� �� ���������� ������������������ x-�����, �� ���������� �� ���� ���������
        for (int kb = 0; kb < block[z_on]; ++kb) {
            for (int k = 0; k < dim[z_on]; ++k) {
                for (int jb = 0; jb < block[y_on]; ++jb) {
                    for (int j = 0; j < dim[y_on]; ++j) {
                        for (int ib = 0; ib < block[x_on]; ++ib) {
                            if (_idp(ib, jb, kb) == 0) { // ������� id �������� ���������������� ��������
                                // ���� �� �������, �� �� ��������� "� ����"
                                for (int i = 0; i < dim[x_on]; ++i) {
                                    buffer_recv[i] = next_values[_i(i, j, k)];
                                }
                            }
                            else {
                                MPI_Recv(buffer_recv, buffer_length, MPI_DOUBLE, _idp(ib, jb, kb), _idp(ib, jb, kb), MPI_COMM_WORLD, &status);
                            }

                            for (int i = 0; i < dim[x_on]; i++) {
                                //printf("%.6e ", buffer_recv[i]);
                                output << buffer_recv[i] << " ";
                            }

                            if (ib + 1 == block[x_on]) {
                                output << "\n";
                                //printf("\n");
                            }
                            else {
                                output << " ";
                                //printf(" ");
                            }
                        }

                    }
                }
                output << "\n";
                // printf("\n");

            }
        }
        output.close();
    }

    free(buffer_send);
    free(buffer_recv);
    free(next_values);
    free(values);

    MPI_Finalize();
}
