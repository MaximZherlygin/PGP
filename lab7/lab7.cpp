#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <time.h>
#include "mpi.h"

using namespace std;

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

// макросы _i, чтобы у нас можно было использовать в индексации -1 и n
#define _i(i, j, k) (((k) + 1) * (dim[y_on] + 2) * (dim[x_on] + 2) + ((j) + 1) * (dim[x_on] + 2) + (i) + 1)
#define _idp(i, j, k) ((k) * block[y_on] * block[x_on] + (j) * block[x_on] + (i))
#define _idb(i, j) (j * buffer_length + i)

int main(int argc, char** argv) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int dim[3];
    int block[3];


    string out_filename; // название файла ввода
    double eps; // точночть вычислений (эпсиллон)
    double l[3];
    double h[3];
    double u[6];
    double u_start; // x, y, z - размер области
    // x, y, z - переменная в формуле
    // down up left right front back
    double *values;
    double *temporary_values;
    double *next_values;
    // data и next - массивы под данные
    // temp - в каждой итерации их можно менять местами между собой
    // буфера, а потом буфер посылаем другому процессу (копируем данные через буфер)

    MPI_Status status;  // фиктивная status
    MPI_Init(&argc, &argv); // Инициализация MPI
    int id, proccess_count;  // id - номер процесса, numproc - количество процессов
                             // машины, на котором будет наш процесс
    MPI_Comm_size(MPI_COMM_WORLD, &proccess_count); // Получаем количество процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &id); // Получаем количество процессов

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

    h[x_on] = l[x_on] / (block[x_on] * dim[x_on]);
    h[y_on] = l[y_on] / (block[y_on] * dim[y_on]);
    h[z_on] = l[z_on] / (block[z_on] * dim[z_on]);

    // использовать буферизацию, буферизированный Bsend и буферизированную отправку данных
    // нам надо определить размер буфера. Определим максимальный размер
    int buffer_size = max(max(dim[x_on], dim[y_on]), dim[z_on]) * max(max(dim[x_on], dim[y_on]), dim[z_on]);

    // принимает кол-во эл-тов, тип эл-тов, коммуникатор и размер буфера, который нам надо
    // отправить через буферизированный Sendrecv

    // выделяем в памяти соотвтествующее buffer_size
    double* send = (double*)malloc(buffer_size * sizeof(double));
    double* recv = (double*)malloc(buffer_size * sizeof(double));

    // b (0-2) - сетка трёхмерная
    int i_b = id % block[x_on];
    int j_b = (id / block[x_on]) % block[y_on];
    int k_b = id / (block[x_on] * block[y_on]);

    // выделим данные под наши "данные"
    // n+2 - организовать фиктивные ячейки из методички
    // (не учитываем границы)
    values = (double*)malloc((dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2) * sizeof(double));
    next_values = (double*)malloc((dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2) * sizeof(double));
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

    // если у нас будут граничные элементы
    // то на границах мы задаём то, как было на границах

    // её делаем в начале тк она никогда не меняется
    // код предназначался для решения нестационарных задач и границы были зависимы от времени
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

    

    while (true) {
        // обмен данными
        // если индекс нашего потока не крайний
        if (block[x_on] > 1) {
            // во избежание dead lock-a мы проходимся волной по всему направлению
            // первый элемент
            if (i_b == 0) {
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int j = 0; j < dim[y_on]; j++) {
                        // мы копируем одну границу нашей области в буфер
                        send[_idb(j, k)] = values[_i(dim[x_on] - 1, j, k)];
                    }
                }
                // и с помощью Sendrecv отправляем буфер процессу по ib+1 вправо
                // для предотрващения путаницы мы используем id - кто отправляет данные
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b + 1, j_b, k_b),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b + 1, j_b, k_b),
                             _idp(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int j = 0; j < dim[y_on]; j++) {
                        values[_i(dim[x_on], j, k)] = recv[_idb(j, k)];
                    }
                }
            } else if (i_b + 1 == block[x_on]) { // последний элемент
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int j = 0; j < dim[y_on]; j++) {
                        send[_idb(j, k)] = values[_i(0, j, k)];
                    }
                }
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b - 1, j_b, k_b),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b - 1, j_b, k_b),
                             _idp(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int j = 0; j < dim[y_on]; j++) {
                        values[_i(-1, j, k)] = recv[_idb(j, k)];
                    }
                }
            } else { // промежуточный элемент
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int j = 0; j < dim[y_on]; j++) {
                        send[_idb(j, k)] = values[_i(dim[x_on] - 1, j, k)];
                    }
                }
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b + 1, j_b, k_b),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b - 1, j_b, k_b),
                             _idp(i_b - 1, j_b, k_b), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int j = 0; j < dim[y_on]; j++) {
                        values[_i(-1, j, k)] = recv[_idb(j, k)];
                        send[_idb(j, k)] = values[_i(0, j, k)];
                    }
                }
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b - 1, j_b, k_b),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b + 1, j_b, k_b),
                             _idp(i_b + 1, j_b, k_b), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int j = 0; j < dim[y_on]; j++) {
                        values[_i(dim[x_on], j, k)] = recv[_idb(j, k)];
                    }
                }
            }
        }

        // аналогично для остальных

        if (block[y_on] > 1) {
            if (j_b == 0) {
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        send[_idb(i, k)] = values[_i(i, dim[y_on] - 1, k)];
                    }
                }
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b, j_b + 1, k_b),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b, j_b + 1, k_b),
                             _idp(i_b, j_b + 1, k_b), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        values[_i(i, dim[y_on], k)] = recv[_idb(i, k)];
                    }
                }
            }
            else if (j_b + 1 == block[y_on]) {
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        send[_idb(i, k)] = values[_i(i, 0, k)];
                    }
                }
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b, j_b - 1, k_b),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b, j_b - 1, k_b),
                             _idp(i_b, j_b - 1, k_b), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        values[_i(i, -1, k)] = recv[_idb(i, k)];
                    }
                }
            }
            else {
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        send[_idb(i, k)] = values[_i(i, dim[y_on] - 1, k)];
                    }
                }
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b, j_b + 1, k_b),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b, j_b - 1, k_b),
                             _idp(i_b, j_b - 1, k_b), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        values[_i(i, -1, k)] = recv[_idb(i, k)];
                        send[_idb(i, k)] = values[_i(i, 0, k)];
                    }
                }
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b, j_b - 1, k_b),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b, j_b + 1, k_b),
                             _idp(i_b, j_b + 1, k_b), MPI_COMM_WORLD, &status);
                for (int k = 0; k < dim[z_on]; k++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        values[_i(i, dim[y_on], k)] = recv[_idb(i, k)];
                    }
                }
            }
        }

        if (block[z_on] > 1) {
            if (k_b == 0) {
                for (int j = 0; j < dim[y_on]; j++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        send[_idb(i, j)] = values[_i(i, j, dim[z_on] - 1)];
                    }
                }
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b, j_b, k_b + 1),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b, j_b, k_b + 1),
                             _idp(i_b, j_b, k_b + 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < dim[y_on]; j++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        values[_i(i, j, dim[z_on])] = recv[_idb(i, j)];
                    }
                }
            }
            else if (k_b + 1 == block[z_on]) {
                for (int j = 0; j < dim[y_on]; j++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        send[_idb(i, j)] = values[_i(i, j, 0)];
                    }
                }
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b, j_b, k_b - 1),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b, j_b, k_b - 1),
                             _idp(i_b, j_b, k_b - 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < dim[y_on]; j++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        values[_i(i, j, -1)] = recv[_idb(i, j)];
                    }
                }
            }
            else {
                for (int j = 0; j < dim[y_on]; j++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        send[_idb(i, j)] = values[_i(i, j, dim[z_on] - 1)];
                    }
                }
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b, j_b, k_b + 1),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b, j_b, k_b - 1),
                             _idp(i_b, j_b, k_b - 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < dim[y_on]; j++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        values[_i(i, j, -1)] = recv[_idb(i, j)];
                        send[_idb(i, j)] = values[_i(i, j, 0)];
                    }
                }
                MPI_Sendrecv(send, buffer_size, MPI_DOUBLE, _idp(i_b, j_b, k_b - 1),
                             id, recv, buffer_size, MPI_DOUBLE, _idp(i_b, j_b, k_b + 1),
                             _idp(i_b, j_b, k_b + 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < dim[y_on]; j++) {
                    for (int i = 0; i < dim[x_on]; i++) {
                        values[_i(i, j, dim[z_on])] = recv[_idb(i, j)];
                    }
                }
            }
        }

        // Организуем основной вычислительный цикл
        // То, ради чего мы написали столько кода
        double diff = 0.0;
        double h2[3];
        h2[x_on] = h[x_on] * h[x_on];
        h2[y_on] = h[y_on] * h[y_on];
        h2[z_on] = h[z_on] * h[z_on];

        for (int k = 0; k < dim[z_on]; k++) {
            for (int j = 0; j < dim[y_on]; j++) {
                for (int i = 0; i < dim[x_on]; i++) {
                    next_values[_i(i, j, k)] = 0.5 * ((values[_i(i + 1, j, k)] + values[_i(i - 1, j, k)]) / h2[x_on] +
                                                      (values[_i(i, j + 1, k)] + values[_i(i, j - 1, k)]) / h2[y_on] +
                                                      (values[_i(i, j, k + 1)] + values[_i(i, j, k - 1)]) / h2[z_on]) /
                                                      (1 / h2[x_on] + 1 / h2[y_on] + 1 / h2[z_on]);
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

    // основной цикл завершён

    // Построчный вывод каждого блока

    // Построчный вывод каждого блока

    if (id != 0) {
        for (int k = 0; k < dim[z_on]; k++) {
            for (int j = 0; j < dim[y_on]; j++) {
                for (int i = 0; i < dim[x_on]; i++) {
                    send[i] = next_values[_i(i, j, k)];
                }
                MPI_Send(send, max(dim[x_on], max(dim[y_on], dim[z_on])), MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
            }
        }
    } else {
        std::fstream out_stream(out_filename, ios::out);
        out_stream << scientific << setprecision(6);

        for (int kb = 0; kb < block[z_on]; kb++) {
            for (int k = 0; k < dim[z_on]; k++) {
                for (int jb = 0; jb < block[y_on]; jb++) {
                    for (int j = 0; j < dim[y_on]; j++) {
                        for (int ib = 0; ib < block[x_on]; ++ib) {

                            if (_idp(ib, jb, kb) == 0) {
                                for (int i = 0; i < dim[x_on]; i++)
                                    recv[i] = next_values[_i(i, j, k)];

                            } else {
                                MPI_Recv(recv, max(dim[x_on], max(dim[y_on], dim[z_on])), MPI_DOUBLE, _idp(ib, jb, kb),
                                         _idp(ib, jb, kb), MPI_COMM_WORLD, &status);
                            }

                            for (int i = 0; i < dim[x_on]; i++)
                                out_stream << recv[i] << " ";

                            if (ib + 1 == block[x_on]) {
                                out_stream << "\n";
                            } else {
                                out_stream << " ";
                            }
                        }
                    }
                }
                out_stream << "\n";
            }
        }
        out_stream.close();
    }

    MPI_Finalize();
    free(next_values);
    free(send);
    free(values);
    free(recv);
    return 0;
}
