#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <time.h>
#include "mpi.h"
const int x_dim = 0;
const int y_dim = 1;
const int z_dim = 2;

using namespace std;

// направления
enum {
    down, up, lft, rght, front, back
};

// макросы _i, чтобы у нас можно было использовать в индексации -1 и n
#define _i(i, j, k) (((k) + 1) * (nb[y_dim] + 2) * (nb[x_dim] + 2) + ((j) + 1) * (nb[x_dim] + 2) + (i) + 1)

#define _idp(i, j, k) ((k) * n[y_dim] * n[x_dim] + (j) * n[x_dim] + (i))
#define _idb(i, j) (j * buffer_length + i)

int main(int argc, char** argv) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    // передаём аргументы для MPI_Init
    int nb[3];
    int n[3];
    //int it; //it - счётсик итераций
    int id, numproc, proc_name_len; // id - номер процесса, numproc - количество процессов
    char proc_name[MPI_MAX_PROCESSOR_NAME]; // у каждого процесса можно получить имя
    // машины, на котором будет наш процесс
    string output_name; // название файла ввода
    double e; // точночть вычислений (эпсиллон)
    double l[3], h[3], bc[6], bc_start; // x, y, z - размер области
    // x, y, z - переменная в формуле
    // down up left right front back
    double* data, * temp, * next; //*buff;
    // data и next - массивы под данные
    // temp - в каждой итерации их можно менять местами между собой
    // buff - переменная буфера. для организации общения нам нужно копировать границу
    // буфера, а потом буфер посылаем другому процессу (копируем данные через буфер)

    MPI_Status status;  // фиктивная status
    MPI_Init(&argc, &argv); // Инициализация MPI
    MPI_Comm_size(MPI_COMM_WORLD, &numproc); // Получаем количество процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &id); // Получаем количество процессов
    MPI_Get_processor_name(proc_name, &proc_name_len); // Получаем имя процессов

    if (id == 0) {
        cin >> n[x_dim] >> n[y_dim] >> n[z_dim];
        cin >> nb[x_dim] >> nb[y_dim] >> nb[z_dim];
        cin >> output_name >> e;
        cin >> l[x_dim] >> l[y_dim] >> l[z_dim];
        cin >> bc[down] >> bc[up] >> bc[lft] >> bc[rght] >> bc[front] >> bc[back];
        cin >> bc_start;
    }

    MPI_Bcast(n, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nb, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&e, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(l, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(bc, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_start, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // выделим данные под наши "данные" 
    // n+2 - организовать фиктивные ячейки из методички
    // (не учитываем границы)
    data = (double*)malloc(sizeof(double) * (nb[0] + 2) * (nb[1] + 2) * (nb[2] + 2));
    next = (double*)malloc(sizeof(double) * (nb[0] + 2) * (nb[1] + 2) * (nb[2] + 2));

    // использовать буферизацию, буферизированный Bsend и буферизированную отправку данных
    // нам надо определить размер буфера. Определим максимальный размер
    int buffer_length = max(nb[x_dim], max(nb[y_dim], nb[z_dim])); // берём буффер достаточно большим, чтобы уместить всё
    int buffer_size = buffer_length * buffer_length;

    // принимает кол-во эл-тов, тип эл-тов, коммуникатор и размер буфера, который нам надо
    // отправить через буферизированный Sendrecv

    // выделяем в памяти соотвтествующее buffer_size
    double* buffer_recv = (double*)malloc(sizeof(double) * buffer_size);
    double* buffer_send = (double*)malloc(sizeof(double) * buffer_size);

    // b (0-2) - сетка трёхмерная
    int ib = id % n[x_dim];
    int jb = (id / n[x_dim]) % n[y_dim];
    int kb = id / (n[x_dim] * n[y_dim]);

    h[x_dim] = l[x_dim] / (n[x_dim] * nb[x_dim]);
    h[y_dim] = l[y_dim] / (n[y_dim] * nb[y_dim]);
    h[z_dim] = l[z_dim] / (n[z_dim] * nb[z_dim]);

    for (int k = 0; k <= nb[z_dim]; k++) {					// Инициализация блока
        for (int j = 0; j <= nb[y_dim]; j++) {
            for (int i = 0; i <= nb[x_dim]; i++) {
                data[_i(i, j, k)] = bc_start;
            }
        }
    }

    // если у нас будут граничные элементы
    // то на границах мы задаём то, как было на границах

    // её делаем в начале тк она никогда не меняется
    // код предназначался для решения нестационарных задач и границы были зависимы от времени
    if (ib == 0) {
        for (int k = 0; k < nb[z_dim]; ++k)
            for (int j = 0; j < nb[y_dim]; ++j) {
                data[_i(-1, j, k)] = bc[lft];
                next[_i(-1, j, k)] = bc[lft];
            }
    }

    if (jb == 0) {
        for (int k = 0; k < nb[z_dim]; ++k)
            for (int i = 0; i < nb[x_dim]; ++i) {
                data[_i(i, -1, k)] = bc[front];
                next[_i(i, -1, k)] = bc[front];
            }
    }

    if (kb == 0) {
        for (int j = 0; j < nb[y_dim]; ++j)
            for (int i = 0; i < nb[x_dim]; ++i) {
                data[_i(i, j, -1)] = bc[down];
                next[_i(i, j, -1)] = bc[down];
            }
    }

    if ((ib + 1) == n[x_dim]) {
        for (int k = 0; k < nb[z_dim]; ++k)
            for (int j = 0; j < nb[y_dim]; ++j) {
                data[_i(nb[0], j, k)] = bc[rght];
                next[_i(nb[0], j, k)] = bc[rght];
            }
    }

    if ((jb + 1) == n[y_dim]) {
        for (int k = 0; k < nb[z_dim]; ++k)
            for (int i = 0; i < nb[x_dim]; ++i) {
                data[_i(i, nb[1], k)] = bc[back];
                next[_i(i, nb[1], k)] = bc[back];
            }
    }

    if ((kb + 1) == n[z_dim]) {
        for (int j = 0; j < nb[y_dim]; ++j)
            for (int i = 0; i < nb[x_dim]; ++i) {
                data[_i(i, j, nb[z_dim])] = bc[up];
                next[_i(i, j, nb[z_dim])] = bc[up];
            }
    }

    double* globalMax = (double*)malloc(sizeof(double) * numproc);

    while (true) {
        // обмен данными
        // если индекс нашего потока не крайний
        if (n[x_dim] > 1) {
            // во избежание dead lock-a мы проходимся волной по всему направлению
            // первый элемент
            if (ib == 0) {
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int j = 0; j < nb[y_dim]; ++j) {
                        // мы копируем одну границу нашей области в буфер
                        buffer_send[_idb(j, k)] = data[_i(nb[x_dim] - 1, j, k)];
                    }
                }
                // и с помощью Sendrecv отправляем буфер процессу по ib+1 вправо
                // для предотрващения путаницы мы используем id - кто отправляет данные
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib + 1, jb, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib + 1, jb, kb), _idp(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int j = 0; j < nb[y_dim]; ++j) {
                        data[_i(nb[x_dim], j, k)] = buffer_recv[_idb(j, k)];
                    }
                }
            }
            // последний элемент
            else if (ib + 1 == n[x_dim]) {
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int j = 0; j < nb[y_dim]; ++j) {
                        buffer_send[_idb(j, k)] = data[_i(0, j, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib - 1, jb, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib - 1, jb, kb), _idp(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int j = 0; j < nb[y_dim]; ++j) {
                        data[_i(-1, j, k)] = buffer_recv[_idb(j, k)];
                    }
                }
            }
            // промежуточный элемент
            else {
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int j = 0; j < nb[y_dim]; ++j) {
                        buffer_send[_idb(j, k)] = data[_i(nb[x_dim] - 1, j, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib + 1, jb, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib - 1, jb, kb), _idp(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int j = 0; j < nb[y_dim]; ++j) {
                        data[_i(-1, j, k)] = buffer_recv[_idb(j, k)];
                        buffer_send[_idb(j, k)] = data[_i(0, j, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib - 1, jb, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib + 1, jb, kb), _idp(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int j = 0; j < nb[y_dim]; ++j) {
                        data[_i(nb[x_dim], j, k)] = buffer_recv[_idb(j, k)];
                    }
                }
            }
        }

        // аналогично для остальных

        if (n[y_dim] > 1) {
            if (jb == 0) {
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        buffer_send[_idb(i, k)] = data[_i(i, nb[y_dim] - 1, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb + 1, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb + 1, kb), _idp(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        data[_i(i, nb[y_dim], k)] = buffer_recv[_idb(i, k)];
                    }
                }
            }
            else if (jb + 1 == n[y_dim]) {
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        buffer_send[_idb(i, k)] = data[_i(i, 0, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb - 1, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb - 1, kb), _idp(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        data[_i(i, -1, k)] = buffer_recv[_idb(i, k)];
                    }
                }
            }
            else {
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        buffer_send[_idb(i, k)] = data[_i(i, nb[y_dim] - 1, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb + 1, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb - 1, kb), _idp(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        data[_i(i, -1, k)] = buffer_recv[_idb(i, k)];
                        buffer_send[_idb(i, k)] = data[_i(i, 0, k)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb - 1, kb), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb + 1, kb), _idp(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
                for (int k = 0; k < nb[z_dim]; ++k) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        data[_i(i, nb[y_dim], k)] = buffer_recv[_idb(i, k)];
                    }
                }
            }
        }

        if (n[z_dim] > 1) {
            if (kb == 0) {
                for (int j = 0; j < nb[y_dim]; ++j) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        buffer_send[_idb(i, j)] = data[_i(i, j, nb[z_dim] - 1)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb + 1), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb + 1), _idp(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < nb[y_dim]; ++j) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        data[_i(i, j, nb[z_dim])] = buffer_recv[_idb(i, j)];
                    }
                }
            }
            else if (kb + 1 == n[z_dim]) {
                for (int j = 0; j < nb[y_dim]; ++j) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        buffer_send[_idb(i, j)] = data[_i(i, j, 0)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb - 1), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb - 1), _idp(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < nb[y_dim]; ++j) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        data[_i(i, j, -1)] = buffer_recv[_idb(i, j)];
                    }
                }
            }
            else {
                for (int j = 0; j < nb[y_dim]; ++j) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        buffer_send[_idb(i, j)] = data[_i(i, j, nb[z_dim] - 1)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb + 1), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb - 1), _idp(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < nb[y_dim]; ++j) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        data[_i(i, j, -1)] = buffer_recv[_idb(i, j)];
                        buffer_send[_idb(i, j)] = data[_i(i, j, 0)];
                    }
                }
                MPI_Sendrecv(buffer_send, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb - 1), id,
                    buffer_recv, buffer_size, MPI_DOUBLE, _idp(ib, jb, kb + 1), _idp(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
                for (int j = 0; j < nb[y_dim]; ++j) {
                    for (int i = 0; i < nb[x_dim]; ++i) {
                        data[_i(i, j, nb[z_dim])] = buffer_recv[_idb(i, j)];
                    }
                }
            }
        }

        // Организуем основной вычислительный цикл
        // То, ради чего мы написали столько кода
        double localMax = 0.0;

        for (int k = 0; k < nb[z_dim]; ++k) {
            for (int j = 0; j < nb[y_dim]; ++j) {
                for (int i = 0; i < nb[x_dim]; ++i) {
                    double val = 0.5 * (
                        (data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (h[x_dim] * h[x_dim]) +
                        (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (h[y_dim] * h[y_dim]) +
                        (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (h[z_dim] * h[z_dim])
                        ) / (1 / (h[x_dim] * h[x_dim]) + 1 / (h[y_dim] * h[y_dim]) + 1 / (h[z_dim] * h[z_dim]));

                    next[_i(i, j, k)] = val;
                    localMax = max(localMax, abs(val - data[_i(i, j, k)]));
                }
            }
        }

        // как работает allgather
        // он получает значения от одного процесса и отдаёт их всем остальным
        // в таком случае получается, что накапливаются значения во всех остальных процессах, включая тот самый
        // и в итоге мы получаем массив отсортированных по процессам значений
        // то есть первый процесс процесс - одно значение, второй процесс - второе и т.д.
        // таким образом мы можем накопить побочный максимум во всех остальных процессах и найти наибольший из них
        // если самый больший максимум окажется больше epsillon = 1e-7^10 - мы продолжаем работу, так как epsillon не достигнут
        // MPI_ALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm)
        // sendbuf - указатель на буфер, sendcount - сколько слов отправляется, sendtype - тип отправленного (int, double),
        // recvbuf - куда будем принимать, recvcount - сколько слов
        // мы принимаем от одного другого процесса, включая этот(проверить!), recvtype - тип полученного (int, double), comm - коммуникатор.
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
            //fprintf(stderr, "localMax = %lf\n", localMax);
            //fflush(stderr);
        }
    }

    // основной цикл завершён

    // Построчный вывод каждого блока

    // Построчный вывод каждого блока

    if (id != 0) {
        for (int k = 0; k < nb[z_dim]; ++k) {
            for (int j = 0; j < nb[y_dim]; ++j) {
                for (int i = 0; i < nb[x_dim]; ++i) {
                    buffer_send[i] = next[_i(i, j, k)];
                }
                MPI_Send(buffer_send, buffer_length, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
            }
        }
    }
    else {
        std::fstream output(output_name, ios::out);
        output << scientific << setprecision(6);
        // чтобы не перепутать последовательность x-строк, мы проходимся по всем процессам
        for (int kb = 0; kb < n[z_dim]; ++kb) {
            for (int k = 0; k < nb[z_dim]; ++k) {
                for (int jb = 0; jb < n[y_dim]; ++jb) {
                    for (int j = 0; j < nb[y_dim]; ++j) {
                        for (int ib = 0; ib < n[x_dim]; ++ib) {
                            if (_idp(ib, jb, kb) == 0) { // считаем id текущего просметриваемого процесса
                                // если он нулевой, то мы заполняем "в себя"
                                for (int i = 0; i < nb[x_dim]; ++i) {
                                    buffer_recv[i] = next[_i(i, j, k)];
                                }
                            }
                            else {
                                MPI_Recv(buffer_recv, buffer_length, MPI_DOUBLE, _idp(ib, jb, kb), _idp(ib, jb, kb), MPI_COMM_WORLD, &status);
                            }

                            for (int i = 0; i < nb[x_dim]; i++) {
                                //printf("%.6e ", buffer_recv[i]);
                                output << buffer_recv[i] << " ";
                            }

                            if (ib + 1 == n[x_dim]) {
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
    free(next);
    free(data);

    MPI_Finalize();
}
