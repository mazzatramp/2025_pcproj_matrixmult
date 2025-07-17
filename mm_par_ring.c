#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#define MATRIX_RAND_MAX 10

void fill_rand(int *data, int len) {
    for (int i = 0; i < len; i++) {
        data[i] = rand() % MATRIX_RAND_MAX;
    }
}

void swap_ptrs(int **a, int **b) {
    int *c = *a;
    *a = *b;
    *b = c; 
}

int compute_cell(int* row, int* col, int len) {
    int sum = 0;
    for (int i=0; i<len; ++i) {
        sum += row[i]*col[i];
    }
    return sum;
}

int print_arr(int* a, int len) {
    for (int i=0; i<len; ++i) {
        printf("%d ",a[i]);
    }
    printf("\n");
}

void chain_array_print(int rank, int *a, int len, char prefix) {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Status status;

    const int next = (rank+1)%len;
    const int prev = (rank-1)<0 ? len-1 : rank-1;

    if (rank == 0) {
        printf("%c%d: ", prefix, rank);
        print_arr(a, len);
        MPI_Send(NULL, 0, MPI_INT, next, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(NULL, 0, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
        printf("%c%d: ", prefix, rank);
        print_arr(a, len);
        if (rank != len -1) {
            MPI_Send(NULL, 0, MPI_INT, next, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int my_rank, n_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Status status;

    //random seed different for each process
    srand(time(NULL) + my_rank * 2435); //unimportant magic number

    const int next = (my_rank+1) % n_procs;
    const int prev = (my_rank-1)<0 ? n_procs-1 : my_rank-1;

    // The assumpion is that n_procs = rows = cols

    int *columnB = (int*)malloc(n_procs * sizeof(int));
    int *rowA = (int*)malloc(n_procs * sizeof(int));
    int *buffer = (int*)malloc(n_procs * sizeof(int));

    fill_rand(columnB, n_procs);
    fill_rand(rowA, n_procs);

    if (my_rank == 0) {
        printf("Matrix A:\n");
    }
    chain_array_print(my_rank, rowA, n_procs, 'A');
    if (my_rank == 0) {
        printf("Matrix B:\n");
    }
    chain_array_print(my_rank, columnB, n_procs, 'B');
    int *columnC = (int*)malloc(n_procs * sizeof(int));

    for (int step = 0; step < n_procs; step++) {
        int i_row = (my_rank + step) % n_procs;
        columnC[i_row] = compute_cell(columnB, rowA, n_procs);

        MPI_Sendrecv(
            rowA, n_procs, MPI_INT, next, 0, //send
            buffer, n_procs, MPI_INT, prev, 0, //recv
            MPI_COMM_WORLD, &status
        );

        swap_ptrs(&rowA, &buffer);
    }

    if (my_rank == 0) {
        printf("Matrix C:\n");
    }
    chain_array_print(my_rank, columnC, n_procs, 'C');

    free(rowA);
    free(columnB);
    free(columnC);
    free(buffer);

    MPI_Finalize();
    return 0;
}