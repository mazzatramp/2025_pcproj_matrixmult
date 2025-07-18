#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#define MATRIX_RAND_MAX 10

typedef struct {
    MPI_Comm comm;
    int size;
    int curr;
} Processes;

int next(Processes procs) {
    return (procs.curr + 1) % procs.size;;
}

int prev(Processes procs) {
    return (procs.curr - 1 + procs.size) % procs.size;
}

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

void print_array(int* a, int len) {
    for (int i=0; i<len; ++i) {
        printf("%d ",a[i]);
    }
    printf("\n");
}

void signal_next(Processes procs) {
    MPI_Send(NULL, 0, MPI_INT, next(procs), 0, procs.comm);
}

void wait_prev(Processes procs) {
    MPI_Recv(NULL, 0, MPI_INT, prev(procs), 0, procs.comm, MPI_STATUS_IGNORE);
}

void chain_array_print(char *prefix, int *arr, int len, Processes procs) {
    // IMPORTANT: Without this barrier, last process might print the last elemnt
    // of the array after the first process started printing another array
    MPI_Barrier(procs.comm); 

    if (procs.curr == 0) {
        printf("%s%d: ", prefix, procs.curr);
        print_array(arr, len);
    } else {
        wait_prev(procs);
        printf("%s%d: ", prefix, procs.curr);
        print_array(arr, len);
    }

    // Check if this process' next wraps around to the start.
    // Using next() ensures compatiblity with other chaing topologies, assuming 0 is always the first process
    if (next(procs) > 0) {
        signal_next(procs);
    }
}

Processes init(int argc, char **argv) {
    Processes procs;
    procs.comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(procs.comm, &procs.curr);
    MPI_Comm_size(procs.comm, &procs.size);
    srand(time(NULL) + procs.curr * 2435); // Different seed for each process
    return procs;
}

void carousel_left_step(int **main_mem, int **recv_buffer, int len, Processes procs) {

    MPI_Sendrecv(
        *main_mem, len, MPI_INT, prev(procs), 0, //send to previous
        *recv_buffer, len, MPI_INT, next(procs), 0, //recv from next
        procs.comm, MPI_STATUS_IGNORE
    );

    swap_ptrs(main_mem, recv_buffer);
}

int main(int argc, char **argv) {

    Processes procs = init(argc, argv);

    const int matrix_size = procs.size;
    int *rowA = (int*)malloc(matrix_size * sizeof(int));
    int *colB = (int*)malloc(matrix_size * sizeof(int));
    int *colC = (int*)malloc(matrix_size * sizeof(int));
    int *buffer = (int*)malloc(matrix_size * sizeof(int));

    fill_rand(rowA, matrix_size);
    fill_rand(colB, matrix_size);

    int row = procs.curr;
    do {
        colC[row] = compute_cell(colB, rowA, matrix_size);
        carousel_left_step(&rowA, &buffer, matrix_size, procs);
        row = (row + 1) % matrix_size;
    } while (row != procs.curr);

    chain_array_print("A", rowA, matrix_size, procs);
    chain_array_print("B", colB, matrix_size, procs);
    chain_array_print("C", colC, matrix_size, procs);

    free(rowA);
    free(colB);
    free(colC);
    free(buffer);

    MPI_Finalize();
    return 0;
}