#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#define MATRIX_RAND_MAX 10
#define SEED_PROCESS_BIAS 2435 // Bias to ensure much different seeds for each process

typedef struct {
    MPI_Comm comm;
    int size;
    int myrank;
} Processes;

int next(Processes procs) {
    return (procs.myrank + 1) % procs.size;;
}

int prev(Processes procs) {
    return (procs.myrank - 1 + procs.size) % procs.size;
}

void fill_rand(int *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        data[i] = rand() % MATRIX_RAND_MAX;
    }
}

int compute_cell(int* row, int* col, size_t len) {
    int sum = 0;
    for (size_t i=0; i<len; ++i) {
        sum += row[i]*col[i];
    }
    return sum;
}

void print_array(int* a, size_t len) {
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

    if (procs.myrank == 0) {
        printf("%s%d: ", prefix, procs.myrank);
        print_array(arr, len);
    } else {
        wait_prev(procs);
        printf("%s%d: ", prefix, procs.myrank);
        print_array(arr, len);
    }

    // Check if this process' next wraps around to the start.
    // Using next() ensures compatiblity with other chaing topologies, assuming 0 is always the first process
    if (next(procs) != 0) {
        signal_next(procs);
    }
}

Processes init(int argc, char **argv) {
    Processes procs;
    procs.comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(procs.comm, &procs.myrank);
    MPI_Comm_size(procs.comm, &procs.size);
    srand(time(NULL) + procs.myrank * SEED_PROCESS_BIAS); // Different seed for each process
    return procs;
}

void getRow(int *rowA, size_t matrix_size, Processes procs) {
    if (procs.myrank == 0) {
        fill_rand(rowA, matrix_size);
    } else {
        MPI_Recv(rowA, matrix_size, MPI_INT, prev(procs), 0, procs.comm, MPI_STATUS_IGNORE);
    }
}

void handleRow(int *rowA, size_t matrix_size, Processes procs) {
    if (next(procs) != 0) {
        MPI_Send(rowA, matrix_size, MPI_INT, next(procs), 0, procs.comm);
    } else {
        static int c = 0;
        printf("A%d: ", c++);
        print_array(rowA, matrix_size);
    }
}

/***
This program performs a parallel matrix multiplication using a linear array topology.
The first process generates all the rows of A, sending them to the next process. Each
process computes some columns of C, using the moving rows of A and the static columns
of B. Each process has a number of columns of B in memory, generated randomly. When the
last process receives a row of A, after using it to compute cells of C, it prints it out.

Next steps:
- Allow for rectangular matrices
- Read matrix from file and scatter it across processes
***/
int main(int argc, char **argv) {
    
    Processes procs = init(argc, argv);

    const size_t matrix_size = 1000;
    size_t ncols = matrix_size / procs.size;
    if (procs.myrank < matrix_size % procs.size) {
        ncols++;
    }
    printf("Process %d has %zu columns\n", procs.myrank, ncols);

    MPI_Barrier(procs.comm);
    int *colsB  = (int*)malloc(matrix_size * ncols * sizeof(int));
    int *colsC  = (int*)malloc(matrix_size * ncols * sizeof(int));
    int *rowA   = (int*)malloc(matrix_size * sizeof(int));
    int *buffer = (int*)malloc(matrix_size * sizeof(int));
    int hasRow = 0;

    // To be substituted with a proper scattering of the matrix A
    fill_rand(colsB, matrix_size * ncols);

    for (int row = 0; row < matrix_size; row++) {
        getRow(rowA, matrix_size, procs);
        for (int col = 0; col < ncols; col++) {
            colsC[col*matrix_size + row] = compute_cell(rowA, colsB + col*matrix_size, matrix_size);
        }
        handleRow(rowA, matrix_size, procs);
    }

    chain_array_print("B", colsB, matrix_size * ncols, procs);
    chain_array_print("C", colsC, matrix_size * ncols, procs);

    free(rowA);
    free(colsB);
    free(colsC);
    free(buffer);

    MPI_Finalize();
    return 0;
}