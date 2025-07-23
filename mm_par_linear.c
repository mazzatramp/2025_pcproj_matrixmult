#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#define MATRIX_RAND_MAX 10
#define SEED_PROCESS_BIAS 2435 // Bias to ensure much different seeds for each process
#define ROOT_RANK 0

typedef struct {
    MPI_Comm comm;
    int size;
    int myrank;
} MpiContext;

typedef struct {
    int *data;
    size_t height;
    size_t width;
} Matrix;

enum ChunkType {
    SINGLE_ROW, //Each chunk is a row
    SINGLE_COL, //Each chunk is a column
    SCATTER_ROWS, //rows are split among processes
    SCATTER_COLS //columns are split among processes
};

int next(MpiContext procs) { //Ring
    return (procs.myrank + 1) % procs.size;
}

int prev(MpiContext procs) { //Ring
    return (procs.myrank - 1 + procs.size) % procs.size;
}


int get_process_chunk_size(size_t rank, size_t nprocs, size_t tot_size) {
    size_t chunk_size = tot_size / nprocs;

    if (rank < tot_size % nprocs) {
        chunk_size++;
    }
    return chunk_size;
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

int signal_next(MpiContext procs) {
    MPI_Send(NULL, 0, MPI_INT, next(procs), 0, procs.comm);
    return 1;
}

int wait_signal_prev(MpiContext procs) {
    MPI_Recv(NULL, 0, MPI_INT, prev(procs), 0, procs.comm, MPI_STATUS_IGNORE);
    return 1;
}

MpiContext init(int argc, char **argv) {
    MpiContext procs;
    procs.comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(procs.comm, &procs.myrank);
    MPI_Comm_size(procs.comm, &procs.size);
    srand(time(NULL) + procs.myrank * SEED_PROCESS_BIAS); // Different seed for each process
    return procs;
}

void recv_row(Matrix *entire, Matrix *chunk, MpiContext procs) {
    if (procs.myrank == 0) {
        static int row = 0;
        if (row < entire->height) {
            chunk->data = entire->data + row * entire->width;
            row++;
        }
    } else {
        MPI_Recv(chunk->data, chunk->width, MPI_INT, prev(procs), 0, procs.comm, MPI_STATUS_IGNORE);
    }
}

void send_row(Matrix *chunk, MpiContext procs) {
    if (next(procs) != 0) {
        MPI_Send(chunk->data, chunk->width, MPI_INT, next(procs), 0, procs.comm);
    }
}

//TODO: generalize for all chunk types
void get_scattering_offsets(int *counts, int *displacements, Matrix *unchunked, MpiContext procs) {
    displacements[0] = 0;
    for (int i = 0; i < procs.size; i++) {
        size_t chunk_width = get_process_chunk_size(i, procs.size, unchunked->width);
        counts[i] = unchunked->height * chunk_width;
        if (i == 0) continue;
        displacements[i] = displacements[i - 1] + counts[i - 1];
    }
}

void gen_matrix(Matrix *matrix) {
    fill_rand(matrix->data, matrix->height * matrix->width);
}

void scatter_matrix(Matrix *entire, Matrix *chunk, MpiContext procs) {

    int *counts = (int*)malloc(procs.size * sizeof(int));
    int *displacements = (int*)malloc(procs.size * sizeof(int));
    get_scattering_offsets(counts, displacements, entire, procs);

    MPI_Scatterv(entire->data, counts, displacements, MPI_INT,
                 chunk->data, counts[procs.myrank], MPI_INT,
                 ROOT_RANK, procs.comm);

    free(displacements);
    free(counts);
}

void gather_matrix(Matrix *chunk, Matrix *entire, MpiContext procs) {
    int *counts = (int*)malloc(procs.size * sizeof(int));
    int *displacements = (int*)malloc(procs.size * sizeof(int));
    get_scattering_offsets(counts, displacements, entire, procs);

    MPI_Gatherv(chunk->data, counts[procs.myrank], MPI_INT,
                entire->data, counts, displacements, MPI_INT,
                ROOT_RANK, procs.comm);

    free(counts);
    free(displacements);    
}

void compute_matrix_chunk(Matrix *A, Matrix *chunkA, Matrix *chunkB, Matrix *chunkC, MpiContext procs) {
    for (int row = 0; row < A->height; row++) {
        recv_row(A, chunkA, procs);

        for (int col = 0; col < chunkC->width; col++) {
            chunkC->data[row * chunkC->width + col] = compute_cell(chunkA->data, chunkB->data + col * chunkA->height, chunkA->width);
        }

        send_row(chunkA, procs);
    }
}

Matrix prep_chunk(Matrix *matrix, enum ChunkType type, MpiContext procs) {
    Matrix chunk;
    size_t nrows = 0, chunk_width = 0;
    if (type == SINGLE_ROW) {
        nrows = 1;
        chunk_width = matrix->width;
    } else if (type == SINGLE_COL) {
        nrows = matrix->height;
        chunk_width = 1;
    } else if (type == SCATTER_ROWS) {
        nrows = get_process_chunk_size(procs.myrank, procs.size, matrix->height);
        chunk_width = matrix->width;
    } else if (type == SCATTER_COLS) {
        nrows = matrix->height;
        chunk_width = get_process_chunk_size(procs.myrank, procs.size, matrix->width);
    }
    chunk.height = nrows;
    chunk.width = chunk_width;
    chunk.data = (int*)malloc(chunk.height * chunk.width * sizeof(int));
    return chunk;
}

void print_matrix_col_major(Matrix *matrix) {
    for (size_t col = 0; col < matrix->width; col++) {
        for (size_t row = 0; row < matrix->height; row++) {
            printf("%d ", matrix->data[row * matrix->width + col]);
        }
        printf("\n");
    }
}

void print_matrix_row_major(Matrix *matrix) {
    for (size_t row = 0; row < matrix->height; row++) {
        for (size_t col = 0; col < matrix->width; col++) {
            printf("%d ", matrix->data[row * matrix->width + col]);
        }
        printf("\n");
    }
}

/***
This program performs a parallel matrix multiplication using a linear array topology.
The first process generates all the rows of A, sending them to the next process. Each
process computes some columns of C, using the moving rows of A and the static columns
of B. Each process has a number of columns of B in memory, generated randomly. When the
last process receives a row of A, after using it to compute cells of C, it prints it out.

Next steps:
- Add error handling
- Read matrixes from file
***/
int main(int argc, char **argv) {
    
    //read sizes from command line
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <rowsA> <colsA/rowsB> <colsB>\n", argv[0]);
        return EXIT_FAILURE;
    }

    MpiContext procs = init(argc, argv);

    Matrix A = {NULL, atoi(argv[1]), atoi(argv[2])};
    Matrix B = {NULL, atoi(argv[2]), atoi(argv[3])};
    Matrix C = {NULL, atoi(argv[1]), atoi(argv[3])};

    if (procs.myrank == ROOT_RANK) {
        A.data = (int*)malloc(B.height * B.width * sizeof(int));
        B.data = (int*)malloc(B.height * B.width * sizeof(int));
        C.data = (int*)malloc(C.height * C.width * sizeof(int));
        gen_matrix(&A);
        gen_matrix(&B);
    }

    // ATTENTION ------------------------------------------------
    // The code is not generalized for any other chunck type yet
    Matrix chunkA = prep_chunk(&A, SINGLE_ROW,   procs);
    Matrix chunkB = prep_chunk(&B, SCATTER_COLS, procs);
    Matrix chunkC = prep_chunk(&C, SCATTER_COLS, procs);
    //-----------------------------------------------------------

    scatter_matrix(&B, &chunkB, procs);
    compute_matrix_chunk(&A, &chunkA, &chunkB, &chunkC, procs);
    gather_matrix(&chunkC, &C, procs);

    if (procs.myrank == ROOT_RANK) {
        printf("Matrix A:\n");
        print_matrix_row_major(&A);
        printf("Matrix B:\n");
        print_matrix_col_major(&B);
        printf("Matrix C:\n");
        print_matrix_col_major(&C);
    }

    if (procs.myrank != ROOT_RANK)
        free(chunkA.data);
    free(chunkB.data);
    free(chunkC.data);

    if (procs.myrank == ROOT_RANK) {
        free(A.data);
        free(B.data);
        free(C.data);
    }

    MPI_Finalize();
    return 0;
}