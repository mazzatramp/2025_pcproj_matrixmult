#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    int rows;
    int cols;
    int* data;
} Matrix;

#define MATRIX_RAND_MAX 100
#define ROWS_A 2000
#define COLS_A 2000

Matrix* create_matrix(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (int*)malloc(rows * cols * sizeof(int));
    return m;
}

void free_matrix(Matrix* m) {
    free(m->data);
    m->data = NULL;
    free(m);
}

int* cell(Matrix *m, int row, int col) {
    return &m->data[row * m->cols + col];
}

void mult(Matrix* A, Matrix* B, Matrix* C) {
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        printf("Can't multiply as dimensions don't match.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            *cell(C, i, j) = 0;
            for (int k = 0; k < A->cols; k++) {
                *cell(C, i, j) += *cell(A, i, k) * *cell(B, k, j);
            }
        }
    }
}

void print_matrix(Matrix* m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%d ", *cell(m, i, j));
        }
        printf("\n");
    }
}

void fill_matrix(Matrix* m, int value) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            *cell(m, i, j) = value;
        }
    }
}

void rand_matrix(Matrix* m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            *cell(m, i, j) = rand() % MATRIX_RAND_MAX;
        }
    }
}

int main(int argc, char** argv) {
    
    Matrix* A = create_matrix(ROWS_A, COLS_A);
    Matrix* B = create_matrix(COLS_A, ROWS_A);
    Matrix* C = create_matrix(ROWS_A, ROWS_A);

    rand_matrix(A);
    rand_matrix(B);
    fill_matrix(C, 0);

    mult(A, B, C);

    //print_matrix(A);
    //print_matrix(B);
    //print_matrix(C);

    //dangling pointers
    free_matrix(A);
    free_matrix(B);
    free_matrix(C); 

    return 0;
}
