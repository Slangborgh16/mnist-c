#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>

typedef struct Matrix {
    double** values;
    int rows;
    int cols;
} Matrix;

Matrix* matrixCreate(int rows, int cols);
void matrixFree(Matrix* matrix);
void matrixRandomize(Matrix* matrix);
void matrixFill(Matrix* matrix, double val);
int matrixCheckDimensions(Matrix* matrix1, Matrix* matrix2);

Matrix* matrixAdd(Matrix* matrix1, Matrix* matrix2);
Matrix* matrixSubtract(Matrix* matrix1, Matrix* matrix2);
Matrix* matrixDot(Matrix* matrix1, Matrix* matrix2);
Matrix* matrixTranspose(Matrix* matrix);

#endif
