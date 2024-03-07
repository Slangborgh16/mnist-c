#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Matrix {
    double** values;
    int rows;
    int cols;
} Matrix;

Matrix* matrixCreate(int rows, int cols);
void matrixFree(Matrix* matrix);
void matrixRandomize(Matrix* matrix);
void matrixFill(Matrix* matrix, double val);
Matrix* matrixCopy(Matrix* matrix);
void matrixPrint(Matrix* matrix);
int matrixCheckDimensions(Matrix* matrix1, Matrix* matrix2);

Matrix* matrixAdd(Matrix* matrix1, Matrix* matrix2);
Matrix* matrixSubtract(Matrix* matrix1, Matrix* matrix2);
Matrix* matrixScalarProduct(Matrix* matrix, double factor);
Matrix* matrixDot(Matrix* matrix1, Matrix* matrix2);
Matrix* matrixHadamard(Matrix* matrix1, Matrix* matrix2);
Matrix* matrixTranspose(Matrix* matrix);
Matrix* matrixRowAvg(Matrix* matrix);

#endif
