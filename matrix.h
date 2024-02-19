#ifndef MATRIX_H
#define MATRIX_H
#include <math.h>

int matrixAdd(int rows, int cols, double a[rows][cols], double b[rows][cols], \
        double output[rows][cols]);
int dotProduct(int aRows, int aCols, double a[aRows][aCols], \
        int bRows, int bCols, double b[bRows][bCols], double output[aRows][bCols]);
int relu(int rows, double input[rows], double output[rows]);
int softmax(int rows, double input[rows], double output[rows]);

#endif
