#ifndef MATRIX_H
#define MATRIX_H


int matrixAdd(int rows, int cols, int a[rows][cols], int b[rows][cols], int output[rows][cols]);
int dotProduct(int aRows, int aCols, int a[aRows][aCols], \
        int bRows, int bCols, int b[bRows][bCols], int output[aRows][bCols]);

#endif
