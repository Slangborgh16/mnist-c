#include "matrix.h"

int matrixAdd(int rows, int cols, int a[rows][cols], int b[rows][cols], int output[rows][cols]) {
    for (int m = 0; m < rows; m++) {
        for (int n = 0; n < rows; n++) {
            output[m][n] = a[m][n] + b[m][n];
        }
    }
}


int dotProduct(int aRows, int aCols, int a[aRows][aCols], int bRows, \
        int bCols, int b[bRows][bCols], int output[aRows][bCols]) {
    if (aCols != bRows) return -1;

    for (int m = 0; m < aRows; m++) {
        for (int n = 0; n < bCols; n++){
            int productSum = 0;
            for (int i = 0; i < bRows; i++) {
                productSum += a[m][i] * b[i][n];
            }
            output[m][n] = productSum;
        }
    }
    return 0;
}
