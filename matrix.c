#include "matrix.h"

int matrixAdd(int rows, int cols, int a[rows][cols], int b[rows][cols], int output[rows][cols]) {
    for (int m = 0; m < rows; m++) {
        for (int n = 0; n < rows; n++) {
            output[m][n] = a[m][n] + b[m][n];
        }
    }
    return 0;
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



int relu(int rows, double input[rows], double output[rows]) {
    for (int i = 0; i < rows; i++)
        output[i] = fmax(0.00f, input[i]);

    return 0;
}


int softmax(int rows, double input[rows], double output[rows]) {
    double summation = 0.00f;
    for (int i = 0; i < rows; i++) {
        double val = powf(M_E, input[i]);
        summation += val;
        output[i] = val;
    }

    for (int i = 0; i < rows; i++) {
        output[i] /= summation;
    }

    return 0;
}
