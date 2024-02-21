#include "neuralnet.h"

int vecAdd(int cols, double a[cols], double b[cols], double output[cols]) {
    for (int i = 0; i < cols; i++) {
        output[i] = a[i] + b[i];
    }
    return 0;
}


int vecSubtract(int cols, double a[cols], double b[cols], double output[cols]) {
    for (int i = 0; i < cols; i++) {
        output[i] = a[i] - b[i];
    }
    return 0;
}


int matrixNormalize(int rows, double input[rows], double output[rows], double maximum) {
    for (int i = 0; i < rows; i++) {
        output[i] = input[i] / maximum;
    }
    return 0;
}


int initializeWeights(int rows, int cols, double input[rows][cols]) {
    for (int m = 0; m < rows; m++) {
        for (int n = 0; n < cols; n++) {
            input[m][n] = (double)rand() / RAND_MAX - 0.5;
        }
    }
    return 0;
}


int dotProduct(int rows, int cols, double matrix[rows][cols], \
        double vec[cols], double output[rows]) {
    for (int m = 0; m < rows; m++) {
        double productSum = 0;
        for (int n = 0; n < cols; n++){
            productSum += matrix[m][n] * vec[n];
        }
        output[m] = productSum;
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

int forwardprop(int rows, int cols, double weights[rows][cols], double vec[cols], \
        double bias[rows], double output[rows]) {
    dotProduct(rows, cols, weights, vec, output);
    vecAdd(cols, output, bias, output);
    return 0;
}
