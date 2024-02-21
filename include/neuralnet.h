#ifndef NEURALNET_H
#define NEURALNET_H
#include <stdlib.h>
#include <math.h>

int vecAdd(int cols, double a[cols], double b[cols], double output[cols]);
int vecSubtract(int cols, double a[cols], double b[cols], double output[cols]);
int matrixNormalize(int rows, double input[rows], double output[rows], double maximum);
int initializeWeights(int rows, int cols, double input[rows][cols]);
int dotProduct(int rows, int cols, double matrix[rows][cols], double vec[cols], double output[rows]);
int relu(int rows, double input[rows], double output[rows]);
int softmax(int rows, double input[rows], double output[rows]);
int forwardprop(int rows, int cols, double weights[rows][cols], double vec[cols], \
        double bias[rows], double output[rows]);

#endif
