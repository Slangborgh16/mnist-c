#ifndef NEURALNET_H
#define NEURALNET_H

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

typedef struct Network {
    int inputNodes;
    int hiddenNodes;
    int outputNodes;

    double* z1;
    double* z2;

    double* a0;
    double* a1;
    double* a2;

    double** w1;
    double** w2;

    double* b1;
    double* b2;
} Network;

void vecAdd(int cols, double* a, double* b, double* output);
void vecSubtract(int cols, double* a, double* b, double* output);
void vecNormalize(int cols, uint8_t* input, double* output, uint8_t maximum);

double** createMatrix(int rows, int cols);
void freeMatrix(int rows, int cols, double** matrix);
void initializeWeights(int rows, int cols, double** matrix);

void matDotVec(int rows, int cols, double** matrix, double* vec, double* output);
void dotProduct(int rows1, int cols1, int rows2, int cols2, \
        double** mat1, double** mat2, double** output);
void matTranspose(int rows1, int cols1, int rows2, int cols2, double** matrix, double** output);

void relu(int cols, double input[cols], double output[cols]);
void dRelu(int cols, double input[cols], double output[cols]);
void softmax(int classes, double input[classes], double output[classes]);
double crossEntropy(int classes, double* label, double* prediction);
void forwardprop(Network* nnet);

#endif
