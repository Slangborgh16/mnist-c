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

double** createMat(int rows, int cols);
void freeMat(int rows, int cols, double** mat);
void initializeWeights(int rows, int cols, double** mat);

void matDotVec(int rows, int cols, double** mat, double* vec, double* output);
void dotProduct(int rows1, int cols1, int rows2, int cols2, \
        double** mat1, double** mat2, double** output);
void matTranspose(int rows, int cols, double** mat, double** output);

void relu(int cols, double input[cols], double output[cols]);
void dRelu(int cols, double input[cols], double output[cols]);
void softmax(int classes, double input[classes], double output[classes]);
double crossEntropy(int classes, double* label, double* prediction);
void forwardprop(Network* nnet);

#endif
