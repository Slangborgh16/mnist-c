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

    double* inputLayer;
    double* hiddenLayer;
    double* outputLayer;

    double** weights1;
    double** weights2;

    double* bias1;
    double* bias2;
} Network;

int vecAdd(int cols, double* a, double* b, double* output);
int vecSubtract(int cols, double* a, double* b, double* output);
int vecNormalize(int cols, uint8_t* input, double* output, uint8_t maximum);

double** initializeWeights(int rows, int cols);
void freeWeights(int rows, int cols, double** weights);

int matDotVec(int rows, int cols, double** matrix, double* vec, double* output);
int dotProduct(int rows1, int cols1, int rows2, int cols2, \
        double** mat1, double** mat2, double** output);
int matTranspose(int rows1, int cols1, int rows2, int cols2, double** matrix, double** output);

int relu(int cols, double input[cols], double output[cols]);
int dRelu(int cols, double input[cols], double output[cols]);
int softmax(int cols, double input[cols], double output[cols]);
double crossEntropy(int classes, double* label, double* prediction);
int forwardprop(Network* nnet);

#endif
