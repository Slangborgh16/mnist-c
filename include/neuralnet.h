#ifndef NEURALNET_H
#define NEURALNET_H

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

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
int dotProduct(int rows, int cols, double** matrix, double* vec, double* output);
int relu(int cols, double input[cols], double output[cols]);
int softmax(int cols, double input[cols], double output[cols]);
int crossEntropy(int classes, double* label, double* prediction, double* output);
int forwardprop(Network* nnet);

#endif
