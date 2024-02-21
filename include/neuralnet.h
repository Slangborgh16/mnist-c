#ifndef NEURALNET_H
#define NEURALNET_H
#include <stdlib.h>
#include <math.h>

typedef struct Network {
    int inputNodes;
    int hiddenNodes;
    int outputNodes;

    double inputLayer[inputNodes];
    double hiddenLayer[hiddenNodes];
    double outputLayer[outputNodes];

    double weights1[hiddenNodes][inputNodes];
    double weights2[outputNodes][hiddenNodes];

    double bias1[hiddenNodes];
    double bias2[outputNodes];
} Network;

int vecAdd(int cols, double a[cols], double b[cols], double output[cols]);
int vecSubtract(int cols, double a[cols], double b[cols], double output[cols]);
int vecNormalize(int cols, double input[cols], double output[cols], double maximum);
int initializeWeights(int rows, int cols, double input[rows][cols]);
int dotProduct(int rows, int cols, double matrix[rows][cols], double vec[cols], double output[rows]);
int relu(int cols, double input[cols], double output[cols]);
int softmax(int cols, double input[cols], double output[cols]);
int forwardprop(Network* nnet) {

#endif
