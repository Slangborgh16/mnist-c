#ifndef NEURALNET_H
#define NEURALNET_H
#include <stdlib.h>
#include <math.h>

typedef struct Network {
    int numInputNodes;
    int numHiddenNodes;
    int numOutputNodes;

    double inputLayer[numInputNodes];
    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputNodes];

    double weights1[numHiddenNodes][numInputNodes];
    double weights2[numOutputNodes][numHiddenNodes];

    double bias1[numHiddenNodes];
    double bias2[numOutputNodes];
} Network;

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
