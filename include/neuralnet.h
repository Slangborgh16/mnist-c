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

    Matrix* z1;
    Matrix* z2;

    Matrix* a0;
    Matrix* a1;
    Matrix* a2;

    Matrix* w1;
    Matrix* w2;

    Matrix* b1;
    Matrix* b2;
} Network;

void relu(int cols, double input[cols], double output[cols]);
void dRelu(int cols, double input[cols], double output[cols]);
void softmax(int classes, double input[classes], double output[classes]);
double crossEntropy(int classes, double* label, double* prediction);
void forwardprop(Network* nnet);

#endif
