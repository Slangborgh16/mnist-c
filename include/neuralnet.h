#ifndef NEURALNET_H
#define NEURALNET_H

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "matrix.h"

typedef struct Network {
    int inputNodes;
    int hiddenNodes;
    int outputNodes;

    Matrix* w1;
    Matrix* w2;

    Matrix* b1;
    Matrix* b2;
} Network;

void relu(Matrix* input, Matrix* output);
void dRelu(Matrix* input, Matrix* output);
void softmax(Matrix* input, Matrix* output);
double crossEntropy(Matrix* predictions, Matrix* labels);
Matrix* forwardprop(Network* nnet, Matrix* input);

#endif
