#ifndef NEURALNET_H
#define NEURALNET_H

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "matrix.h"

typedef struct Network {
    Matrix* w1;
    Matrix* w2;

    Matrix* b1;
    Matrix* b2;
} Network;

Network* neuralNetCreate(int inputNodes, int hiddenNodes, int outputNodes);
void neuralNetFree(Network* nnet);

Matrix* relu(Matrix* matrix);
Matrix* dRelu(Matrix* matrix);
Matrix* softmax(Matrix* matrix);
Matrix* dSoftmax(Matrix* matrix);
double crossEntropy(Matrix* predictions, Matrix* labels);

Matrix* forwardprop(Network* nnet, Matrix* input);
void updateNetwork(Network* nnet, Matrix* dW1, Matrix* dW2, \
        Matrix* dB1, Matrix* dB2, double learningRate);
void backprop(Network* nnet, Matrix* input, Matrix* onehot, double learningRate);

#endif
