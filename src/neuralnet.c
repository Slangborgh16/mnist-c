#include "neuralnet.h"

void relu(int cols, double input[cols], double output[cols]) {
    for (int i = 0; i < cols; i++)
        output[i] = fmax(0.00f, input[i]);
}


void dRelu(int cols, double input[cols], double output[cols]) {
    for (int i = 0; i < cols; i++)
        output[i] = input[i] > 0;
}


void softmax(int classes, double input[classes], double output[classes]) {
    double summation = 0.00f;
    for (int i = 0; i < classes; i++) {
        double val = exp(input[i]);
        summation += val;
        output[i] = val;
    }

    for (int i = 0; i < classes; i++) {
        output[i] /= summation;
    }
}


double crossEntropy(int classes, double* label, double* prediction) {
    int classId = 0;
    while (label[classId] == 0)
        classId++;
    
    return -log(prediction[classId]);
}


void forwardprop(Network* nnet) {
    int iNodes = nnet->inputNodes;
    int hNodes = nnet->hiddenNodes;
    int oNodes = nnet->outputNodes;

    double* z1 = nnet->z1;
    double* z2 = nnet->z2;

    double* a0 = nnet->a0;
    double* a1 = nnet->a1;
    double* a2 = nnet->a2;

    double** w1 = nnet->w1;
    double** w2 = nnet->w2;

    double* b1 = nnet->b1;
    double* b2 = nnet->b2;

    matDotVec(hNodes, iNodes, w1, a0, z1);
    vecAdd(hNodes, z1, b1, z1);
    relu(hNodes, z1, a1);
    matDotVec(oNodes, hNodes, w2, a1, z2);
    vecAdd(oNodes, z2, b2, z2);
    softmax(oNodes, z2, a2);
}
