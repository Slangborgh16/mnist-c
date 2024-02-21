#include "neuralnet.h"

int vecAdd(int cols, double a[cols], double b[cols], double output[cols]) {
    for (int i = 0; i < cols; i++) {
        output[i] = a[i] + b[i];
    }
    return 0;
}


int vecSubtract(int cols, double a[cols], double b[cols], double output[cols]) {
    for (int i = 0; i < cols; i++) {
        output[i] = a[i] - b[i];
    }
    return 0;
}


int vecNormalize(int cols, double input[cols], double output[cols], double maximum) {
    for (int i = 0; i < cols; i++) {
        output[i] = input[i] / maximum;
    }
    return 0;
}


int initializeWeights(int rows, int cols, double input[rows][cols]) {
    for (int m = 0; m < rows; m++) {
        for (int n = 0; n < cols; n++) {
            input[m][n] = (double)rand() / RAND_MAX - 0.5;
        }
    }
    return 0;
}


int dotProduct(int rows, int cols, double matrix[rows][cols], \
        double vec[cols], double output[rows]) {
    for (int m = 0; m < rows; m++) {
        double productSum = 0;
        for (int n = 0; n < cols; n++){
            productSum += matrix[m][n] * vec[n];
        }
        output[m] = productSum;
    }
    return 0;
}


int relu(int cols, double input[cols], double output[cols]) {
    for (int i = 0; i < cols; i++)
        output[i] = fmax(0.00f, input[i]);

    return 0;
}


int softmax(int cols, double input[cols], double output[cols]) {
    double summation = 0.00f;
    for (int i = 0; i < cols; i++) {
        double val = powf(M_E, input[i]);
        summation += val;
        output[i] = val;
    }

    for (int i = 0; i < cols; i++) {
        output[i] /= summation;
    }

    return 0;
}


int forwardprop(Network* nnet) {
    int iNodes = nnet->inputNodes;
    int hNodes = nnet->hiddenNodes;
    int oNodes = nnet->outputNodes;

    double* iLayer = nnet->inputLayer;
    double* hLayer = nnet->hiddenLayer;
    double* oLayer = nnet->outputLayer;

    double* w1 = nnet->weights1;
    double* w2 = nnet->weights2;

    double* b1 = nnet->bias1;
    double* b2 = nnet->bias2;

    dotProduct(hNodes, iNodes, w1, iLayer, hLayer);
    vecAdd(hNodes, hLayer, b1, hLayer);
    reul(hNodes, hLayer, hLayer);
    dotProduct(oNodes, hNodes, w2, hLayer, oLayer);
    vecAdd(oNodes, oLayer, b2, oLayer);
    softmax(oNodes, oLayer, oLayer);

    return 0;
}
