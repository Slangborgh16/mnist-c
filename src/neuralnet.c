#include "neuralnet.h"

int vecAdd(int cols, double* a, double* b, double* output) {
    for (int i = 0; i < cols; i++) {
        output[i] = a[i] + b[i];
    }
    return 0;
}


int vecSubtract(int cols, double* a, double* b, double* output) {
    for (int i = 0; i < cols; i++) {
        output[i] = a[i] - b[i];
    }
    return 0;
}


int vecNormalize(int cols, uint8_t* input, double* output, uint8_t maximum) {
    for (int i = 0; i < cols; i++) {
        output[i] = (double)input[i] / maximum;
    }
    return 0;
}


double** initializeWeights(int rows, int cols) {
    srand(time(NULL));

    double** weights = (double**)malloc(sizeof(double*) * rows);
    for (int i = 0; i < rows; i++) {
        weights[i] = (double*)malloc(sizeof(double) * cols);
        for (int j = 0; j < cols; j++)
            weights[i][j] = (double)rand() / RAND_MAX - 0.5;
    }

    return weights;
}


void freeWeights(int rows, int cols, double** weights) {
    for (int i = 0; i < rows; i++)
        free(weights[i]);
    free(weights);
}

int matDotVec(int rows, int cols, double** matrix, double* vec, double* output) {
    for (int m = 0; m < rows; m++) {
        double productSum = 0;
        for (int n = 0; n < cols; n++){
            productSum += matrix[m][n] * vec[n];
        }
        output[m] = productSum;
    }
    return 0;
}


int dotProduct(int rows1, int cols1, int rows2, int cols2, \
        double** mat1, double** mat2, double** output) {
    if (cols1 != rows2) return -1;

    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            double productSum = 0;
            for (int k = 0; k < rows2; k++) {
                productSum += mat1[i][k] * mat2[k][j];
            }
            output[i][j] = productSum;
        }
    }

    return 0;
}


int matTranspose(int rows1, int cols1, int rows2, int cols2, double** matrix, double** output) {
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            output[j][i] = matrix[i][j];
        }
    }
    return 0;
}


int relu(int cols, double input[cols], double output[cols]) {
    for (int i = 0; i < cols; i++)
        output[i] = fmax(0.00f, input[i]);

    return 0;
}


int dRelu(int cols, double input[cols], double output[cols]) {
    for (int i = 0; i < cols; i++)
        output[i] = input[i] > 0;
    return 0;
}


int softmax(int cols, double input[cols], double output[cols]) {
    double summation = 0.00f;
    for (int i = 0; i < cols; i++) {
        double val = exp(input[i]);
        summation += val;
        output[i] = val;
    }

    for (int i = 0; i < cols; i++) {
        output[i] /= summation;
    }

    return 0;
}


double crossEntropy(int classes, double* label, double* prediction) {
    int classId = 0;
    while (label[classId] == 0)
        classId++;
    
    return -log(prediction[classId]);
}


int forwardprop(Network* nnet) {
    int iNodes = nnet->inputNodes;
    int hNodes = nnet->hiddenNodes;
    int oNodes = nnet->outputNodes;

    double* iLayer = nnet->inputLayer;
    double* hLayer = nnet->hiddenLayer;
    double* oLayer = nnet->outputLayer;

    double** w1 = nnet->weights1;
    double** w2 = nnet->weights2;

    double* b1 = nnet->bias1;
    double* b2 = nnet->bias2;

    matDotVec(hNodes, iNodes, w1, iLayer, hLayer);
    vecAdd(hNodes, hLayer, b1, hLayer);
    relu(hNodes, hLayer, hLayer);
    matDotVec(oNodes, hNodes, w2, hLayer, oLayer);
    vecAdd(oNodes, oLayer, b2, oLayer);
    softmax(oNodes, oLayer, oLayer);

    return 0;
}
