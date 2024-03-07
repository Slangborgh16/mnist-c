#include "neuralnet.h"

Network* neuralNetCreate(int inputNodes, int hiddenNodes, int outputNodes) {
    Network* nnet = (Network*)malloc(sizeof(Network));

    nnet->w1 = matrixCreate(hiddenNodes, inputNodes);
    nnet->w2 = matrixCreate(outputNodes, hiddenNodes);
    matrixRandomize(nnet->w1);
    matrixRandomize(nnet->w2);

    nnet->b1 = matrixCreate(hiddenNodes, 1);
    nnet->b2 = matrixCreate(outputNodes, 1);
    matrixFill(nnet->b1, 0.0);
    matrixFill(nnet->b2, 0.0);

    return nnet;
}


void neuralNetFree(Network* nnet) {
    matrixFree(nnet->w1);
    matrixFree(nnet->w2);
    matrixFree(nnet->b1);
    matrixFree(nnet->b2);

    free(nnet);
    nnet = NULL;
}


Matrix* relu(Matrix* matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;

    Matrix* reluMatrix = matrixCreate(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            reluMatrix->values[i][j] = fmax(0.0, matrix->values[i][j]);
    }

    return reluMatrix;
}


Matrix* dRelu(Matrix* matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;

    Matrix* dReluMatrix = matrixCreate(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            dReluMatrix->values[i][j] = matrix->values[i][j] > 0.0;
    }

    return dReluMatrix;
}


Matrix* softmax(Matrix* matrix) {
    // Applies softmax to each column of the matrix
    int rows = matrix->rows;
    int cols = matrix->cols;

    Matrix* softmaxMatrix = matrixCreate(rows, cols);

    for (int j = 0; j < cols; j++) {
        double max = 0.0;
        double sum = 0.0;

        for (int i = 0; i < rows; i++)
            max = fmax(matrix->values[i][j], max);

        for (int i = 0; i < rows; i++) {
            double val = exp(matrix->values[i][j] - max);
            sum += val;
            softmaxMatrix->values[i][j] = val;
        }

        for (int i = 0; i < rows; i++)
            softmaxMatrix->values[i][j] /= sum;
    }

    return softmaxMatrix;
}


Matrix* dSoftmax(Matrix* matrix) {
    // Softmax derivative: a(1 - a)
    Matrix* ones = matrixCreate(matrix->rows, matrix->cols);
    matrixFill(ones, 1.0);

    Matrix* difference = matrixSubtract(ones, matrix);
    matrixFree(ones);
    Matrix* derivative = matrixHadamard(matrix, difference);
    matrixFree(difference);

    return derivative;
}


Matrix* biasAdd(Matrix* z, Matrix* bias) {
    int rows = z->rows;
    int cols = z->cols;

    Matrix* sum = matrixCreate(rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            sum->values[i][j] = z->values[i][j] + bias->values[i][0];
    }

    return sum;
}


double crossEntropy(Matrix* predictions, Matrix* labels) {
    // Labels and predictions are expected to be the columns
    if (!matrixCheckDimensions(predictions, labels)) {
        printf("Cross entropy dimension error. Predictions: %dx%d, Labels: %dx%d\n", \
                predictions->rows, predictions->cols, labels->rows, labels->cols);
        exit(EXIT_FAILURE);
    }

    int rows = predictions->rows;
    int cols = predictions->cols;
    double avgCrossEntropy = 0.0;

    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            if (labels->values[i][j] == 0)
                continue;

            avgCrossEntropy += -1 * log(predictions->values[i][j]);
            break;
        }
    }

    avgCrossEntropy /= cols;
    return avgCrossEntropy;
}


Matrix* forwardprop(Network* nnet, Matrix* input) {
    Matrix* w1 = nnet->w1;
    Matrix* w2 = nnet->w2;

    Matrix* b1 = nnet->b1;
    Matrix* b2 = nnet->b2;

    Matrix* z1 = matrixDot(w1, input);
    Matrix* z1Biased = biasAdd(z1, b1);
    Matrix* a1 = relu(z1Biased);
    Matrix* z2 = matrixDot(w2, a1);
    Matrix* z2Biased = biasAdd(z2, b2);
    Matrix* a2 = softmax(z2Biased);

    matrixFree(z1);
    matrixFree(z1Biased);
    matrixFree(a1);
    matrixFree(z2);
    matrixFree(z2Biased);

    return a2;
}


void updateNetwork(Network* nnet, Matrix* dW1, Matrix* dW2, \
        Matrix* dB1, Matrix* dB2, double learningRate) {
    Matrix* product = NULL;
    Matrix* updatedParam = NULL;

    product = matrixScalarProduct(dW1, learningRate);
    updatedParam = matrixSubtract(nnet->w1, product);
    matrixFree(product);
    matrixFree(nnet->w1);
    nnet->w1 = updatedParam;

    product = matrixScalarProduct(dW2, learningRate);
    updatedParam = matrixSubtract(nnet->w2, product);
    matrixFree(product);
    matrixFree(nnet->w2);
    nnet->w2 = updatedParam;
    
    product = matrixScalarProduct(dB1, learningRate);
    updatedParam = matrixSubtract(nnet->b1, product);
    matrixFree(product);
    matrixFree(nnet->b1);
    nnet->b1 = updatedParam;
    
    product = matrixScalarProduct(dB2, learningRate);
    updatedParam = matrixSubtract(nnet->b2, product);
    matrixFree(product);
    matrixFree(nnet->b2);
    nnet->b2 = updatedParam;
}


void backprop(Network* nnet, Matrix* input, Matrix* onehot, double learningRate, int batchSize) {
    // I don't like how these lines are identical to the forwardprop function
    // Need to come up with a better solution
    Matrix* w1 = nnet->w1;
    Matrix* w2 = nnet->w2;

    Matrix* b1 = nnet->b1;
    Matrix* b2 = nnet->b2;

    Matrix* z1 = matrixDot(w1, input);
    Matrix* z1Biased = biasAdd(z1, b1);
    Matrix* a1 = relu(z1Biased);
    Matrix* z2 = matrixDot(w2, a1);
    Matrix* z2Biased = biasAdd(z2, b2);
    Matrix* a2 = softmax(z2Biased);

    // Last layer adjustments
    Matrix* dZ2 = matrixSubtract(a2, onehot);
    Matrix* a1T = matrixTranspose(a1);
    Matrix* dZ2DotA1T = matrixDot(dZ2, a1T);
    Matrix* dW2 = matrixScalarProduct(dZ2DotA1T, 1.0 / (double)batchSize);
    Matrix* dB2 = matrixRowAvg(dZ2);

    // First layer adjustments
    Matrix* w2T = matrixTranspose(w2);
    Matrix* w2TDotdZ2 = matrixDot(w2T, dZ2);
    Matrix* dReluZ1 = dRelu(z1);
    Matrix* dZ1 = matrixHadamard(w2TDotdZ2, dReluZ1);
    Matrix* a0T = matrixTranspose(input);
    Matrix* dZ1DotA0T = matrixDot(dZ1, a0T);
    Matrix* dW1 = matrixScalarProduct(dZ1DotA0T, 1.0 / (double)batchSize);
    Matrix* dB1 = matrixRowAvg(dZ1);

    updateNetwork(nnet, dW1, dW2, dB1, dB2, learningRate);
    // matrixPrint(dW1);

    matrixFree(z1);
    matrixFree(z1Biased);
    matrixFree(a1);
    matrixFree(z2);
    matrixFree(z2Biased);
    matrixFree(a2);

    matrixFree(dZ2);
    matrixFree(a1T);
    matrixFree(dZ2DotA1T);
    matrixFree(dW2);
    matrixFree(dB2);

    matrixFree(w2T);
    matrixFree(w2TDotdZ2);
    matrixFree(dReluZ1);
    matrixFree(dZ1);
    matrixFree(a0T);
    matrixFree(dZ1DotA0T);
    matrixFree(dW1);
    matrixFree(dB1);
}
