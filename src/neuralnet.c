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


void relu(Matrix* input, Matrix* output) {
    if (!matrixCheckDimensions(input, output)) {
        printf("ReLU dimension error. Input: %dx%d, Output: %dx%d\n", \
                input->rows, input->cols, output->rows, output->cols);
        exit(EXIT_FAILURE);
    }

    int rows = input->rows;
    int cols = input->cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            output->values[i][j] = fmax(0.0, input->values[i][j]);
    }
}


void dRelu(Matrix* input, Matrix* output) {
    if (!matrixCheckDimensions(input, output)) {
        printf("ReLU derivative dimension error. Input: %dx%d, Output: %dx%d\n", \
                input->rows, input->cols, output->rows, output->cols);
        exit(EXIT_FAILURE);
    }

    int rows = input->rows;
    int cols = input->cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            output->values[i][j] = input->values[i][j] > 0;
    }
}


void softmax(Matrix* input, Matrix* output) {
    // Applies softmax to each column of the matrix

    if (!matrixCheckDimensions(input, output)) {
        printf("Softmax dimension error. Input: %dx%d, Output: %dx%d\n", \
                input->rows, input->cols, output->rows, output->cols);
        exit(EXIT_FAILURE);
    }

    int rows = input->rows;
    int cols = input->cols;

    for (int j = 0; j < cols; j++) {
        double sum = 0.0;

        for (int i = 0; i < rows; i++) {
            double val = exp(input->values[i][j]);
            sum += val;
            output->values[i][j] = val;
        }

        for (int i = 0; i < rows; i++)
            output->values[i][j] /= sum;
    }
}


Matrix* dSoftmax(Matrix* matrix) {
    // Softmax derivative: a(1 - a)
    Matrix* ones = matrixCreate(input->rows, input->cols);
    matrixFill(ones, 1.0);

    Matrix* difference = matrixSubtract(ones, input);
    matrixFree(ones);
    Matrix* derivative = matrixHadamard(input, difference);
    matrixFree(difference);

    return derivative;
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
    Matrix* a1 = matrixAdd(z1, b1);
    relu(a1, a1);
    Matrix* z2 = matrixDot(w2, a1);
    Matrix* a2 = matrixAdd(z2, b2);
    softmax(a2, a2);

    matrixFree(z1);
    matrixFree(a1);
    matrixFree(z2);

    return a2;
}


Matrix* backprop(Network* nnet, Matrix* input, Matrix* onehot) {
    Matrix* w1 = nnet->w1;
    Matrix* w2 = nnet->w2;

    Matrix* b1 = nnet->b1;
    Matrix* b2 = nnet->b2;

    Matrix* z1 = matrixDot(w1, input);
    Matrix* a1 = matrixAdd(z1, b1);
    relu(a1, a1);
    Matrix* z2 = matrixDot(w2, a1);
    Matrix* a2 = matrixAdd(z2, b2);
    softmax(a2, a2);

    // Last layer adjustments
    // Weight adjustment: (a - t)(a_prev) where t is onehot encoded label
    // Bias adjustment: a - t
    Matrix* a2Error = matrixSubtract(a2, onehot);
    Matrix* w2Adj = matrixDot(w2Transpose, a2Error);

    // First layer adjustments
    //
}
