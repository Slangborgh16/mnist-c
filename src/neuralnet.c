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
        double sum = 0.0;

        for (int i = 0; i < rows; i++) {
            double val = exp(matrix->values[i][j]);
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
    Matrix* z1Biased = matrixAdd(z1, b1);
    Matrix* a1 = relu(z1Biased);
    Matrix* z2 = matrixDot(w2, a1);
    Matrix* z2Biased = matrixAdd(z2, b2);
    Matrix* a2 = softmax(z2Biased);

    matrixFree(z1);
    matrixFree(z1Biased);
    matrixFree(a1);
    matrixFree(z2);
    matrixFree(z2Biased);

    return a2;
}


Matrix* backprop(Network* nnet, Matrix* input, Matrix* onehot) {
    // I don't like how these lines are straight from the forwardprop function
    // Need to come up with a better solution
    Matrix* w1 = nnet->w1;
    Matrix* w2 = nnet->w2;

    Matrix* b1 = nnet->b1;
    Matrix* b2 = nnet->b2;

    Matrix* z1 = matrixDot(w1, input);
    Matrix* z1Biased = matrixAdd(z1, b1);
    Matrix* a1 = relu(z1Biased);
    Matrix* z2 = matrixDot(w2, a1);
    Matrix* z2Biased = matrixAdd(z2, b2);
    Matrix* a2 = softmax(z2Biased);

    // Last layer adjustments
    // Bias adjustment: a2 - t -- Where t is one-hot encoded label
    // Weight adjustment: (a2 - t) dot a1^T -- At least I think...
    Matrix* b2Adjustment = matrixSubtract(a2, onehot);
    Matrix* a1Transpose = matrixTranspose(a1);
    Matrix* w2Adjustment = matrixDot(b2Adjustment, a1Transpose);
    matrixFree(a1Transpose);

    // First layer adjustments
    // Bias adjustment: (w2^T dot b2Adjustment) hadamard dRelu(z1)
    // Weight adjustment: b1Adjustment dot a0^T
    Matrix* w2Transpose = matrixTranspose(w2);
    Matrix* w2DotB2Adjustment = matrixDot(w2Transpose, b2Adjustment);
    Matrix* z1ReluDerivative = dRelu(z1);
    Matrix* b1Adjustment = matrixHadamard(w2DotB2Adjustment, z1ReluDerivative);
    matrixFree(w2DotB2Adjustment);
    matrixFree(z1ReluDerivative);
    Matrix* a0Transpose = matrixTranspose(input);
    Matrix* w1Adjustment = matrixDot(b1Adjustment, a0Transpose);
    matrixFree(a0Transpose);

    matrixFree(z1);
    matrixFree(z1Biased);
    matrixFree(a1);
    matrixFree(z2);
    matrixFree(z2Biased);
    matrixFree(a2);

    // PLACEHOLDER SO I CAN TEST COMPILE
    return w1;
}
