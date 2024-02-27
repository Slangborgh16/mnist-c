#include "neuralnet.h"

void vecAdd(int cols, double* a, double* b, double* output) {
    for (int i = 0; i < cols; i++) {
        output[i] = a[i] + b[i];
    }
}


void vecSubtract(int cols, double* a, double* b, double* output) {
    for (int i = 0; i < cols; i++) {
        output[i] = a[i] - b[i];
    }
}


void vecNormalize(int cols, uint8_t* input, double* output, uint8_t maximum) {
    for (int i = 0; i < cols; i++) {
        output[i] = (double)input[i] / maximum;
    }
}


double** createMat(int rows, int cols) {
    double** mat = (double**)malloc(sizeof(double*) * rows);
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(sizeof(double) * cols);
    }

    return mat;
}


void freeMat(int rows, int cols, double** mat) {
    for (int i = 0; i < rows; i++)
        free(mat[i]);
    free(mat);
}


void initializeWeights(int rows, int cols, double** mat) {
    srand(time(NULL));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = (double)rand() / RAND_MAX - 0.5;
        }
    }
}


void matDotVec(int rows, int cols, double** mat, double* vec, double* output) {
    for (int m = 0; m < rows; m++) {
        double productSum = 0;
        for (int n = 0; n < cols; n++){
            productSum += mat[m][n] * vec[n];
        }
        output[m] = productSum;
    }
}


void dotProduct(int rows1, int cols1, int rows2, int cols2, \
        double** mat1, double** mat2, double** output) {
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            double productSum = 0;
            for (int k = 0; k < rows2; k++) {
                productSum += mat1[i][k] * mat2[k][j];
            }
            output[i][j] = productSum;
        }
    }
}


void matTranspose(int rows, int cols, double** mat, double** output) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j][i] = mat[i][j];
        }
    }
}


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
