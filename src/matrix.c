#include "matrix.h"

Matrix* matrixCreate(int rows, int cols) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));

    matrix->rows = rows;
    matrix->cols = cols;

    matrix->values = (double**)malloc(sizeof(double*) * rows);
    for (int i = 0; i < rows; i++)
        matrix->values[i] = (double*)malloc(sizeof(double) * cols);

    return matrix;
}


void matrixFree(Matrix* matrix) {
    int rows = matrix->rows;

    for (int i = 0; i < rows; i++)
        free(matrix->values[i]);

    free(matrix->values);
    free(matrix);
    matrix = NULL;
}


void matrixRandomize(Matrix* matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            matrix->values[i][j] = (double)rand() / RAND_MAX - 0.5;
    }
}


void matrixFill(Matrix* matrix, double val) {
    int rows = matrix->rows;
    int cols = matrix->cols;

    for (int i = 0; i < rows; i++)
        memset(matrix->values[i], val, sizeof(double) * cols);
}


Matrix* matrixCopy(Matrix* matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;

    Matrix* newMatrix = matrixCreate(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j ++)
            newMatrix->values[i][j] = matrix->values[i][j];
    }

    return newMatrix;
}


void matrixPrint(Matrix* matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%.4f ", matrix->values[i][j]);

        printf("\n\n");
    }
}


int matrixCheckDimensions(Matrix* matrix1, Matrix* matrix2) {
    // Returns 1 if dimensions match and 0 if not
    return (matrix1->rows == matrix2->rows) && (matrix1->cols == matrix2->cols);
}


Matrix* matrixAdd(Matrix* matrix1, Matrix* matrix2) {
    if (!matrixCheckDimensions(matrix1, matrix2)) {
        printf("Matrix addition dimension error: %dx%d + %dx%d\n", \
                matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols);
        exit(EXIT_FAILURE);
    }

    int rows = matrix1->rows;
    int cols = matrix1->cols;

    Matrix* sum = matrixCreate(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            sum->values[i][j] = matrix1->values[i][j] + matrix2->values[i][j];
    }

    return sum;
}


Matrix* matrixSubtract(Matrix* matrix1, Matrix* matrix2) {
    if (!matrixCheckDimensions(matrix1, matrix2)) {
        printf("Matrix subtraction dimension error: %dx%d - %dx%d\n", \
                matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols);
        exit(EXIT_FAILURE);
    }

    int rows = matrix1->rows;
    int cols = matrix1->cols;

    Matrix* difference = matrixCreate(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            difference->values[i][j] = matrix1->values[i][j] - matrix2->values[i][j];
    }

    return difference;
}


Matrix* matrixDot(Matrix* matrix1, Matrix* matrix2) {
    if (matrix1->cols != matrix2->rows) {
        printf("Matrix dot product dimension error: %dx%d • %dx%d\n", \
                matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols);
        exit(EXIT_FAILURE);
    }

    int rows1 = matrix1->rows;
    int rows2 = matrix2->rows;
    int cols2 = matrix2->cols;

    Matrix* product = matrixCreate(rows1, cols2);

    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            double sum = 0;
            for (int k = 0; k < rows2; k++)
                sum += matrix1->values[i][k] * matrix2->values[k][j];
            product->values[i][j] = sum;
        }
    }

    return product;
}


// Element-wise multiplication
Matrix* matrixHadamard(Matrix* matrix1, Matrix* matrix2) {
    if (!matrixCheckDimensions(matrix1, matrix2)) {
        printf("Matrix Hadamard product dimension error: %dx%d - %dx%d\n", \
                matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols);
        exit(EXIT_FAILURE);
    }

    int rows = matrix1->rows;
    int cols = matrix1->cols;

    Matrix* hadamardProduct = matrixCreate(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            hadamardProduct->values[i][j] = matrix1->values[i][j] * matrix2->values[i][j];
    }

    return hadamardProduct;
}


Matrix* matrixTranspose(Matrix* matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;

    Matrix* transpose = matrixCreate(cols, rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            transpose->values[j][i] = matrix->values[i][j];
    }

    return transpose;
}
