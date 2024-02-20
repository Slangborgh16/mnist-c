#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mnist.h"
#include "matrix.h"


int main() {
    const char trainingLabelsPath[] = "dataset/train-labels-idx1-ubyte";
    const char trainingImagesPath[] = "dataset/train-images-idx3-ubyte";

    LabelData trainingLabels;
    ImageData trainingImages;

    if (loadLabels(trainingLabelsPath, &trainingLabels) == -1) {
        printf("Error reading labels\n");
        exit(EXIT_FAILURE);
    }

    if (loadImages(trainingImagesPath, &trainingImages) == -1) {
        free(trainingLabels.labels);
        printf("Error loading images\n");
        exit(EXIT_FAILURE);
    }

    uint32_t rows = trainingImages.numRows;
    uint32_t cols = trainingImages.numCols;
    uint32_t numPixels = rows * cols;

    double inputLayer[numPixels];
    double weightsI_H[20][numPixels];
    double biasI_H[20];
    double hiddenLayer[20];
    double weightsH_O[10][20];
    double biasH_O[10];
    double outputLayer[10];

    free(trainingLabels.labels);
    free(trainingImages.pixelData);
    exit(EXIT_SUCCESS);
}
