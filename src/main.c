#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mnist.h"
#include "neuralnet.h"

#define HIDDEN_NODES 20
#define OUTPUT_NODES 10

int main() {
    srand(time(NULL));

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

    Network nnet;
    nnet.inputNodes = numPixels;
    nnet.hiddenNodes = HIDDEN_NODES;
    nnet.outputNodes = OUTPUT_NODES;

    nnet.inputLayer = (double*)malloc(sizeof(double) * numPixels);
    nnet.hiddenLayer = (double*)malloc(sizeof(double) * HIDDEN_NODES);
    nnet.outputLayer  = (double*)malloc(sizeof(double) * OUTPUT_NODES);

    nnet.weights1 = initializeWeights(HIDDEN_NODES, numPixels);
    nnet.weights2 = initializeWeights(OUTPUT_NODES, HIDDEN_NODES);

    nnet.bias1 = (double*)malloc(sizeof(double) * HIDDEN_NODES);
    nnet.bias2 = (double*)malloc(sizeof(double) * OUTPUT_NODES);
    memset(nnet.bias1, 0, sizeof(double) * HIDDEN_NODES);
    memset(nnet.bias2, 0, sizeof(double) * OUTPUT_NODES);


    // Test to see if weights initialized correctly
    printf("Input-Hidden Weights\n");
    for (int m = 0; m < HIDDEN_NODES; m++) {
        for (int n = 0; n < numPixels; n++) {
            printf("%.3f ", nnet.weights1[m][n]);
        }
        printf("\n");
    }

    printf("\nHidden-Output Weights\n");
    for (int m = 0; m < OUTPUT_NODES; m++) {
        for (int n = 0; n < HIDDEN_NODES; n++) {
            printf("%.3f ", nnet.weights2[m][n]);
        }
        printf("\n");
    }

    free(trainingLabels.labels);
    free(trainingImages.pixelData);

    free(nnet.inputLayer);
    free(nnet.hiddenLayer);
    free(nnet.outputLayer);
    free(nnet.bias1);
    free(nnet.bias2);
    freeWeights(HIDDEN_NODES, numPixels, nnet.weights1);
    freeWeights(OUTPUT_NODES, HIDDEN_NODES, nnet.weights2);

    exit(EXIT_SUCCESS);
}
