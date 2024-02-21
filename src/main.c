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

    double inputLayer[numPixels];
    double weightsI_H[HIDDEN_NODES][numPixels];
    double biasI_H[HIDDEN_NODES] = {0};
    double hiddenLayer[HIDDEN_NODES];
    double weightsH_O[HIDDEN_NODES][OUTPUT_NODES];
    double biasH_O[OUTPUT_NODES] = {0};
    double outputLayer[OUTPUT_NODES];

    initializeWeights(HIDDEN_NODES, numPixels, weightsI_H);
    initializeWeights(OUTPUT_NODES, HIDDEN_NODES, weightsH_O);

    // Test to see if weights initialized correctly
    for (int m = 0; m < HIDDEN_NODES; m++) {
        for (int n = 0; n < OUTPUT_NODES; n++) {
            printf("%.3f ", weightsH_O[m][n]);
        }
        printf("\n");
    }
    free(trainingLabels.labels);
    free(trainingImages.pixelData);
    exit(EXIT_SUCCESS);
}
