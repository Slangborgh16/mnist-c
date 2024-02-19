#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mnist.h"


int main() {
    const char trainingLabelsPath[] = "dataset/train-labels-idx1-ubyte";
    const char trainingImagesPath[] = "dataset/train-images-idx3-ubyte";
    const char testingLabelsPath[] = "dataset/t10k-labels-idx1-ubyte";
    const char testingImagesPath[] = "dataset/t10k-images-idx3-ubyte";

    LabelData trainingLabels;
    ImageData trainingImages;
    LabelData testingLabels;
    ImageData testingImages;

    if (loadLabels(trainingLabelsPath, &trainingLabels) == -1) {
        printf("Error reading training labels\n");
        exit(EXIT_FAILURE);
    }

    if (loadLabels(testingLabelsPath, &testingLabels) == -1) {
        printf("Error reading testing labels\n");
        exit(EXIT_FAILURE);
    }

    printf("There are %u training labels and %u testing labels.\n", trainingLabels.numLabels, testingLabels.numLabels);

    free(trainingLabels.labels);
    free(testingLabels.labels);
    exit(EXIT_SUCCESS);
}
