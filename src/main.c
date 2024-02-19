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
        printf("Error reading training labels\n");
        exit(EXIT_FAILURE);
    }

    if (loadImages(trainingImagesPath, &trainingImages) == -1) {
        free(trainingLabels.labels);
        printf("Error loading images\n");
        exit(EXIT_FAILURE);
    }

    free(trainingLabels.labels);
    free(trainingImages.pixelData);
    exit(EXIT_SUCCESS);
}
