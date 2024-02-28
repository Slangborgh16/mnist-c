#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "mnist.h"
#include "matrix.h"
#include "neuralnet.h"

#define HIDDEN_NODES 20
#define OUTPUT_NODES 10
#define BATCH_SIZE 10

void shuffleArray(int arr[], int size);

int main() {
    srand(time(NULL));

    const char trainingLabelsPath[] = "dataset/train-labels-idx1-ubyte";
    const char trainingImagesPath[] = "dataset/train-images-idx3-ubyte";

    LabelData trainingLabels;
    ImageData trainingImages;

    loadLabels(trainingLabelsPath, &trainingLabels);
    loadImages(trainingImagesPath, &trainingImages);

    int imageIndices[trainingImages.numImages];
    for (int i = 0; i < trainingImages.numImages; i++)
        imageIndices[i] = i;

    uint32_t numPixels = trainingImages.numPixels;

    Network* nnet = neuralNetCreate(numPixels, HIDDEN_NODES, OUTPUT_NODES);
    
    
    shuffleArray(imageIndices, trainingImages.numImages);
    int imageId = imageIndices[0];

    Matrix* input = imgToMatrix(&trainingImages, imageId);
    Matrix* output = forwardprop(nnet, input);

    printImage(input);
    int width = (int)log10(trainingImages.numImages) + 1;
    printf("Image ID: %0*d     Label: %d\n", width, imageId + 1, getLabel(&trainingLabels, imageId));

    Matrix* oneHotLabel = oneHotEncode(&trainingLabels, imageId);

    printf("\nOutput:\n");
    Matrix* output_transpose = matrixTranspose(output);
    matrixPrint(output_transpose);

    free(trainingLabels.labels);
    free(trainingImages.pixelData);

    neuralNetFree(nnet);

    matrixFree(oneHotLabel);
    matrixFree(input);
    matrixFree(output);
    matrixFree(output_transpose);

    exit(EXIT_SUCCESS);
}


// Fisher-Yates shuffle
void shuffleArray(int arr[], int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
