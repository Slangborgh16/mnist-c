#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
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
    int numImages = trainingImages.numImages;
    uint32_t numPixels = trainingImages.numPixels;

    int imageIndices[numImages];
    for (int i = 0; i < numImages; i++)
        imageIndices[i] = i;
    shuffleArray(imageIndices, numImages);

    Network* nnet = neuralNetCreate(numPixels, HIDDEN_NODES, OUTPUT_NODES);
    
    for (int i = 0; i < 100; i++) {
        printf("\033[H\033[J");
        int imageId = imageIndices[i];

        Matrix* input = imgToMatrix(&trainingImages, imageId);
        Matrix* output = forwardprop(nnet, input);
        Matrix* oneHotLabel = oneHotEncode(&trainingLabels, imageId);

        int width = (int)log10(numImages) + 1;
        printImage(input);
        printf("Image ID: %0*d     Label: %d\n", width, imageId + 1, getLabel(&trainingLabels, imageId));

        printf("\nOutput:\n");
        Matrix* output_transpose = matrixTranspose(output);
        matrixPrint(output_transpose);

        matrixFree(input);
        matrixFree(output);
        matrixFree(oneHotLabel);
        matrixFree(output_transpose);

        sleep(1);
    }

    neuralNetFree(nnet);

    free(trainingLabels.labels);
    free(trainingImages.pixelData);
    trainingLabels.labels = NULL;
    trainingImages.pixelData = NULL;

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
