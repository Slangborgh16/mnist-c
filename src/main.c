#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

    Network nnet;
    nnet.inputNodes = numPixels;
    nnet.hiddenNodes = HIDDEN_NODES;
    nnet.outputNodes = OUTPUT_NODES;

    nnet.w1 = matrixCreate(HIDDEN_NODES, numPixels);
    nnet.w2 = matrixCreate(OUTPUT_NODES, HIDDEN_NODES);
    matrixRandomize(nnet.w1);
    matrixRandomize(nnet.w2);

    nnet.b1 = matrixCreate(HIDDEN_NODES, 1);
    nnet.b2 = matrixCreate(OUTPUT_NODES, 1);
    matrixFill(nnet.b1, 0.00);
    matrixFill(nnet.b2, 0.00);
    
    
    shuffleArray(imageIndices, trainingImages.numImages);

    Matrix* input = imgToMatrix(&trainingImages, imageIndices[0]);
    Matrix* output = forwardprop(&nnet, input);

    printImage(input);
    printf("Image ID: %d\n", imageIndices[0]);

    Matrix* oneHotLabel = oneHotEncode(&trainingLabels, imageIndices[0]);

    printf("\nOutput:\n");
    Matrix* output_transpose = matrixTranspose(output);
    matrixPrint(output_transpose);

    free(trainingLabels.labels);
    free(trainingImages.pixelData);

    matrixFree(nnet.w1);
    matrixFree(nnet.w2);
    matrixFree(nnet.b1);
    matrixFree(nnet.b2);

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
