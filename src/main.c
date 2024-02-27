#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mnist.h"
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

    if (loadLabels(trainingLabelsPath, &trainingLabels) == -1) {
        printf("Error reading labels\n");
        exit(EXIT_FAILURE);
    }

    if (loadImages(trainingImagesPath, &trainingImages) == -1) {
        free(trainingLabels.labels);
        printf("Error loading images\n");
        exit(EXIT_FAILURE);
    }

    int imageIndices[trainingImages.numImages];
    for (int i = 0; i < trainingImages.numImages; i++)
        imageIndices[i] = i;

    uint32_t rows = trainingImages.numRows;
    uint32_t cols = trainingImages.numCols;
    uint32_t numPixels = rows * cols;

    Network nnet;
    nnet.inputNodes = numPixels;
    nnet.hiddenNodes = HIDDEN_NODES;
    nnet.outputNodes = OUTPUT_NODES;

    nnet.z1 = (double*)malloc(sizeof(double) * HIDDEN_NODES);
    nnet.z2  = (double*)malloc(sizeof(double) * OUTPUT_NODES);

    nnet.a0 = (double*)malloc(sizeof(double) * numPixels);
    nnet.a1 = (double*)malloc(sizeof(double) * HIDDEN_NODES);
    nnet.a2  = (double*)malloc(sizeof(double) * OUTPUT_NODES);

    nnet.w1 = createMat(HIDDEN_NODES, numPixels);
    nnet.w2 = createMat(OUTPUT_NODES, HIDDEN_NODES);
    initializeWeights(HIDDEN_NODES, numPixels, nnet.w1);
    initializeWeights(OUTPUT_NODES, HIDDEN_NODES, nnet.w2);

    nnet.b1 = (double*)malloc(sizeof(double) * HIDDEN_NODES);
    nnet.b2 = (double*)malloc(sizeof(double) * OUTPUT_NODES);
    memset(nnet.b1, 0, sizeof(double) * HIDDEN_NODES);
    memset(nnet.b2, 0, sizeof(double) * OUTPUT_NODES);
    
    
    shuffleArray(imageIndices, trainingImages.numImages);
    printf("Image ID: %d\n", imageIndices[0]);

    double inputLabel[10];
    int label = oneHotEncode(&trainingLabels, imageIndices[0], inputLabel);
    printf("Input: %d\n[ ", label);
    for (int i = 0; i < 10; i++)
        printf("%d ", (int)inputLabel[i]);
    printf("]\n");

    uint8_t* inputImage = readImage(&trainingImages, imageIndices[0]);
    vecNormalize(nnet.inputNodes, inputImage, nnet.a0, 255);
    forwardprop(&nnet);
    printf("\nOutput:\n[ ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", nnet.a2[i]);
    printf("]\n");

    free(trainingLabels.labels);
    free(trainingImages.pixelData);

    free(nnet.z1);
    free(nnet.z2);
    free(nnet.a0);
    free(nnet.a1);
    free(nnet.a2);
    free(nnet.b1);
    free(nnet.b2);
    freeMat(HIDDEN_NODES, numPixels, nnet.w1);
    freeMat(OUTPUT_NODES, HIDDEN_NODES, nnet.w2);

    exit(EXIT_SUCCESS);
}


// Fisher-Yates shuffle
void shuffleArray(int arr[], int size) {
    srand(time(NULL));
    
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
