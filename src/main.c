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
#define NUM_EPOCHS 1

void shuffleArray(int arr[], const int size);
void train(Network* nnet, Dataset* dataset, const int epochs, double learningRate);

int main() {
    srand(time(NULL));

    const double learningRate = 1.0;
    const char trainingLabelsPath[] = "dataset/train-labels-idx1-ubyte";
    const char trainingImagesPath[] = "dataset/train-images-idx3-ubyte";

    Dataset training;

    loadDataset(&training, trainingLabelsPath, trainingImagesPath);
    uint32_t numPixels = training.imageRows * training.imageCols;


    Network* nnet = neuralNetCreate(numPixels, HIDDEN_NODES, OUTPUT_NODES);
    
    train(nnet, &training, NUM_EPOCHS, learningRate);

    /*
    for (int i = 0; i < 100; i++) {
        printf("\033[H\033[J");
        int imageId = imageIndices[i];

        Matrix* input = imgToMatrix(&training, imageId);
        Matrix* output = forwardprop(nnet, input);
        Matrix* oneHotLabel = oneHotEncode(&training, imageId);

        int width = (int)log10(numImages) + 1;
        printImage(&training, imageId);
        printf("Image ID: %0*d     Label: %d\n", width, imageId + 1, getLabel(&training, imageId));

        printf("\nOutput:\n");
        Matrix* output_transpose = matrixTranspose(output);
        matrixPrint(output_transpose);

        matrixFree(input);
        matrixFree(output);
        matrixFree(oneHotLabel);
        matrixFree(output_transpose);

        sleep(1);
    }
    */
    Matrix* input = imgToMatrix(&training, 0);
    Matrix* output = forwardprop(nnet, input);
    Matrix* oneHotLabel = oneHotEncode(&training, 0);

    int width = (int)log10(training.numImages) + 1;
    printImage(&training, 0);
    printf("Image ID: %0*d     Label: %d\n", width, 0 + 1, getLabel(&training, 0));

    printf("\nOutput:\n");
    Matrix* output_transpose = matrixTranspose(output);
    matrixPrint(output_transpose);

    matrixFree(input);
    matrixFree(output);
    matrixFree(oneHotLabel);
    matrixFree(output_transpose);

    neuralNetFree(nnet);

    free(training.labels);
    free(training.pixelData);
    training.labels = NULL;
    training.pixelData = NULL;

    exit(EXIT_SUCCESS);
}


// Fisher-Yates shuffle
void shuffleArray(int arr[], const int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}


void train(Network* nnet, Dataset* dataset, const int epochs, double learningRate) {
    int numImages = dataset->numImages;
    int imageIndices[numImages];

    for (int i = 0; i < numImages; i++)
        imageIndices[i] = i;

    for (int i = 0; i < epochs; i++) {
        printf("Epoch #%d\n", i);

        shuffleArray(imageIndices, numImages);

        int batchIndex = 0;                         // Index of current batch
        int batchSize = BATCH_SIZE * sizeof(int);   // Size of the current batch

        while(batchIndex < numImages - 1) {
            // printf("batchIndex: %d\n", batchIndex);
            int* batchAddr = imageIndices + batchIndex;
            Matrix* input = imagesToMatrix(dataset, batchAddr, batchSize);
            Matrix* oneHotLabels = oneHotEncodeLabels(dataset, batchAddr, batchSize);

            backprop(nnet, input, oneHotLabels, learningRate, batchSize);

            matrixFree(input);
            matrixFree(oneHotLabels);

            if ((numImages - 1 - batchIndex) < BATCH_SIZE)
                batchSize = (numImages - 1 - batchIndex) * sizeof(int);

            batchIndex += batchSize;
        }
    }
}
