#include "mnist.h"

void loadDataset(Dataset* dataset, const char* labelsPath, const char* imagesPath) {
    // Load labels
    FILE* labelFd = fopen(labelsPath, "r");

    if (labelFd == NULL) {
        printf("Error opening labels file: %s\nerrno: %d\n", labelsPath, errno);
        exit(EXIT_FAILURE);
    }

    uint32_t numImages;

    fseek(labelFd, sizeof(uint32_t), SEEK_CUR);
    fread(&numImages, sizeof(uint32_t), 1, labelFd);
    numImages = be32toh(numImages);

    dataset->numImages = numImages;
    dataset->labels = (uint8_t*)malloc(sizeof(uint8_t) * numImages);
    fread(dataset->labels, sizeof(uint8_t), numImages, labelFd);

    if (fclose(labelFd) == EOF) {
        printf("Error closing labels file.\nerrno: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    // Load images
    FILE* imageFd = fopen(imagesPath, "r");

    if (imageFd == NULL) {
        printf("Error opening image file: %s\nerrno: %d\n", imagesPath, errno);
        exit(EXIT_FAILURE);
    }

    uint32_t rows;
    uint32_t cols;
    uint32_t numPixels;

    fseek(imageFd, sizeof(uint32_t) * 2, SEEK_CUR);
    fread(&rows, sizeof(uint32_t), 1, imageFd);
    fread(&cols, sizeof(uint32_t), 1, imageFd);
    rows = be32toh(rows);
    cols = be32toh(cols);
    numPixels = rows * cols;

    dataset->imageRows = rows;
    dataset->imageCols = cols;
    dataset->pixelData = (uint8_t*)malloc(sizeof(uint8_t) * numPixels * numImages);
    fread(dataset->pixelData, sizeof(uint8_t), numPixels * numImages, imageFd);

    if (fclose(imageFd) == EOF) {
        printf("Error closing image file.\nerrno: %d\n", errno);
        exit(EXIT_FAILURE);
    }
}


int getLabel(Dataset* dataset, const int index) {
    int label = *(dataset->labels + sizeof(uint8_t) * index);    
    return label;
}


Matrix* oneHotEncode(Dataset* dataset, const int index) {
    int label = getLabel(dataset, index);    
    int classes = 10;
    Matrix* onehot = matrixCreate(classes, 1);

    for (int i = 0; i < classes; i++)
        onehot->values[i][0] = 0.0;

    onehot->values[label][0] = 1.0;

    return onehot;
}


Matrix* imgToMatrix(Dataset* dataset, const int index) {
    if (index >= dataset->numImages || index < 0) {
        printf("Image index %d out of range. Max index: %d\n", index, dataset->numImages - 1);
        exit(EXIT_FAILURE);
    }

    int numPixels = dataset->imageRows * dataset->imageCols;
    uint8_t* pixels = dataset->pixelData + sizeof(uint8_t) * numPixels * index;

    // Create a column vector for the pixel data
    Matrix* img = matrixCreate(numPixels, 1);

    for (int i = 0; i < numPixels; i++)
        img->values[i][0] = (double)pixels[i] / 255;

    return img;
}


void printImage(Dataset* dataset, const int index) {
    int rows = dataset->imageRows;
    int cols = dataset->imageCols;
    uint8_t* pixels = dataset->pixelData + sizeof(uint8_t) * rows * cols * index;

    int x = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val = (double)pixels[x] / 255.0;

            if (val > 0.75) {
                printf(" ");
            } else if (val > 0.5) {
                printf("░");
            } else if (val > 0.25) {
                printf("▒");
            } else if (val > 0.0) {
                printf("▓");
            } else {
                printf("█");
            }

            x++;
        }
        printf("\n");
    }
}
