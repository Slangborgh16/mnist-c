#include "mnist.h"

void loadLabels(const char* labelsPath, struct LabelData* labelData) {
    FILE* fd = fopen(labelsPath, "r");

    if (fd == NULL) {
        printf("Error opening file: %s\nerrno: %d\n", labelsPath, errno);
        exit(EXIT_FAILURE);
    }

    uint32_t magicNumber;
    uint32_t numLabels;

    fread(&magicNumber, sizeof(uint32_t), 1, fd);
    fread(&numLabels, sizeof(uint32_t), 1, fd);

    labelData->magicNumber = be32toh(magicNumber);
    labelData->numLabels = be32toh(numLabels);
    labelData->labels = (uint8_t*)malloc(sizeof(uint8_t) * labelData->numLabels);

    fread(labelData->labels, sizeof(uint8_t), labelData->numLabels, fd);

    if (fclose(fd) == EOF) {
        printf("Error closing file.\nerrno: %d\n", errno);
        exit(EXIT_FAILURE);
    }
}


void loadImages(const char* imagesPath, struct ImageData* imageData) {
    FILE* fd = fopen(imagesPath, "r");

    if (fd == NULL) {
        printf("Error opening file: %s\nerrno: %d\n", imagesPath, errno);
        exit(EXIT_FAILURE);
    }

    uint32_t magicNumber;
    uint32_t numImages;
    uint32_t numRows;
    uint32_t numCols;
    uint32_t totalPixels;

    fread(&magicNumber, sizeof(uint32_t), 1, fd);
    fread(&numImages, sizeof(uint32_t), 1, fd);
    fread(&numRows, sizeof(uint32_t), 1, fd);
    fread(&numCols, sizeof(uint32_t), 1, fd);

    imageData->magicNumber = be32toh(magicNumber);
    imageData->numImages = be32toh(numImages);
    imageData->numPixels = be32toh(numRows) * be32toh(numCols);
    totalPixels = imageData->numImages * imageData->numPixels;
    imageData->pixelData = (uint8_t*)malloc(sizeof(uint8_t) * totalPixels);

    fread(imageData->pixelData, sizeof(uint8_t), totalPixels, fd);

    if (fclose(fd) == EOF) {
        printf("Error closing file.\nerrno: %d\n", errno);
        exit(EXIT_FAILURE);
    }
}


Matrix* imgToMatrix(ImageData* imageData, const int index) {
    if (index >= imageData->numImages || index < 0) {
        printf("Image index %d out of range. Max index: %d\n", index, imageData->numImages - 1);
        exit(EXIT_FAILURE);
    }

    int numPixels = imageData->numPixels;
    uint8_t* pixels = imageData->pixelData + sizeof(uint8_t) * numPixels * index;

    // Create a column vector for the pixel data
    Matrix* img = matrixCreate(numPixels, 1);

    for (int i = 0; i < numPixels; i++)
        img->values[i][0] = (double)pixels[i] / 255;

    return img;
}


uint8_t* readImage(ImageData* imageData, const int index) {
    if (index >= imageData->numImages || index < 0) return NULL;

    uint8_t* pixelDataPtr = imageData->pixelData + sizeof(uint8_t) * imageData->numPixels * index;
    return pixelDataPtr;
}


Matrix* oneHotEncode(LabelData* labelData, const int index) {
    int label = *(labelData->labels + sizeof(uint8_t) * index);    
    int classes = 10;
    Matrix* onehot = matrixCreate(classes, 1);

    for (int i = 0; i < classes; i++)
        onehot->values[i][0] = 0.0;

    onehot->values[label][0] = 1.0;

    return onehot;
}
