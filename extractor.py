import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

## EXTRACTION PROCESS
# 1- Provide toBeExtracted folder
# 2- Load images to extract features from
# 3- Extract several defect features from each image
# 4- Save all defect features with alpha channel into a folder

## GENERATION PROCESS
# Provide forConstruction folder
# Randomly select n images from the folder
# Average/combine/merge n images to have a base image
# Select m extracted defect layers
# Apply random amount of elastic transformations to all m images
# Merge background and defect layers
# OPTIONAL/TEST: Apply 3x3 Gaussian to defect layer.
# Put result image into generated folder.

## OPTIONAL STEPS
# Try if generated image is classified as defect, otherwise repeat defect layer addition process


def LoadImagesFromFolder(folderPath, extension):
    imageFiles = []
    for file in os.listdir(folderPath):
        if file.endswith(extension):
            filePath = os.path.join(folderPath, file)
            imageFiles.append(filePath)
    return imageFiles

def ExtractFeaturesFromImageCV(imagePath, featureLayersToExtract, kernelSize, featureSavePath):
    showResult = False
    # Load image
    rawImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    gaussianImage = cv2.GaussianBlur(rawImage, (25, 25), 0)

    hfFeaturesImage = cv2.subtract(rawImage, gaussianImage)  # Could be one of the feature layers

    # Feature 1
    f1a = cv2.add(rawImage, hfFeaturesImage)
    _, f1b = cv2.threshold(f1a, 0, 255, cv2.THRESH_OTSU)
    f1bKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    f1c = cv2.morphologyEx(f1b, cv2.MORPH_CLOSE, f1bKernel)
    f1d = cv2.bitwise_not(f1c)

    feature1Alpha = cv2.merge([rawImage, rawImage, rawImage, f1d], 4)
    cv2.imwrite(os.path.join(featureSavePath, os.path.basename(imagePath)), feature1Alpha)

    if (showResult):
        f, axarr = plt.subplots(2)
        axarr[0].imshow(rawImage)
        axarr[1].imshow(cv2.GaussianBlur(feature1Alpha, (3,3), 0))
        plt.show()


if __name__ == '__main__':
    folderToExtractImages = r'G:\Selim\Thesis\Code\3MSet_Mid\Defect'
    folderToSaveFeatures = r'G:\Selim\Thesis\Code\3MSet_Mid\Features'
    imageFiles = LoadImagesFromFolder(folderToExtractImages, '.png')

    for image in imageFiles:
        ExtractFeaturesFromImageCV(image, 0, 0, folderToSaveFeatures)


#rawImage = cv2.imread('image.bmp', cv2.IMREAD_UNCHANGED)


#ret, thresholdedImage = cv2.threshold(rawImage, 115, 255, cv2.THRESH_BINARY_INV)

#openingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#openImage = cv2.morphologyEx(thresholdedImage, cv2.MORPH_OPEN, openingKernel)
##openImage = cv2.morphologyEx(openImage, cv2.MORPH_OPEN, openingKernel)

#dilationKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#dilatedImage = cv2.morphologyEx(openImage, cv2.MORPH_DILATE, dilationKernel)

#rawImageAlpha = cv2.cvtColor(rawImage, cv2.COLOR_GRAY2BGRA)
#dilatedImageAlpha = cv2.cvtColor(dilatedImage, cv2.COLOR_GRAY2BGRA)


#imageAdd = cv2.addWeighted(rawImageAlpha, 0.5, dilatedImageAlpha, 0.5, 0)
##imageAdd = cv2.add(rawImageAlpha, dilatedImageAlpha)

#plt.imshow(rawImageAlpha, cmap = 'gray', interpolation='none')
#plt.xticks([]), plt.yticks([])
#plt.show()

#plt.imshow(imageAdd, cmap = 'gray', interpolation='none')
#plt.xticks([]), plt.yticks([])
#plt.show()