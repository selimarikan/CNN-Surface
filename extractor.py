import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from enum import Enum
import math

def LoadImagesFromFolder(folderPath, extension):
    imageFiles = []
    for file in os.listdir(folderPath):
        if file.endswith(extension):
            filePath = os.path.join(folderPath, file)
            imageFiles.append(filePath)
    return imageFiles

def ExtractFeaturesFromImageCV(imagePath, featureLayersToExtract, kernelSize, featureSavePath):
    showResult = False
    rawImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    gaussianImage = cv2.GaussianBlur(rawImage, (25, 25), 0)

    hfFeaturesImage = cv2.subtract(rawImage, gaussianImage)  # Could be one of the feature layers

    # Feature 1
    f1a = cv2.add(rawImage, hfFeaturesImage)
    _, f1b = cv2.threshold(f1a, 0, 255, cv2.THRESH_OTSU)
    f1bKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    f1c = cv2.morphologyEx(f1b, cv2.MORPH_CLOSE, f1bKernel)
    f1d = cv2.bitwise_not(f1c)

    # & each channel with the mask so that unnecessary pixels are deleted
    feature1Alpha = cv2.merge([rawImage & f1d, rawImage & f1d, rawImage & f1d, f1d], 4)
    cv2.imwrite(os.path.join(featureSavePath, os.path.basename(imagePath)), feature1Alpha)

    if (showResult):
        f, axarr = plt.subplots(2)
        axarr[0].imshow(rawImage)
        axarr[1].imshow(cv2.GaussianBlur(feature1Alpha, (3,3), 0))
        plt.show()

class Transformation(Enum):
    ROTATE = 0
    SCALE = 1
    MIRROR = 2
    SHARPEN = 3
    SMOOTH = 4
    BRIGHTNESS = 5
    #SHIFT = 6

def TransformImage(image, transformation):
    rows, cols, ch = image.shape

    if (transformation == Transformation.ROTATE):
        rotationMode = np.random.randint(1, 4) # 90, 180, 270
        Mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90 * rotationMode, 1)
        return cv2.warpAffine(image, Mat, (cols, rows))

    if (transformation == Transformation.SCALE):
        scaleRatio = (np.random.rand(1, 1) / 8.0) + 1.0 # Between 1 and 1.125 (1 + 1/8)
        resizedImage = cv2.resize(image, None, fx=scaleRatio, fy=scaleRatio, interpolation=cv2.INTER_LINEAR)
        resizedRows, resizedCols, ch = resizedImage.shape
        # Crop the center
        rowDiff2 = int(math.floor((resizedRows - rows) / 2))
        colDiff2 = int(math.floor((resizedCols - cols) / 2))
        # Not very good
        return resizedImage[rowDiff2:rows + rowDiff2, colDiff2:cols + colDiff2]
        #return image

    if (transformation == Transformation.MIRROR):
        flipMode = np.random.randint(0, 2) # X or Y
        return cv2.flip(image, flipMode)

    if (transformation == Transformation.SHARPEN):
        kernels = [1, 3]
        kernelSize = kernels[np.random.randint(0, len(kernels))]
        blurred = cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)
        return cv2.addWeighted(image, 1.4, blurred, -0.4, 0)

    if (transformation == Transformation.SMOOTH):
        kernels = [1]
        kernelSize = kernels[np.random.randint(0, len(kernels))]
        return cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)

    if (transformation == Transformation.BRIGHTNESS):
        multRatio = ((np.random.rand(1, 1) - 0.5) / 2.0) # -0.25 to 0.25
        return cv2.addWeighted(image, 1.0, image, multRatio, 0)

    #if (transformation == Transformation.SHIFT):
     #   shiftX = np.random.randint(-21, 21)
      #  shiftY = np.random.randint(-21, 21)


def TestTransformation(imagePath, transformation):
    image = cv2.imread(imagePath)
    cv2.imshow('RawImage', image)
    transformedImage = TransformImage(image, Transformation(transformation))

    cv2.imshow(str(Transformation(transformation)), transformedImage)
    cv2.waitKey(0)

def GenerateNonDefectImage(backgroundFiles, backgroundImageCount, transformationCount):
    showResult = False
    # Parameters
    generateBgndImageCount = backgroundImageCount
    generateTransformationCount = transformationCount

    backgroundFileIndices = np.linspace(0, len(backgroundFiles) - 1, len(backgroundFiles)).astype('uint8')

    # Pick random n background images and fuse them
    bgndImagesToProcess = []

    # Select background images
    while (generateBgndImageCount > 0):
        # Pick random index
        imageIndex = np.random.randint(0, len(backgroundFileIndices))
        bgndImagesToProcess.append(cv2.imread(bgndFiles[backgroundFileIndices[imageIndex]]))
        # Remove used image index so it will not be picked again
        backgroundFileIndices = np.delete(backgroundFileIndices, imageIndex)
        generateBgndImageCount = generateBgndImageCount - 1

    # Fuse background images
    backgroundImage = bgndImagesToProcess[0]
    for iImage in xrange(len(bgndImagesToProcess) - 1):
        # Fix to 1/len multipliers to have equal weights
        backgroundImage = cv2.addWeighted(backgroundImage, 0.5, bgndImagesToProcess[iImage + 1], 0.5, 0)

    if (showResult):
        # Create appropriate sized subplot layout
        f, axarr = plt.subplots(max(generateBgndImageCount, generateFeatureImageCount) + 1, 3)

        # Visualize background images
        for iImage in xrange(len(bgndImagesToProcess)):
            axarr[iImage, 0].imshow(bgndImagesToProcess[iImage])
            axarr[iImage, 0].set_title('Background image ' + str(iImage))
        axarr[len(axarr) - 1, 0].imshow(backgroundImage)
        axarr[len(axarr) - 1, 0].set_title('Generated background image')

    # Unsharp mask the background
    blurred = cv2.GaussianBlur(backgroundImage, (5, 5), 0)
    backgroundImage = cv2.addWeighted(backgroundImage, 1.5, blurred, -0.5, 0)

    for iTrafo in xrange(generateTransformationCount):
        trafoType = np.random.randint(0, 5)
        backgroundImage = TransformImage(backgroundImage, Transformation(trafoType))

    return backgroundImage

def GenerateDefectImage(featureFiles, backgroundFiles, backgroundImageCount, featureImageCount, transformationCount):
    showResult = False
    # Parameters
    generateBgndImageCount = backgroundImageCount
    generateFeatureImageCount = featureImageCount
    generateTransformationCount = transformationCount

    # To overcome array copying problem in each function call
    featureFileIndices = np.linspace(0, len(featureFiles) - 1, len(featureFiles)).astype('uint8')
    backgroundFileIndices = np.linspace(0, len(backgroundFiles) - 1, len(backgroundFiles)).astype('uint8')

    # Pick random n background images and fuse them
    bgndImagesToProcess = []

    # Select background images
    while (generateBgndImageCount > 0):
        # Pick random index
        imageIndex = np.random.randint(0, len(backgroundFileIndices))
        bgndImagesToProcess.append(cv2.imread(bgndFiles[backgroundFileIndices[imageIndex]]))
        # Remove used image index so it will not be picked again
        backgroundFileIndices = np.delete(backgroundFileIndices, imageIndex)
        generateBgndImageCount = generateBgndImageCount - 1

    # Fuse background images
    backgroundImage = bgndImagesToProcess[0]
    for iImage in xrange(len(bgndImagesToProcess) - 1):
        # Fix to 1/len multipliers to have equal weights
        backgroundImage = cv2.addWeighted(backgroundImage, 0.5, bgndImagesToProcess[iImage + 1], 0.5, 0)

    if (showResult):
        # Create appropriate sized subplot layout
        f, axarr = plt.subplots(max(generateBgndImageCount, generateFeatureImageCount) + 1, 3)

        # Visualize background images
        for iImage in xrange(len(bgndImagesToProcess)):
            axarr[iImage, 0].imshow(bgndImagesToProcess[iImage])
            axarr[iImage, 0].set_title('Background image ' + str(iImage))
        axarr[len(axarr) - 1, 0].imshow(backgroundImage)
        axarr[len(axarr) - 1, 0].set_title('Generated background image')

    # Pick random m feature images and fuse them
    featureImagesToProcess = []

    # Select feature images
    while (generateFeatureImageCount > 0):
        # Pick random index
        imageIndex = np.random.randint(0, len(featureFileIndices))
        featureImagesToProcess.append(cv2.imread(featureFiles[featureFileIndices[imageIndex]], cv2.IMREAD_UNCHANGED))
        # Remove used image so it will not be picked again
        featureFileIndices = np.delete(featureFileIndices, imageIndex)
        generateFeatureImageCount = generateFeatureImageCount - 1

    # Fuse background images
    featureImage = featureImagesToProcess[0]
    for iImage in xrange(len(featureImagesToProcess) - 1):
        # Fix to 1/len multipliers to have equal weights
        featureImage = cv2.add(featureImage, featureImagesToProcess[iImage + 1])  # cv2.addWeighted(featureImage, 1.0, featureImagesToProcess[iImage + 1], 1.0, 0)

    # Try blurring
    featureImage = cv2.GaussianBlur(featureImage, (1, 1), 0)

    # Unsharp mask the background
    blurred = cv2.GaussianBlur(backgroundImage, (5, 5), 0)
    backgroundImage = cv2.addWeighted(backgroundImage, 1.5, blurred, -0.5, 0)

    if (showResult):
        # Visualize feature images
        for iImage in xrange(len(featureImagesToProcess)):
            axarr[iImage, 1].imshow(featureImagesToProcess[iImage])
            axarr[iImage, 1].set_title('Feature image ' + str(iImage))
        axarr[len(axarr) - 1, 1].imshow(featureImage)
        axarr[len(axarr) - 1, 1].set_title('Generated feature image')

    # Final fusion of background and features
    generatedImage = backgroundImage
    height, width, chn = backgroundImage.shape

    # Transform featureImage
    # for iTrafo in xrange(generateTransformationCount):
    #     trafoType = np.random.randint(0, 5)
    #     featureImage = TransformImage(featureImage, Transformation(trafoType))

    for c in range(0, 3):
        alpha = featureImage[:, :, 3] / 255.0
        color = featureImage[:, :, c] * alpha
        beta = backgroundImage[:, :, c] * (1.0 - alpha)

        generatedImage[:, :, c] = color + beta

    for iTrafo in xrange(generateTransformationCount):
        trafoType = np.random.randint(0, 5)
        generatedImage = TransformImage(generatedImage, Transformation(trafoType))

    if (showResult):
        axarr[0, 2].imshow(generatedImage)
        plt.show()

    return generatedImage

class ExecutionMode(Enum):
    EXTRACT = 0,
    GENERATE = 1,
    TRANSFORM = 2,
    TESTTRANSFORM = 3

if __name__ == '__main__':
    mode = ExecutionMode.TESTTRANSFORM
    basePath = r'C:\Users\Selim\Documents\GitHub\Files\3MSet_Large\\'

    if (mode == ExecutionMode.EXTRACT):
        folderToExtractImages = os.path.join(basePath, 'Defect')
        folderToSaveFeatures = os.path.join(basePath, 'Features')
        imageFiles = LoadImagesFromFolder(folderToExtractImages, '.png')

        for image in imageFiles:
            ExtractFeaturesFromImageCV(image, 0, 0, folderToSaveFeatures)

    if (mode == ExecutionMode.GENERATE):
        generateCount = 500
        folderToFeatures = os.path.join(basePath, 'Features')
        folderToBgndImages = os.path.join(basePath, 'NonDefect')
        folderToSaveGeneratedDefectImages = os.path.join(basePath, 'GeneratedDefect')
        folderToSaveGeneratedNonDefectImages = os.path.join(basePath, 'GeneratedNonDefect')
        featureFiles = LoadImagesFromFolder(folderToFeatures, '.png')
        bgndFiles = LoadImagesFromFolder(folderToBgndImages, '.png')

        # Parameters
        generateBgndImageCount = 2
        generateFeatureImageCount = 2
        generateTransformationCount = 2


        for iGenerate in xrange(0, generateCount):
            image = GenerateDefectImage(featureFiles, bgndFiles, generateBgndImageCount, generateFeatureImageCount, generateTransformationCount)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            cv2.imwrite(os.path.join(folderToSaveGeneratedDefectImages, 'Image_' + str(iGenerate) + '.png'), image)

        for iGenerate in xrange(0, generateCount):
            image = GenerateNonDefectImage(bgndFiles, 3, generateTransformationCount)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            cv2.imwrite(os.path.join(folderToSaveGeneratedNonDefectImages, 'Image_' + str(iGenerate) + '.png'), image)

    if (ExecutionMode.TRANSFORM):
        generateCount = 500
        transformationCount = 3
        folderToDefectImages = os.path.join(basePath, 'Defect')
        folderToBgndImages = os.path.join(basePath, 'NonDefect')
        folderToSaveGeneratedDefectImages = os.path.join(basePath, 'GeneratedDefect')
        folderToSaveGeneratedNonDefectImages = os.path.join(basePath, 'GeneratedNonDefect')
        defectImages = LoadImagesFromFolder(folderToDefectImages, '.png')
        nonDefectImages = LoadImagesFromFolder(folderToBgndImages, '.png')

        defectFileIndices = np.linspace(0, len(defectImages) - 1, len(defectImages)).astype('uint8')
        nonDefectFileIndices = np.linspace(0, len(nonDefectImages) - 1, len(nonDefectImages)).astype('uint8')

        # Select feature images
        while (generateCount > 0):
            # Pick random index
            defectImageIndex = np.random.randint(0, len(defectFileIndices))
            nonDefectImageIndex = np.random.randint(0, len(nonDefectFileIndices))
            defectImage = cv2.imread(defectImages[defectImageIndex], cv2.IMREAD_COLOR)
            nonDefectImage = cv2.imread(nonDefectImages[nonDefectImageIndex], cv2.IMREAD_COLOR)

            for iTrafo in xrange(transformationCount):
                trafoTypeA = np.random.randint(0, 5)
                trafoTypeB = np.random.randint(0, 5)
                defectImage = TransformImage(defectImage, Transformation(trafoTypeA))
                nonDefectImage = TransformImage(nonDefectImage, Transformation(trafoTypeB))

            defectImage = cv2.cvtColor(defectImage, cv2.COLOR_BGRA2GRAY)
            cv2.imwrite(os.path.join(folderToSaveGeneratedDefectImages, 'Image_' + str(generateCount) + '.png'), defectImage)
            nonDefectImage = cv2.cvtColor(nonDefectImage, cv2.COLOR_BGRA2GRAY)
            cv2.imwrite(os.path.join(folderToSaveGeneratedNonDefectImages, 'Image_' + str(generateCount) + '.png'), nonDefectImage)

            generateCount = generateCount - 1




    if (ExecutionMode.TESTTRANSFORM):
        imageFolder = os.path.join(basePath, 'NonDefect')
        imageFiles = LoadImagesFromFolder(imageFolder, '.png')

        for iCount in xrange(0, 6):
            imageIndex = np.random.randint(0, len(imageFiles))
            #trafoType = np.random.randint(0, 6)
            TestTransformation(imageFiles[imageIndex], iCount) #trafoType
