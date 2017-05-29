import os
import cv2
import math
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt

def GetFilesFromFolder(folderPath, extension):
    files = []
    for file in os.listdir(folderPath):
        if file.endswith(extension):
            filePath = os.path.join(folderPath, file)
            files.append(filePath)
    return files

def ExtractImageFeaturesCV(imagePath, featureLayersToExtract, kernelSize, featureSavePath, isDefect):
    showResult = False
    rawImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    gaussianImage = cv2.GaussianBlur(rawImage, (51, 51), 0)

    hfFeaturesImage = cv2.subtract(rawImage, gaussianImage)  # Could be one of the feature layers

    if showResult:
        cv2.imshow('HF features', hfFeaturesImage)
    # Feature 1
    f1a = cv2.add(rawImage, hfFeaturesImage)
    _, f1b = cv2.threshold(f1a, 0, 255, cv2.THRESH_OTSU)
    f1b = cv2.bitwise_not(f1b)

    if isDefect:
        f1bKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        f1c = cv2.morphologyEx(f1b, cv2.MORPH_OPEN, f1bKernel)
        f1d = f1c #cv2.bitwise_not(f1c)
    else:
        f1bKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        f1c = cv2.morphologyEx(f1b, cv2.MORPH_OPEN, f1bKernel)
        f1dKernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (117, 117))
        f1d = cv2.morphologyEx(f1c, cv2.MORPH_DILATE, f1dKernel)

    if showResult:
        cv2.imshow('f1b', f1b)
        cv2.imshow('f1c', f1c)
        cv2.imshow('f1d', f1d)

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
    SHIFT = 6

def MergeLayersAlpha(backgroundImage, foregroundImage, shiftX = 0, shiftY = 0):
    generatedImage = backgroundImage
    rows, cols, ch = foregroundImage.shape
    mat = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
    foregroundImage = cv2.warpAffine(foregroundImage, mat, (cols, rows))
    for c in range(0, 3):
        alpha = foregroundImage[:, :, 3] / 255.0
        color = foregroundImage[:, :, c] * alpha
        beta = backgroundImage[:, :, c] * (1.0 - alpha)

        generatedImage[:, :, c] = color + beta
    return generatedImage

def UnsharpMask(image):
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

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

    if (transformation == Transformation.SHIFT):
        shiftX = np.random.randint(-41, 41)
        shiftY = np.random.randint(-41, 41)
        mat = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        return cv2.warpAffine(image, mat, (cols, rows))

def GenerateNonDefectImage(backgroundFiles, backgroundImageCount, transformationCount):
    showResult = False
    # Parameters
    generateBgndImageCount = backgroundImageCount
    generateTransformationCount = transformationCount

    backgroundFileIndices = np.linspace(0, len(backgroundFiles) - 1, len(backgroundFiles)).astype('uint8')

    bgndImagesToProcess = []     # Pick random n background images and fuse them

    # Select background images
    while (generateBgndImageCount > 0):
        # Pick random index
        imageIndex = np.random.randint(0, len(backgroundFileIndices))
        bgndImagesToProcess.append(cv2.imread(backgroundFiles[backgroundFileIndices[imageIndex]]))
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

    backgroundImage = UnsharpMask(backgroundImage)

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

    # Hold indices of images to do array operations
    featureFileIndices = np.linspace(0, len(featureFiles) - 1, len(featureFiles)).astype('uint8')
    backgroundFileIndices = np.linspace(0, len(backgroundFiles) - 1, len(backgroundFiles)).astype('uint8')

    bgndImagesToProcess = []     # Pick random n background images and fuse them

    # Select background images
    while (generateBgndImageCount > 0):
        # Pick random index
        imageIndex = np.random.randint(0, len(backgroundFileIndices))
        bgndImagesToProcess.append(cv2.imread(backgroundFiles[backgroundFileIndices[imageIndex]]))
        # Remove used image index so it will not be picked again
        backgroundFileIndices = np.delete(backgroundFileIndices, imageIndex)
        generateBgndImageCount = generateBgndImageCount - 1

    # Fuse background images
    backgroundImage = bgndImagesToProcess[0]
    weight = 1.0 / len(bgndImagesToProcess)
    for iImage in xrange(len(bgndImagesToProcess) - 1):
        backgroundImage = cv2.addWeighted(backgroundImage, weight + (iImage * weight), bgndImagesToProcess[iImage + 1], weight, 0)

    if (showResult):
        f, axarr = plt.subplots(max(generateBgndImageCount, generateFeatureImageCount) + 1, 3)

        # Visualize background images
        for iImage in xrange(len(bgndImagesToProcess)):
            axarr[iImage, 0].imshow(bgndImagesToProcess[iImage])
            axarr[iImage, 0].set_title('Background image ' + str(iImage))
        axarr[len(axarr) - 1, 0].imshow(backgroundImage)
        axarr[len(axarr) - 1, 0].set_title('Generated background image')

    featureImagesToProcess = []     # Pick random m feature images and fuse them

    # Select feature images
    while (generateFeatureImageCount > 0):
        # Pick random index
        imageIndex = np.random.randint(0, len(featureFileIndices))
        featureImagesToProcess.append(cv2.imread(featureFiles[featureFileIndices[imageIndex]], cv2.IMREAD_UNCHANGED))
        # Remove used image so it will not be picked again
        featureFileIndices = np.delete(featureFileIndices, imageIndex)
        generateFeatureImageCount = generateFeatureImageCount - 1

    # Fuse foreground images
    featureImage = featureImagesToProcess[0]
    for iImage in xrange(len(featureImagesToProcess) - 1):
        imageToBeMerged = featureImagesToProcess[iImage + 1]
        rows, cols, ch = imageToBeMerged.shape
        shiftX = np.random.randint(-41, 41)
        shiftY = np.random.randint(-41, 41)
        mat = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        imageToBeMerged = cv2.warpAffine(imageToBeMerged, mat, (cols, rows))
        featureImage = MergeLayersAlpha(featureImage, imageToBeMerged)
        #featureImage = cv2.add(featureImage, imageToBeMerged)  # cv2.addWeighted(featureImage, 1.0, featureImagesToProcess[iImage + 1], 1.0, 0)

    # Try blurring
    featureImage = cv2.GaussianBlur(featureImage, (1, 1), 0)

    backgroundImage = UnsharpMask(backgroundImage)

    if (showResult):
        # Visualize feature images
        for iImage in xrange(len(featureImagesToProcess)):
            axarr[iImage, 1].imshow(featureImagesToProcess[iImage])
            axarr[iImage, 1].set_title('Feature image ' + str(iImage))
        axarr[len(axarr) - 1, 1].imshow(featureImage)
        axarr[len(axarr) - 1, 1].set_title('Generated feature image')

    # Final fusion of background and features
    height, width, chn = backgroundImage.shape

    # Transform featureImage
    # for iTrafo in xrange(generateTransformationCount):
    #     trafoType = np.random.randint(0, 5)
    #     featureImage = TransformImage(featureImage, Transformation(trafoType))

    generatedImage = MergeLayersAlpha(backgroundImage, featureImage)

    for iTrafo in xrange(generateTransformationCount):
        trafoType = np.random.randint(0, 5)
        generatedImage = TransformImage(generatedImage, Transformation(trafoType))

    if (showResult):
        axarr[0, 2].imshow(generatedImage)
        plt.show()

    return generatedImage