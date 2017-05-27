import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from enum import Enum
import math
from functions import GetFilesFromFolder
from functions import TransformImage
from functions import Transformation
from functions import ExtractImageFeaturesCV
from functions import GenerateDefectImage
from functions import GenerateNonDefectImage

def TestTransformation(imagePath, transformation):
    image = cv2.imread(imagePath)
    cv2.imshow('RawImage', image)
    transformedImage = TransformImage(image, Transformation(transformation))

    cv2.imshow(str(Transformation(transformation)), transformedImage)
    cv2.waitKey(0)

class ExecutionMode(Enum):
    EXTRACT = 0,
    GENERATE = 1,
    TRANSFORM = 2,
    TESTEXTRACT = 3,
    TESTTRANSFORM = 4,

if __name__ == '__main__':
    mode = ExecutionMode.GENERATE
    basePath = r'C:\Users\Selim\Documents\GitHub\Files\3MSet_Large\\'

    if (mode == ExecutionMode.EXTRACT):
        folderToExtractImages = os.path.join(basePath, 'Defect')
        folderToSaveFeatures = os.path.join(basePath, 'Features')
        if not os.path.exists(folderToSaveFeatures):
            os.makedirs(folderToSaveFeatures)
        imageFiles = GetFilesFromFolder(folderToExtractImages, '.png')

        for image in imageFiles:
            ExtractImageFeaturesCV(image, 0, 0, folderToSaveFeatures)

    if (mode == ExecutionMode.GENERATE):
        generateCount = 50
        folderToFeatures = os.path.join(basePath, 'Features')
        folderToBgndImages = os.path.join(basePath, 'NonDefect')
        folderToSaveGeneratedDefectImages = os.path.join(basePath, 'GeneratedDefect')
        if not os.path.exists(folderToSaveGeneratedDefectImages):
            os.makedirs(folderToSaveGeneratedDefectImages)
        folderToSaveGeneratedNonDefectImages = os.path.join(basePath, 'GeneratedNonDefect')
        if not os.path.exists(folderToSaveGeneratedNonDefectImages):
            os.makedirs(folderToSaveGeneratedNonDefectImages)
        featureFiles = GetFilesFromFolder(folderToFeatures, '.png')
        bgndFiles = GetFilesFromFolder(folderToBgndImages, '.png')

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

    if (mode == ExecutionMode.TRANSFORM):
        generateCount = 10
        transformationCount = 3
        folderToDefectImages = os.path.join(basePath, 'Defect')
        folderToBgndImages = os.path.join(basePath, 'NonDefect')
        folderToSaveGeneratedDefectImages = os.path.join(basePath, 'GeneratedDefect')
        folderToSaveGeneratedNonDefectImages = os.path.join(basePath, 'GeneratedNonDefect')
        defectImages = GetFilesFromFolder(folderToDefectImages, '.png')
        nonDefectImages = GetFilesFromFolder(folderToBgndImages, '.png')

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

    if (mode == ExecutionMode.TESTEXTRACT):
        folderToExtractImages = os.path.join(basePath, 'Defect')
        folderToSaveFeatures = os.path.join(basePath, 'FeaturesTest')
        if not os.path.exists(folderToSaveFeatures):
            os.makedirs(folderToSaveFeatures)

        # TODO Fix extraction for col_6
        imagePath = os.path.join(folderToExtractImages, 'cell_3M_TM1_Middle_Test01_Row_15_Col_8.png')  # col_8

        ExtractImageFeaturesCV(imagePath, 0, 0, folderToSaveFeatures)


    if (mode == ExecutionMode.TESTTRANSFORM):
        imageFolder = os.path.join(basePath, 'NonDefect')
        imageFiles = GetFilesFromFolder(imageFolder, '.png')

        for iCount in xrange(6, 7):
            imageIndex = np.random.randint(0, len(imageFiles))
            #trafoType = np.random.randint(0, 6)
            TestTransformation(imageFiles[imageIndex], iCount) #trafoType