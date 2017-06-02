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
    TESTGENERATE = 4,
    TESTTRANSFORM = 5,

if __name__ == '__main__':
    mode = ExecutionMode.TESTGENERATE
    basePath = r'C:\Users\Selim\Documents\GitHub\Files\3MSet_Large\\'

    if (mode == ExecutionMode.EXTRACT):
        folderToExtractImages = os.path.join(basePath, 'Defect')
        folderToExtractNonFeatures = os.path.join(basePath, 'NonDefect')
        folderToSaveFeatures = os.path.join(basePath, 'Features')
        folderToSaveNonFeatures = os.path.join(basePath, 'NonFeatures')
        if not os.path.exists(folderToSaveFeatures):
            os.makedirs(folderToSaveFeatures)
        if not os.path.exists(folderToSaveNonFeatures):
            os.makedirs(folderToSaveNonFeatures)
        imageFiles = GetFilesFromFolder(folderToExtractImages, '.png')
        bgndFiles = GetFilesFromFolder(folderToExtractNonFeatures, '.png')

        for image in imageFiles:
            ExtractImageFeaturesCV(image, folderToSaveFeatures, isDefect=True)

        for image in bgndFiles:
            ExtractImageFeaturesCV(image, folderToSaveNonFeatures, isDefect=False)

    if (mode == ExecutionMode.GENERATE):
        generateDefectCount = 500
        generateNonDefectCount = 2000
        folderToFeatures = os.path.join(basePath, 'Features')
        folderToNonFeatures = os.path.join(basePath, 'NonFeatures')
        folderToBgndImages = os.path.join(basePath, 'NonDefect')
        folderToSaveGeneratedDefectImages = os.path.join(basePath, 'GeneratedDefect')
        if not os.path.exists(folderToSaveGeneratedDefectImages):
            os.makedirs(folderToSaveGeneratedDefectImages)
        folderToSaveGeneratedNonDefectImages = os.path.join(basePath, 'GeneratedNonDefect')
        if not os.path.exists(folderToSaveGeneratedNonDefectImages):
            os.makedirs(folderToSaveGeneratedNonDefectImages)
        featureFiles = GetFilesFromFolder(folderToFeatures, '.png')
        nonFeatureFiles = GetFilesFromFolder(folderToNonFeatures, '.png')
        bgndFiles = GetFilesFromFolder(folderToBgndImages, '.png')

        # Parameters
        generateBgndImageCount = 1
        generateFeatureImageCount = 1
        generateTransformationCount = 2

        for iGenerate in xrange(0, generateDefectCount):
            image = GenerateDefectImage(featureFiles, bgndFiles, generateBgndImageCount, generateFeatureImageCount, generateTransformationCount, isDefect=True)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            cv2.imwrite(os.path.join(folderToSaveGeneratedDefectImages, 'Image_' + str(iGenerate) + '.png'), image)

        for iGenerate in xrange(0, generateNonDefectCount):
            image = GenerateDefectImage(nonFeatureFiles, bgndFiles, generateBgndImageCount, generateFeatureImageCount, generateTransformationCount, isDefect=False)
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
        folderToExtractFeatures = os.path.join(basePath, 'DefectTest')
        folderToExtractNonFeatures = os.path.join(basePath, 'NonDefectTest')
        folderToSaveFeatures = os.path.join(basePath, 'FeaturesTest')
        folderToSaveNonFeatures = os.path.join(basePath, 'NonFeaturesTest')
        if not os.path.exists(folderToSaveFeatures):
            os.makedirs(folderToSaveFeatures)
        if not os.path.exists(folderToSaveNonFeatures):
            os.makedirs(folderToSaveNonFeatures)
        featureImagePath = os.path.join(folderToExtractFeatures, 'cell_3M_TM1_Middle_Test01_Row_15_Col_8.png')  # col_8
        nonFeatureImagePath = os.path.join(folderToExtractNonFeatures, 'cell_3M_TM1_Middle_Test01_Row_92_Col_5.png')
        #ExtractImageFeaturesCV(featureImagePath, 0, 0, folderToSaveFeatures, isDefect=True)
        ExtractImageFeaturesCV(nonFeatureImagePath, 0, 0, folderToSaveNonFeatures, isDefect=False)

    if (mode == ExecutionMode.TESTGENERATE):
        folderToFeatures = os.path.join(basePath, 'Features')
        folderToNonFeatures = os.path.join(basePath, 'NonFeatures')
        folderToBgndImages = os.path.join(basePath, 'NonDefect')
        folderToSaveTestDefectImages = os.path.join(basePath, 'TestDefect')
        folderToSaveTestNonDefectImages = os.path.join(basePath, 'TestNonDefect')
        if not os.path.exists(folderToSaveTestDefectImages):
            os.makedirs(folderToSaveTestDefectImages)
        if not os.path.exists(folderToSaveTestNonDefectImages):
            os.makedirs(folderToSaveTestNonDefectImages)
        featureFiles = GetFilesFromFolder(folderToFeatures, '.png')
        nonFeatureFiles = GetFilesFromFolder(folderToNonFeatures, '.png')
        bgndFiles = GetFilesFromFolder(folderToBgndImages, '.png')

        # Parameters
        generateBgndImageCount = 1
        generateFeatureImageCount = 1
        generateTransformationCount = 2

        imageDefect = GenerateDefectImage(featureFiles, bgndFiles, generateBgndImageCount, generateFeatureImageCount, generateTransformationCount)
        imageDefect = cv2.cvtColor(imageDefect, cv2.COLOR_BGRA2GRAY)
        cv2.imwrite(os.path.join(folderToSaveTestDefectImages, 'Image_.png'), imageDefect)

    if (mode == ExecutionMode.TESTTRANSFORM):
        imageFolder = os.path.join(basePath, 'NonDefect')
        imageFiles = GetFilesFromFolder(imageFolder, '.png')

        for iCount in xrange(6, 7):
            imageIndex = np.random.randint(0, len(imageFiles))
            #trafoType = np.random.randint(0, 6)
            TestTransformation(imageFiles[imageIndex], iCount) #trafoType
