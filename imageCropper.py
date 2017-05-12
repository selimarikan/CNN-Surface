import skimage
import skimage.io
import os
import math

imageRootDir = "G:\Selim\Thesis\Code\SmashFilmIndonesia\BF"
overlapPercentage = 5

def GridCropImage(imagePath, outName, outDir, cellHeight, cellWidth):
    #img = Image.open(imagePath)
    saveExtension = '.png'
    img = skimage.io.imread(imagePath, True, 'pil')

    (imageHeight, imageWidth) = img.shape
    print('ImageWidth: ' + str(imageWidth))
    print('ImageHeight: ' + str(imageHeight))

    # Use floor, skip the right remaining pixels
    gridColCount = math.floor(imageWidth / (cellWidth * (100 - overlapPercentage) / 100))
    gridRowCount = math.floor(imageHeight / (cellHeight * (100 - overlapPercentage) / 100))

    print('ColCount: ' + str(gridColCount))
    print('RowCount: ' + str(gridRowCount))

    for iCellRow in list(range(gridRowCount)):
        for iCellCol in list(range(gridColCount)):
            print(str((((iCellRow * gridColCount) + iCellCol) / (gridColCount * gridRowCount)) * 100) + '%')

            colFrom = math.floor(iCellCol * (cellWidth * (100 - overlapPercentage) / 100))
            colTo = colFrom + cellWidth
            rowFrom = math.floor(iCellRow * (cellHeight * (100 - overlapPercentage) / 100))
            rowTo = rowFrom + cellHeight

            imageCrop = img[rowFrom:rowTo, colFrom:colTo]
            print('Image index (col, row): ' + str(iCellCol) + ':' + str(iCellRow))
            print('ColFrom-To: ' + str(colFrom) + ':' + str(colTo))
            print('RowFrom-To: ' + str(rowFrom) + ':' + str(rowTo))
            print(imageCrop.shape)
            fileName = os.path.join(outDir, 'cell_' + str(outName) + '_Row_' + str(iCellRow) + '_Col_' + str(iCellCol) + saveExtension)
            skimage.io.imsave(fileName, imageCrop, 'pil')

if __name__ == '__main__':
    index = 0
    for file in os.listdir(imageRootDir):
        if file.endswith('.bmp'):
            filePath = os.path.join(imageRootDir, file)
            dirPath = os.path.join(imageRootDir, 'image_'+ str(index))

            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            print(filePath)
            GridCropImage(filePath, index, dirPath, 256, 256)
            index = index + 1