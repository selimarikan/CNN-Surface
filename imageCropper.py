import skimage
import skimage.io
import os
import math

imageRootDir = 'g:/Selim/Thesis/Defect-Free/Stream'
overlapPercentage = 5

def GridCropImage(imagePath, outName, outDir, cellHeight, cellWidth):
    #img = Image.open(imagePath)
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
            fileName = os.path.join(outDir, 'cell_' + outName + '_Row_' + str(iCellRow) + '_Col_' + str(iCellCol) + '.bmp')
            skimage.io.imsave(fileName, imageCrop, 'pil')

if __name__ == '__main__':
    GridCropImage(imageRootDir + '/4096_3M_TM1_Left_Test01_LeftCleaned.bmp', '3M_TM1_Left_Test01', imageRootDir + '/3MTM1Export4/', 256, 256)