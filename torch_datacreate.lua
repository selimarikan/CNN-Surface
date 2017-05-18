require 'torch'
require 'xlua'
require 'image'
require 'nn'
require 'dataset' -- local

local extension = 'png'
local imageSize = 64

-- Load images works, loads a table of tensors
defectImages = LoadImages('G:/Selim/Thesis/Code/3MSet_Large_Augmented/Defect/', extension)
nonDefectImages = LoadImages('G:/Selim/Thesis/Code/3MSet_Large_Augmented/NonDefect/', extension)
print('Images loaded. ' .. #defectImages .. ' defectImages and ' .. #nonDefectImages .. ' nonDefectImages')

-- Create empty label tensors
defectLabels = torch.ones(#defectImages)
nonDefectLabels = torch.ones(#nonDefectImages):add(1)
print('Labels created. ' .. defectLabels:size(1) .. ' defectLabels and ' .. nonDefectLabels:size(1) .. ' nonDefectLabels')

-- Concat images and labels for t7 creation
allImages = TableConcat(defectImages, nonDefectImages)
allLabels = torch.cat(defectLabels, nonDefectLabels, 1)
print('Concat done. ' .. #allImages .. ' allImages and ' .. allLabels:size(1) .. ' allLabels')

-- Scale images to imageSize x imageSize
for i = 1, #allImages do
	allImages[i] = image.scale(allImages[i], imageSize, imageSize, 'bilinear')
end
print('Image resize done.') 

-- Convert table of images to a tensor
imgTensor = TableToTensor(allImages)
labelTensor = torch.Tensor(allLabels)
print('Tensor assignment done.')

-- Split data into training and test
trainData, trainLabels, testData, testLabels = SplitDataset(imgTensor, labelTensor, 0.5)

trainDataExport = {
	data = trainData,
	labels = trainLabels
}

testDataExport = {
	data = testData,
	labels = testLabels
}

print('Saving datasets...')

setName = 'defectAndNonDefectLargeAug' .. tostring(imageSize)
torch.save(setName .. '-train.t7', trainDataExport, 'ascii')
torch.save(setName .. '-test.t7', testDataExport, 'ascii')

print('Datasets saved. \nALL DONE!')