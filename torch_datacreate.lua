require 'torch'
require 'xlua'
require 'image'
require 'nn'
require 'dataset' -- local

local basePath = 'G:/Selim/Thesis/Code/3MSet_All/'
local extension = 'png'
local imageSize = 64

-- Load images works, loads a table of tensors
defectImages = LoadImages(basePath .. 'Defect/', extension)
augDefectImages = LoadImages(basePath .. 'DefectAug/', extension)
nonDefectImages = LoadImages(basePath .. 'NonDefect/', extension)
augNonDefectImages = LoadImages(basePath .. 'NonDefectAug/', extension)
print('Images loaded. ' .. #defectImages .. ' defectImages, ' .. #augDefectImages .. ' augDefectImages, ' .. #nonDefectImages .. ' nonDefectImages, ' .. #augNonDefectImages .. ' augNonDefectImages')

-- Create empty label tensors
defectLabels = torch.ones(#defectImages)
augDefectLabels = torch.ones(#augDefectImages)
nonDefectLabels = torch.ones(#nonDefectImages):add(1)
augNonDefectLabels = torch.ones(#augNonDefectImages):add(1)
print('Labels created. ' .. defectLabels:size(1) .. ' defectLabels, ' .. augDefectLabels:size(1) .. ' augDefectLabels' .. nonDefectLabels:size(1) .. ' nonDefectLabels, ' .. augNonDefectLabels:size(1))

-- Concat images and labels for t7 creation
allImages = TableConcat(defectImages, nonDefectImages)
augAllImages = TableConcat(augDefectImages, augNonDefectImages)
allLabels = torch.cat(defectLabels, nonDefectLabels, 1)
augAllLabels = torch.cat(augDefectLabels, augNonDefectLabels, 1)
print('Concat done. ' .. #allImages .. ' allImages, ' .. #augAllImages .. ' augAllImages, ' .. allLabels:size(1) .. ' allLabels, ' .. augAllLabels:size(1) .. ' augAllLabels')

-- Scale images to imageSize x imageSize
for i = 1, #allImages do
	allImages[i] = image.scale(allImages[i], imageSize, imageSize, 'bilinear')
end
for i = 1, #augAllImages do
	augAllImages[i] = image.scale(augAllImages[i], imageSize, imageSize, 'bilinear')
end
print('Image resize done.') 

-- Convert table of images to a tensor
imgTensor = TableToTensor(allImages)
augImgTensor = TableToTensor(augAllImages)
labelTensor = torch.Tensor(allLabels)
augLabelTensor = torch.Tensor(augAllLabels)
print('Tensor assignment done.')

-- Split data into training and test
trainData, trainLabels, testData, testLabels = SplitDataset(imgTensor, labelTensor, 0.5)
trainData = torch.cat(trainData, augImgTensor, 1)
trainLabels = torch.cat(trainLabels, augLabelTensor, 1)

trainDataExport = {
	data = trainData,
	labels = trainLabels
}

testDataExport = {
	data = testData,
	labels = testLabels
}

print('Created dataset with ' .. trainData:size(1) .. ' trainingData and ' .. testData:size(1) .. ' testData')
print('Saving datasets...')

setName = 'defectAndNonDefectAllAug' .. tostring(imageSize)
torch.save(setName .. '-train.t7', trainDataExport, 'ascii')
torch.save(setName .. '-test.t7', testDataExport, 'ascii')

print('Datasets saved: ' .. setName .. ' \nALL DONE!')