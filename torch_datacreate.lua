require 'torch'
require 'xlua'
require 'image'
require 'nn'
require 'dataset' -- local

local basePath = 'G:/Selim/Thesis/Code/3MSet_All/'
local extension = 'png'
local imageSize = 64
local setName = 'defectAndNonDefectAllAug10k' .. tostring(imageSize)
local augmentation = 1

-- Load images works, loads a table of tensors
defectImages = LoadImages(basePath .. 'Defect/', extension)
if augmentation then
	augDefectImages = LoadImages(basePath .. 'GeneratedDefect/', extension)
end
nonDefectImages = LoadImages(basePath .. 'NonDefect/', extension)
if augmentation then
	augNonDefectImages = LoadImages(basePath .. 'GeneratedNonDefect/', extension)
end
print('Images loaded. ') -- .. #defectImages .. ' defectImages, ' .. #augDefectImages .. ' augDefectImages, ' .. #nonDefectImages .. ' nonDefectImages, ' .. #augNonDefectImages .. ' augNonDefectImages')

-- Create empty label tensors
defectLabels = torch.ones(#defectImages)
if augmentation then
	augDefectLabels = torch.ones(#augDefectImages)
end
nonDefectLabels = torch.ones(#nonDefectImages):add(1)
if augmentation then
	augNonDefectLabels = torch.ones(#augNonDefectImages):add(1)
end
print('Labels created. ') -- .. defectLabels:size(1) .. ' defectLabels, ' .. augDefectLabels:size(1) .. ' augDefectLabels' .. nonDefectLabels:size(1) .. ' nonDefectLabels, ' .. augNonDefectLabels:size(1))

-- Concat images and labels for t7 creation
allImages = TableConcat(defectImages, nonDefectImages)
if augmentation then
	augAllImages = TableConcat(augDefectImages, augNonDefectImages)
end
allLabels = torch.cat(defectLabels, nonDefectLabels, 1)
if augmentation then
	augAllLabels = torch.cat(augDefectLabels, augNonDefectLabels, 1)
end
print('Concat done. ') -- .. #allImages .. ' allImages, ' .. #augAllImages .. ' augAllImages, ' .. allLabels:size(1) .. ' allLabels, ' .. augAllLabels:size(1) .. ' augAllLabels')

-- Scale images to imageSize x imageSize
for i = 1, #allImages do
	allImages[i] = image.scale(allImages[i], imageSize, imageSize, 'bilinear')
end
if augmentation then
	for i = 1, #augAllImages do
		augAllImages[i] = image.scale(augAllImages[i], imageSize, imageSize, 'bilinear')
	end
end
print('Image resize done.') 

-- Convert table of images to a tensor
imgTensor = TableToTensor(allImages)
if augmentation then
	augImgTensor = TableToTensor(augAllImages)
end
labelTensor = torch.Tensor(allLabels)
if augmentation then
	augLabelTensor = torch.Tensor(augAllLabels)
end
print('Tensor assignment done.')

-- Split data into training and test
trainData, trainLabels, testData, testLabels = SplitDataset(imgTensor, labelTensor, 0.5)
if augmentation then
	trainData = torch.cat(trainData, augImgTensor, 1)
	trainLabels = torch.cat(trainLabels, augLabelTensor, 1)
end

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

torch.save(setName .. '-train.t7', trainDataExport, 'ascii')
torch.save(setName .. '-test.t7', testDataExport, 'ascii')

print('Datasets saved: ' .. setName .. ' \nALL DONE!')