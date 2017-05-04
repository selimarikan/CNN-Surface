require 'torch'
require 'xlua'
require 'image'
require 'nn'
require 'dataset' -- local

local extension = 'png'

-- Load images works, loads a table of tensors
defectImages = LoadImages('DefectMidSet/Defect/', extension)
nonDefectImages = LoadImages('DefectMidSet/NonDefect/', extension)
print('Images loaded.')

-- Create empty label tensors
defectLabels = torch.ones(#defectImages)
nonDefectLabels = torch.ones(#nonDefectImages):add(1)

-- Concat images and labels for t7 creation
allImages = TableConcat(defectImages, nonDefectImages)
allLabels = torch.cat(defectLabels, nonDefectLabels, 1)
print('Concat done.')

-- Scale images down to 256x256
for i = 1, #allImages do
	allImages[i] = image.scale(allImages[i], 256, 256, 'bilinear')
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

setName = 'defectAndNonDefectSmall'
torch.save(setName .. '-train.t7', trainDataExport, 'ascii')
torch.save(setName .. '-test.t7', testDataExport, 'ascii')

print('Datasets saved. \nALL DONE!')