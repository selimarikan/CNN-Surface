require 'torch'
require 'nn'
require 'xlua'
require 'image'
require 'os' -- platform specific, for COPY operation
require 'dataset' -- local

useCUDA = 1

if useCUDA then
	require 'cutorch'
	require 'cunn'
end

local netName = 'defectAndNonDefectLarge'
local netImageSize = '64'

local classes = {'defect', 'nonDefect'}

local baseImagePath = 'g:\\Selim\\Thesis\\Code\\CNN-Surface\\UnlabeledImages'
local outputPath = 'g:\\Selim\\Thesis\\Code\\CNN-Surface\\LabeledImages'
local imageExtension = 'png'

-- TODO: Load dataset identity file or identity image?
mean = 0.50032
stdv = 0.05167

print('Loading neural network...')
netFileName = netName .. netImageSize .. '.net'
net = torch.load(netFileName, 'ascii')
if useCUDA then
	net = net:cuda()
end
print('Loaded network.')

print('Loading images...')
-- Load
local unlabeledImagePaths = GetImagesInDirectory(baseImagePath, imageExtension)
local unlabeledImages = LoadImages(baseImagePath, imageExtension)

-- Resize
for iImage = 1, #unlabeledImages do
	unlabeledImages[iImage] = image.scale(unlabeledImages[iImage], netImageSize, netImageSize, 'bilinear')
end

-- Table -> tensor
unlabeledImagesTensor = TableToTensor(unlabeledImages)
print(unlabeledImagesTensor[1][1][1][1])
unlabeledImagesTensor:add(-mean)
unlabeledImagesTensor:div(stdv)
print(unlabeledImagesTensor[1][1][1][1])

if useCUDA then
	unlabeledImagesTensor = unlabeledImagesTensor:cuda()
end
print('Loaded images.')

mainTimer = torch.Timer()

print('Classifying...')
for iImage = 1, #unlabeledImages do
	local timer = torch.Timer()
	local result = net:forward(unlabeledImagesTensor[iImage])
	local val, idx = torch.max(result, 1)
	print('[' .. iImage .. '/' .. #unlabeledImages .. '] Predicted result: ' .. classes[idx[1]] .. ' in ' .. timer:time().real .. ' seconds')

	local command = 'xcopy /s/y ' .. baseImagePath .. '\\' .. GetFileName(unlabeledImagePaths[iImage]) .. ' ' .. outputPath .. '\\' .. classes[idx[1]] .. '\\'
	print(command)
	os.execute(command)
end

print('Finished ' .. #unlabeledImages .. ' images in ' .. mainTimer:time().real .. ' seconds')
print('\nAll done!')