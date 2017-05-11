require 'torch'
require 'nn'
require 'image'

useCUDA = 1

if useCUDA then
	require 'cutorch'
	require 'cunn'
end

netName = 'defectAndNonDefectLarge'
netImageSize = '64'

classes = {'defect', 'nonDefect'}

mean = 0.50032
stdv = 0.05167

print('Loading neural network...')
netFileName = netName .. netImageSize .. '.net'
net = torch.load(netFileName, 'ascii')
if useCUDA then
	net = net:cuda()
end
print('Loaded network.')

print('Loading image...')
-- Load test image 
testImage = image.load('test1.png', 1, 'float')
testImage = image.scale(testImage, netImageSize, netImageSize, 'bilinear')

testImage:add(-mean)
testImage:div(stdv)

if useCUDA then
	testImage = testImage:cuda()
end
print('Loaded image.')

print('Classifying...')
timer = torch.Timer()
result = net:forward(testImage)
val, idx = torch.max(result, 1)
print(result)
print(classes)
print('Predicted result: ' .. classes[idx[1]] .. ' in ' .. timer:time().real .. ' seconds')


print('\nAll done!')