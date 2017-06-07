require 'torch'
require 'nn'
nninit = require 'nninit'

useCUDA = 1

if useCUDA then
  require 'cutorch'
  require 'cunn'
end

setName = 'defectAndNonDefectAllAug10k'
setImageSize = '64'

print('Loading training data...')
trainSetFileName = setName .. setImageSize .. '-train.t7'
trainSet = torch.load(trainSetFileName, 'ascii')
classes = {'defect', 'nonDefect'}

setmetatable(trainSet,
{ __index = function(t, i) 
        return {
                t.data[i],
                t.labels[i]
            } end });
function trainSet:size()
    return self.data:size(1)
end
trainSet.data = trainSet.data:double()

print('Loaded training data. (' .. trainSetFileName .. ' with ' .. trainSet.data:size(1) .. ' images)')

mean, stdv = {}, {}
for i = 1, trainSet.data:size(2) do --for each channel
    mean[i] = trainSet.data:select(2, i):mean()
    print('Channel ' .. i .. ' Mean: ' .. mean[i])
    trainSet.data:select(2, i):add(-mean[i]) -- subtract the mean
    
    stdv[i] = trainSet.data:select(2, i):std()
    print('Channel ' .. i .. ' Stddev: ' .. stdv[i])
    trainSet.data:select(2, i):div(stdv[i]) -- stddev scaling
end

if useCUDA then
  trainSet.data = trainSet.data:cuda()
end

print('Using ' .. trainSet.data:size(3) .. 'x' .. trainSet.data:size(4) .. ' images.')

net = nn.Sequential()

--input 1x64x64
net:add(nn.SpatialConvolution(trainSet.data:size(2), 32, 3, 3, 1, 1, 1, 1)) -- :init('weight', nninit.normal, 0, 0.1)
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

net:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1)) -- :init('weight', nninit.normal, 0, 0.1)
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- kWxkH regions by step size dWxdH

net:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1)) -- :init('weight', nninit.normal, 0, 0.1)
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

imageSize = tonumber(setImageSize)
outMul = imageSize / 8

net:add(nn.View(32*outMul*outMul))

net:add(nn.Linear(32*outMul*outMul, 4096)) --:init('weight', nninit.kaiming, { dist = 'uniform', gain = {'relu'}})
net:add(nn.ReLU())
net:add(nn.Dropout(0.5))
net:add(nn.Linear(4096, 4096)) -- :init('weight', nninit.kaiming, { dist = 'uniform', gain = {'relu'}})
net:add(nn.ReLU())
net:add(nn.Dropout(0.5))
net:add(nn.Linear(4096, 4096)) -- :init('weight', nninit.kaiming, { dist = 'uniform', gain = {'relu'}})
net:add(nn.ReLU())
net:add(nn.Dropout(0.5))

net:add(nn.Linear(4096, #classes)) -- :init('weight', nninit.sparse, 0.1)
net:add(nn.LogSoftMax())

if useCUDA then
  net = net:cuda()
end

--print(net)

print('Created neural network.')

-- Define loss function
criterion = nn.CrossEntropyCriterion()

if useCUDA then
  criterion = criterion:cuda()
end

-- Train the NN
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.learningRateDecay = 0.0000195
trainer.maxIteration = 50

errorRates = {}
learningRates = {}

trainingTimer = torch.Timer()

-- START TRAINING 
local iteration = 1
local currentLearningRate = trainer.learningRate
local module = trainer.module
local criterion = trainer.criterion
local dataset = trainSet

local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
if not trainer.shuffleIndices then
      for t = 1,dataset:size() do
         shuffledIndices[t] = t
      end
end
print(shuffledIndices)
print("# StochasticGradient: training")

while true do
      local currentError = 0
      for t = 1,dataset:size() do
         local example = dataset[shuffledIndices[t]]
         local input = example[1]
         local target = example[2]
         print(#example)
         currentError = currentError + criterion:forward(module:forward(input), target)

         module:updateGradInput(input, criterion:updateGradInput(module.output, target))
         module:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)

         if trainer.hookExample then
            trainer.hookExample(self, example)
         end
      end

      currentError = currentError / dataset:size()

      if trainer.hookIteration then
         trainer.hookIteration(self, iteration, currentError)
      end

      if trainer.verbose then
         print("Iteration: " .. iteration .. "# current error = " .. currentError)
      end
      
      errorRates[iteration] = currentError
      learningRates[iteration] = currentLearningRate

      iteration = iteration + 1
      currentLearningRate = trainer.learningRate/(1+iteration*trainer.learningRateDecay)
      if trainer.maxIteration > 0 and iteration > trainer.maxIteration then
         print("# StochasticGradient: you have reached the maximum number of iterations")
         print("# training error = " .. currentError)
         break
      end
end

print('Training complete in ' .. trainingTimer:time().real .. ' seconds')

-- Prepare the test set
testSet = torch.load(setName .. setImageSize .. '-test.t7', 'ascii')
testSet.data = testSet.data:double()

if useCUDA then
  testSet.data = testSet.data:cuda()
end

print('Loaded test data.')

for i = 1, trainSet.data:size(2) do -- for each color channel
    local channel = testSet.data:select(2, i)
    channel:add(-mean[i])
    channel:div(stdv[i])
    print(string.format('channel %d: mean = %f stdv = %f', i, channel:mean(), channel:std()))
end

-- Evaluate success rate of the training set
correctInTrainingSet = 0

for i=1,trainSet.data:size(1) do
    local groundtruth = trainSet.labels[i]
    local prediction = net:forward(trainSet.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correctInTrainingSet = correctInTrainingSet + 1
    end
end
print(correctInTrainingSet, 100*correctInTrainingSet/trainSet.data:size(1) .. ' % success in training set')

-- Evaluate success rate of the test set
correctInTestSet = 0 

for i=1,testSet.data:size(1) do
    local groundtruth = testSet.labels[i]
    local prediction = net:forward(testSet.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correctInTestSet = correctInTestSet + 1
    end
end
print(correctInTestSet, 100*correctInTestSet/testSet.data:size(1) .. ' % success in test set')

class_performance = {0, 0}
class_count = {0, 0}
for i=1,testSet.data:size(1) do
    local groundtruth = testSet.labels[i]
    local prediction = net:forward(testSet.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    class_count[groundtruth] = class_count[groundtruth] + 1
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end
for i=1,#classes do
    print(classes[i], 100*class_performance[i]/class_count[i] .. ' %')
end
print(class_performance)
print(class_count)

print('Saving net...')
netFileName = setName .. setImageSize .. '.net'
torch.save(netFileName, net, 'ascii')
print('Net saved to : ' .. netFileName .. '\nALL DONE!')