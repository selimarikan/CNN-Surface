require 'torch'
require 'nn'

useCUDA = 1

if useCUDA then
  require 'cutorch'
  require 'cunn'
end

setName = 'daisyAndDandelion'

trainSet = torch.load(setName .. '-train.t7', 'ascii')
classes = {'daisy', 'dandelion'}

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

print('Loaded training data.')

-- Data conditioning: 0-mean and 1-stddev
mean = {}
stdv = {}
for i = 1, trainSet.data:size(2) do --for each color channel
    -- Selects column 1 -> Data
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

net = nn.Sequential()

--input 1x64x64
net:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1)) -- nInputPlane, nOutputPlane, kW, kH
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

net:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- kWxkH regions by step size dWxdH

net:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

net:add(nn.View(16*8*8))

net:add(nn.Linear(16*8*8, 512))
net:add(nn.ReLU())
net:add(nn.Linear(512, 512))
net:add(nn.ReLU())
net:add(nn.Linear(512, 512))
net:add(nn.ReLU())

net:add(nn.Linear(512, #classes))
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
trainer.learningRate = 0.005
trainer.learningRateDecay = 0.0001
trainer.maxIteration = 25

errorRates = {}
learningRates = {}

timer = torch.Timer()

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

print("# StochasticGradient: training")

while true do
      local currentError = 0
      for t = 1,dataset:size() do
         local example = dataset[shuffledIndices[t]]
         local input = example[1]
         local target = example[2]

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

print('Training complete.')

print('Training completed in: ' .. timer:time().real .. ' seconds')

-- Prepare the test set
testSet = torch.load(setName .. '-test.t7', 'ascii')
testSet.data = testSet.data:double()

if useCUDA then
  testSet.data = testSet.data:cuda()
end

print('Loaded test data.')

for i = 1, 3 do -- for each color channel
    local channel = testSet.data:select(2, i)
    channel:add(-mean[i])
    channel:div(stdv[i])
    print(string.format('channel %d: mean = %f stdv = %f', i, channel:mean(), channel:std()))
end

timer = torch.Timer()

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

print('Test of training set completed in: ' .. timer:time().real .. ' seconds')

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
torch.save(setName .. '.net', net, 'ascii')
print('Net saved. \nALL DONE!')
