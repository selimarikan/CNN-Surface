require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'

useCUDA = 1

if useCUDA then
  require 'cutorch'
  require 'cunn'
end

basePath = ''
setName = 'defectAndNonDefectMedium'

model = loadcaffe.load('VGG_CNN_M_128_deploy.prototxt', 'VGG_CNN_M_128.caffemodel')

trainingSetPath = basePath .. setName .. '-train.t7'

trainSet = torch.load(trainingSetPath)
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

requiredW, requiredH = 224, 224
print('trainSet size' .. tostring(#trainSet.data))

local tL = torch.Tensor(trainSet.data:size(1), 3, requiredW, requiredH)
print('tL size' .. tostring(#tL))

for iItem = 1, ((#tL)[1]) do
    local ta = trainSet.data[iItem]
    local t1 = torch.Tensor(3, (#ta)[2], (#ta)[3])
    t1[1] = ta[1]:clone()
    t1[2] = t1[1]:clone()
    t1[3] = t1[1]:clone()
    
    tL[iItem] = image.scale(t1, requiredW, requiredH)
end
trainSet.data = tL
print(#trainSet.data)

mean, stdv = {}, {}
for i = 1, trainSet.data:size(2) do --for each channel
    mean[i] = trainSet.data:select(2, i):mean()
    print('Channel ' .. i .. ' Mean: ' .. mean[i])
    trainSet.data:select(2, 1):add(-mean[i]) -- subtract the mean
    
    stdv[i] = trainSet.data:select(2, i):std()
    print('Channel ' .. i .. ' Stddev: ' .. stdv[i])
    trainSet.data:select(2, i):div(stdv[i]) -- stddev scaling
end

if useCUDA then
  trainSet.data = trainSet.data:cuda()
end

model:remove()
model:remove()

model:add(nn.Linear(128, 2))
model:add(nn.SoftMax())

if useCUDA then
  	model = model:cuda()
end

criterion = nn.CrossEntropyCriterion()

if useCUDA then
	criterion = criterion:cuda()
end

trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.1
trainer.learningRateDecay = 0.05
trainer.maxIteration = 4

errorRates = {}
learningRates = {}

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

correctInTrainingSet = 0
print(trainSet.data:size(1))
for i=1,trainSet.data:size(1) do
    local groundtruth = trainSet.labels[i]
    local prediction = model:forward(trainSet.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correctInTrainingSet = correctInTrainingSet + 1
    end
end
print(correctInTrainingSet, 100*correctInTrainingSet/trainSet.data:size(1) .. ' % success in training set')