
% Load vehicle dataset ground truth.
data = load('vehicleDatasetGroundTruth.mat');
data
vehicleDataset = data.vehicleDataset;
% Display first few rows of the data set.
vehicleDataset(1:4,:)


%Display one of the images from the data set to understand the type of images it contains.
% Add the fullpath to the local vehicle data folder.
vehicleDataset.imageFilename = fullfile(pwd, vehicleDataset.imageFilename);

% Read one of the images.
I = imread(vehicleDataset.imageFilename{10});

% Insert the ROI labels.
I = insertShape(I, 'Rectangle', vehicleDataset.vehicle{10});

% Resize and display image.
I = imresize(I,3);
figure
imshow(I)



% Set random seed to ensure example training reproducibility.
rng(0);

% Randomly split data into a training and test set.
shuffledIdx = randperm(size(vehicleDataset,1));
idx = floor(0.6 * size(vehicleDataset,1));
trainingData = vehicleDataset(shuffledIdx(1:idx),:);
testData = vehicleDataset(shuffledIdx(idx+1:end),:);

%Configure Training Options
% Options for step 1.
options = trainingOptions('sgdm', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);


 % Train Faster R-CNN detector.
    %  * Use 'resnet50' as the feature extraction network. 
    %  * Adjust the NegativeOverlapRange and PositiveOverlapRange to ensure
    %    training samples tightly overlap with ground truth.
[detectorFasterRCNN, info] = trainFasterRCNNObjectDetector(trainingData, 'resnet50', options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1]);

save detectorFasterRCNN


%FROM HERE ON WE TEST THE TRAINED FASTER-RCNN

% Read a test image.
signCount = size(vehicleDataset)
numberSignCount = signCount(1)
FP=0;
TN=0;
load('detectorFasterRCNN')
for k=1:numberSignCount
    I = imread(testData.imageFilename{k});
    
    % Run the detector.
    [bboxes, scores, labels] = detect(detectorFasterRCNN,I);
    
    %calculation of the sensitivity
    [bestScore, best] = max(scores)
    if   bestScore > 0.8
        FP = FP + 1;
        %title(sprintf('%s, score %.3f', 'FP', bestScore))
        %tts(robotVoice);
    else
        TN=TN + 1;
        %title(sprintf('%s, score %.3f', labelStr, bestScore))
        %tts(robotVoice);
        label = 'Negatives';
    end
    
    % Annotate detections in the image.
    I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
    figure
    imshow(I)
end
Sp= TN/(TN+FP)