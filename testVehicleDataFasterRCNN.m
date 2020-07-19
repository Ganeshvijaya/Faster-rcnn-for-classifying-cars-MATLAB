% Load vehicle dataset ground truth.
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;
% Display first few rows of the data set.
vehicleDataset(1:4,:)


% Set random seed to ensure example training reproducibility.
rng(0);

% Randomly split data into a training and test set.
shuffledIdx = randperm(height(vehicleDataset));
idx = floor(0.6 * height(vehicleDataset));
trainingData = vehicleDataset(shuffledIdx(1:idx),:);
testData = vehicleDataset(shuffledIdx(idx+1:end),:);

% Read a test image.
I = imread(testData.imageFilename{1});

% Run the detector.
[bboxes,scores] = detect(detectorFasterRCNN,I);

% Annotate detections in the image.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)