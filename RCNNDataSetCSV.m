load ('pedestrianArrayAll.mat')
pedestrianArrayAll

filename = fullfile('PedestrainCrossing.csv');
pedestrian = readtable(filename,'ReadRowNames',false)


pedestrian(1:5,1:4)
pedestrian.imageFilename; 
numberSignCount = size(pedestrian.imageFilename)
SignCount = numberSignCount(1)

pedestrianArray = [1 2 3 4] 

for k=1:SignCount
    %[1×4 double]
    pedestrianArray (1,1)=  pedestrian.Var2(k);
    pedestrianArray (1,2)=  pedestrian.Var3(k);
    pedestrianArray (1,3)=  pedestrian.Var6(k);
    pedestrianArray (1,4)=  pedestrian.Var7(k);
    imageFilename{k} = pedestrian.imageFilename{k};
    pedestrianArray
    pedestrian1{k} = pedestrianArray;
    
    
end

imageFilename1 = imageFilename(:);
pedestrian2=pedestrian1(:);

pedestrianArrayAll = table(imageFilename1,pedestrian2)



pedestrianArrayAll(1:4,:)
save pedestrianArrayAll
%imageFilename               pedestrian 