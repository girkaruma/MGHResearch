function testNeuralNet()
    M = 2;
    %initialW1 = ones(2, M);
%     initialW1 = [1, 2; 3, 4];
    %initialW2 = ones(M, 3);
%     initialW2 = [1, 2, 3; 4, 5, 6];
    %hiddenNodes = [2 3 4 5 6 7 8 9 10];
    % hiddenNodes = [1 2 5 10 25 50 100];
    hiddenNodes = [10]
     errorMatrix = zeros(1, length(hiddenNodes));
     errorTraining = zeros(1, length(hiddenNodes));
%    hiddenNodes = [25];
    bestError = 1;
    bestw1 = [];
    bestw2 = [];
    bestM = 0;
    dataTrain = importdata(strcat('toy_multiclass_2_train.csv'));
    %Xtrain = dataTrain(:,1:784);
    Xtrain = dataTrain(:,1:2);
    [N,D] = size(Xtrain);
    %Ytrain = dataTrain(:,785);
    Ytrain = dataTrain(:,3);
    dataValidate = importdata(strcat('toy_multiclass_2_validate.csv'));
    %Xvalidate = dataValidate(:,1:784);
    %Yvalidate = dataValidate(:,785);
     Xvalidate = dataValidate(:,1:2);
     Yvalidate = dataValidate(:,3);
    K = 3;
    %randomness_threshold = 1000;
    for i=1:length(hiddenNodes)
        initialW1 = 2*randi([0 1],D,hiddenNodes(i))-ones(D,hiddenNodes(i));
        initialW2 = 2*randi([0 1],hiddenNodes(i),K)-ones(hiddenNodes(i),K);
        
%         initialW1 = randi([-randomness_threshold randomness_threshold],D,hiddenNodes(i));
%         initialW2 = randi([-randomness_threshold randomness_threshold],hiddenNodes(i),K);
        %initialW1 = rand(D,hiddenNodes(i));
        %initialW2 = rand(hiddenNodes(i),K); % UNCOMMENT LATER
%         initialW1 = [1, 2; 3, 4];
%         initialW2 = [1, 2, 3; 4, 5, 6];
        i
        [w1, w2, error] = gradientDescentNoLoops(Xtrain, Ytrain, initialW1, initialW2, 0.01, 0.001*hiddenNodes(i)*K, 0, hiddenNodes(i),K);
        error
        %[w1, w2, error] = stochasticGradientDescentMatrix(Xtrain, Ytrain, initialW1, initialW2, 0.01, 0.001, 0, hiddenNodes(i));
        %[w1, w2, e] = gradientDescent(Xtrain, Ytrain, initialW1, initialW2, 0.01, 0.001, 0, hiddenNodes(i));
        errorTraining(1, i) = getClassificationError(Xtrain, Ytrain, max(Ytrain), w1, w2, hiddenNodes(i));
        error = getClassificationError(Xvalidate, Yvalidate, max(Yvalidate), w1, w2, hiddenNodes(i));
        errorMatrix(1, i) = error;
        if error < bestError
            bestError = error;
            bestw1 = w1;
            bestw2 = w2;
            bestM = hiddenNodes(i);
        end
    end
     data = importdata(strcat('toy_multiclass_2_test.csv'));
     X = data(:,1:2);
     Y = data(:,3);
     error = getClassificationError(X, Y, max(Y), bestw1, bestw2, bestM)
    bestM
    hiddenNodes
    errorMatrix
    scatter(hiddenNodes, errorTraining, 100, 'b')
    hold on
    scatter(hiddenNodes, errorMatrix, 100, 'r')
   % stochasticgradientdescent(initialW1, initialW2, 0.1, 0.001, 0, M);
end
function classificationError = getClassificationError(X, Y, K, w1, w2, M)
    misclassifiedPoints = 0;
    for i=1:length(Y)
        highestK = 0;
        highestPredictionValue = 0;
        xPoint = X(i, :);
        for j=1:K
            if h(xPoint, w1, w2, j, M) > highestPredictionValue
                highestK = j;
                highestPredictionValue = h(xPoint, w1, w2, j, M);
            end
        end
        if Y(i) ~= highestK
            misclassifiedPoints = misclassifiedPoints + 1;
        end
    end
    classificationError = misclassifiedPoints/length(Y);
end
function prediction = h(x, w1, w2, k, M)
    outerSum = 0;
    D = length(x);
    for j=1:M
        innerSum = 0;
        for i = 1:D 
            innerSum = innerSum + w1(i,j)*x(i);
        end
        outerSum = outerSum + sigmoid(innerSum) * w2(j,k);
    end 
    prediction = sigmoid(outerSum);
end
function sig = sigmoid(z)
    sig = 1/(1 + exp(0 - z));
end

