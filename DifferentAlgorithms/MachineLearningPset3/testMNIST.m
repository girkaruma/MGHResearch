function testMNIST()
    M = 2;
    %initialW1 = ones(2, M);
    initialW1 = [1, 2; 3, 4];
    %initialW2 = ones(M, 3);
    initialW2 = [1, 2, 3; 4, 5, 6];
    hiddenNodes = [1 2 5 10 25 50 100];
    bestError = 1;
    bestw1 = [];
    bestw2 = [];
    bestM = 0;
    dataTrain = importdata(strcat('mnist_train.csv'));
    Xtrain = dataTrain(:,1:784);
    %Xtrain = dataTrain(:,1:2);
    [N,D] = size(Xtrain);
    Ytrain = dataTrain(:,785);
    %Ytrain = dataTrain(:,3);
    dataValidate = importdata(strcat('mnist_validate.csv'));
    Xvalidate = dataValidate(:,1:784);
    Yvalidate = dataValidate(:,785);
    K = 6;
    lambda = 0;
    for i=1:length(hiddenNodes)
        initialW1 = 2*randi([0 1],D,hiddenNodes(i))-ones(D,hiddenNodes(i));
        initialW2 = 2*randi([0 1],hiddenNodes(i),K)-ones(hiddenNodes(i),K);
%         initialW1 = rand(D,hiddenNodes(i));
%         initialW2 = rand(hiddenNodes(i),K);
%         initialW1 = [1, 2; 3, 4];
%         initialW2 = [1, 2, 3; 4, 5, 6];
        MValue = hiddenNodes(i) 
        [w1, w2, e] = gradientDescentNoLoops(Xtrain, Ytrain, initialW1, initialW2, 0.01, 0.001*sqrt(hiddenNodes(i)*K), lambda, hiddenNodes(i),K);
        trainingError = getClassificationError(Xtrain, Ytrain, max(Ytrain), w1, w2, hiddenNodes(i))
        validateError = getClassificationError(Xvalidate, Yvalidate, max(Yvalidate), w1, w2, hiddenNodes(i))
        if validateError < bestError
            bestError = validateError;
            bestw1 = w1;
            bestw2 = w2;
            bestM = hiddenNodes(i);
        end
    end
    data = importdata(strcat('mnist_test.csv'));
    X = data(:,1:784);
    Y = data(:,785);
    error = getClassificationError(X, Y, max(Y), bestw1, bestw2, bestM)
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