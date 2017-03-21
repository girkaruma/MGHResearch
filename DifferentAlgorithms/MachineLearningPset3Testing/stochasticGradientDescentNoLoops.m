%function minimum = stochasticGradientDescentMatrix(w1, w2, stepSize, convergenceThreshold, lambda, M)
function [w1, w2, classificationError] = stochasticGradientDescentNoLoops(X, Y, w1, w2, stepSize, convergenceThreshold, lambda, M,K)
    %disp('init');
    tic
    oldY = Y;
    newY = zeros(length(Y), 3);
    for i=1:length(Y)
        newY(i, round(Y(i))) = 1;
    end
    Y = newY;
    %getClassificationError(X, oldY, max(oldY), w1, w2, M)
    [N,~] = size(X);
    %BREAK
    sample_index = randi([1,N]);
    x = X(sample_index,:);
    y = Y(sample_index,:);
    [w1new, w2new, w1, w2] = stochasticGradientDescentCalculation(x, y, w1, w2, lambda, M, stepSize);
    n = 1;
    objectiveFunction(X, Y, w1, w2, lambda, M);
    while abs(objectiveFunction(X, Y, w1new, w2new, lambda, M) - objectiveFunction(X, Y, w1, w2, lambda, M)) >= convergenceThreshold
        step_factor = 1;
        w1 = w1new;
        w2 = w2new;
        sample_index = randi([1,N]);
        x = X(sample_index,:);
        y = Y(sample_index,:);
        [w1new, w2new, w1, w2] = stochasticGradientDescentCalculation(x,y, w1, w2, lambda, M, stepSize*step_factor);
        
        %objectiveFunction(X, Y, w1, w2, lambda, M);
        n = n + 1;
    end
    objectiveFunction(X, Y, w1, w2, lambda, M)
    toc
    n = n
    classificationError = getClassificationError(X, oldY, max(oldY), w1, w2, M);
end

function [w1new, w2new, w1, w2] = stochasticGradientDescentCalculation(x, y,w1, w2,lambda, M,stepSize)
    w1new = w1 - stepSize * stochasticGradientFunctionw1(x, y, w1, w2,lambda, M);
    w2new = w2 - stepSize * stochasticGradientFunctionw2(x, y, w1, w2,lambda, M);
end

function objValue = objectiveFunction(X, Y, w1, w2, lambda, M)
    objValue = findFinalCost(X, Y, w1, w2, lambda, M);
end

function finalCost = findFinalCost(X, Y, w1, w2, lambda, M)
    w1sum = norm(w1,2)^2;
    w2sum = norm(w2,2)^2;
    finalCost = lossFunction(X, Y, w1, w2, M) + lambda * (w1sum + w2sum);
end
function loss = lossFunction(X, Y, w1, w2, M)
    [N, D] = size(X);
    [N, K] = size(Y);
    H = HTotal(X, w1, w2, M, K);
    kVect = ones(K,1);
    nVect = ones(1,N);
    loss = nVect*((0 - Y).*log2(H) - (1 - Y).*log2(1 - H))*kVect;
end

function prediction = h(x, w1, w2, M, K) %should return a kx1 vector
    prediction = logsig(logsig(x*w1)*w2)';
end

function predictionTotal = HTotal(X, w1, w2, M, K) %should return a Nxk vector
    predictionTotal = logsig(logsig(X*w1)*w2);
end

function sig = sigmoid(z)
    sig = 1/(1 + exp(0 - z));
end

function gradient = stochasticGradientFunction(initialGuess,X, ~, w1, w2,lambda, M)
    gradient = computeGradientCentralDifferences(initialGuess', 0.01);
end

function P = getP(X, Y, w1, w2, M)
    [N,D] = size(X);
    [N, K] = size(Y);
    predictionTotal = HTotal(X, w1, w2, M, K); 
    P = predictionTotal - Y;
end

function w2gradient = stochasticGradientFunctionw2(X, Y, w1, w2,lambda, M)
    [N, D] = size(X);
    [N, K] = size(Y);
    w2gradient = 2*lambda*w2;
    P = getP(X, Y, w1, w2, M); % should be Kx1 (in old method) and NxK in new method
    Z = getZ(X,w1); %should be 1xM (in old method) and NxM in new method
    w2gradient = w2gradient + Z'*P;
end

function w1gradient = stochasticGradientFunctionw1(X, Y, w1, w2,lambda, M)
    [N, D] = size(X);
    [N, K] = size(Y);
    %kVector = ones(1, K);
    Z = getZ(X,w1); %NxM matrix
    innerProduct = getP(X, Y, w1, w2, M)*w2'.*(Z.*(1-Z));
    w1gradient = 2*lambda*w1 + X'*innerProduct;
end

function Z = getZ(x,w1)
    Z = logsig(x * w1);
end

function classificationError = getClassificationError(X, Y, K, w1, w2, M)
    misclassifiedPoints = 0;
    for i=1:length(Y)
        highestK = 0;
        highestPredictionValue = 0;
        xPoint = X(i, :);
        for j=1:K
            if h(xPoint, w1, w2, M, K) > highestPredictionValue
                highestK = j;
                highestPredictionValue = h(xPoint, w1, w2, M, K);
            end
        end
        if Y(i) ~= highestK
            misclassifiedPoints = misclassifiedPoints + 1;
        end
    end
    classificationError = misclassifiedPoints/length(Y);
end