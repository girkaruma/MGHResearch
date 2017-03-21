function minimum = stochasticgradientdescent(w1, w2, stepSize, convergenceThreshold, lambda, M)
    %disp('init');
    data = importdata(strcat('toy_multiclass_1_train.csv'));
    X = data(:,1:2);
    Y = data(:,3);
    newY = zeros(length(Y), 3);
    for i=1:length(Y)
        newY(i, round(Y(i))) = 1;
    end
    Y = newY;
    [N, D] = size(X);
    sample_index = randi([1,N]);
    x = X(sample_index,:);
    y = Y(sample_index,:);
    [w1new, w2new, w1, w2] = stochasticGradientDescentCalculation(x, y, w1, w2, lambda, M, stepSize);
    n = 1;
    objectiveFunction(w1, w2, M)
    while abs(objectiveFunction(w1new, w2new, M) - objectiveFunction(w1, w2, M)) >= convergenceThreshold
        step_factor = 50/n;
        w1 = w1new;
        w2 = w2new;
        sample_index = randi([1,N]);
        x = X(sample_index,:);
        y = Y(sample_index,:);
        [w1new, w2new, w1, w2] = stochasticGradientDescentCalculation(x,y, w1, w2, lambda, M, stepSize*step_factor);
        objectiveFunction(w1new, w2new, M)
        n = n + 1;
    end
    objectiveFunction(w1, w2, M)
end

function [w1new, w2new, w1, w2] = stochasticGradientDescentCalculation(x, y,w1, w2,lambda, M,stepSize)
    w1new = w1 - stepSize * stochasticGradientFunctionw1(x, y, w1, w2,lambda, M);
    w2new = w2 - stepSize * stochasticGradientFunctionw2(x, y, w1, w2,lambda, M);
end

function objValue = objectiveFunction(w1, w2, M)
    data = importdata(strcat('toy_multiclass_1_train.csv'));
    X = data(:,1:2);
    Y = data(:,3);
    newY = zeros(length(Y), 3);
    for i=1:length(Y)
        newY(i, round(Y(i))) = 1;
    end
    Y = newY;
    lambda = 0;
    objValue = findFinalCost(X, Y, w1, w2, lambda, M);
end

function finalCost = findFinalCost(X, Y, w1, w2, lambda, M)
    [w1i, w1j] = size(w1);
    w1sum = 0;
    for i=1:w1i
        for j=1:w1j
            w1sum = w1sum + w1(i, j)^2;
        end
    end
    [w2i, w2j] = size(w2);
    w2sum = 0;
    for i=1:w2i
        for j=1:w2j
            w2sum = w2sum + w2(i, j)^2;
        end
    end
    finalCost = lossFunction(X, Y, w1, w2, M) + lambda * (w1sum + w2sum);
end
function loss = lossFunction(X, Y, w1, w2, M)
    loss = 0;
    [N, D] = size(X);
    [N, K] = size(Y);
    for i=1:N
        for j=1:K
            loss = loss + (0 - Y(i,j))*log2(h(X(i, :), w1, w2, j, M)) - (1 - Y(i,j))*log2(1 - h(X(i, :), w1, w2, j, M));
        end
    end
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

function gradient = stochasticGradientFunction(initialGuess,X, Y, w1, w2,lambda, M)
    gradient = computeGradientCentralDifferences(initialGuess', 0.01);
end

function w2gradient = stochasticGradientFunctionw2(x, y, w1, w2,lambda, M)
    [~, D] = size(x);
    [~, K] = size(y);
    w2gradient = 2*lambda*w2;
    Z = getZ(x,w1,M);
    A2 = Z * w2;
    for b=1:M,
        for c=1:K,
            w2gradient(b,c) = w2gradient(b,c) - y(c)*sigmoid(-A2(c))*Z(b)+(1-y(c))*sigmoid(A2(c))*Z(b);
        end
    end
end

function w1gradient = stochasticGradientFunctionw1(x, y, w1, w2,lambda, M)
    [~, D] = size(x);
    [~, K] = size(y);
    w1gradient = 2*lambda*w1;
    Z = getZ(x,w1,M);
    A2 = Z * w2;
    for k=1:K,
        for b=1:M,
            for a=1:D,
                mult_term = w2(b,k)*Z(b)*(1-Z(b))*x(a);
                w1gradient(a,b) = w1gradient(a,b) + (-y(k)*sigmoid(-A2(k)) + (1-y(k))*sigmoid(A2(k)))*mult_term;
            end
        end
    end
    
end

function Z = getZ(x,w1,M)
    Z = zeros(1,M);
    for j=1:M,
        Z(j) = sigmoid( x * w1(:,j));
    end
end