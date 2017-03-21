function [w1, w2, classificationError] = gradientDescentNoLoops(X, Y, w1, w2, stepSize, convergenceThreshold, lambda, M,K)
    %disp('init');
    oldY = Y;
    newY = zeros(length(Y), K);
    for i=1:length(Y)
        newY(i, round(Y(i))) = 1;
    end
    Y = newY;
    tic
    %getClassificationError(X, oldY, max(oldY), w1, w2, M)
    [w1new, w2new, w1, w2] = gradientDescentCalculation(X, Y, w1, w2, lambda, M, stepSize);
    n = 0;
    objectiveFunction(X, Y, w1, w2, M)
    while abs(objectiveFunction(X, Y, w1new, w2new, M) - objectiveFunction(X, Y, w1, w2, M)) >= convergenceThreshold
        w1 = w1new;
        w2 = w2new;
        [w1new, w2new, w1, w2] = gradientDescentCalculation(X, Y, w1, w2, lambda, M, stepSize);
        objectiveFunction(X, Y, w1, w2, M);
        %objectiveFunction(w1new, w2new, M)
        n = n + 1;
        %getClassificationError(X, Y, max(Y), w1, w2, M)
    end
    %objectiveFunction(w1, w2, M)
    toc
    n=n
    classificationError = getClassificationError(X, oldY, max(oldY), w1, w2, M);
end

function [w1new, w2new, w1, w2] = gradientDescentCalculation(X, Y,w1, w2,lambda, M,stepSize)
    w1new = w1 - stepSize * gradientFunctionw1(X, Y, w1, w2,lambda, M);
    w2new = w2 - stepSize * gradientFunctionw2(X, Y, w1, w2,lambda, M);
end

function objValue = objectiveFunction(X, Y, w1, w2, M)
%     newY = zeros(length(Y), 3);
%     for i=1:length(Y)
%         newY(i, round(Y(i))) = 1;
%     end
%     Y = newY;
    lambda = 0; % CHANGE THIS
    objValue = findFinalCost(X, Y, w1, w2, lambda, M);
end

function finalCost = findFinalCost(X, Y, w1, w2, lambda, M)
    w1sum = norm(w1,2);
    w2sum = norm(w2,2);
    finalCost = lossFunction(X, Y, w1, w2, M) + lambda * (w1sum + w2sum);
end
% function loss = lossFunction(X, Y, w1, w2, M)
%     loss = 0;
%     [N, D] = size(X);
%     [N, K] = size(Y);
%     for i=1:N
%         h_i = h(X(i, :), w1, w2, M, K);
% %         for j=1:K
% %             loss = loss + (0 - Y(i,j))*log2(h_i(j)) - (1 - Y(i,j))*log2(1 - h_i(j));
% %         end
%         loss = loss + (0 - Y(i,:))*log2(h_i) - (1 - Y(i,:))*log2(1 - h_i);
%     end
% end
function loss = lossFunction(X, Y, w1, w2, M)
    [N, D] = size(X);
    [N, K] = size(Y);
    H = HTotal(X, w1, w2, M, K);
    %         for j=1:K
    %             loss = loss + (0 - Y(i,j))*log2(h_i(j)) - (1 - Y(i,j))*log2(1 - h_i(j));
    %         end
    kVect = ones(K,1);
    nVect = ones(1,N);
    loss = nVect*((0 - Y).*log2(H) - (1 - Y).*log2(1 - H))*kVect;
end

% function prediction = h(x, w1, w2, k, M)
%     outerSum = 0;
%     D = length(x);
%     for j=1:M
%         innerSum = 0;
%         for i = 1:D 
%             innerSum = innerSum + w1(i,j)*x(i);
%         end
%         outerSum = outerSum + sigmoid(innerSum) * w2(j,k);
%     end 
%     prediction = sigmoid(outerSum);
% end

function prediction = h(x, w1, w2, M, K) %should return a kx1 vector
    prediction = logsig(logsig(x*w1)*w2)';
end

function predictionTotal = HTotal(X, w1, w2, M, K) %should return a Nxk vector
    predictionTotal = logsig(logsig(X*w1)*w2);
end

function sig = sigmoid(z)
    sig = 1/(1 + exp(0 - z));
end

function gradient = gradientFunction(initialGuess,X, Y, w1, w2,lambda, M)
    gradient = computeGradientCentralDifferences(initialGuess', 0.01);
end

% function P = getP(X, Y, w1, w2, M, i)
%     [N,D] = size(X);
%     [N, K] = size(Y);
%     prediction = h(X(i,:), w1, w2, M, K); %fill in later
%     %P = -Y(i,:)'.*(1-prediction) + (1-Y(i,:))'.*prediction;
%     P = prediction - Y(i,:)';
% end
    
function P = getP(X, Y, w1, w2, M)
    [N,D] = size(X);
    [N, K] = size(Y);
    predictionTotal = HTotal(X, w1, w2, M, K); 
    P = predictionTotal - Y;
end

% function w2gradient = gradientFunctionw2(X, Y, w1, w2,lambda, M)
%     [N, D] = size(X);
%     [N, K] = size(Y);
%     w2gradient = 2*lambda*w2;
%     P = getP(X, Y, w1, w2, M); % should be Kx1 (in old method) and NxK in new method
%     Z = getZ(X,w1); %should be 1xM (in old method) and NxM in new method
%     for i=1:N
%         w2gradient = w2gradient + Z'*P';
%     end
% end

function w2gradient = gradientFunctionw2(X, Y, w1, w2,lambda, M)
    [N, D] = size(X);
    [N, K] = size(Y);
    w2gradient = 2*lambda*w2;
    P = getP(X, Y, w1, w2, M); % should be Kx1 (in old method) and NxK in new method
    Z = getZ(X,w1); %should be 1xM (in old method) and NxM in new method
    w2gradient = w2gradient + Z'*P;
end


% function w1gradient = gradientFunctionw1(X, Y, w1, w2,lambda, M)
%     [N, D] = size(X);
%     [N, K] = size(Y);
%     kVector = ones(1, K);
%     w1gradient = 2*lambda*w1;
%     for i=1:N
%         hidden_sig = getZ(X,w1,i); % 1xM vector
%         innerProduct = w2'.*(getP(X, Y, w1, w2, M, i)*(hidden_sig.*(1-hidden_sig)));
%         w1gradient = w1gradient + X(i, :)'*kVector*innerProduct; 
%     end   
% end

function w1gradient = gradientFunctionw1(X, Y, w1, w2,lambda, M)
    [N, D] = size(X);
    [N, K] = size(Y);
    %kVector = ones(1, K);
    Z = getZ(X,w1); %NxM matrix
    innerProduct = getP(X, Y, w1, w2, M)*w2'.*(Z.*(1-Z));
    w1gradient = 2*lambda*w1 + X'*innerProduct;
end

% function Z = getZ(X,w1,M,i)
%     Z = zeros(1,M);
%     for j=1:M,
%         Z(j) = sigmoid( X(i,:) * w1(:,j));
%     end
% end
function Z = getZ(X,w1)
    Z = logsig(X * w1);
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