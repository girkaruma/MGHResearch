function [w1, w2, classificationError] = gradientDescent(X, Y, w1, w2, stepSize, convergenceThreshold, lambda, M,K)
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
    [w1i, w1j] = size(w1);
    w1sum = norm(w1,2);
    [w2i, w2j] = size(w2);
    w2sum = norm(w2,2);
    finalCost = lossFunction(X, Y, w1, w2, M) + lambda * (w1sum + w2sum);
end
function loss = lossFunction(X, Y, w1, w2, M)
    loss = 0;
    [N, D] = size(X);
    [N, K] = size(Y);
    for i=1:N
        h_i = h(X(i, :), w1, w2, M, K);
%         for j=1:K
%             loss = loss + (0 - Y(i,j))*log2(h_i(j)) - (1 - Y(i,j))*log2(1 - h_i(j));
%         end
        loss = loss + (0 - Y(i,:))*log2(h_i) - (1 - Y(i,:))*log2(1 - h_i);
    end
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

function prediction = HTotal(X, w1, w2, M, K) %should return a Nxk vector
    prediction = logsig(logsig(x*w1)*w2)';
end

function sig = sigmoid(z)
    sig = 1/(1 + exp(0 - z));
end

function gradient = gradientFunction(initialGuess,X, Y, w1, w2,lambda, M)
    gradient = computeGradientCentralDifferences(initialGuess', 0.01);
end

function P = getP(X, Y, w1, w2, M, i)
    [N,D] = size(X);
    [N, K] = size(Y);
    prediction = h(X(i,:), w1, w2, M, K); %fill in later
    %P = -Y(i,:)'.*(1-prediction) + (1-Y(i,:))'.*prediction;
    P = prediction - Y(i,:)';
    
end
function w2gradient = gradientFunctionw2(X, Y, w1, w2,lambda, M)
    [N, D] = size(X);
    [N, K] = size(Y);
    w2gradient = 2*lambda*w2;
    for i=1:N
        P = getP(X, Y, w1, w2, M, i); % should be Kx1
        Z = getZ(X,w1,i); %should be 1xM
        w2gradient = w2gradient + Z'*P'; 
    end
    %Z = getZ(X,w1,M,i);
end
% function w2gradient = gradientFunctionw2(X, Y, w1, w2,lambda, M)
%     [N, D] = size(X);
%     [N, K] = size(Y);
%     w2gradient = 2*lambda*w2;
%     for i=1:N,
%         Z = getZ(X,w1,M,i);
%         A2 = Z * w2;
%         for b=1:M,
%             for c=1:K,
%                 w2gradient(b,c) = w2gradient(b,c) - Y(i,c)*sigmoid(-A2(c))*Z(b)+(1-Y(i,c))*sigmoid(A2(c))*Z(b);
%             end
%         end
%     end
% end

function w1gradient = gradientFunctionw1(X, Y, w1, w2,lambda, M)
    [N, D] = size(X);
    [N, K] = size(Y);
    kVector = ones(1, K);
    w1gradient = 2*lambda*w1;
    for i=1:N
        hidden_sig = getZ(X,w1,i); % 1xM vector
        innerProduct = w2'.*(getP(X, Y, w1, w2, M, i)*(hidden_sig.*(1-hidden_sig)));
        w1gradient = w1gradient + X(i, :)'*kVector*innerProduct; 
    end   
end
% function w1gradient = gradientFunctionw1(X, Y, w1, w2,lambda, M)
%     [N, D] = size(X);
%     [N, K] = size(Y);
%     w1gradient = 2*lambda*w1;
%     for i=1:N,
%         Z = getZ(X,w1,M,i);
%         A2 = Z * w2;
%         for k=1:K,
%             for b=1:M,
%                 for a=1:D,
%                     mult_term = w2(b,k)*Z(b)*(1-Z(b))*X(i,a);
%                     w1gradient(a,b) = w1gradient(a,b) + (-Y(i,k)*sigmoid(-A2(k)) + (1-Y(i,k))*sigmoid(A2(k)))*mult_term;
%                 end
%             end
%         end
%     end    
% end

% function Z = getZ(X,w1,M,i)
%     Z = zeros(1,M);
%     for j=1:M,
%         Z(j) = sigmoid( X(i,:) * w1(:,j));
%     end
% end
function Z = getZ(X,w1,i)
    Z = logsig(X(i,:) * w1);
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