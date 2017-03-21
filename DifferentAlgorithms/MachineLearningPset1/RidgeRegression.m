function RidgeRegression()
    data = importdata('regress_train.txt');
    X = data(1,:);
    Y = data(2,:);
    data1 = importdata('regress_validate.txt');
    X1 = data1(1,:);
    Y1 = data1(2,:);
    data2 = importdata('regress_test.txt');
    X2 = data2(1,:);
    Y2 = data2(2,:);
    minM = 0;
    minW = 0;
    lambda = 0.1;
    error = 1000000;
    for i=1:9
        phi = designMatrix(X, i);
        w = largeRegression(phi, Y, lambda);
        newError = computeRidgeRegression(X1, Y1, i, w, lambda)
        if newError < error
            error = newError;
            minM = i;
            minW = w;
        end
    end
    Uma = computeRidgeRegression(X2, Y2, 3, minW, 0.1)
%      M = 7;
%      lambda = 0;
%      phi = designMatrix(X, M);
%      w1 = largeRegression(phi, Y, lambda);
%     lambda = 0.00001;
%     phi = designMatrix(X, M);
%     w2 = largeRegression(phi, Y, lambda);
%     lambda = 0.0001;
%     phi = designMatrix(X, M);
%     w3 = largeRegression(phi, Y, lambda);
%     lambda = 0.1;
%     phi = designMatrix(X, M);
%     w4 = largeRegression(phi, Y, lambda);
%     lambda = 1;
%     phi = designMatrix(X, M);
%     w5 = largeRegression(phi, Y, lambda);
    %w = computeRidgeRegressionDerivative(X, Y, M, lambda);
    %computeRidgeRegression(X, Y, M, w, lambda)
    
    %Plot of graphs
    figure;
    plot(X2, Y2, 'o', 'MarkerSize', 10);
    xlabel('x');
    ylabel('y');
    hold all
    x = -3:.01:2;
    M = 3;
    y = minW'*designMatrix(x,M)';
    plot(x,y)
%     hold all
%     x = -3:.01:2;
%     y = w2'*designMatrix(x,M)';
%     plot(x,y)
%     hold all
%     x = -3:.01:2;
%     y = w3'*designMatrix(x,M)';
%     plot(x,y)
%     hold all
%     x = -3:.01:2;
%     y = w4'*designMatrix(x,M)';
%     plot(x,y)
%     hold all
%     x = -3:.01:2;
%     y = w5'*designMatrix(x,M)';
%     plot(x,y)
end
function error = computeRidgeRegression(X, Y, M, w, lambda)
    error = 0;
    N = length(X);
    for i=1:N
        hypothesis = 0;
        for j=0:M
            hypothesis = hypothesis + w(j+1)*basisFunction(X(i), j);
        end
        error = error + (Y(i) - hypothesis)^2;
    end
    weightMagnitude = 0;
    for i=1:length(w)
        weightMagnitude = weightMagnitude + w(i)^2;
    end
    %error = error + lambda * weightMagnitude;
    error = 0.5 * error;
end
function optimalW = computeRidgeRegressionDerivative(X, Y, M, lambda)
    numerator = zeros(M+1,1);
    denominator = lambda;
    N = length(X);
    for i = 1:N
        phiN = zeros(M+1, 1);
        for j=0:M
            phiN(j+1, 1) = basisFunction(X(i), j);
            denominator = denominator + (basisFunction(X(i), j))^2;
        end 
        numerator = numerator + Y(i) * phiN;
    end
    optimalW = numerator/denominator;
end
function val = basisFunction(x, m)
    val = x^m;
end
function optimalW = largeRegression(phi, Y, lambda)
    xSize = size(transpose(phi)*phi);
    identity = eye(xSize(2), xSize(1));
    optimalW = inv(lambda*identity + transpose(phi)*phi)*transpose(phi)*Y';
end
function phi = designMatrix(x, order)
    M = order;
    phi = zeros(length(x), M+1);
    for i=1:length(x)
        for j=0:M
            phi(i, j+1)=basisFunction(x(i),j);
        end
    end
end

