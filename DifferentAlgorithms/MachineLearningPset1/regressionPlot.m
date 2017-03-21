function regressionPlot(X,Y,order)
% X is an array of N data points (one dimensional for now), that is, Nx1
% Y is a Nx1 column vector of data values
% order is the order of the highest order polynomial in the basis functions
figure;

plot(X, Y, 'o', 'MarkerSize', 10);
xlabel('x');
ylabel('y');

% You will need to write the designMatrix and regressionFit functions

% constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
%phi = designMatrix(X,order);
% compute the weight vector
%w = regressionFit(X, Y, phi);
w = findWeightVector(X, Y, order);

hold all

x = 0:.01:1;
y = w'*designMatrix(x,order);
plot(x,y)
end
function weightVector = findWeightVector(X, Y, M)
    phi = computePhi(X, M);
    weightVector = inv(transpose(phi) * phi)*transpose(phi)*transpose(Y);
end
function phi = computePhi(X, M)
    N = length(X);
    phi = zeros(N, M);
    for r = 1:N
        for c = 0:M
            phi(r, c+1) = basisFunction(X(r), c);
        end
    end
end
function phi = designMatrix(x, order)
    M = order;
    phi = zeros(M+1, length(x));
    for i=1:length(x)
        for j=0:M
            phi(j+1,i)=basisFunction(x(i),j);
        end
    end
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