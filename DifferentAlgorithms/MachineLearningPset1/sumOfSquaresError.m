function sumOfSquaresError()
    data = importdata('curvefitting.txt');
    X = data(1,:);
    Y = data(2,:);
    w = [0.3137 7.9854 -25.4261 17.3741];
    error = computeSumOfSquaresError(X, Y, 3, w)
    derivativeVector = derivativeOfSSE(X, Y, 3, w)
end
function derivativeVector = derivativeOfSSE(X, Y, M, w)
    N = length(X);
    derivativeVector = zeros(M+1, 1);
    for i=1:N
        hypothesis = 0;
        phiN = zeros(M+1, 1);
        for j=0:M
            phiN(j+1, 1) = basisFunction(X(i), j);
            hypothesis = hypothesis + w(j+1)*basisFunction(X(i), j);
        end
        derivativeVector = derivativeVector + (Y(i) - hypothesis)*phiN;
    end
end
function error = computeSumOfSquaresError(X, Y, M, w)
    error = 0;
    N = length(X);
    for i=1:N
        hypothesis = 0;
        for j=0:M
            hypothesis = hypothesis + w(j+1)*basisFunction(X(i), j);
        end
        error = error + (Y(i) - hypothesis)^2;
    end
    error = 0.5 * error;
end
function val = basisFunction(x, m)
    val = x^m;
end

