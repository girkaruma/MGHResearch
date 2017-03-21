function computeMaxWeightVector()
    data = importdata('curvefitting.txt');
    X = data(1,:);
    Y = data(2,:);
    for M=1:9
        w = findWeightVector(X, Y, M);
        computeSumOfSquaresError(X, Y, M, w)
    end
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