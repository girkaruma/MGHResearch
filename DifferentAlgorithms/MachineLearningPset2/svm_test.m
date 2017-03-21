function [geometricMargin, supportVectors, cl] = svm_test(name, C, bandwidth)
disp('======Training======');
% load data from csv files
data = importdata(strcat('data/data_',name,'_train.csv'));
X = data(:,1:2);
% for p=5:8
%     X(:,p) = getFeatureScaling(X(:,p));
% end 
Y = data(:,3);
n = length(Y);
% Carry out training, primal and/or dual
optim_ver = ver('optim');
optim_ver = str2double(optim_ver.Version);
if optim_ver >= 6
    opts = optimset('Algorithm', 'interior-point-convex');
else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
end
H = zeros(n, n);
K = computeKernel(X, n, bandwidth);
for a = 1:n
    for b = 1:n
        H(a, b) = Y(a)*Y(b)*X(a,:)*X(b,:)';
        %H(a, b) = Y(a)*Y(b)*K(a, b);
    end
end
f = -1 * ones(n, 1);
Aeq = Y';
beq = 0;
lb = zeros(1, n);
ub = C * ones(1,n);
sol = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], opts);
%weights = [0 0];
weights = [0 0];
for i = 1:length(sol)
    weights = weights + Y(i) * sol(i) * X(i, :);
end
weights
geometricMargin = 1/(weights*weights');
M = 0;
ySum = 0;
supportVectors = 0;
supportSum = 0;
for i = 1:length(sol)
%     if sol(i) > 0.01 
%         ySum = ySum + Y(i,:);
%         supportSum = supportSum + sol(i)*Y(i,:)*X(i,:)*X(i,:)';
%         M = M + 1;
%     end
    if sol(i) > 0.01
        supportVectors = supportVectors + 1;
    end
    %supportSum = supportSum + sol(i)*Y(i)*X(i,:)*X(i,:)';
end
supportVectors;
w0 = (ySum - supportSum)/(1.0 * M);
w0 = getIntercept(X, Y, sol, C)
%[weights, w0] = svm_obj(1, strcat('data/data_',name,'_train.csv'));
getClassificationErrorRate(X, Y, weights', w0)
% Define the predictSVM(x) function, which uses trained parameters
%%% TODO %%%


hold on;
% plot training results
plotDecisionBoundary(X, Y, @(x)predictSVM(x,weights', w0), [-1, 0, 1], 'SVM Train');

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X = validate(:,1:2);
% for p=5:8
%     X(:,p) = getFeatureScaling(X(:,p));
% end 
Y = validate(:,3);
cl = getClassificationErrorRate(X, Y, weights', w0);
% plot validation results
plotDecisionBoundary(X, Y, @(x)predictSVM(x,weights', w0), [-1, 0, 1], 'SVM Validate');
end
function errorRate = getClassificationErrorRate(X, Y, w, w0)
    error = 0;
    for i=1:length(Y)
        val = predictSVM(X(i, :)', w, w0);
        if val <= 0
            val = -1;
        else
            val = 1;
        end
        if val ~= Y(i)
            error = error + 1;
        end
    end
    errorRate = error/(1.0 * length(Y));
end
function kernelMatrix = computeKernel(X, n, dev)
    kernelMatrix = zeros(n, n);
    for i = 1:n
        for j = 1:n
            kernelMatrix(i, j) = exp(0 - (X(i,:)-X(j,:))*(X(i,:)-X(j,:))'/(2.0* dev^2));
        end
    end
end
function featureScaling = getFeatureScaling(XValue)
    featureScaling = zeros(length(XValue), 1);
    xmin = min(XValue);
    xmax = max(XValue);
    for i=1:length(XValue)
        featureScaling(i, 1) = (XValue(i, 1) - xmin)/(xmax - xmin);
    end
end

