function [c1, c2] = lr_test(name, lambda)
disp('======Training======');
% load data from csv files

data = importdata(strcat('data/data_',name,'_train.csv'));

X = data(:,1:11);
for p=5:8
    X(:,p) = getFeatureScaling(X(:,p));
end 
Y = data(:,12);
fun = @(w)(sum(log2(1 + exp(-Y.*(X*w(1:11)' + w(12))))) + lambda * w(1:11)*w(1:11)');
gradientSolution = fminunc(fun,[0 0 0 0 0 0 0 0 0 0 0 0]); 
gradientSolution = gradientSolution';
c1 = getClassificationErrorRate(X, Y, gradientSolution(1:11), gradientSolution(12))
%gradientSolution = gradientDescent([1 1 0.5]', 0.0001, 0.000001);
% Carry out training.
%%% TODO %%%

% Define the predictLR(x) function, which uses trained parameters
%%% TODO %%%

hold on;
% plot training results
%plotDecisionBoundary(X, Y, @(x)predictLR(x,gradientSolution(1:2), gradientSolution(3)), [0, 0], 'LR Train');

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X = validate(:,1:11);
for p=5:8
    X(:,p) = getFeatureScaling(X(:,p));
end
Y = validate(:,12);
c2 = getClassificationErrorRate(X, Y, gradientSolution(1:11), gradientSolution(12))
% plot validation results
%plotDecisionBoundary(X, Y, @(x)predictLR(x,gradientSolution(1:2), gradientSolution(3)), [0, 0], 'LR Validate');
end

function errorRate = getClassificationErrorRate(X, Y, w, w0)
    error = 0;
    for i=1:length(Y)
        val = predictLR(X(i, :)', w, w0);
        if val < 0
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
function featureScaling = getFeatureScaling(XValue)
    featureScaling = zeros(length(XValue), 1);
    xmin = min(XValue);
    xmax = max(XValue);
    for i=1:length(XValue)
        featureScaling(i, 1) = (XValue(i, 1) - xmin)/(xmax - xmin);
    end
end
% function error_logistic_regression = logistic_regression(X, Y, w, w0)
%     error_logistic_regression = 0;
%     for i = 1:length(Y) 
%         error_i = log10(1 + exp(-Y(i)*(X(i)'*w + w0)));
%         error_logistic_regression = error_logistic_regression + error_i;
%     end
%     error_logistic_regression = error_logistic_regression + lambda * (w.*w);
% end 
