function svm_test1(name,C)
disp('======Training======');
% load data from csv files
data = importdata(strcat('data/data_',name,'_train.csv'));
X = data(:,1:2);
Y = data(:,3);
% X = [1 2; 2 2; 0 0; -2 3];
% Y = [1 1 -1 -1]';

% Carry out training, primal and/or dual
%%% TODO %%%
[ w, intercept] = svm_obj(C,strcat('data/data_',name,'_train.csv'));

weights = [intercept w]

% Define the predictSVM(x) function, which uses trained parameters
%%% TODO %%%


hold on;
% plot training results
plotDecisionBoundary2(X, Y, @predictSVM2, [-1, 0, 1], 'SVM Train',weights,[]);


disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X = validate(:,1:2);
Y = validate(:,3);

% plot validation results
plotDecisionBoundary2(X, Y, @predictSVM2, [-1, 0, 1], 'SVM Validate',weights,[]);

