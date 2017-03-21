features_file = 'whole_dll_features_with_text.csv';
features = csvread(features_file);
labels = features(:,1);
observations = features(:,2:end);
test_indices = load('test_indices.mat');
training_indices = load('training_indices.mat');

training_labels = labels(training_indices.training);
test_labels = labels(test_indices.test); %hold out set
training_features = observations(training_indices.training,:);
test_features = observations(test_indices.test,:); % hold out set

K = 5;
partition = cvpartition(training_labels, 'kFold', K);

weighted = false;
distanceFxn = 'euclidean';
num_neighbors = 3;

for t = 1:K
    validation_features = training_features(partition.test(t),:);
    validation_labels = training_labels(partition.test(t));
    train_features = training_features(partition.training(t),:);
    train_labels = training_labels(partition.training(t));

    % check that training and test data have same number of observations
    if size(train_features, 2) ~= size(validation_features, 2)
        error ('incorrect dimensions for train or validation data');
    end

    %distanceFxn = 'cosine'; %might want to pass this in somewhere as parameter

    if (weighted)
        ratio = sum(train_labels == 1) / sum(train_labels == -1);
    else
        ratio = 1;
    end

    [distances, indices] = pdist2(train_features, validation_features, distanceFxn, 'Smallest',num_neighbors);

    % get the nearest labels
    KNN_labels = {};
    for i = 1:size(validation_features, 1)
        labels = [];
        for j = 1:num_neighbors
            labels(j) = train_labels(indices(j, i));
            if labels(j) == 0
                labels(j) == -1;
            end
        end
        KNN_labels{i} = labels;
    end

    %predict labels
    predicted_labels = [];
    for i = 1:size(validation_features, 1)
        if (sum(KNN_labels{i} == 1)*ratio > sum(KNN_labels{i} == -1))
            predicted_labels(i) = 1;
        else
            predicted_labels(i) = 0;
        end
    end

    num_correct = 0;
    tp = 0;
    fp = 0;
    tn = 0;
    fn = 0;
    for i = 1:length(predicted_labels)
        if validation_labels(i) == 1
            if predicted_labels(i) == 1
                tp = tp+ 1;
                num_correct = num_correct+ 1;
            else
                fn = fn+ 1;
            end
        else
            if predicted_labels(i) == 1
                fp = fp+ 1;
            else
                tn = tn+ 1;
                num_correct = num_correct+ 1;
            end
        end
    end

    tp
    fp
    tn
    fn
    %accuracy = num_correct/length(predictions)
    accuracy_2 = (tp + tn) / (fp + fn + tp + tn)

end