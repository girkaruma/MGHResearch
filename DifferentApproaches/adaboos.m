features_file = 'whole_dll_features_with_text.csv';
features = csvread(features_file);
labels = features(:,1);
%filenames = features(:,2);
observations = features(:,3:5);
test_indices = load('test_indices.mat');
training_indices = load('training_indices.mat');

training_labels = labels(training_indices.training);
test_labels = labels(test_indices.test); %hold out set
training_features = observations(training_indices.training,:);
test_features = observations(test_indices.test,:); % hold out set

K = 5;
partition = cvpartition(training_labels, 'kFold', K);

for j=1:K
	validation_features = training_features(partition.test(j),:);
	validation_labels = training_labels(partition.test(j));
	train_features = training_features(partition.training(j),:);
	train_labels = training_labels(partition.training(j));


	model = fitensemble(train_features, train_labels, 'AdaBoostM1', 100,'Tree');
	predictions = predict(model, validation_features);

	num_correct = 0;
	tp = 0;
	fp = 0;
	tn = 0;
	fn = 0;
	for i = 1:length(predictions)
		if validation_labels(i) == 1
			if predictions(i) == 1
				tp = tp+ 1;
				num_correct = num_correct+ 1;
			else
				fn = fn+ 1;
			end
		else
			if predictions(i) == 1
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
	accuracy = num_correct/length(predictions)
	accuracy_2 = (tp + tn) / (fp + fn + tp + tn)
end

final_tp = 0;
final_tn = 0;
final_fp = 0;
final_fn = 0;

final_model = fitensemble(training_features, training_labels, 'AdaBoostM1', 100,'Tree');
final_predictions = predict(final_model, test_features);

for i = 1:length(final_predictions)
	if test_labels(i) == 1
		if final_predictions(i) == 1
			final_tp = final_tp+ 1;
		else
			final_fn = final_fn+ 1;
		end
	else
		if final_predictions(i) == 1
			final_fp = final_fp+ 1;
		else
			final_tn = final_tn+ 1;
		end
	end
end

final_tp
final_fp
final_tn
final_fn
final_accuracy = (final_tp + final_tn) / (final_fp + final_fn + final_tp + final_tn)