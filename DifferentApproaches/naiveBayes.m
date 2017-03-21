features_file = 'whole_dll_features_with_text.csv';
features = csvread(features_file);
labels = features(:,1);
%filenames = features(:,2);
observations = features(:,2:end);
K = 5;
partition = cvpartition(labels, 'kFold', K);

for j=1:K
	test_features = observations(partition.test(j),:);
	test_labels = labels(partition.test(j));
	training_features = observations(partition.training(j),:);
	training_labels = labels(partition.training(j));


	model = NaiveBayes.fit(training_features, training_labels, 'Distribution', 'mn');
	predictions = predict(model, test_features);

	num_correct = 0;
	tp = 0;
	fp = 0;
	tn = 0;
	fn = 0;
	for i = 1:length(predictions)
		if test_labels(i) == 1
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
