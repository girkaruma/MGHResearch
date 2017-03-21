function gradientCentralDifferences()
    point = [3 0];
    stepSize = 10;
    gradient = computeGradientCentralDifferences(point, stepSize)
end
function gradient = computeGradientCentralDifferences(point, stepSize)
    gradient = zeros(length(point), 1);
    for i=1:length(point)
        pointAdd = transpose(point);
        pointSubtract = transpose(point);
        pointAdd(i, 1) = pointAdd(i, 1) + 0.5 * stepSize;
        pointSubtract(i, 1) = pointSubtract(i, 1) - 0.5 * stepSize;
        gradient(i, 1) = (objectiveFunction(pointAdd) - objectiveFunction(pointSubtract))/stepSize;
    end
end
function objValue = objectiveFunction(point)
    x = point(1);
    y = point(2);
    objValue = (x+4)*(x-3)*(x+1)*(x-2);
end

