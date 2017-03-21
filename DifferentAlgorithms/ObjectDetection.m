function [maxArea, hasObject] = ObjectDetection(imageToBeRead)
%clear all
%close all
%load 2010_02_05_Hole_ET.mat;
%sonar = ans;
%stbd = uint8(sonar.stbd*255);
%partialSonarImage = stbd(2100:2499,2150:2649);
partialSonarImage = imread(imageToBeRead);
partialSonarImage = rgb2gray(partialSonarImage);
[x, y, z] = size(partialSonarImage);
highlightedImage = zeros(x, y);
highlightedImage = im2uint8(highlightedImage);
dilatedObjectHighlights = zeros(x, y);
dilatedObjectHighlights = im2uint8(dilatedObjectHighlights);
%g = im2uint8(mat2gray(log(1 + double(partialSonarImage))));
%g = rgb2gray(partialSonarImage);
g = partialSonarImage;
g = histeq(g, 256);
o = fspecial('disk', 0.5);
aq = imfilter(g, o);
p = fspecial('unsharp');
ai = imfilter(aq, p);
a = histeq(ai, 256);
a = adpmedian(a, 9);
a = wiener2(a);
a = im2bw(a, 0.41);
aia = bayesEstimateDenoise(double(ai), 'sigmaSpatial', 0, 'windowSize', 3, 'sigmaFactor', 2);
for row = 1: x
    for col = 1: y
        if a(row, col) == 0
            a(row, col) = 1;
        else
            a(row, col) = 0;
        end
    end
end
a = bwareaopen(a, 70, 4);
[L, n] = bwlabeln(a, 4);
stats = regionprops(L,'Area');
for i = 1:n
    [r, c] = find(L == i);
    x1 = min(c); x2 = max(c); y1 = min(r); y2 = max(r);
    if x1 - 20 >= 1
        x1 = x1 - 20;
    else
        x1 = 1;
    end
    if y1 - 5 >= 1
        y1 = y1 - 5;
    else
        y1 = 1;
    end
    if y2 + 10 <= x
        y2 = y2 + 10;
    else
        y2 = x;
    end
    if x2 + 50 <= y
        x2 = x2 + 50;
    else
        x2 = y;
    end
    for ro = x1:x2
        for co = y1: y2
            if ai(co, ro) == 0
                aia(co, ro) = 0;
            end
        end
    end
end
aa = histeq(aia, 256);
aa = adpmedian(aa, 9);
aa = wiener2(aa);
aa = im2bw(aa, 0.41);
for row = 1: x
    for col = 1: y
        if aa(row, col) == 0
            aa(row, col) = 1;
        else
            aa(row, col) = 0;
        end
    end
end
aa = bwareaopen(aa, 25, 4);
aa = bwmorph(aa, 'bridge', 2);
[L, n] = bwlabeln(aa, 4);
stats = regionprops(L,'Area');
% figure, imshow(aa)
% figure, imshow(partialSonarImage)
% hold on
% if n > 2
%     hasObject = 1;
% else
%     hasObject = 0;
% end
hasObject = n;
maxArea = 0;
for i = 1:n
    [r, c] = find(L == i);    
    x1 = min(c); x2 = max(c); y1 = min(r); y2 = max(r);
    if x1 - 20 >= 1
        x1 = x1 - 20;
    else
        x1 = 1;
    end
    if y1 - 5 >= 1
        y1 = y1 - 5;
    else
        y1 = 1;
    end
    if y2 + 10 <= x
        y2 = y2 + 10;
    else
        y2 = x;
    end
    if x2 + 50 <= y
        x2 = x2 + 50;
    else
        x2 = y;
    end
    if maxArea < (x2 - x1) * (y2 - y1)
        maxArea = (x2 - x1) * (y2 - y1);
    end
%     plot([x1, x2] , [y1, y1])
%     hold on
%     plot([x2, x2] , [y1, y2])
%     hold on
%     plot([x1, x1], [y1, y2])
%     hold on
%     plot([x1, x2], [y2, y2])
       umag = zeros(x, y);
       umag = im2uint8(umag);
       for jj = x1:x2
           for kk = y1:y2
               umag(kk, jj) = partialSonarImage(kk, jj);
           end
       end
       for abc = 1: x
           for cdf = 1:y
               if umag(abc, cdf) < 30
                   umag(abc, cdf) = 0;
               else
                   if umag(abc, cdf) > 200
                       umag(abc, cdf) = 255;
                   else
                       umag(abc, cdf) = 0;
                   end
               end               
           end
       end
       umag = bwareaopen(umag, 5, 4);
       umag = bwmorph(umag, 'dilate', 8);
       umag = bwmorph(umag, 'thin', 2);
       [LL, nn] = bwlabeln(umag, 4);
       stats = regionprops(L,'Area');
       [re, ce] = find(LL == 1);
       x11 = min(ce); x22 = max(ce); y11 = min(re); y22 = max(re);
       for myR = x11: x22
           for myC = y11: y22
               if umag(myC, myR) == 1
                   dilatedObjectHighlights(myC, myR) = 255;
               end
           end
       end
end

for pi = 1: x
    for po = 1:y
        if dilatedObjectHighlights(pi, po) == 255
            dilatedObjectHighlights(pi, po) = partialSonarImage(pi, po);    
        end
    end
end

dilatedObjectHighlights = im2bw(dilatedObjectHighlights, 0.69);
%figure, imshow(dilatedObjectHighlights)
close all
end

