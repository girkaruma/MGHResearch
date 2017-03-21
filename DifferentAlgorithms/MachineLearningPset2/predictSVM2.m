function [ y ] = predictSVM2( weights, X , vo)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
X = [1; X];
y = weights*X;
end

