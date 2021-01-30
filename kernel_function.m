function [ k ] = kernel_function( x, y, s )
%KERNEL_FUNCTION Summary of this function goes here
%   Detailed explanation goes here

    % polynomial kernel
%     k =  (x'*y+4)^2;

    % gaussian kernel
    k =  exp(-norm(x - y)^2 / (2*s*s));
end

