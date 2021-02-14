% Copyright (c) 2021 Denis C. ILIE-ABLACHIM <denis.ilie_ablachim@upb.ro>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
% 
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

clc;
clear;

sigma = 8;
gamma = 5;
n_components = 100;
n_nonzero_coefs = 20;
n_iterations = 10;

% Load training and testing data
% DataPath   = 'YaleB_Jiang';
DataPath   = 'ar_face_db';
% DataPath   = 'caltech101_db';
load(fullfile('dbs', DataPath));

% Data preprocessing
% - data normalization is needed -
% y_train = TrLabel;
% y_test = TtLabel;
% X_train = TrData;
% X_test = TtData;
X_train = normcol_equal(X_train);
X_test = normcol_equal(X_test);

n_classes = length(unique(y_train));
n_signals = size(X_train, 2);

A_all = cell(1, n_classes);
X_all = cell(1, n_classes);
for i_class = 1:n_classes
    A_all{i_class} = normcol_equal(randn(n_signals / n_classes, n_components));
end

% start waitbar
trainTime = 0;
wb = waitbar(0, '[Kernel AK-SVD] Training...');
ompparams = {'checkdict', 'off'};

% Calculate Kernel matrix
K_all = zeros(n_signals, n_signals);
for row = 1:n_signals
   for col = 1:n_signals
       x = X_train(:,row);
       y = X_train(:,col);
       K_all(row, col) = kernel_function(x, y, sigma);
   end
end

for i_iter = 1:n_iterations
    tmpTime = tic;
    for i_class = 1:n_classes
        A = A_all{i_class};
        K = K_all(y_train==i_class, y_train==i_class);
        
        % coding method
        X_all{i_class} = omp_sparse(A'*K, A'*K*A, n_nonzero_coefs, ompparams{:});
        X = X_all{i_class};
        
        % learning method
        Kc = [];
        for tmp_i_class = 1:n_classes
           if tmp_i_class ~= i_class
              Kc = [Kc K_all(y_train==i_class, y_train==tmp_i_class)*A_all{tmp_i_class}];
           end
        end

        E = eye(size(K, 1)) - A * X;
        for i_atom = 1:size(A, 2)
            [~, atom_usages, ~] = find(X(i_atom,:));
            
            if (isempty(atom_usages))
                A(:, i_atom) = randn(size(A,1), 1);
                A(:, i_atom) = A(:, i_atom) / sqrt(A(:, i_atom)' * K * A(:, i_atom));
            else
                F = E(:, atom_usages) + A(:, i_atom) * X(i_atom, atom_usages);
                a = K * F * X(i_atom, atom_usages)' - 2 * gamma * (Kc * Kc') * A(:, i_atom);
                A(:, i_atom) = a / sqrt(a' * K * a);
                X(i_atom, atom_usages) = F' * K * A(:, i_atom);
                E(:, atom_usages) = F - A(:, i_atom) * X(i_atom, atom_usages);
            end
        end
        
        A_all{i_class} = A;
        X_all{i_class} = X;
    end
    trainTime = trainTime + toc(tmpTime);
    
    % update waitbar
    waitbar(i_iter/n_iterations, wb, sprintf('[Kernel AK-SVD] Training - Remaining time: %d [sec]',...
            round(trainTime/i_iter*(n_iterations - i_iter))));
end

% close waitbar
close(wb);


testTime = 0;
wb = waitbar(0, '[Kernel AK-SVD] Testing...');

% Calculate test Kernel matrix
K_all_test = zeros(n_signals, size(X_test, 2));
for row = 1:n_signals
   for col = 1:size(X_test, 2)
       x = X_train(:,row);
       y = X_test(:,col);
       K_all_test(row, col) = kernel_function(x, y, sigma);
   end
end

Errs = [];
prediction = [];
for i_test = 1:size(X_test, 2)

    tmpTime = tic;
    errs = [];
    k_y_y = kernel_function(X_test(:,i_test), X_test(:,i_test), sigma);
   
    for i_class = 1:n_classes
        A = A_all{i_class};
        K = K_all(y_train==i_class, y_train==i_class);
        k = K_all_test(y_train==i_class, i_test);
        
        x = omp_sparse(A'*k, A'*K*A, n_nonzero_coefs, ompparams{:});
        err = k_y_y - 2*k'*A*x + x'*A'*K*A*x;
        errs = [errs err];
    end
   
    Errs = [Errs; errs];
    [~, index] = min(errs);
    prediction = [prediction index];
    testTime = testTime + toc(tmpTime);
   
   % update waitbar
   waitbar(i_test/size(X_test, 2), wb, sprintf('[Kernel AK-SVD] Testing - Remaining time: %d [sec]',...
           round(testTime/i_test*(size(X_test, 2) - i_test))));
end

% close waitbar
close(wb);

% compute problem accuracy
accuracy = sum(y_test==prediction)/size(X_test,2);

fprintf("[Kernel AK-SVD] Accuracy: %f\n", accuracy);
fprintf('[Kernel AK-SVD] Training time: %f [sec]\n', trainTime);
fprintf('[Kernel AK-SVD] Testing time: %f [sec]\n', testTime);
fprintf('\n');

