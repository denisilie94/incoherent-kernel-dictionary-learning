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
close all;

gamma = 4;
n_components = 40;
n_nonzero_coefs = 20;
n_iterations = 10;

% Load training and testing data
DataPath   = 'YaleB_Jiang';
% DataPath   = 'ar_face_db';
% DataPath   = 'caltech101_db';
load(fullfile('dbs', DataPath));

% Data preprocessing
% y_train = TrLabel;
% y_test = TtLabel;
% X_train = TrData;
% X_test = TtData;

% Dataset properties
n_classes = length(unique(y_train));

X_all = cell(1, n_classes);
D_all = cell(1, n_classes);
for i_class = 1:n_classes
    D_all{i_class} = normcol_equal(randn(size(X_train,1), n_components));
end

% Start waitbar
trainTime = 0;
wb = waitbar(0, '[AK-SVD] Training...');

for i_iter = 1:n_iterations
    tmpTime = tic;
    for i_class = 1:n_classes
        % Coding method
        X_all{i_class} = omp(X_train(:, y_train==i_class), D_all{i_class}, n_nonzero_coefs);
        
        % Learning method
        Y = X_train(:, y_train==i_class);
        D = D_all{i_class};
        X = X_all{i_class};
        E = Y - D * X;
        
        Dc = [];
        for tmp_i_class = 1:n_classes
           if tmp_i_class ~= i_class
              Dc = [Dc D_all{tmp_i_class}];
           end
        end
        
        for i_atom = 1:size(D, 2)
            [~, atom_usages, ~] = find(X(i_atom,:));
            
            if (isempty(atom_usages))
                D(:, i_atom) = randn(size(D,1), 1);
                D(:, i_atom) = D(:, i_atom) / norm(D(:, i_atom));
            else
                F = E(:, atom_usages) + D(:, i_atom) * X(i_atom, atom_usages);
                d = F * X(i_atom, atom_usages)' - 2 * gamma * (Dc * Dc') * D(:, i_atom);
                D(:, i_atom) = d / norm(d);
                X(i_atom, atom_usages) = F' * D(:, i_atom);
                E(:, atom_usages) = F - D(:, i_atom) * X(i_atom, atom_usages); 
            end
        end
        
        D_all{i_class} = D;
        X_all{i_class} = X;
    end
    trainTime = trainTime + toc(tmpTime);
    
    % Update waitbar
    waitbar(i_iter/n_iterations, wb, sprintf('[AK-SVD] Training - Remaining time: %d [sec]',...
            round(trainTime/i_iter*(n_iterations - i_iter))));
end

% Close waitbar
close(wb);


% Start waitbar
testTime = 0;
wb = waitbar(0, '[AK-SVD] Testing...');

Errs = [];
prediction = [];
for i_test = 1:size(X_test, 2)

    tmpTime = tic;
    errs = [];
    for i_class = 1:n_classes
        x = omp(X_test(:, i_test), D_all{i_class}, n_nonzero_coefs);
        errs = [errs norm(X_test(:, i_test) - D_all{i_class} * x)];
    end
   
    Errs = [Errs; errs];
    [~, index] = min(errs);
    prediction = [prediction index];
    testTime = testTime + toc(tmpTime);
   
   % Update waitbar
   waitbar(i_test/size(X_test, 2), wb, sprintf('[AK-SVD] Testing - Remaining time: %d [sec]',...
           round(testTime/i_test*(size(X_test, 2) - i_test))));
end

% Close waitbar
close(wb);

% Compute problem accuracy
accuracy = sum(y_test==prediction)/size(X_test,2);

fprintf("[AK-SVD] Accuracy: %f\n", accuracy);
fprintf('[AK-SVD] Training time: %f [sec]\n', trainTime);
fprintf('[AK-SVD] Testing time: %f [sec]\n', testTime);
fprintf('\n');

