v_features_dim = 128;

tic;
clc;
clearvars -except v_features_dim;
% AR Face Database
dir_path = '/media/denisilie94/D/Dictionary Learning/AR Face Database/';
files = dir(fullfile(dir_path, '*.png'));
n = length(files);

ar_labels   = zeros(1, n);
ar_features = zeros(v_features_dim, n);


for i = 1:n
    
    if files(i).name(1) == 'm'
        ar_labels(i) = 100 + str2double(files(i).name(3:5));
    else
        ar_labels(i) = 100 + str2double(files(i).name(3:5));
    end
    
    filename = strcat(dir_path, '/', files(i).name);
    img = imread(filename);
    [rows, cols] = size(img);
    
    ar_features(:,i) = extract_features(img, rows, cols, v_features_dim);
    fprintf('AR database %d / %d completed\n', i, n);
end

save('ar_features.mat','ar_features', 'ar_labels')




% -------------------------------------------------------------------------

clc;
clearvars -except v_features_dim;
% Caltech101 Database
dir_path = '/media/denisilie94/D/Dictionary Learning/Caltech101/101_ObjectCategories';
dirFolders = dir(dir_path);
n = length(dirFolders);

position    = 0;
label       = 0;
no_of_elems = 8677;

caltech101_labels   = zeros(1, no_of_elems);
caltech101_features = zeros(v_features_dim, no_of_elems);


for i = 1:n
    
    if (~strcmp(dirFolders(i).name, '.') && ~strcmp(dirFolders(i).name, '..'))
        tmp_dir_path = strcat(dir_path, '/', dirFolders(i).name);     
        files = dir(fullfile(tmp_dir_path, '*.jpg'));
                
        for j = 1:length(files)
            
            caltech101_labels(position) = label;        
            
            filename = strcat(tmp_dir_path, '/', files(j).name);
            img = imread(filename);
            [rows, cols] = size(img);

            caltech101_features(:,position) = extract_features(img, rows, cols, v_features_dim);
            fprintf('Caltech101 database %d / %d folders completed with %d / %d img completed\n', i - 2, n - 2, j, length(files));
            
            position = position + 1;
         end
        
        label = label + 1;
    end  
end

save('caltech101_features.mat','caltech101_features', 'caltech101_labels')




% -------------------------------------------------------------------------

clc;
clearvars -except v_features_dim;
% Caltech256 Database
dir_path = '/media/denisilie94/D/Dictionary Learning/Caltech256';
dirFolders = dir(dir_path);

position    = 0;
no_of_elems = 30608;

caltech256_labels   = zeros(1, no_of_elems);
caltech256_features = zeros(v_features_dim, no_of_elems);

for i = 1:length(dirFolders)
    
    if (~strcmp(dirFolders(i).name, '.') && ~strcmp(dirFolders(i).name, '..'))
        tmp_dir_path = strcat(dir_path, '/', dirFolders(i).name);     
        files = dir(fullfile(tmp_dir_path, '*.jpg'));
        
        for j = 1:length(files)
            
            caltech256_labels(position) = str2double(dirFolders(i).name(1:3));
            
            filename = strcat(tmp_dir_path, '/', files(j).name);
            img = imread(filename);
            [rows, cols] = size(img);

            caltech256_features(:,position) = extract_features(img, rows, cols, v_features_dim);
            fprintf('Caltech256 database %d / %d folders completed with %d / %d img completed\n', i - 2, length(dirFolders) - 2, j, length(files));
            
            position = position + 1;
        end
    end  
end

save('caltech256_features.mat','caltech256_features', 'caltech256_labels')




% -------------------------------------------------------------------------

clc;
clearvars -except v_features_dim;
% YaleB Cropped Database
dir_path = '/media/denisilie94/D/Dictionary Learning/Extended Yale B/CroppedYale';
dirFolders = dir(dir_path);

for i = 1:length(dirFolders)
    
    if (~strcmp(dirFolders(i).name, '.') && ~strcmp(dirFolders(i).name, '..'))
        tmp_dir_path = strcat(dir_path, '/', dirFolders(i).name);     
        files = dir(fullfile(tmp_dir_path, '*.pgm'));
        
        for j = 1:length(files)
            
            yaleBcropped_label(j) = str2double(dirFolders(3).name(end-1:end));
            
            filename = strcat(tmp_dir_path, '/', files(j).name);
            img = imread(filename);
            [rows, cols] = size(img);

            yaleBcropped_features(:,j) = extract_features(img, rows, cols, v_features_dim);
            fprintf('YaleB cropped database %d / %d folders completed with %d / %d img completed\n', i - 2, length(dirFolders) - 2, j, length(files));
        end
    end  
end

save('yaleBcropped_features.mat','yaleBcropped_features', 'yaleBcropped_labels')




% -------------------------------------------------------------------------

clc;
clearvars -except v_features_dim;
% YaleB Database
dir_path = '/media/denisilie94/D/Dictionary Learning/Extended Yale B/ExtendedYaleB';
dirFolders = dir(dir_path);

for i = 1:length(dirFolders)
    
    if (~strcmp(dirFolders(i).name, '.') && ~strcmp(dirFolders(i).name, '..'))
        tmp_dir_path = strcat(dir_path, '/', dirFolders(i).name);     
        files = dir(fullfile(tmp_dir_path, '*.pgm'));
        
        for j = 1:length(files)
            
            yaleB_label(j) = str2double(dirFolders(3).name(end-1:end));
            
            filename = strcat(tmp_dir_path, '/', files(j).name);
            img = imread(filename);
            [rows, cols] = size(img);

            yaleB_features(:,j) = extract_features(img, rows, cols, v_features_dim);
            fprintf('YaleB database %d / %d folders completed with %d / %d img completed\n', i - 2, length(dirFolders) - 2, j, length(files));
        end
    end  
end

save('yaleB_features.mat','yaleB_features', 'yaleB_labels')
toc;

