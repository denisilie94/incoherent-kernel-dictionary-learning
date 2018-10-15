clc;
clear;

dir_path = '/media/denisilie94/D/Dictionary Learning/AR Face Database/';
d = uigetdir(dir_path, 'Select a folder');
files = dir(fullfile(d, '*.raw'));

rows = 768;
cols = 576;

for n = 1:length(files)
    filename = strcat(dir_path, '/', files(n).name);
    
    f = fopen(filename, 'r');
    I = fread(f, rows * cols, 'uint8=>uint8');
    Z = reshape(I,rows,cols);
    Z = Z';
    
    imwrite(Z, strrep(filename, 'raw', 'png'));
    fclose(f);
    delete(filename);
end