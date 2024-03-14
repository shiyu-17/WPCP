% Specify the path to your JSON file
json_file_path = 'E:\Mycode\WiSPPN\raw\raw1\1.json';

% Load and process the JSON file
fid = fopen(json_file_path); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
val = jsondecode(str);
joints = val.bodies(1).joints;

x = joints(1:3:end);
y = joints(2:3:end);
c = joints(3:3:end);

jointsVector = [x; y; c; c];

jointsMatrix = zeros([17, 17, 4]);

for row = 1:18
    for column = 1:18
        if row == column
            jointsMatrix(row, column, :) = [x(row), y(row), c(row), c(row)];
        else
            jointsMatrix(row, column, :) = [x(row)-x(column), y(row)-y(column), c(row)*c(column), c(row)*c(column)];
        end 
    end
end

% Specify the path to the image file you want to add
image_path = 'E:\Mycode\WiSPPN\raw\raw1\01.jpg';
frame = imread(image_path);

% Load the raw1.mat file
load('E:\Mycode\WiSPPN\raw\raw1.mat', 'mat2');

% Extract the first 5 time steps
csi_serial = mat2(20:25, :, :, :);

% Save only jointsMatrix and jointsVector
save('01.mat', 'jointsVector', 'jointsMatrix','frame', 'csi_serial', '-v7.3');