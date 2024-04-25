% Specify the path to your JSON file
json_file_path = 'E:\dataset\pic_960\keypoints.json';
output_folder = 'E:\dataset\pic_960\keypoints';
% Load and process the JSON file
try
    fid = fopen(json_file_path);
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    val = jsondecode(str);
catch
    error('Error loading or decoding JSON file.');
end

% Loop through all annotations
annotations = val.annotations;
for i = 1:numel(annotations)
    annotation = annotations{i};  % 使用大括号索引
    keypoints = annotation.keypoints;
    pose = annotation.pose;

    % Construct joints vector and matrix
    x = keypoints(1:3:end);
    y = keypoints(2:3:end);
    c = keypoints(3:3:end);
    jointsVector = [x; y; c; c];

    jointsMatrix = zeros([17, 17, 4]);
    for row = 1:17
        for column = 1:17
            if row == column
                jointsMatrix(row, column, :) = [x(row), y(row), c(row), c(row)];
            else
                jointsMatrix(row, column, :) = [x(row)-x(column), y(row)-y(column), c(row)*c(column), c(row)*c(column)];
            end
        end
    end

    % Specify the image path based on the pose
    %image_path = strcat('E:\dataset\pic_960\csi_pic', pose, '.jpg');
    %frame = imread(image_path);

    % Load the raw1.mat file
    %load('E:\Mycode\WiSPPN\raw\raw1.mat', 'mat2');

    % Extract the first 5 time steps
    %csi_serial = mat2(20:25, :, :, :);

    % Save only jointsMatrix and jointsVector
    save(fullfile(output_folder, strcat(pose, '.mat')),'jointsVector', 'jointsMatrix', '-v7.3');
end
