clc
load('E:\Mycode\WiSPPN\examples\oct17set1_fw_1814.mat', 'csi_serial');
disp(size(csi_serial))
%disp(csi_serial);

load('E:\Mycode\WiSPPN\raw\raw1.mat', 'mat2');
matrix_size = size(mat2);
disp(matrix_size);

% Extract the first 5 time steps
csi = mat2(1:5, :, :, :);
disp(size(csi))
disp(csi);

