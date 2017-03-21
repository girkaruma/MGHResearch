% fileID = fopen('2804506.txt');
% tmp = textscan(fileID, '%s');
% fclose(fileID);
% celldisp(tmp)

1,  info=nii_read_header('2804506.nii')