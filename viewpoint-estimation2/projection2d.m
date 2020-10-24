function fov = projection2d(imgpath, theta, tx, ty, axis)
img = niftiread(imgpath);
img = img(:, :, 1:512);
%volumeViewer(img);

if axis == "z"
    imgr = imrotate3(img, theta, [0 0 1]);
    imgrp = squeeze(sum(imgr, 2));
    imgrps = rescale(imgrp);
    imgrps8 = uint8(255 * imgrps);
    img2d = transpose(imgrps8);
    fov = img2d(55+ty:454+ty, 64+tx:463+tx);
    %imwrite(img2d, strcat(num2str(theta), ".png"));

elseif axis == "y"
    imgr = imrotate3(img, theta, [0 1 0]);
    imgrp = squeeze(sum(imgr, 2));
    imgrps = rescale(imgrp);
    imgrps8 = uint8(255 * imgrps);
    img2d = transpose(imgrps8);
    fov = img2d(55+ty:454+ty, 64+tx:463+tx);
    %imwrite(img2d, strcat(num2str(theta), ".png"));

elseif axis == "x"
    imgr = imrotate3(img, theta, [1 0 0]);
    imgrp = squeeze(sum(imgr, 2));
    imgrps = rescale(imgrp);
    imgrps8 = uint8(255 * imgrps);
    img2d = transpose(imgrps8);
    fov = img2d(55+ty:454+ty, 64+tx:463+tx);
    %imwrite(img2d, strcat(num2str(theta), ".png"));
else
    error("Unrecognized axis")
end 
