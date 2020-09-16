function img2d = projection2d(imgpath, angle, axis)
img = niftiread(imgpath);
img = img(:, :, 1:512);
%volumeViewer(img);

if axis == "z"
    imgr = imrotate3(img, angle, [0 0 1]);
    imgrp = squeeze(sum(imgr, 2));
    imgrps = rescale(imgrp);
    imgrps8 = uint8(255 * imgrps);
    img2d = imgrps8;
    img2d = transpose(img2d);
    %imwrite(img2d, strcat(num2str(angle), ".png"));

elseif axis == "y"
    imgr = imrotate3(img, angle, [0 1 0]);
    imgrp = squeeze(sum(imgr, 2));
    imgrps = rescale(imgrp);
    imgrps8 = uint8(255 * imgrps);
    img2d = imgrps8;
    img2d = transpose(img2d);
    %imwrite(img2d, strcat(num2str(angle), ".png"));
elseif axis == "x"
    imgr = imrotate3(img, angle, [1 0 0]);
    imgrp = squeeze(sum(imgr, 2));
    imgrps = rescale(imgrp);
    imgrps8 = uint8(255 * imgrps);
    img2d = imgrps8;
    img2d = transpose(img2d);
    %imwrite(img2d, strcat(num2str(angle), ".png"));
else
    error("Unrecognized axis")
end 