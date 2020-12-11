function fov = views(imgpath)
img = niftiread(imgpath);
img = img(:, :, 1:500);

rotationAngles = 0 :10 :360;
for angle = rotationAngles
    imgr = imrotate3(img, -angle, [0 0 1], "crop");
    imgrp = squeeze(sum(imgr, 2));
    imgrps = rescale(imgrp);
    imgrps8 = uint8(255 * imgrps);
    fov = imgrps8(55:454, 64:463);
    imwrite(transpose(fov), strcat("rz/", num2str(angle), ".png"));
end
