function [p,val] = checkdigit(cam, all_theta)

fprintf('Press any key to take a picture.\n')
pause;
pause(5)
for i = 1:10
    b = snapshot(cam);
end
b = imresize(b, [20,20]);
b = rgb2gray(b);
b = imcomplement(b);
b = mat2gray(b);
imshow(b)
b = b'
b = (b(:))'
[p,val] = predictOneVsAll(all_theta,b);
if(p>0)
    p
else
    0
end
fprintf('%f percent probability', val(p))
val
end