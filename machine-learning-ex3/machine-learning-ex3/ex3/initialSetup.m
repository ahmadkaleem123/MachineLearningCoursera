function[f] = initialSetup()
rpi = raspi('192.168.0.101','pi','raspberry');
ard = arduino('COM3', 'Uno', 'Libraries', 'Servo');
s1= servo(rpi, 18, 'MinPulseDuration', 600*10^-6, 'MaxPulseDuration', 2300*10^-6);
s2 = servo(ard,'D9', 'MinPulseDuration', 544*10^-6, 'MaxPulseDuration', 2400*10^-6);
cam = cameraboard(rpi,'Resolution','1280x720','FrameRate',30);
cam.HorizontalFlip = true;
cam.VerticalFlip = true;
cam.Brightness = 65;
cam.Contrast = 50;
system(rpi, 'sudo shutdown -h now');
end