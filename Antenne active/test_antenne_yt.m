clc;
clear all;

N = 2;
lambda = 1;
d = 0.5;

max_angle = 90;
max_rad = max_angle*pi/180;

b = 2*pi/lambda;
alpha = -b*d*cos(max_rad);

phi = linspace(0,2*pi,1000);
epsi = alpha +b*d*cos(phi);


AF = sin(N*epsi/2) ./ (N*sin(epsi/2));
AF_abs = abs(AF);


E_norm = sin(phi);
pattern = abs(E_norm);


polar(phi,AF_abs.*pattern)



