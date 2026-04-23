clear; clc; close all;

gm = fegeometry("cube.stl");
gm = generateMesh(gm,Hmax=0.1,GeometricOrder="linear");
pdemesh(gm,FaceAlpha=0.5)

% gm = generateMesh(gm,Hmax=0.3, ...
%                   GeometricOrder="linear");
% pdemesh(gm)
