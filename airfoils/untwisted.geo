SetFactory("OpenCASCADE");

v() = ShapeFromFile("/Users/tanaythakur/Desktop/University of Michigan/Aero 623/Project4/aero623-project-2/airfoils/NACA9512_0deg_10pc.step");

Printf("USING UPDATED GEO FILE");

bb() = BoundingBox Volume{v()};
Printf("Bounding box: xmin=%g xmax=%g ymin=%g ymax=%g zmin=%g zmax=%g", bb(0), bb(3), bb(1), bb(4), bb(2), bb(5));

Mesh.CharacteristicLengthMin = 32.5;
Mesh.CharacteristicLengthMax = 65;

Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;

Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 2;

Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;

// periodic top/bottom in z-direction
Periodic Surface {3} = {4} Translate {0, 0, 1000};

Physical Volume("Fluid") = {v()};
Physical Surface("Boundary") = Surface{:};

Mesh.Binary = 0;
Mesh.MshFileVersion = 2.2;

Printf("Number of tetrahedra: %g", #ElementsOfType(4));
Printf("Number of 2nd order tetrahedra: %g", #ElementsOfType(11));