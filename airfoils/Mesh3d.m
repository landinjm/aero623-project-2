function Mesh3d(filename,hmax)
    % will read an stl file and write into a gri
    model = femodel(Geometry=filename+".stl");
    mesh = generateMesh(model,Hmax=hmax,Hmin=0.001,Hgrad=1.2,GeometricOrder="quadratic");
    Mesh = mesh.Geometry.Mesh;
    disp('Size of mesh.Nodes:')
    disp(size(Mesh.Nodes))

    disp('Size of mesh.Elements:')
    disp(size(Mesh.Elements))
    pdemesh(mesh,FaceAlpha=0.5,ElementLabels="off",NodeLabels="off")
    write_gri_3d(filename+".gri",Mesh.Nodes',Mesh.Elements')
end

% for dbugging from stl:
%   pdegplot(model, 'FaceLabels', 'on', 'FaceAlpha', 0.5)
% for twisted:
%   add Hface={[3,11] 0.5}, into generate mesh
