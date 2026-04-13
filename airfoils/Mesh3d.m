function Mesh3d(filename,hmax)
    % will read an stl file and write into a gri
    model = femodel(Geometry=filename+".stl");
    mesh = generateMesh(model,Hmax=hmax,Hmin=0.001,Hgrad=1.2,GeometricOrder="linear");
    Mesh = mesh.Geometry.Mesh;
    pdemesh(mesh,FaceAlpha=0.5,ElementLabels="off",NodeLabels="off")
    write_gri_3d(filename+".gri",Mesh.Nodes',Mesh.Elements')
end