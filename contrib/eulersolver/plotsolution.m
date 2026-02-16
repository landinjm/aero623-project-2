function plotsolution(Mesh,U,sname)
% Makes a contour plot of field data
% Mesh, U = mesh and solution
% sname = name of scalar field to plot (see below)

% Calculate some useful scalars
gamma = 1.4;
q = sqrt(U(:,2).^2 + U(:,3).^2)./U(:,1);
p = (gamma-1)*(U(:,4) - 0.5*U(:,1).*q.^2);
c = sqrt(gamma*p./U(:,1));
M = q./c;

switch lower(sname)
  case 'mach'
    S = M;
  case 'pressure'
    S = p;
  case 'entropy'
    S = p./(U(:,1).^gamma);
end

% Plot final solution
patch('Vertices',Mesh.Node,'Faces',Mesh.Elem,'FaceVertexCData',S,'FaceColor', 'flat','EdgeColor','none');
colorbar; colormap('jet');
xlabel('$x$', 'interpreter', 'latex');
ylabel('$y$', 'interpreter', 'latex');
title(sname, 'interpreter', 'latex');
axis equal;
