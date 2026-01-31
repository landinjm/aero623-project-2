function [x, cp, n, I] = getcp(Mesh, U, Minf)
% Obtains the pressure coefficient and normal vectors on walls
% Mesh, U = mesh and solution
% Minf = free-stream Mach number
% x = x-coordinates of points on walls
% cp = pressure coefficient at points
% n = normal vectors, pointing into the airfoil, at points
% I = point id flag: 1 = upper, 2 = lower wall surface

V = Mesh.Node; BE = Mesh.BE; Nb = size(BE,1);
g = 1.4; a = 0.; % angle of attack is not relevant for pinf calc
Uinf = [1, Minf*cos(a), Minf*sin(a), 1./((g-1)*g) + 0.5*Minf^2];
pinf = (g-1)*(Uinf(4) - 0.5*(Uinf(2)^2 + Uinf(3)^2)/Uinf(1));
wall = [];
for i = 1:length(Mesh.B.title)
  if (strcmp(lower(Mesh.B.title{i}), 'upper') == 1), wall = [wall,i]; end;
  if (strcmp(lower(Mesh.B.title{i}), 'lower') == 1), wall = [wall,i]; end;
end

% boundary-edge flux contributions
x = []; cp = []; n = zeros(Nb,2); j = 0; I = [];
for i = 1:Nb,
  eL = BE(i,3); ib = BE(i,4);
  if ismember(ib,wall)
    j = j+1;
    Ub = U(eL,:);
    p = (g-1)*(Ub(4) - 0.5*(Ub(2)^2 + Ub(3)^2)/Ub(1));
    cp = [cp, 2/(g*Minf^2)*(p/pinf - 1)];
    X = V(BE(i,1:2),:); dX = X(2,:)-X(1,:);
    x = [x, 0.5*sum(X(:,1))];
    n(j,:) = [dX(2), -dX(1)];
    if (ib==wall(1)), I(j) = 1; else I(j) = 2; end;
  end
end
n = n(1:j,:);
