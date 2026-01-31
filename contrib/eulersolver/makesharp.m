function makesharp(ref)
% meshes a sharp 2D object
% The output file is sharp#.gri, where # is ref
% ref is the refinement number (0,1,2,etc.)
% elements are quadrilaterals

% baseline spacing parameters
nx0 = 12;   % in x direction
nr0 = 6;    % outward in r

% x spacing
a = 1; b = 1.7; xmax = 1;
xv = (logspace(a,b,nx0) - 10^a)/(10^b-10^a)*xmax;
xv = spaceq(xv, ref);
nx = length(xv);

% r spacing
a = 1; b = 2;
rv = (logspace(a,b,nr0) - 10^a)/(10^b-10^a);
rv = spaceq(rv, ref);
nr = length(rv);

% displacements to farfield
dv = [-0.05, 0; 0, 0.8];

% number of nodes and elements
nnode = (2*nx-1)*nr;
nelem = 2*(nx-1)*(nr-1);

% open file for writing
fname = sprintf('sharp%d.gri', ref);
fid = fopen(fname, 'w');

fprintf(fid, '%d %d 2\n', nnode, nelem);

%%%%%%%%%%%%%%%%%%
% NODE LIST
%%%%%%%%%%%%%%%%%%
V = zeros(nx,nr,2);
for ir = 1:nr % top half
  for ix = 1:nx
    x = xv(ix); y = ygeom(x); r = rv(ir);
    phix = x/xmax; phir = r;
    x = x + (dv(1,1)*(1-phix) + dv(2,1)*phix)*phir;
    y = y + (dv(1,2)*(1-phix) + dv(2,2)*phix)*phir;
    V(ix,ir,:) = [x,y];
    fprintf(fid, '%20.15f %20.15f\n', x, y);
  end
end
for ir = 1:nr % bottom half
  for ix = 2:nx
    xy = V(ix,ir,:);
    fprintf(fid, '%20.15f %20.15f\n', xy(1), -xy(2));
  end
end



%%%%%%%%%%%%%%%%%%%
% BOUNDARY FACES
%%%%%%%%%%%%%%%%%%%
nbgroup = 4;
fprintf(fid, '%d\n', nbgroup);
nn1 = nx*nr;
nn2 = (nx-1)*nr;

% upper, lower, inflow, outflow

fprintf(fid, '%d 2 upper\n', nx-1);
for i = 1:(nx-1), fprintf(fid, '%10d %10d\n', i, i+1); end

fprintf(fid, '%d 2 lower\n', nx-1);
j=1; for i = 1:(nx-1), k=nn1+i; fprintf(fid, '%10d %10d\n', j,k); j=k; end

fprintf(fid, '%d 2 inflow\n', 2*(nx-1));
for i = 1:(nx-1), fprintf(fid, '%10d %10d\n', nn1+1-i, nn1-i); end
j=nn1-nx+1; for i = 1:(nx-1), k=nn1+nn2-(nx-1)+i; fprintf(fid, '%10d %10d\n', j, k); j=k; end

fprintf(fid, '%d 2 outflow\n', 2*(nr-1));
for i = 1:(nr-1), fprintf(fid, '%10d %10d\n', nx*i, nx*(i+1)); end
for i = 1:(nr-1), fprintf(fid, '%10d %10d\n', nn1+(nx-1)*i, nn1+(nx-1)*(i+1)); end


%%%%%%%%%%%%%%%%%%%
% ELEMENTS
%%%%%%%%%%%%%%%%%%%

% one group
fprintf(fid, '%d %d QuadLagrange\n', nelem, 1);
for ir = 1:(nr-1)
  for ix = 1:(nx-1)
    in = (ir-1)*nx + ix;
    fprintf(fid, '%d %d %d %d\n', in, in+1, in+nx, in+nx+1);
  end
end
for ir = 1:(nr-1)
  fprintf(fid, '%d %d %d %d\n', (ir-1)*nx+1, ir*nx+1, nn1+(ir-1)*(nx-1)+1, nn1+ir*(nx-1)+1);
  for ix = 1:(nx-2)
    in = nn1 + (ir-1)*(nx-1) + ix;
    fprintf(fid, '%d %d %d %d\n', in, in+nx-1, in+1, in+nx);
  end
end



fclose(fid);


%------------------------------------------------
function r = spaceq(re, ref)
nsub = 2^ref;
nre = length(re) - 1;
nr  = nsub*nre;
r = zeros(1, nr+1);
for k=0:nre-1,
  for j = 1:nsub,
    f = (j-1.0)/(nsub);
    r(k*nsub+j) = re(k+1)*(1.0-f) + re(k+2)*f;
  end
end
r(nr+1) = re(nre+1);


%--------------------------------
function [y, y_x] = ygeom(x)
% geometry of object; TODO: put in the correct geometry here
c = 1; y = 0.1*x.*(1-x/c);
y_x = 0.1 - 0.2*x/c;
