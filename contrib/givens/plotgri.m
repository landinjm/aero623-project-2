function plotgri(grifile)

% Read mesh (gri file)
fid = fopen(grifile, 'r');
% Read in nodes
A = fscanf(fid,'%d', 3);
nnode    = A(1);
nelemtot = A(2);
dim      = A(3);
V = zeros(nnode, dim);
for inode = 1:nnode,
  A = fscanf(fid, '%lf', 2);
  V(inode,:) = A(1:2)';
end
% Read through boundary info
A = fscanf(fid, '%d', 1);
nbfgrp = A(1);
for ibfgrp = 1:nbfgrp,
  fgets(fid);
  sline = fgets(fid);
  [nbface, nnode, title] = strread(sline, '%d %d %s');
  for ibface = 1:nbface,
    A = fscanf(fid, '%d', nnode);
  end
end

% Read in elements and plot edges
figure(1); clf; hold on;
curtot = 0;
E2N = zeros(nelemtot, 6);
while (curtot ~= nelemtot),
  fgets(fid);
  sline = fgets(fid);
  [nelem, p, sbasis] = strread(sline, '%d %d %s');
  switch sbasis{1}
    case 'TriLagrange'
      nnode = (p+1)*(p+2)/2;
      nedge = 3;
      fvec = zeros(3,p+1);
      fvec(1,:) = [1:(p+1)];
      v = p+1; d = p;
      for k=1:(p+1), fvec(2,k) = v; v = v+d; d = d-1; end;
      v = 1; d = p+1;
      for k=1:(p+1), fvec(3,k) = v; v = v+d; d = d-1; end;
    otherwise
      error('element type not understood');
  end

  for elem = 1:nelem,
    A = fscanf(fid, '%d', nnode);  % HO nodes included
    for edge=1:nedge,
      plotedge(V(A(fvec(edge,:)),:));
    end
  end
  curtot = curtot + nelem;
end
fclose(fid);
nelem = nelemtot;
axis equal
set(gca, 'XTick', []);
set(gca, 'YTick', []);
set(gca, 'XTickLabel', '');
set(gca, 'YTickLabel', '');
box off; axis off
set(gca, 'LineWidth', 0.5);

%------------------------------------------------
function plotedge(V);
x = V(:,1);
y = V(:,2);
plot(x,y, 'k-', 'linewidth', 1);
