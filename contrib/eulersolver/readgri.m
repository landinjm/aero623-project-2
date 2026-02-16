function Mesh = readgri(grifile)
% function Mesh = readgri(grifile)
%
% This function reads a text .gri file into a data structure called
% Mesh.
%
% INPUTS:
%   grifile = name of .gri file
%
% OUTPUTS:
%   Mesh = data structure:
%    Mesh.dim   = dimension
%    Mesh.nNode = number of nodes
%    Mesh.Node  = nNode x dim array of node coordinates
%    Mesh.nElem  = number of elements
%    Mesh.B = boundary information
%         B.nbfgrp = number of boundary face groups
%         B.nbface = [nbfgrp] number of faces in each group
%         B.nnode = [nbfgrp] number of nodes per face in each group
%         B.title = [nbfgrp] title of each group
%         B.nodes = cell array of boundary nodes
%    Mesh.QBasis = basis type for elements
%    Mesh.QOrder = order type for elements
%    Mesh.Elem   = array of element node numbers (1-based)
%    Mesh.IE = [niedge x 4] array giving (n1, n2, elem1, elem2)
%              information for each interior edge
%    Mesh.BE = [nbedge x 4] array giving (n1, n2, elem, bgroup)
%              information for each boundary edge
%

% initialize Mesh structure
Mesh = struct;

% open the file for writing
fid = fopen(grifile, 'r');

% Read in nodes
A = fscanf(fid,'%d', 3);
Mesh.nNode    = A(1);
nelemtot = A(2);
Mesh.Dim = A(3);
Mesh.Node = zeros(Mesh.nNode, Mesh.Dim);
for inode = 1:Mesh.nNode,
  A = fscanf(fid, '%lf', Mesh.Dim);
  Mesh.Node(inode,:) = A(1:Mesh.Dim)';
end

% Read boundary info
A = fscanf(fid, '%d', 1);
B.nbfgrp = A(1);
B.nbface = zeros(B.nbfgrp,1);
B.nnode  = zeros(B.nbfgrp,1);
B.title  = cell(B.nbfgrp,1);
B.nodes  = cell(B.nbfgrp,1);
for ibfgrp = 1:B.nbfgrp,
  fgets(fid);
  sline = fgets(fid);
  [B.nbface(ibfgrp), B.nnode(ibfgrp), B.title(ibfgrp)] = strread(sline, '%d %d %s');
  N = zeros(B.nbface(ibfgrp), B.nnode(ibfgrp));
  for ibface = 1:B.nbface(ibfgrp),
    A = fscanf(fid, '%d', B.nnode(ibfgrp));
    N(ibface,:) = A';
  end
  B.nodes{ibfgrp} = N;
end
Mesh.B = B;

% Read in elements
fgets(fid);
sline = fgets(fid);
[nelem, p, sbasis] = strread(sline, '%d %d %s');
Mesh.nElem = nelem;
Mesh.QBasis = sbasis{1};
Mesh.QOrder = p;
switch sbasis{1}
  case 'TriLagrange'
    nnode = (p+1)*(p+2)/2;
  case 'QuadLagrange'
    nnode = (p+1)*(p+1);
  otherwise
    error('element type not understood');
end

E = zeros(nelem, nnode);

for elem = 1:nelem,
  E(elem,:) = fscanf(fid, '%d', nnode);
end

% reorder quads to be counterclockwise
if (size(E,2)==4), E = [E(:,1:2), E(:,4), E(:,3)]; end

Mesh.Elem = E;

% close file
fclose(fid);

% perform an edge hash to identify interior and boundary edges
nelem = size(E,1);            % number of elements
nnode = max(max(E));          % number of nodes
H = sparse(nnode,nnode);        % Create a hash list to identify edges
IE = zeros(ceil(nelem*4/2), 4); % (over) allocate interior edge array
niedge = 0;


% Loop over elements and identify all edges
for elem = 1:nelem,
  nv = E(elem,:);
  lnv = length(nv);
  for edge = 1:lnv,
    n1 = nv(edge);
    n2 = nv(mod(edge,lnv)+1);
    if (H(n2,n1) == 0),
      H(n1,n2) = elem;
    else
      elemR = H(n2,n1);
      niedge = niedge+1;
      IE(niedge,:) = [n1,n2, elem, elemR];
      H(n2,n1) = 0;
    end
  end
end

Mesh.IE = IE(1:niedge,:);  % clip IE

% find boundary edges
if isempty(B.nodes)
  [I,J] = find(triu(H)>0);
  BE = [I, J, zeros(size(I)), zeros(size(I))];
  for b = 1:size(I,1), BE(b, 3) = H(I(b),J(b)); end;
else
  nb0 = 0; nb = 0;
  for g = 1:length(B.nodes), nb0 = nb0 + size(B.nodes{g}, 1); end;
  BE = zeros(nb0, 4);
  for g = 1:length(B.nodes)
    Bi = B.nodes{g};
    for b = 1:size(Bi,1)
      n1 = Bi(b,1); n2 = Bi(b,2);
      if (H(n1,n2) == 0), nt = n1; n1 = n2; n2 = nt; end
      nb = nb + 1;
      BE(nb,:) = [n1, n2, H(n1,n2), g];
    end
  end
end

Mesh.BE = BE;
