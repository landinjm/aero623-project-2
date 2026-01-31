function [U, Rhist] = solve(Mesh, Minf, aoa, niter, U0)
% Euler solver
% Mesh = computational mesh
% Minf, aoa = Mach number and angle of attack
% niter = max number of iterations
% U0 = initial condition, or [] to initialize here
% Outputs: U = solved state, Rhist = residual history
% If convergence fails, and iterations are not maxed out, try
% decreasing the CFL or adjusting the IC (see repmat below).

Normals = getNormals(Mesh); % build normals
Uinf = getUinf(Minf, aoa) ; % freestream boundary
U = U0; if isempty(U), U = repmat(getUinf(1.1, aoa), Mesh.nElem, 1); end

% time step loop
CFL = 0.5; Rtol = 1e-6; Rhist = [];
for iiter = 1:niter
  
  % calculate residual
  [R, dtA] = calcRes(Mesh, Normals, Uinf, U);
  
  % check convergence
  if (mod(iiter,10) == 0)
    Rnorm = norm(R); Rhist = [Rhist, Rnorm];
    fprintf(1, 'iiter = %5d, Rnorm = %10.4e\n', iiter, Rnorm);
    if (Rnorm < Rtol), break; end;
  end
  
  % update the state
  dtA = 2*CFL./dtA;
  for k = 1:4, U(:,k) = U(:,k) - dtA.*R(:,k); end;
  
end


%-------------------------------------
function Normals = getNormals(Mesh)
V = Mesh.Node; IE = Mesh.IE; BE = Mesh.BE;
inormal = [V(IE(:,2),2)-V(IE(:,1),2), V(IE(:,1),1)-V(IE(:,2),1)];
ilength = sqrt(inormal(:,1).^2 + inormal(:,2).^2);
Normals.inormal = [inormal(:,1)./ilength, inormal(:,2)./ilength];
Normals.ilength = ilength;

% Calculate outward-pointing bedge normals and lengths
bnormal = [V(BE(:,2),2)-V(BE(:,1),2), V(BE(:,1),1)-V(BE(:,2),1)];
blength = sqrt(bnormal(:,1).^2 + bnormal(:,2).^2);
Normals.bnormal = [bnormal(:,1)./blength, bnormal(:,2)./blength];
Normals.blength = blength;

%-------------------------------------
function [gamma] = getParam()
gamma = 1.4; 

%-------------------------------------
function Uinf = getUinf(Minf,a)
g = getParam();
Uinf = [1, Minf*cos(a), Minf*sin(a), 1./((g-1)*g) + 0.5*Minf^2];

%-------------------------------------
function [R,dtA] = calcRes(Mesh, Normals, Uinf, U)
V = Mesh.Node; E = Mesh.Elem; IE = Mesh.IE; BE = Mesh.BE;
Ne = size(E,1); Ni = size(IE,1); Nb = size(BE,1);
gam = getParam(); % gamma
BN = getBname(Mesh); % boundary names
R = zeros(Ne, 4); % residual
dtA = zeros(Ne, 1); % inverse time step + area combination
% interior-edge flux contributions
for i = 1:Ni,
  eL = IE(i,3); eR = IE(i,4); ilen = Normals.ilength(i);
  [F, smag] = FluxFunction(U(eL,:),U(eR,:),Normals.inormal(i,:),gam);
  R(eL,:) = R(eL,:) + F*ilen;
  R(eR,:) = R(eR,:) - F*ilen;
  dtA(eL) = dtA(eL) + smag*ilen;
  dtA(eR) = dtA(eR) + smag*ilen;
end
% boundary-edge flux contributions
for i = 1:Nb,
  eL = BE(i,3); ib = BE(i,4); blen = Normals.blength(i);
  if (ib == BN.inflow), % inflow: use flux function
    [F, smag] = FluxFunction(U(eL,:),Uinf,Normals.bnormal(i,:),gam);
  elseif (ib == BN.outflow), % outflow: assume supersonic
    [F, smag] = FluxFunction(U(eL,:),U(eL,:),Normals.bnormal(i,:),gam);
  elseif ((ib == BN.upper) || (ib == BN.lower)), % wall
    [F, smag] = WallFlux(U(eL,:),Normals.bnormal(i,:),gam);
  else
    error('unsupported boundary condition');
  end
  R(eL,:) = R(eL,:) + F*blen;
  dtA(eL) = dtA(eL) + smag*blen;
end


%-------------------------------------
function BN = getBname(Mesh)
BN.inflow = -1; BN.outflow = -1; BN.upper = -1; BN.lower = -1;
for i = 1:length(Mesh.B.title)
  switch lower(Mesh.B.title{i})
    case 'inflow'
      BN.inflow = i;
    case 'outflow'
      BN.outflow = i;
    case 'upper'
      BN.upper = i;
    case 'lower'
      BN.lower = i;
    otherwise
      error 'unrecognized boundary name'
  end
end


%---------------------------------------------------
function [F, smag] = FluxFunction(UL, UR, n, gam)
% Calculates Euler flux using the HLLE flux function.  Returns the
% flux (F) and the maximum propagation speed (smag)

rL = UL(1);
uL = UL(2)/rL;
vL = UL(3)/rL;
unL = uL*n(1) + vL*n(2);
qL = sqrt(UL(2)^2 + UL(3)^2)/rL;
pL = (gam-1)*(UL(4) - 0.5*rL*qL^2);
if ((pL<0) || (rL<0)), disp('Non-physical state!'); UL; end
rHL = UL(4) + pL;
cL = sqrt(gam*pL/rL);
sLmin = min(0, unL - cL);
sLmax = max(0, unL + cL);
sLmag = abs(unL) + cL;

FL = zeros(size(UL));
FL(1) = rL*unL;
FL(2) = UL(2)*unL + pL*n(1);
FL(3) = UL(3)*unL + pL*n(2);
FL(4) = rHL*unL;

rR = UR(1);
uR = UR(2)/rR;
vR = UR(3)/rR;
unR = uR*n(1) + vR*n(2);
qR = sqrt(UR(2)^2 + UR(3)^2)/rR;
pR = (gam-1)*(UR(4) - 0.5*rR*qR^2);
if ((pR<0) || (rR<0)), disp('Non-physical state!'); UR; end
rHR = UR(4) + pR;
cR = sqrt(gam*pR/rR);
sRmin = min(0, unR - cR);
sRmax = max(0, unR + cR);
sRmag = abs(unR) + cR;

FR = zeros(size(UR));
FR(1) = rR*unR;
FR(2) = UR(2)*unR + pR*n(1);
FR(3) = UR(3)*unR + pR*n(2);
FR(4) = rHR*unR;

smag = max(sLmag, sRmag);

sLRmin = min( sLmin, sRmin );
sLRmax = max( sLmax, sRmax );

F = 0.5*(FL + FR) - 0.5*(sLRmax+sLRmin)/(sLRmax - sLRmin)*(FR - FL) + sLRmax*sLRmin/(sLRmax - sLRmin)*(UR-UL);


%---------------------------------------------------
function [F, smag] = WallFlux(UL, n, gam)
% Calculates solid wall Euler flux

rL  = UL(1);
uL  = UL(2)/rL;
vL  = UL(3)/rL;
unL = uL*n(1) + vL*n(2);
qL  = sqrt(UL(2)^2 + UL(3)^2)/rL;
utL = sqrt(qL^2 - unL^2);
pL = (gam-1)*(UL(4) - 0.5*rL*utL^2);
rHL = UL(4) + pL;
cL = sqrt(gam*pL/rL);

smag = abs(unL) + cL;

F = zeros(size(UL));
F(2) = pL*n(1);
F(3) = pL*n(2);
