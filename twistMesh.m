clc; clear all;
gri = readGri("/Users/tanaythakur/Desktop/University of Michigan/Aero 623/Project4/aero623-project-2/untwisted.gri");

% sanity checks
size(gri.nodes)
size(gri.elements)
size(gri.periodic_pairs)

for g = 1:numel(gri.boundary_groups)
    fprintf('%s: %d faces, %d nodes/face\n', gri.boundary_groups(g).name, gri.boundary_groups(g).n_faces, gri.boundary_groups(g).n_nodes);
end

%% Mesh twisting

nodes0 = gri.nodes;
nodes = nodes0;

% freeze every boundary node except airfoil
fixed_names = ["Top","Bottom","InletSide","OutletSide","Front","Back"];

fixed_nodes = [];

for g = 1:numel(gri.boundary_groups)
    if any(strcmp(gri.boundary_groups(g).name, fixed_names))
        fixed_nodes = [fixed_nodes; gri.boundary_groups(g).faces(:)];
    end
end

fixed_nodes = unique(fixed_nodes);

all_nodes = (1:gri.n_nodes).';
moving_nodes = setdiff(all_nodes, fixed_nodes);

% coordinates
x = nodes(:,1);
y = nodes(:,2);
z = nodes(:,3);

% twist parameters
theta_max = deg2rad(9.5);

% choose twist axis/center: adjust these after checking geometry
xc = 500;     % mid-chord if chord is 1000
yc = -50;    % mid-height-ish
zmin = min(z);
zmax = max(z);

% twist angle varies along z
eta = (z - zmin) / (zmax - zmin);
theta = theta_max * (eta - 0.5);

fprintf("Theta stats (radians): min = %.6e, max = %.6e\n", min(theta), max(theta));
fprintf("Theta stats (degrees): min = %.6f, max = %.6f\n", rad2deg(min(theta)), rad2deg(max(theta)));

zvals = gri.nodes(:,3);

figure;
scatter(zvals, rad2deg(theta), 5, 'filled');
xlabel('z');
ylabel('theta (deg)');
title('Twist distribution along span');
grid on;

% only twist interior nodes for first test
idx = moving_nodes;

X = x(idx) - xc;
Y = y(idx) - yc;
ct = cos(theta(idx));
st = sin(theta(idx));

nodes(idx,1) = xc + X .* ct - Y .* st;
nodes(idx,2) = yc + X .* st + Y .* ct;

gri.nodes = nodes;

write_gri_struct("twisted_from_untwisted.gri", gri);

%% Twist check outputs
pairs = gri.periodic_pairs;
p1 = pairs(:,1);
p2 = pairs(:,2);

offsets = gri.nodes(p2,:) - gri.nodes(p1,:);
fprintf("Periodic offset min/max:\n");
disp([min(offsets); max(offsets)]);

check_tet_volumes(gri)

%% visual inspection 
gri0 = readGri("untwisted.gri");
griT = readGri("twisted_from_untwisted.gri");

airfoil_group_id = find(strcmp({gri0.boundary_groups.name}, "Airfoil"));

airfoil_faces = gri0.boundary_groups(airfoil_group_id).faces;
airfoil_nodes = unique(airfoil_faces(:));

p0 = gri0.nodes(airfoil_nodes,:);
pT = griT.nodes(airfoil_nodes,:);

figure;
scatter3(p0(:,1), p0(:,2), p0(:,3), 10, 'filled');
hold on;
scatter3(pT(:,1), pT(:,2), pT(:,3), 10, 'filled');
axis equal;
grid on;
xlabel('x');
ylabel('y');
zlabel('z');
legend('Untwisted airfoil nodes','Twisted airfoil nodes');
title('Airfoil boundary nodes before and after twist');


disp_mag = vecnorm(pT - p0, 2, 2);

figure;
scatter3(p0(:,1), p0(:,2), p0(:,3), 20, disp_mag, 'filled');
axis equal;
grid on;
colorbar;
xlabel('x');
ylabel('y');
zlabel('z');
title('Airfoil node displacement magnitude from twist');

top_id = find(strcmp({gri0.boundary_groups.name}, "Top"));
bottom_id = find(strcmp({gri0.boundary_groups.name}, "Bottom"));

top_nodes = unique(gri0.boundary_groups(top_id).faces(:));
bottom_nodes = unique(gri0.boundary_groups(bottom_id).faces(:));

top_motion = max(vecnorm(griT.nodes(top_nodes,:) - gri0.nodes(top_nodes,:), 2, 2));
bottom_motion = max(vecnorm(griT.nodes(bottom_nodes,:) - gri0.nodes(bottom_nodes,:), 2, 2));

fprintf("Max top boundary motion = %.3e\n", top_motion);
fprintf("Max bottom boundary motion = %.3e\n", bottom_motion);
airfoil_motion = max(vecnorm(gri.nodes(airfoil_nodes,:) - gri0.nodes(airfoil_nodes,:), 2, 2));
fprintf("Max airfoil motion = %.3e\n", airfoil_motion);

%% plot a closer section
gri0 = readGri("untwisted.gri");
griT = readGri("twisted_from_untwisted.gri");

airfoil_id = find(strcmp({gri0.boundary_groups.name}, "Airfoil"));
airfoil_nodes = unique(gri0.boundary_groups(airfoil_id).faces(:));

p0 = gri0.nodes(airfoil_nodes,:);
pT = griT.nodes(airfoil_nodes,:);

zvals = p0(:,3);
zmin = min(zvals);
zmax = max(zvals);
zmid = 0.5*(zmin + zmax);

slice_z = [zmin, zmid, zmax];
tol = 20;

figure;
hold on;
grid on;
axis equal;

for k = 1:length(slice_z)
    mask = abs(zvals - slice_z(k)) < tol;

    scatter(p0(mask,1), p0(mask,2), 20, 'filled');
    scatter(pT(mask,1), pT(mask,2), 20, 'filled');
end

xlabel('x');
ylabel('y');
legend('Untwisted','Twisted');
title('Airfoil slices before and after twist');

scale = 20;

figure;
scatter3(p0(:,1), p0(:,2), p0(:,3), 10, 'filled');
hold on;
quiver3(p0(:,1), p0(:,2), p0(:,3), ...
        scale*(pT(:,1)-p0(:,1)), ...
        scale*(pT(:,2)-p0(:,2)), ...
        scale*(pT(:,3)-p0(:,3)), ...
        0);
axis equal;
grid on;
xlabel('x');
ylabel('y');
zlabel('z');
title('Amplified airfoil displacement vectors');

%% plot full twisted mesh
griT = readGri("twisted_from_untwisted.gri");

figure;
hold on;
axis equal;
grid on;
xlabel('x');
ylabel('y');
zlabel('z');
title('Twisted mesh boundary surface');

for g = 1:numel(griT.boundary_groups)
    faces6 = griT.boundary_groups(g).faces;
    faces3 = faces6(:,1:3);   % use corner nodes only for plotting
    trisurf(faces3, griT.nodes(:,1), griT.nodes(:,2), griT.nodes(:,3), ...
        'FaceAlpha', 0.25, 'EdgeColor', 'k');
end

view(3);

figure;
hold on;
axis equal;
grid on;
xlabel('x');
ylabel('y');
zlabel('z');
title('Twisted airfoil surface only');

airfoil_id = find(strcmp({griT.boundary_groups.name}, "Airfoil"));
faces6 = griT.boundary_groups(airfoil_id).faces;
faces3 = faces6(:,1:3);

trisurf(faces3, griT.nodes(:,1), griT.nodes(:,2), griT.nodes(:,3), ...
    'FaceAlpha', 0.8, 'EdgeColor', 'k');

view(3);

%% compare twist and untwist airfoil
gri0 = readGri("untwisted.gri");
griT = readGri("twisted_from_untwisted.gri");

airfoil_id = find(strcmp({gri0.boundary_groups.name}, "Airfoil"));
faces3 = gri0.boundary_groups(airfoil_id).faces(:,1:3);

figure;
hold on;
axis equal;
grid on;
xlabel('x');
ylabel('y');
zlabel('z');
title('Untwisted vs twisted airfoil surface');

h1 = trisurf(faces3, gri0.nodes(:,1), gri0.nodes(:,2), gri0.nodes(:,3), ...
    'FaceColor', 'blue', 'FaceAlpha', 0.25, 'EdgeColor', 'none');

h2 = trisurf(faces3, griT.nodes(:,1), griT.nodes(:,2), griT.nodes(:,3), ...
    'FaceColor', 'red', 'FaceAlpha', 0.6, 'EdgeColor', 'k');

legend([h1 h2], {'Untwisted','Twisted'});

view(3);

%% more chcecks
% displacement magnitude for all nodes
d = vecnorm(griT.nodes - gri0.nodes, 2, 2);

fprintf("Max node displacement = %.6e\n", max(d));
fprintf("Mean node displacement = %.6e\n", mean(d));

figure;
scatter3(gri0.nodes(:,1), gri0.nodes(:,2), gri0.nodes(:,3), 8, d, 'filled');
axis equal; grid on; colorbar;
xlabel('x'); ylabel('y'); zlabel('z');
title('Node displacement magnitude from twist');

airfoil_id = find(strcmp({gri0.boundary_groups.name}, "Airfoil"));
airfoil_nodes = unique(gri0.boundary_groups(airfoil_id).faces(:));

d_airfoil = vecnorm(griT.nodes(airfoil_nodes,:) - gri0.nodes(airfoil_nodes,:), 2, 2);

fprintf("Min airfoil displacement = %.6e\n", min(d_airfoil));
fprintf("Max airfoil displacement = %.6e\n", max(d_airfoil));
fprintf("Mean airfoil displacement = %.6e\n", mean(d_airfoil));

boundary_nodes = [];
for g = 1:numel(gri0.boundary_groups)
    boundary_nodes = [boundary_nodes; gri0.boundary_groups(g).faces(:)];
end
boundary_nodes = unique(boundary_nodes);

interior_nodes = setdiff((1:gri0.n_nodes).', boundary_nodes);

d_interior = vecnorm(griT.nodes(interior_nodes,:) - gri0.nodes(interior_nodes,:), 2, 2);

fprintf("Max interior displacement = %.6e\n", max(d_interior));
fprintf("Mean interior displacement = %.6e\n", mean(d_interior));


%% Helper functions
function write_gri_struct(filename, gri)
    fid = fopen(filename, 'w');
    assert(fid ~= -1, 'Could not open output file: %s', filename);

    fprintf(fid, '%d %d %d\n', gri.n_nodes, gri.n_elements, gri.dim);

    for i = 1:gri.n_nodes
        fprintf(fid, '%.16g %.16g %.16g\n', gri.nodes(i,1), gri.nodes(i,2), gri.nodes(i,3));
    end

    fprintf(fid, '%d\n', numel(gri.boundary_groups));

    for g = 1:numel(gri.boundary_groups)
        bg = gri.boundary_groups(g);
        fprintf(fid, '%d %d %s\n', bg.n_faces, bg.n_nodes, bg.name);
        for j = 1:bg.n_faces
            fprintf(fid, '%d %d %d %d %d %d\n', bg.faces(j,:));
        end
    end

    fprintf(fid, '%d %d %s\n', gri.n_elements, gri.q_order, gri.basis_function);

    for i = 1:gri.n_elements
        fprintf(fid, '%d %d %d %d %d %d %d %d %d %d\n', gri.elements(i,:));
    end

    fprintf(fid, '%d PeriodicGroup\n', gri.n_periodic_groups);
    fprintf(fid, '%d Translational\n', size(gri.periodic_pairs,1));
    for i = 1:size(gri.periodic_pairs,1)
        fprintf(fid, '%d %d\n', gri.periodic_pairs(i,1), gri.periodic_pairs(i,2));
    end

    fclose(fid);
end

function check_tet_volumes(gri)
    nodes = gri.nodes;
    elems = gri.elements(:,1:4);

    n = size(elems,1);
    vols = zeros(n,1);

    for i = 1:n
        p1 = nodes(elems(i,1),:);
        p2 = nodes(elems(i,2),:);
        p3 = nodes(elems(i,3),:);
        p4 = nodes(elems(i,4),:);

        J = [p2-p1; p3-p1; p4-p1]';
        vols(i) = det(J)/6;
    end

    fprintf("Min signed volume = %.6e\n", min(vols));
    fprintf("Max signed volume = %.6e\n", max(vols));
    fprintf("Number of inverted/zero tets = %d\n", sum(vols <= 0));
end