function write_gri_from_gmsh(gri_filename, msh)
% Write .gri from Gmsh mesh data.
%
% Inputs:
%   gri_filename : output .gri file name
%   msh          : struct from read_gmsh_v22()
%
% Assumes:
%   msh.nodes  = [nNodes x 3]
%   msh.tet10  = [nTet x 10]
%   msh.tri6   = [nTri x 6] boundary faces

    nodes = msh.nodes;
    elements = msh.tet10;
    boundary_faces_q2 = msh.tri6;

    n_nodes = size(nodes, 1);
    n_elements = size(elements, 1);
    dim = size(nodes, 2);

    assert(dim == 3, 'Unsupported dimension.');

    % --- classify boundary faces by location ---
    face_pts = permute(reshape(nodes(boundary_faces_q2', :), [6, size(boundary_faces_q2,1), 3]), [2 1 3]);

    xq = face_pts(:,:,1);
    yq = face_pts(:,:,2);
    zq = face_pts(:,:,3);

    tol = 1e-6;

    top = max(zq(:));
    bot = min(zq(:));
    right = max(xq(:));
    left = min(xq(:));
    front = max(yq(:));
    back = min(yq(:));

    % Use corner nodes only for planar-face classification
    zc = zq(:,1:3);
    xc = xq(:,1:3);
    yc = yq(:,1:3);

    top_mask = all(abs(zc - top) < tol, 2);
    bottom_mask = all(abs(zc - bot) < tol, 2);

    inlet_side_mask = all(abs(xc - left) < tol, 2);
    outlet_side_mask = all(abs(xc - right) < tol, 2);

    front_mask = all(abs(yc - front) < tol, 2);
    back_mask = all(abs(yc - back) < tol, 2);

    combined_mask = top_mask | bottom_mask | inlet_side_mask | outlet_side_mask | front_mask | back_mask;
    airfoil_mask = ~combined_mask;

    % --- periodic node matching from top/bottom boundary faces ---
    top_nodes = unique(boundary_faces_q2(top_mask, :));
    bottom_nodes = unique(boundary_faces_q2(bottom_mask, :));

    assert(numel(top_nodes) == numel(bottom_nodes), ...
        'Top and bottom periodic node counts do not match.');

    top_xyz = nodes(top_nodes, :);
    bottom_xyz = nodes(bottom_nodes, :);

    tol_match = 1e-8;

    top_xy = round(top_xyz(:,1:2) / tol_match) * tol_match;
    bottom_xy = round(bottom_xyz(:,1:2) / tol_match) * tol_match;

    top_keys = string(top_xy(:,1)) + "," + string(top_xy(:,2));
    bottom_keys = string(bottom_xy(:,1)) + "," + string(bottom_xy(:,2));

    [tf, loc] = ismember(top_keys, bottom_keys);
    assert(all(tf), 'Some top periodic nodes could not be matched to bottom nodes.');

    bottom_nodes = bottom_nodes(loc);
    bottom_xyz = nodes(bottom_nodes, :);

    xy_err = top_xyz(:,1:2) - bottom_xyz(:,1:2);
    max_xy_err = max(abs(xy_err(:)));
    fprintf('Periodic matching passed. Max XY err = %.3e\n', max_xy_err);

    z_diff = top_xyz(:,3) - bottom_xyz(:,3);
    fprintf('Min dz = %.6f, Max dz = %.6f\n', min(z_diff), max(z_diff));

    % --- bottom-face orientation fix for q2 triangles ---
    % Reverse orientation while preserving q2 edge-node ordering
    % [1 2 3 4 5 6] -> [1 3 2 6 5 4]
    bottom_faces_written = boundary_faces_q2(bottom_mask, :);
    bottom_faces_written = bottom_faces_written(:, [1 3 2 6 5 4]);

    % --- write .gri ---
    fid = fopen(gri_filename, 'w');
    assert(fid ~= -1, 'Could not open output file: %s', gri_filename);

    fprintf(fid, '%d %d %d\n', n_nodes, n_elements, dim);

    for i = 1:n_nodes
        fprintf(fid, '%.16g %.16g %.16g\n', nodes(i,1), nodes(i,2), nodes(i,3));
    end

    n_boundary_ids = 7;
    n_boundary_nodes = 6;
    fprintf(fid, '%d\n', n_boundary_ids);

    for i = 1:n_boundary_ids
        switch i
            case 1
                local_boundary_faces = boundary_faces_q2(top_mask, :);
                boundary_type = "Top";
            case 2
                local_boundary_faces = bottom_faces_written;
                boundary_type = "Bottom";
            case 3
                local_boundary_faces = boundary_faces_q2(inlet_side_mask, :);
                boundary_type = "InletSide";
            case 4
                local_boundary_faces = boundary_faces_q2(outlet_side_mask, :);
                boundary_type = "OutletSide";
            case 5
                local_boundary_faces = boundary_faces_q2(front_mask, :);
                boundary_type = "Front";
            case 6
                local_boundary_faces = boundary_faces_q2(back_mask, :);
                boundary_type = "Back";
            case 7
                local_boundary_faces = boundary_faces_q2(airfoil_mask, :);
                boundary_type = "Airfoil";
        end

        n_boundary_face = size(local_boundary_faces, 1);
        fprintf(fid, '%d %d %s\n', n_boundary_face, n_boundary_nodes, boundary_type);

        for j = 1:n_boundary_face
            fprintf(fid, '%d %d %d %d %d %d\n', local_boundary_faces(j,:));
        end
    end

    % element block
    n_element_groups = 1;
    element_order = 2;
    basis_function = 'TriLagrange';

    for i = 1:n_element_groups
        fprintf(fid, '%d %d %s\n', n_elements, element_order, basis_function);
        for j = 1:n_elements
            fprintf(fid, '%d %d %d %d %d %d %d %d %d %d\n', elements(j,:));
        end
    end

    % periodic block
    fprintf(fid, '%d PeriodicGroup\n', 1);
    fprintf(fid, '%d Translational\n', numel(top_nodes));
    for j = 1:numel(top_nodes)
        fprintf(fid, '%d %d\n', bottom_nodes(j), top_nodes(j));
    end

    fclose(fid);
end