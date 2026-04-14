function write_grid(filename, nodes, elements)
    % modified write_gri from project 1 to be capable of writing our stl
    % files

    n_nodes = size(nodes, 1);
    n_elements = size(elements, 1);
    dim = size(nodes, 2);

    assert(dim == 3, "Unsupported dimension")

    % Create a triangulation object given the nodes and elements
    tria = triangulation(elements, nodes);

    % Grab the boundary faces
    boundary_faces = freeBoundary(tria);

    % Grab the boundary face points
    % The first dimension is the face pair, the second is all the points,
    % and the last dimension is the spatial dimension.
    boundary_faces_flat = boundary_faces(:);
    boundary_face_points = reshape(nodes(boundary_faces_flat, :), [], dim, dim);

    % Grab the boundary faces from each point set mask. We'll start with
    % the periodic one since that's the most important.
    tol = 1e-3;

    x = boundary_face_points(:,:,1);
    y = boundary_face_points(:,:,2);
    z = boundary_face_points(:,:,3);

    top = max(z,[],'all'); bot = min(z,[],'all');
    right = max(x,[],'all'); left = min(x,[],'all');
    front = max(y,[],'all'); back = min(y,[],'all');

    on_top = abs(z - top) < tol;
    on_bottom = abs(z - bot) < tol;

    on_inlet_side = abs(x - left) < tol;
    on_outlet_side = abs(x - right) < tol;

    on_front = abs(y - front) < tol;
    on_back = abs(y - back) < tol;

    top_mask = sum(on_top, 2) == dim;
    bottom_mask = sum(on_bottom, 2) == dim;

    inlet_side_mask = sum(on_inlet_side, 2) == dim;
    outlet_side_mask = sum(on_outlet_side, 2) == dim;

    front_mask = sum(on_front,2) == dim;
    back_mask = sum(on_back,2) == dim;

    combined_mask = top_mask | bottom_mask | ...
                    inlet_side_mask | outlet_side_mask | ...
                    front_mask | back_mask;

    airfoil_mask = ~combined_mask;

    % Finding the periodic node matches. Because we are in 2D this is as 
    % simple as matching node pairs with the same x coordinate.
    top_nodes = unique(boundary_faces(top_mask, :));
    bottom_nodes = unique(boundary_faces(bottom_mask, :));
    assert(size(top_nodes, 1) == size(bottom_nodes, 1), "Invalid periodicity node matching");

    top_xyz    = nodes(top_nodes, :);
    bottom_xyz = nodes(bottom_nodes, :);

    [~, index_top] = sortrows(top_xyz,[1,2]);
    [~, index_bottom] = sortrows(bottom_xyz,[1,2]);
        
    top_nodes    = top_nodes(index_top);
    bottom_nodes = bottom_nodes(index_bottom);

    top_xyz    = nodes(top_nodes, :);
    bottom_xyz = nodes(bottom_nodes, :);

    % Check that the x positions of the node match 1 to 1. If not, 
    % shift them.
    tol = 1e-12;
    if any(top_xyz(:,1:2)-bottom_xyz(:,1:2)>tol)
        warning('Periodic XY do not match')
    
        % x_top    = nodes(top_nodes, 1);
        % x_bottom = nodes(bottom_nodes, 1);
        % diff_x = abs(x_top - x_bottom);
        % z_top    = nodes(top_nodes, 3);
        % z_bottom = nodes(bottom_nodes, 3);
        % diff_z = abs(z_top - z_bottom);
        % if ~all((diff_x < tol) & (diff_z < tol))
        %     warning('Periodic nodes mismatch in x-coordinates! Shifting them');
        % 
        %     % Compute the average x for each pair
        %     x_avg = (x_top + x_bottom) / 2;
        %     % Replace values
        %     nodes(inlet_top_nodes, 1)    = x_avg;
        %     nodes(inlet_bottom_nodes, 1) = x_avg;
        % end
    end

    % Open an output stream
    fstream = fopen(filename, 'w');

    % Write the header info
    fprintf(fstream, '%d %d %d\n', n_nodes, n_elements, dim);

    % Write the node data
    for i = 1:n_nodes
        fprintf(fstream, '%g ', nodes(i, :));
        fprintf(fstream, '\n');
    end

    % Write the number of unique boundary groups (ids).
    % With our given geometry we have 7 groups
    n_boundary_ids = 7;
    n_linear_nodes = dim;
    fprintf(fstream, '%d\n', n_boundary_ids);
    for i = 1:n_boundary_ids
        switch i
            case 1
                n_boundary_face = sum(top_mask);
                local_boundary_faces = boundary_faces(top_mask, :);
                boundary_type = "Top";
            case 2
                % NOTE: We have to flip the direction here?
                n_boundary_face = sum(bottom_mask);
                local_boundary_faces = flip(boundary_faces(bottom_mask, :), 2);
                boundary_type = "Bottom";
            case 3
                n_boundary_face = sum(inlet_side_mask);
                local_boundary_faces = boundary_faces(inlet_side_mask, :);
                boundary_type = "InletSide";
            case 4
                n_boundary_face = sum(outlet_side_mask);
                local_boundary_faces = boundary_faces(outlet_side_mask, :);
                boundary_type = "OutletSide";
            case 5
                n_boundary_face = sum(front_mask);
                local_boundary_faces = boundary_faces(front_mask, :);
                boundary_type = "Front";
            case 6
                n_boundary_face = sum(back_mask);
                local_boundary_faces = boundary_faces(back_mask, :);
                boundary_type = "Back";
            case 7
                n_boundary_face = sum(airfoil_mask);  
                local_boundary_faces = boundary_faces(airfoil_mask, :);
                boundary_type = "Airfoil";
            otherwise
                assert(false, "How did you get here?");
        end

        fprintf(fstream, '%d %d %s\n', n_boundary_face, n_linear_nodes, boundary_type);

        for j = 1:n_boundary_face
            fprintf(fstream, '%d %d %d\n', local_boundary_faces(j, :));
        end
    end

    % Write the element group info
    % Since MATLAB does all the hard work of order the elements with 
    % counter-clockwise connectivity, we don't have to do anything special
    % here. We also don't have anything special with the elements because
    % everything is trilgrange and first order.
    n_element_groups = 1;
    element_order = 1;
    basis_function = 'TriLagrange';

    for i = 1:n_element_groups
        fprintf(fstream, '%d %d %s\n', n_elements, element_order, basis_function);
        for j = 1:n_elements
            fprintf(fstream, '%d %d %d %d\n', ...
                elements(j,1), elements(j,2), elements(j,3), elements(j,4));
        end
    end

    % Write the periodic mapping info
    n_periodic_groups = 1;
    
    periodicity_type = 'Translational';
    fprintf(fstream, '%d PeriodicGroup\n', n_periodic_groups);
    for i = 1:n_periodic_groups
        switch i
            case 1
                n_periodic_node_pairs = size(top_nodes, 1);
                mappings = [bottom_nodes top_nodes];
            % case 2
            %     n_periodic_node_pairs = size(outlet_top_nodes, 1);
            %     mappings = [outlet_top_nodes outlet_bottom_nodes];
            otherwise
                assert(false, "How did you get here?");
        end
        fprintf(fstream, '%d %s\n', n_periodic_node_pairs, periodicity_type);
            for j = 1:n_periodic_node_pairs
                fprintf(fstream, '%d %d\n', mappings(j, 1), mappings(j, 2));
            end
    end
    
    % Clean up
    fclose(fstream);
end