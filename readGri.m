function gri = readGri(filename)
% Reads .gri file written by write_gri_from_gmsh().
%
% Output:
%   gri.nodes              [nNodes x 3]
%   gri.elements           [nElements x 10]
%   gri.boundary_groups    struct array with fields:
%                          name, n_faces, n_nodes, faces
%   gri.periodic_pairs     [nPairs x 2]

    fid = fopen(filename, 'r');
    assert(fid ~= -1, 'Could not open file: %s', filename);

    header = textscan(fid, '%d %d %d', 1);
    n_nodes = header{1};
    n_elements = header{2};
    dim = header{3};

    assert(dim == 3, 'Expected 3D .gri file.');

    nodes = fscanf(fid, '%f', [dim, n_nodes]).';

    n_boundary_groups = fscanf(fid, '%d', 1);

    boundary_groups = struct('name', {}, 'n_faces', {}, 'n_nodes', {}, 'faces', {});

    for g = 1:n_boundary_groups
        n_faces = fscanf(fid, '%d', 1);
        n_nodes_per_face = fscanf(fid, '%d', 1);
        name = strtrim(fgetl(fid));

        faces = fscanf(fid, '%d', [n_nodes_per_face, n_faces]).';

        boundary_groups(g).name = name;
        boundary_groups(g).n_faces = n_faces;
        boundary_groups(g).n_nodes = n_nodes_per_face;
        boundary_groups(g).faces = faces;
    end

    elem_header = textscan(fid, '%d %d %s', 1);
    n_elements_in_group = elem_header{1};
    q_order = elem_header{2};
    basis_function = string(elem_header{3});

    assert(n_elements_in_group == n_elements, 'Element count mismatch.');
    assert(q_order == 2, 'Expected q=2 elements.');
    assert(basis_function == "TriLagrange", 'Expected TriLagrange basis.');

    elements = fscanf(fid, '%d', [10, n_elements]).';

    periodic_header = textscan(fid, '%d %s', 1);
    n_periodic_groups = periodic_header{1};
    periodic_label = string(periodic_header{2});

    assert(periodic_label == "PeriodicGroup", 'Expected PeriodicGroup label.');

    periodic_pairs = [];

    for pg = 1:n_periodic_groups
        pg_header = textscan(fid, '%d %s', 1);
        n_pairs = pg_header{1};
        periodic_type = string(pg_header{2});

        assert(periodic_type == "Translational", 'Expected Translational periodicity.');

        pairs = fscanf(fid, '%d', [2, n_pairs]).';
        periodic_pairs = [periodic_pairs; pairs]; %#ok<AGROW>
    end

    fclose(fid);

    gri.n_nodes = n_nodes;
    gri.n_elements = n_elements;
    gri.dim = dim;
    gri.nodes = nodes;
    gri.boundary_groups = boundary_groups;
    gri.elements = elements;
    gri.q_order = q_order;
    gri.basis_function = basis_function;
    gri.n_periodic_groups = n_periodic_groups;
    gri.periodic_pairs = periodic_pairs;
end