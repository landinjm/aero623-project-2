function msh = read_gmsh_v22(filename)
% Reads an ASCII Gmsh v2.2 .msh file.
%
% Returns struct with:
%   msh.nodes          [nNodes x 3]
%   msh.tet10          [nTet x 10]
%   msh.tri6           [nTri x 6]
%   msh.tri6_phys      [nTri x 1] physical tag for each tri6 face
%   msh.tet10_phys     [nTet x 1] physical tag for each tet10 element

    fid = fopen(filename, 'r');
    assert(fid ~= -1, 'Could not open file: %s', filename);

    msh.nodes = [];
    msh.tet10 = [];
    msh.tri6 = [];
    msh.tri6_phys = [];
    msh.tet10_phys = [];

    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if ~ischar(line)
            break;
        end

        switch line
            case '$MeshFormat'
                fmt = sscanf(fgetl(fid), '%f %d %d');
                version = fmt(1);
                fileType = fmt(2);
                assert(abs(version - 2.2) < 1e-12, 'Expected Gmsh v2.2 file.');
                assert(fileType == 0, 'Expected ASCII .msh file (not binary).');
                endLine = strtrim(fgetl(fid));
                assert(strcmp(endLine, '$EndMeshFormat'), 'Malformed $MeshFormat section.');

            case '$Nodes'
                nNodes = sscanf(fgetl(fid), '%d');
                nodes = zeros(nNodes, 3);
                for i = 1:nNodes
                    vals = sscanf(fgetl(fid), '%d %f %f %f');
                    nodes(i, :) = vals(2:4).';
                end
                msh.nodes = nodes;
                endLine = strtrim(fgetl(fid));
                assert(strcmp(endLine, '$EndNodes'), 'Malformed $Nodes section.');

            case '$Elements'
                nElem = sscanf(fgetl(fid), '%d');

                tet10 = [];
                tet10_phys = [];
                tri6 = [];
                tri6_phys = [];

                for i = 1:nElem
                    vals = sscanf(fgetl(fid), '%d');
                    elemType = vals(2);
                    nTags = vals(3);
                    tags = vals(4:3+nTags);
                    conn = vals(4+nTags:end);

                    physTag = 0;
                    if ~isempty(tags)
                        physTag = tags(1);
                    end

                    switch elemType
                        case 11 % 10-node tetrahedron
                            assert(numel(conn) == 10, 'Unexpected tet10 connectivity length.');
                            tet10(end+1, :) = conn(:).'; %#ok<AGROW>
                            tet10_phys(end+1, 1) = physTag; %#ok<AGROW>

                        case 9 % 6-node second-order triangle
                            assert(numel(conn) == 6, 'Unexpected tri6 connectivity length.');
                            tri6(end+1, :) = conn(:).'; %#ok<AGROW>
                            tri6_phys(end+1, 1) = physTag; %#ok<AGROW>
                    end
                end

                msh.tet10 = tet10;
                msh.tet10_phys = tet10_phys;
                msh.tri6 = tri6;
                msh.tri6_phys = tri6_phys;

                endLine = strtrim(fgetl(fid));
                assert(strcmp(endLine, '$EndElements'), 'Malformed $Elements section.');
        end
    end

    fclose(fid);
end