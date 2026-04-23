filename = "/Users/tanaythakur/Desktop/University of Michigan/Aero 623/Project4/aero623-project-2/untwisted.msh";

debug_gmsh_element_counts(filename);

msh = readGmsh(filename);
msh = readGmsh("untwisted.msh");
writeGmshGri("untwisted.gri", msh);

%% testing
function debug_gmsh_element_counts(filename)
    fid = fopen(filename, 'r');
    assert(fid ~= -1, 'Could not open file: %s', filename);

    counts = containers.Map('KeyType','double','ValueType','double');

    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if strcmp(line, '$Elements')
            nElem = sscanf(fgetl(fid), '%d');
            fprintf('Total element records in file: %d\n', nElem);

            for i = 1:nElem
                vals = sscanf(fgetl(fid), '%d');
                elemType = vals(2);

                if isKey(counts, elemType)
                    counts(elemType) = counts(elemType) + 1;
                else
                    counts(elemType) = 1;
                end
            end

            break
        end
    end

    fclose(fid);

    keysList = sort(cell2mat(keys(counts)));
    fprintf('Element type counts:\n');
    for k = keysList
        fprintf('  type %d -> %d\n', k, counts(k));
    end
end