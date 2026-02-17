function plotgri(Mesh)
% plots a computational mesh

figure(1); clf;
for e = 1:Mesh.nElem,
  I = Mesh.Elem(e,:); I = [I,I(1)];
  plot(Mesh.Node(I,1), Mesh.Node(I,2), 'k-'); hold on;
end

axis equal
set(gca, 'XTick', []);
set(gca, 'YTick', []);
set(gca, 'XTickLabel', '');
set(gca, 'YTickLabel', '');
box off; axis off
set(gca, 'LineWidth', 0.5);
