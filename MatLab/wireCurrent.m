this_time = 900;
junctionCurrent = Output.junctionVoltage./ Output.junctionResistance;
currents = junctionCurrent(this_time,:).';

connectivity = Connectivity;
edgeList = connectivity.EdgeList;
Imat = zeros(connectivity.NumberOfNodes, connectivity.NumberOfNodes);

index = sub2ind(size(Imat), edgeList(1,:),edgeList(2,:));
Imat(index) = currents;
Imat = Imat + Imat.';