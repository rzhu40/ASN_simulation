function [OutputDynamics, SimulationOptions, snapshots] = simulateNetwork(Connectivity, Components, Signals, SimulationOptions, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate network at each time step. Mostly the same as Ido's code.
% Improved the simulation efficiency by change using nodal analysis.
% Enabled multi-electrodes at the same time.
%
% Left the API of snapshots. For later usage of visualize the network.
% ARGUMENTS:
% Connectivity - The Connectivity information of the network. Most
%                importantly the edge list, which shows the connectivity
%                condition between nanowires, and number of nodes and
%                junctions.
% Components - Structure that contains the component properties. Every
%              field is a (E+1)x1 vector. The extra component is the
%              tester resistor connected in series to the voltage and to
%              the network.
% Stimulus - Structure that contains the details of the external stimulus
%            (time axis details, voltage signal).
% SimulationOptions - Structure that contains general simulation details that are indepedent of
%           the other structures (eg, dt and simulation length);
% varargin - if not empty, contains an array of indidces in which a
%            snapshot of the resistances and voltages in the network is
%            requested. This indices are based on the length of the simulation.
% OUTPUT:
% OutputDynamics -- is a struct with the activity of the network
%                    .networkResistance - the resistance of the network (between the two
%                     contacts) as a function of time.
%                    .networkCurrent - the overall current from contact (1) to contact (2) as a
%                     function of time.
% Simulationoptions -- same struct as input, with updated field names
% snapshots - a cell array of structs, holding the resistance and voltage
%             values in the network, at the requested time-stamps.

% REQUIRES:
% updateComponentResistance
% updateComponentState
%
% USAGE:
%{
    Connectivity = getConnectivity(Connectivity);
    contact      = [1,2];
    Equations    = getEquations(Connectivity,contact);
    Components   = initializeComponents(Connectivity.NumberOfEdges,Components)
    Stimulus     = getStimulus(Stimulus);
    
    OutputDynamics = runSimulation(Equations, Components, Stimulus);
%}
%
% Authors:
% Ido Marcus
% Paula Sanz-Leon
% Ruomin Zhu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize:
compPtr         = ComponentsPtr(Components);        % using this matlab-style pointer to pass the Components structure by reference
niterations     = SimulationOptions.NumberOfIterations;
electrodes      = SimulationOptions.electrodes;
numOfElectrodes = SimulationOptions.numOfElectrodes;
E               = Connectivity.NumberOfEdges;
V               = Connectivity.NumberOfNodes;
edgeList        = Connectivity.EdgeList.';
RHS             = zeros(V+numOfElectrodes,1); % the first E entries in the RHS vector.

wireVoltage        = zeros(niterations, V);
electrodeCurrent   = zeros(niterations, numOfElectrodes);
junctionVoltage    = zeros(niterations, E);
junctionResistance = zeros(niterations, E);
junctionFilament   = zeros(niterations, E);
%% If snapshots are requested, allocate memory for them:
if ~isempty(varargin)
    snapshots           = cell(size(varargin{1}));
    snapshots_idx       = sort(varargin{1});
else
    nsnapshots          = 10;
    snapshots           = cell(nsnapshots,1);
    snapshots_idx       = ceil(logspace(log10(1), log10(niterations), nsnapshots));
end
kk = 1; % Counter

%% Solve equation systems for every time step and update:
for ii = 1 : niterations
    % Show progress:
    progressBar(ii,niterations);
    
    % Update resistance values:
    updateComponentResistance(compPtr);
    componentConductance = 1./compPtr.comp.resistance;
    
    % Get LHS (matrix) and RHS (vector) of equation:
    Gmat = zeros(V,V);
    
    % This line can be written in a more efficient vetorized way.
    % Something like:
    % Gmat(edgeList(:,1),edgeList(:,2)) = componentConductance;
    % Gmat(edgeList(:,2),edgeList(:,1)) = componentConductance;
    
    for i = 1:E
        Gmat(edgeList(i,1),edgeList(i,2)) = componentConductance(i);
        Gmat(edgeList(i,2),edgeList(i,1)) = componentConductance(i);
    end
    
    Gmat = diag(sum(Gmat, 1)) - Gmat;
    
    LHS          = zeros(V+numOfElectrodes, V+numOfElectrodes);
    LHS(1:V,1:V) = Gmat;
    for i = 1:numOfElectrodes
        this_elec           = electrodes(i);
        LHS(V+i,this_elec)  = 1;
        LHS(this_elec,V+i)  = 1;
        RHS(V+i)            = Signals{i,1}(ii);
    end
    % Solve equation:
    lhs = sparse(LHS);
    rhs = sparse(RHS);
    sol = lhs\rhs;
    %         sol = LHS\RHS;
    tempWireV = sol(1:V);
    compPtr.comp.voltage = tempWireV(edgeList(:,1)) - tempWireV(edgeList(:,2));
    
    % Update element fields:
    updateComponentState(compPtr, SimulationOptions.dt);    % ZK: changed to allow retrieval of local values
    %[lambda_vals(ii,:), voltage_vals(ii,:)] = updateComponentState(compPtr, Stimulus.dt);
    
    wireVoltage(ii,:)        = sol(1:V);
    electrodeCurrent(ii,:)   = sol(V+1:end);
    junctionVoltage(ii,:)    = compPtr.comp.voltage;
    junctionResistance(ii,:) = compPtr.comp.resistance;
    junctionFilament(ii,:)   = compPtr.comp.filamentState;
    
    
    % Record the activity of the whole network
    if find(snapshots_idx == ii)
        frame.Timestamp  = SimulationOptions.TimeVector(ii);
        frame.Voltage    = compPtr.comp.voltage;
        frame.Resistance = compPtr.comp.resistance;
        frame.OnOrOff    = compPtr.comp.OnOrOff;
        frame.filamentState = compPtr.comp.filamentState;
        %Find Wire Current
        for currWire=1 : Connectivity.NumberOfNodes
            % Find the indices of edges (=intersections) relevant for this
            % vertex (=wire):
            relevantEdges = find(Connectivity.EdgeList(1,:) == currWire | Connectivity.EdgeList(2,:) == currWire);
            
            % Sort them according to physical location (left-to-right, or
            % if the wire is vertical then bottom-up):
            if Connectivity.WireEnds(currWire,1) ~= Connectivity.WireEnds(currWire,3)
                [~,I] = sort(Connectivity.EdgePosition(relevantEdges,1));
            else
                [~,I] = sort(Connectivity.EdgePosition(relevantEdges,2));
            end
            relevantEdges = relevantEdges(I);
            
            % Calculate the current along each section of the wire:
            direction = ((currWire ~= Connectivity.EdgeList(1,relevantEdges))-0.5)*2;
            % Using the convention that currents are defined to flow
            % from wires with lower index to wires with higher index,
            % and that in the field EdgeList the upper row always
            % contains lower indices.
            wireCurrents{currWire} = cumsum(frame.Current(relevantEdges(1:end)).*direction(1:end)');
            % The first element in wireCurrents is the current in the
            % section between relevantEdge(1) and relevantEdge(2). We
            % assume that between relevantEdge(1) and the closest wire
            % end there's no current. Then, acording to a KCL
            % equation, there's also no current from relevantEdge(end)
            % to the other wire end (that's why the end-1 in the
            % expression).
            % The only two exceptions are the two contacts, where the
            % contact point is defined as the wire end closest to
            % relevantEdge(end)
        end
        %Save Wire Currents and Voltages in Adrian's Structure
        frame.WireVoltage{kk}=getAbsoluteVoltage(snapshots{kk}, Connectivity, SimulationOptions.ContactNodes);
        frame.WireCurrents{kk}=wireCurrents;
        %Save Wire Currents and Voltages is normal Structure
        SelSims.Data.WireCurrents{kk}=wireCurrents;
        SelSims.Data.WireVoltages{kk}= frame.WireVoltage{kk};
        snapshots{kk} = frame;
        kk = kk + 1;
    end
end

% Store some important fields
SimulationOptions.SnapshotsIdx = snapshots_idx; % Save these to access the right time from .TimeVector.

% Calculate network resistance and save:
OutputDynamics.electrodeCurrent   = electrodeCurrent;
OutputDynamics.wireVoltage        = wireVoltage;
OutputDynamics.junctionVoltage    = junctionVoltage;
OutputDynamics.junctionResistance = junctionResistance;
OutputDynamics.junctionFilament   = junctionFilament;
end