function [OutputDynamics] = forecast(Connectivity, Components, Signals, SimulationOptions)
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
    
    train_ratio = 1;
    training_length = round(niterations*train_ratio);
    steps = 100;
    forecast_on = true;
    update_weight = false;
    junctionList = [ 222,  302,  419,  497,  572,  584,  605,  627,  829,  831,  847, 854,  986, 1043, 1057, 1104, 1191, 1194, 1240, 1243];
    %% Solve equation systems for every time step and update:
    for ii = 1 : training_length
        % Show progress:
        progressBar(ii,training_length);
        
        % Update resistance values:
        updateComponentResistance(compPtr); 
        componentConductance = 1./compPtr.comp.resistance;
        
        % Get LHS (matrix) and RHS (vector) of equation:
        Gmat = zeros(V,V);
        
        % This line can be written in a more efficient vetorized way.
        % Something like:
%         Gmat(edgeList(:,1),edgeList(:,2)) = diag(componentConductance);
%         Gmat(edgeList(:,2),edgeList(:,1)) = diag(componentConductance);
        
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
        
    end
    measure = junctionVoltage(1:training_length,junctionList)./junctionResistance(1:training_length,junctionList);
    
    weight = getWeight(measure, Signals{1,1}(1:training_length), steps);
    predict = zeros(niterations,1);
    predict(1:training_length) = Signals{1,1}(1:training_length);
    
    for ii = training_length+1 : niterations
        % Show progress:
        progressBar(ii,niterations);
        
        % Update resistance values:
        updateComponentResistance(compPtr); 
        componentConductance = 1./compPtr.comp.resistance;
        
        % Get LHS (matrix) and RHS (vector) of equation:
        Gmat = zeros(V,V);
        
        for i = 1:E
            Gmat(edgeList(i,1),edgeList(i,2)) = componentConductance(i);
            Gmat(edgeList(i,2),edgeList(i,1)) = componentConductance(i);
        end
        
        Gmat = diag(sum(Gmat, 1)) - Gmat;
        
        LHS          = zeros(V+numOfElectrodes, V+numOfElectrodes);
        LHS(1:V,1:V) = Gmat;
        paras = length(junctionList);
        hist = zeros(paras * steps + 2,1);
        hist(1) = 1;
        hist(2) = 0;
        for i = 0:steps-1
            hist(i*paras+3:(i+1)*paras+2) = measure(ii-i-1,:);
        end
        predict(ii) = dot(hist,weight);
        
        for i = 1:numOfElectrodes
            this_elec           = electrodes(i);
            LHS(V+i,this_elec)  = 1;
            LHS(this_elec,V+i)  = 1;
            if i == 1
                if forecast_on
                    RHS(V+i)            = predict(ii);
                else   
                    RHS(V+i)            = Signals{i,1}(ii);
                end
            end
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
        this_measure = junctionVoltage(ii,junctionList)./junctionResistance(ii,junctionList);
        measure = [measure ; this_measure];
        if update_weight
            weight = getWeight(measure, Signals{1,1}(1:training_length), steps);
        end
    end
    
    % Calculate network resistance and save:
    OutputDynamics.electrodeCurrent   = electrodeCurrent;
    OutputDynamics.wireVoltage        = wireVoltage;
    OutputDynamics.junctionVoltage    = junctionVoltage;
    OutputDynamics.junctionResistance = junctionResistance;
    OutputDynamics.junctionFilament   = junctionFilament;
    OutputDynamics.forecast           = predict;
end