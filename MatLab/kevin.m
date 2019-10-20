function [OutputDynamics, SimulationOptions, snapshots] = kevin(Connectivity ,Components, Stimulus, SimulationOptions, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulates the network and finds the resistance between the two contacts
% as a function of time.
%
% ARGUMENTS: 
% Equations - Structure that contains the (abstract) matrix of coefficients
%             (as documented in getEquations) and the number of nodes in
%             the circuit.
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    %% Initialize:
    compPtr         = ComponentsPtr(Components);        % using this matlab-style pointer to pass the Components structure by reference
    niterations     = SimulationOptions.NumberOfIterations; 
    contactNodes    = SimulationOptions.electrodes;
    E               = Connectivity.NumberOfEdges;
    V               = Connectivity.NumberOfNodes;
    edgeList        = Connectivity.EdgeList.';
    RHS             = zeros(V+length(contactNodes),1); 
   wireVoltage        = zeros(niterations, V);
    networkCurrent     = zeros(niterations, 1);
    junctionVoltage    = zeros(niterations, E);
    junctionResistance = zeros(niterations, E);
    junctionFilament   = zeros(niterations, E);
    
    %% Use sparse matrices:
%     Equations.KCLCoeff = sparse(Equations.KCLCoeff);
%     Equations.KVLCoeff = sparse(Equations.KVLCoeff);
  
    
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
    sol(1:V)=0;recordjvoltage=zeros(2*(V),3/8*niterations-3);
    %% Solve equation systems for every time step and update:
    for ii = 1 : niterations
        % Show progress:
        progressBar(ii,niterations);
        
        % Update resistance values:
        updateComponentResistance(compPtr); 
        componentConductance = 1./compPtr.comp.resistance(1:E);
        Gmat = zeros(V,V);
         for i = 1:E
            Gmat(edgeList(i,1),edgeList(i,2)) = componentConductance(i);
            Gmat(edgeList(i,2),edgeList(i,1)) = componentConductance(i);
        end
         
        Gmat         = diag(sum(Gmat, 1)) - Gmat;
        LHS          = zeros(V+length(contactNodes), V+length(contactNodes));
        LHS(1:V,1:V) = Gmat;
        
        % Again, should be able to vectorize 
       
        % Get LHS (matrix) and RHS (vector) of equation:
%         LHS = [Equations.KCLCoeff ./ compPtr.comp.resistance(:,ones(Equations.NumberOfNodes-1,1)).' ; ...
%                Equations.KVLCoeff];
           signal(:,ii)=Stimulus.Signal(:,ii);
 if 1
     
           if ii<3/8*niterations
               signal(1,ii)=Stimulus.Signal(1,ii);
        if ii>=3&&ii<3/8*niterations
            recordjvoltage(1:V,ii-2)=vwire;
%                recordjvoltage(1:262,ii-2)=compPtr.comp.voltage;
               recordjvoltage(V+1:2*(V),ii-2)=lastV;
%                recordjvoltage(263:524,ii-2)=lastV;
        end
           elseif ii==3/8*niterations
%                target=1.3*sawtooth(2*1*pi*Stimulus.Frequency*(Stimulus.TimeAxis+1000));
               target=Stimulus.Signal(1,3:3/8*niterations-1)';
%                target=Stimulus.Signal(3:end-2/7*length(Stimulus.Signal));
                 inputx=[ones(length(target),1),Stimulus.Signal(1,2:3/8*niterations-2)',recordjvoltage'];
% % % %                  inputx(:,1)=Stimulus.TimeAxis(3:3/8*niterations-1);
                [a1,bint,r,rint,state]=regress(target,inputx);
%                 reg = 1e-8;a1 = ((inputx'*inputx + reg*eye(size(inputx'*inputx))) \ (inputx'*target));
%                 rrr=target-inputx*a1;
                signal(1,ii)=Stimulus.Signal(1,ii);
%                 rr(ii-3/8*niterations+1)=signal(ii)-Stimulus.Signal(ii)
% multiple linear regression 
if 0
i=1;
for aa=1:20:800
         iii=0;
         recordv=recordjvoltage(:,aa+iii*800:aa+iii*800+20-1);
         recordv(end+1,:)=Stimulus.Signal(2+iii*800:2+iii*800+20-1);
        targetii=target(aa+iii*800:aa+iii*800+20-1);
        for iii=1:floor(length(target)/(800))-1
        targetii=[targetii; target(aa+iii*800:aa+iii*800+20-1)];
        recordii=recordjvoltage(:,aa+iii*800:aa+iii*800+20-1);
        recordii(end+1,:)=Stimulus.Signal(2+iii*800:2+iii*800+20-1);
        recordv=[recordv,recordii];
  
        end
        inputxii=[ones(length(targetii),1),recordv'];
        [a1(:,i),bint,r,rint,state]=regress(targetii,inputxii);
        i=i+1;
end
end
           elseif ii>3/8*niterations&&ii<=4/8*niterations
                 signal(1,ii)=0.015;
%                  signal(ii)=Stimulus.Signal(ii);
%                rr(ii-5/7*niterations+1)=signal(ii)-Stimulus.Signal(ii);
              elseif ii>4/8*niterations&&ii<4/8*niterations+3
                  signal(1,ii)=Stimulus.Signal(1,ii-4/8*niterations);
              elseif ii>=4/8*niterations+3&&ii<7/8*niterations
                  signal(1,ii)=1.00*[1,signal(1,ii-1),vwire',lastV']*a1;
%                     a1=[a1(:,end) a1(:,1:end-1)];
%                     cc=rem(ceil(ii/20),40)+1;
%                     signal(ii)=[1,vwire',lastV',signal(ii-1)]*a1(:,cc);
                  if ii==4/8*niterations+3
%                   bo=sum([1,compPtr.comp.voltage',lastV']~=inputx(1,:));
%                     bo1=[1,compPtr.comp.voltage',lastV'];bo2=inputx(1,:);
                  end
%                    rr(ii-5/8*niterations+1)=signal(ii)-Stimulus.Signal(ii-4/8*niterations);
           elseif ii>=7/8*niterations
%                signal(ii)=Stimulus.Signal(ii-4/8*niterations);
%                     cc=rem(ceil(ii/20),40)+1;
%                   signal(ii)=[1,vwire',lastV,signal(ii-1)']*a1(:,cc);
                  signal(1,ii)=1.00*[1,signal(1,ii-1),vwire',lastV']*a1;
           end
 end

  for a=1:length(contactNodes)
        LHS(V+a, contactNodes(a)) = 1;
        LHS(contactNodes(a), V+a) = 1;
        RHS(V+a) = signal(a,ii);
        end
  
           lastV=sol(1:V);
%            lastV(SimulationOptions.ContactNodes)=[];
           
        lhs = sparse(LHS);
        rhs = sparse(RHS);
        sol = lhs\rhs;
        
%         lastV=compPtr.comp.voltage;
%         RHS = [RHSZeros ; signal(ii)];
%         vnIs=VnIs(Connectivity, SimulationOptions.ContactNodes,compPtr.comp.resistance,signal(ii));
        % Solve equation:
%         compPtr.comp.voltage = full(LHS\RHS);
            vwire=sol(1:V);
            
           compPtr.comp.voltage(1:E) =  vwire(edgeList(:,1)) -  vwire(edgeList(:,2));
%            vwire(SimulationOptions.ContactNodes)=[];
%         compPtr.comp.voltage=sovle(Connectivity, SimulationOptions.ContactNodes,compPtr.comp.resistance,signal(ii));
        % Update element fields:
        updateComponentState(compPtr, Stimulus.dt);                           
      wireVoltage(ii,:)        = sol(1:V);
        networkCurrent(ii,1:length(contactNodes))       = sol(V+1:end);
%         junctionVoltage(ii,:)    = compPtr.comp.voltage(1:E);
%         junctionResistance(ii,:) = compPtr.comp.resistance(1:E);
%         junctionFilament(ii,:)   = compPtr.comp.filamentState(1:E);
        % Record tester voltage:
%         testerVoltage(ii) = compPtr.comp.voltage(end);
        
        % Record the activity of the whole network
       if find(snapshots_idx == ii) 
                frame.Timestamp  = SimulationOptions.TimeVector(ii);
%                 frame.Voltage    = compPtr.comp.voltage;
                frame.Resistance = compPtr.comp.resistance;
                frame.OnOrOff    = compPtr.comp.OnOrOff;
%                 frame.filamentState = compPtr.comp.filamentState;
                snapshots{kk} = frame;
                kk = kk + 1;
        end
    end
    % Store some important fields
     % Store some important fields
    SimulationOptions.SnapshotsIdx = snapshots_idx; % Save these to access the right time from .TimeVector.

    % Calculate network resistance and save:
    
    OutputDynamics.networkCurrent     = networkCurrent;
    OutputDynamics.wireVoltage        = wireVoltage;
    
%     OutputDynamics.junctionVoltage    = junctionVoltage;
%     OutputDynamics.junctionResistance = junctionResistance;
%     OutputDynamics.junctionFilament   = junctionFilament;
    
    if 1
%         th=inputx*a1-target;M=signal(4/8*niterations+3:7/8*niterations-1)'-target;
     t=0.01:0.01:niterations/100;
     tt=t(1:3/8*niterations-3);
     target=Stimulus.Signal(1,3:3/8*niterations-1);
    figure,subplot(3,1,1),plot(tt,target); set(gca,'FontSize',16);  set(gca,'XMinorTick','on','YMinorTick','on');
    subplot(3,1,2),plot(tt,inputx*a1,'r--'); set(gca,'FontSize',16);  set(gca,'XMinorTick','on','YMinorTick','on');
    subplot(3,1,3),plot(t,signal(1,:)),hold on, plot(t(4/8*niterations+3:7/8*niterations),signal(1,4/8*niterations+3:7/8*niterations),'r');
     plot(t(7/8*niterations:end),signal(1,7/8*niterations:end),'r');set(gca,'XMinorTick','on','YMinorTick','on');set(gca,'FontSize',16);
     %ylim([-4,4]);
%      aaa=signal(4/8*niterations+3:7/8*niterations-1)-target;
%      save dat target;
%      save dat2 aaa
     mse = sum((target-signal(1,4/8*niterations+3:7/8*niterations-1)).^2)./length(target);
     OutputDynamics.mse=mse;
     
%      glo.contactn=mse;
%      accuracy=1-sqrt(mse);
%      disp( ['MSE = ', num2str( mse )] );disp( ['accuracy = ', num2str( accuracy )] );
%      set(gca,'FontSize',16);  set(gca,'XMinorTick','on','YMinorTick','on');
    end
%     ylim([-4,4]);
% %     save dat rr
% %     save dat2 rrr
%     save dat3 a1
%     save dat4 bo1
%     save dat5 bo2
end