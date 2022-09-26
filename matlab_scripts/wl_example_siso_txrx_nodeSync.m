function [phase_diff] = wl_example_siso_txrx_nodeSync(nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ManualRxGainRF, ...
ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % wl_example_siso_txrx_nodeSync.m
    %
    % This example illustrates how to synchronize multiple WARP v3 nodes
    % to eliminate all frequency and timing offsets.
    %
    % Requirements:
    %     - 2 WARP nodes (same hardware generation); 1 RF interface each
    %     - Ether:
    %         - 2 CM-MMCX modules; MMCX coax cable assemblies to connect the CM-MMCX I/O 
    %           and a 2-pin twisted pair cable assembly to route the inter-node trigger
    %         - 2 CM-PLL modules; CM-PLL connector 
    %           (see:  http://warpproject.org/trac/wiki/HardwareUsersGuides/CM-PLL/Connectors#Cables )
    %     - WARPLab 7.6.0 and higher
    %
    % More details on using this example are available on the WARP site:
    %     http://warpproject.org/w/WARPLab/Examples
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Top Level Control Variables
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % External trigger mode requires a connection from the trigger output EXT_OUT_P0 on node 0
    % to EXT_IN_P3 on node 1 (see http://warpproject.org/w/WARPLab/Examples for details)
    %
%     USE_EXTERNAL_TRIGGER = true;

    % To maintain constant phase offsets among nodes sharing an RF reference clock, bypass 
    % wl_initNodes() which executes a reset of the MAX2829 transceivers that forces a re-tune 
    % of the PLL that changes the inter-node phases.
    %
    % NOTE:  This has to be false the first time this script is run otherwise, the script will
    %     not have the "nodes" variable populated.
    %
    BYPASS_INIT_NODES = true;

    % RX variables
%     USE_AGC        = false;
%     ManualRxGainRF = 1;                    % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
%     ManualRxGainBB = 15;                   % Rx Baseband Gain in [0:31] (ignored if USE_AGC is true)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set up the WARPLab experiment
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Create a vector of node objects
    if ( ~BYPASS_INIT_NODES )

        nodes = wl_initNodes(2);

    else 
        % This example assumes that the node is in the state from which it exits initNodes.
        % If the example does not run initNodes to keep the phase offsets constant, then we need
        % to issue a couple of commands to put the node in a known state.
        %

        % Set the transmit delay to zero
        wl_basebandCmd(nodes, 'tx_delay', 0);

        % Disable the buffers and RF interfaces for TX / RX
        wl_basebandCmd(nodes, ifc_ids.RF_ALL, 'tx_rx_buff_dis');
        wl_interfaceCmd(nodes, ifc_ids.RF_ALL, 'tx_rx_dis');
    end


    % Assign roles to the nodes (ie transmitter / receiver)
    node_tx = nodes(1);
    node_rx = nodes(2);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set up Trigger Manager
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Create a UDP broadcast trigger and primary node to be ready for it
    eth_trig = wl_trigger_eth_udp_broadcast;
    nodes.wl_triggerManagerCmd('add_ethernet_trigger', [eth_trig]);

    % Read Trigger IDs into workspace
    trig_in_ids  = wl_getTriggerInputIDs(node_tx);
    trig_out_ids = wl_getTriggerOutputIDs(node_tx);

    % For the transmit node, we will allow Ethernet to trigger the buffer baseband, the AGC, and debug0 
    % (which is mapped to pin 8 on the debug header)
    node_tx.wl_triggerManagerCmd('output_config_input_selection', [trig_out_ids.BASEBAND, trig_out_ids.EXT_OUT_P0], [trig_in_ids.ETH_A]);

    if(USE_EXTERNAL_TRIGGER)
        % For the receive node, we will allow debug3 (mapped to pin 15 on the
        % debug header) to trigger the buffer baseband, and the AGC
        % Note that the below line selects both P0 and P3. This will allow the
        % script to work with either the CM-PLL (where output P0 directly
        % connects to input P0) or the CM-MMCX (where output P0 is usually
        % connected to input P3 since both neighbor ground pins).
        node_rx.wl_triggerManagerCmd('output_config_input_selection', [trig_out_ids.BASEBAND, trig_out_ids.AGC], [trig_in_ids.EXT_IN_P0, trig_in_ids.EXT_IN_P3]);
    else
        node_rx.wl_triggerManagerCmd('output_config_input_selection', [trig_out_ids.BASEBAND, trig_out_ids.AGC], [trig_in_ids.ETH_A]);
    end

    % For the receive node, we enable the debounce circuity on the debug 3 input
    % to deal with the fact that the signal may be noisy.
    node_rx.wl_triggerManagerCmd('input_config_debounce_mode', [trig_in_ids.EXT_IN_P0, trig_in_ids.EXT_IN_P3], true); 

    % Since the debounce circuitry is enabled, there will be a delay at the
    % receiver node for its input trigger. To better align the transmitter and
    % receiver, we can artifically delay the transmitters trigger outputs that
    % drive the buffer baseband and the AGC.
    node_tx.wl_triggerManagerCmd('output_config_delay', [trig_out_ids.EXT_OUT_P0], 0);
    node_tx.wl_triggerManagerCmd('output_config_delay', [trig_out_ids.BASEBAND], 400);     % 62.5ns delay
    node_rx.wl_triggerManagerCmd('output_config_delay', [trig_out_ids.AGC], 3000);          % 3000ns delay


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set up the Interface parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Get IDs for the interfaces on the boards.  
    %
    % NOTE:  This example assumes each board has the same interface capabilities (ie 2 RF 
    %   interfaces; RFA and RFB).  Therefore, we only need to get the IDs from one of the boards.
    %
    ifc_ids = wl_getInterfaceIDs(node_tx);

    % Set the Transmit and Receive interfaces
    %     Transmit from RFA of one node to RFA of the other node
    %
    % NOTE:  Variables are used to make it easier to change interfaces.
    % 
%     RF_TX         = ifc_ids.RF_A;                    % Transmit RF interface
%     RF_RX         = ifc_ids.RF_A;                    % Receive RF interface
% 
%     RF_RX_VEC     = ifc_ids.RF_A;                    % Vector version of transmit RF interface
%     RF_TX_VEC     = ifc_ids.RF_A;                    % Vector version of receive RF interface

    % Set the RF center frequency on all interfaces
    %     - Frequency Band  :  Must be 2.4 or 5, to select 2.4GHz or 5GHz channels
    %     - Channel         :  Must be an integer in [1,11] for BAND = 2.4; [1,23] for BAND = 5
    %
    wl_interfaceCmd(nodes, ifc_ids.RF_ALL, 'channel', BAND, CHANNEL);

    % Set the RX gains on all interfaces or use AGC
    %     - Rx RF Gain      :  Must be an integer in [1:3]
    %     - Rx Baseband Gain:  Must be an integer in [0:31]
    % 
    % NOTE:  The gains may need to be modified depending on your experimental setup
    %
    if(USE_AGC)
        wl_interfaceCmd(nodes, ifc_ids.RF_ALL, 'rx_gain_mode', 'automatic');
        wl_basebandCmd(nodes, 'agc_target', -10);
    else
        wl_interfaceCmd(nodes, ifc_ids.RF_ALL, 'rx_gain_mode', 'manual');
%         wl_interfaceCmd(nodes, ifc_ids.RF_ALL, 'rx_gains', ManualRxGainRF, ManualRxGainBB);
        wl_interfaceCmd(nodes, ifc_ids.RF_ALL_VEC, 'rx_gains', ManualRxGainRF, ManualRxGainBB);
    end

    % Set the TX gains on all interfaces
    %     - Tx Baseband Gain:  Must be an integer in [0:3] for approx [-5, -3, -1.5, 0]dB baseband gain
    %     - Tx RF Gain      :  Must be an integer in [0:63] for approx [0:31]dB RF gain
    % 
    % NOTE:  The gains may need to be modified depending on your experimental setup
    % 
    wl_interfaceCmd(nodes, ifc_ids.RF_ALL, 'tx_gains', 3, 30);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set up the Baseband parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Get the sample frequency from the board
    ts      = 1 / (wl_basebandCmd(nodes(1), 'tx_buff_clk_freq'));

    % Read the maximum I/Q buffer length.  
    %
    % NOTE:  This example assumes that each board has the same baseband capabilities (ie both nodes are 
    %   the same WARP hardware version, for example WARP v3).  This example also assumes that each RF 
    %   interface has the same baseband capabilities (ie the max number of TX samples is the same as the 
    %   max number of RF samples). Therefore, we only need to read the max I/Q buffer length of node_tx RFA.
    % 
    maximum_buffer_len = wl_basebandCmd(node_tx, RF_TX, 'tx_buff_max_num_samples');

    % Set the transmission / receptions lengths (in samples)
    %     See WARPLab user guide for maximum length supported by WARP hardware 
    %     versions and different WARPLab versions.
    %
    tx_length    = 2^15;
    rx_length    = tx_length;

    % Check the transmission length
    if (tx_length > maximum_buffer_len) 
        error('Node supports max transmission length of %d samples.  Requested %d samples.', maximum_buffer_len, tx_length); 
    end

    % Set the length for the transmit and receive buffers based on the transmission length
    wl_basebandCmd(nodes, 'tx_length', tx_length);
    wl_basebandCmd(nodes, 'rx_length', rx_length);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Signal processing to generate transmit signal
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % First generate the preamble for AGC. 
    %     NOTE:  The preamble corresponds to the short symbols from the 802.11a PHY standard
    %
    shortSymbol_freq = [0 0 0 0 0 0 0 0 1+i 0 0 0 -1+i 0 0 0 -1-i 0 0 0 1-i 0 0 0 -1-i 0 0 0 1-i 0 0 0 0 0 0 0 1-i 0 0 0 -1-i 0 0 0 1-i 0 0 0 -1-i 0 0 0 -1+i 0 0 0 1+i 0 0 0 0 0 0 0].';
    shortSymbol_freq = [zeros(32,1);shortSymbol_freq;zeros(32,1)];
    shortSymbol_time = ifft(fftshift(shortSymbol_freq));
    shortSymbol_time = (shortSymbol_time(1:32).')./max(abs(shortSymbol_time));
    shortsyms_rep    = repmat(shortSymbol_time,1,30);
    preamble         = shortsyms_rep;
    preamble         = preamble(:);


    t = [0:ts:((tx_length - length(preamble) - 1))*ts].';      % Create time vector(Sample Frequency is ts (Hz))

    sinusoid  = 0.6 * exp(j*2*pi * 2e6 * t);                   % Create 2 MHz sinusoid

    tx_data   = [preamble; sinusoid];


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Transmit and receive signal using WARPLab
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Transmit IQ data to the TX node
    wl_basebandCmd(node_tx, RF_TX_VEC, 'write_IQ', tx_data(:));

    % Enabled the RF interfaces for TX / RX
    wl_interfaceCmd(node_tx, RF_TX, 'tx_en');
    wl_interfaceCmd(node_rx, RF_RX, 'rx_en');

    % Enable the buffers for TX / RX
    wl_basebandCmd(node_tx, RF_TX, 'tx_buff_en');
    wl_basebandCmd(node_rx, RF_RX, 'rx_buff_en');

    % Send the Ethernet trigger to start the TX
    eth_trig.send();

    % Read the IQ data from the RX node 
    rx_IQ    = wl_basebandCmd(node_rx, RF_RX_VEC, 'read_IQ', 0, rx_length);

    % Disable the buffers and RF interfaces for TX / RX
    wl_basebandCmd(nodes, ifc_ids.RF_ALL, 'tx_rx_buff_dis');
    wl_interfaceCmd(nodes, ifc_ids.RF_ALL, 'tx_rx_dis');

    sampStart = 5000;
    sampEnd = tx_length;
    rx_phases = angle(rx_IQ(sampStart:sampEnd, :));
    tx_phases = repmat(angle(tx_data(sampStart:sampEnd)), 1, length(RF_RX_VEC));
    phase_diff = mean(unwrap(rx_phases) - unwrap(tx_phases));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Visualize results
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tVec = [0:ts:((tx_length -1 )*ts)]*1e6;

    sampStart = 5000;
    sampEnd = tx_length;
    htxt = [];

    figure(1);clf;
    for iii = 1:length(RF_RX_VEC)
        subplot(3,length(RF_RX_VEC),iii)
        plot(tVec(sampStart:sampEnd),real(tx_data(sampStart:sampEnd)),'b')
        ylabel('Amplitude');
        title('Transmitted I Waveform');
        axis([tVec(sampStart) tVec(sampStart+200) -1 1]);
        grid on;

        subplot(3,length(RF_RX_VEC),iii+length(RF_RX_VEC))
        plot(tVec(sampStart:sampEnd),real(rx_IQ(sampStart:sampEnd,iii)), 'r')
        ylabel('Amplitude');
        title('Received I Waveform');
        axis([tVec(sampStart) tVec(sampStart+200) -1 1]);
        grid on;

        subplot(3,length(RF_RX_VEC),iii+2*length(RF_RX_VEC))
        phase_diff__ = unwrap(angle(rx_IQ(sampStart:sampEnd, iii))) - unwrap(angle(tx_data(sampStart:sampEnd)));
        plot(tVec(sampStart:sampEnd), phase_diff__)
        axis tight
        myAxis = axis;

        if(myAxis(4)-myAxis(3) < 2*pi)
            %Zoom out to at least a 2*pi range of angles
            axis([myAxis(1), myAxis(2), mean(myAxis(3:4))-pi, mean(myAxis(3:4))+pi]);
        end

        grid on;
        title('Tx-Rx Phase Offset');
        ylabel('Phase Difference (radians)');
        xlabel('Time (us)');
    end
end