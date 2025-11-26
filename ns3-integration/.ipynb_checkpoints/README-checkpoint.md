# NS-3 Integration (ELITE)

This folder provides compile-ready C++ skeletons to integrate the ELITE controller with ns-3 (with SUMO mobility and IEEE 802.11p), using ns3-ai to exchange observations/actions with the Python controller.

Files
- EliteHeader.h
  - Custom header for data packets. Carries `policyId` (BP/HRF/LDF/LBF), `routeLen`, `route[32]`, `nextIndex`, `msgType`, `ttl`, `pathId`.
- EliteForwarderApp.h/.cc
  - Application that performs hop-by-hop forwarding along a junction path:
    - Maintains neighbor table from CAM receptions (hook `OnCamReceived` in your CAM callback).
    - For each packet, if arrived at current target junction → advance `nextIndex`.
    - Picks next hop: neighbor closer to the next target junction; fallback to the nearest neighbor when a void occurs.
    - Accumulates per-segment metrics (ADca, HCca, RCca, lca) and per-path metrics (ADsd/HCsd/RCsd/PL/RO).
    - Initiates a route via ns3-ai (routeRequestFlag=1) and reports feedback at completion (feedbackFlag=1).
  - Includes placeholders for ns3-ai calls; replace with your environment’s ns3-ai C++ API.

Observer/Action Layout (must match ELITE-fusion/controller_server.py)
- Observation (double array, fixed length)
  1) simTime, vehicleId, srcId, dstId, msgTypeId(0..3), queueLen, bufferBytes, neighborCount
  2) neighbors[64], distToNext[64]
  3) routeRequestFlag, feedbackFlag, success(1/0)
  4) ADsd, HCsd, RCsd, hopCount, PL, RO, nsd
  5) seg_ADca[32], seg_HCca[32], seg_lca[32], seg_RCca[32]
  6) path_len, path_ids[32]
- Action (double array, fixed length)
  - policyId(0=BP,1=HRF,2=LDF,3=LBF), path_len, path_ids[32]

Integration Points
1) Include headers in your scenario:
   - `#include "EliteForwarderApp.h"`
   - `#include "EliteHeader.h"`
   - `#include "ns3-ai header"` (e.g., `ns3/ai-rl.h`) for your ns3-ai build.
2) In your `setupNewWifiNode` lambda (after `bs_container->setupContainer(...)`):
   - Create/install `EliteForwarderApp`:
     ```cpp
     Ptr<EliteForwarderApp> eliteApp = CreateObject<EliteForwarderApp>();
     eliteApp->SetStationId(intVehicleID);
     eliteApp->SetReceivePort(9999);
     eliteApp->SetSendPort(9999);
     eliteApp->SetAiPort(5555);
     // Provide junction coords map if available:
     // std::map<uint32_t, Vector> juncPos = ...;
     // eliteApp->SetSumoJunctionPos(juncPos);
     c.Get(nodeID)->AddApplication(eliteApp);
     eliteApp->SetStartTime(Seconds(0.5));
     eliteApp->SetStopTime(Seconds(simTime));
     // Bridge CAM reception:
     bs_container->addCAMRxCallback ([eliteApp] (unsigned int my_stationID, Ptr<const Packet> p, Address from, Address to, Time rxTime) {
       Vector srcPos; // fill with sender position from your stack
       eliteApp->OnCamReceived(my_stationID, srcPos, rxTime);
     });
     ```
3) Initiate routes for your traffic flows:
   - From the desired source vehicles, call: `eliteApp->RequestRoute(dstStationId, msgTypeId)`.

Python Controller (ELITE)
- Run: `python3 ELITE-fusion/controller_server.py <sumo_net.xml> dtn_out 6000000 300 5555`
  - Loads SUMO net.xml → builds topology
  - Trains/loads four Q-tables → normalization → BP/HRF/LDF/LBF via paper-accurate fuzzy logic
  - APN: updates P(state,action) using R0 = a*g(ADsd)+b*q(HCsd)+γ*l(RCsd)
  - Selects policy based on message type × load level; computes area_path and returns it to ns-3

Notes
- Replace placeholders in `EliteForwarderApp::BuildObservation` and `GetAction` with actual ns3-ai API calls and real data.
- Calibrate 802.11p TxPower/propagation to ≈300 m reliable communication radius.
- For load estimation in APN, provide average buffer usage along baseline path (or report per-hop buffers to controller). 

