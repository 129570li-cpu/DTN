// ns3-integration/EliteForwarderApp.h
// EliteForwarderApp: hop-by-hop forwarding according to ELITE area_path and next target junction.
#pragma once
#include "ns3/application.h"
#include "ns3/address.h"
#include "ns3/socket.h"
#include "ns3/event-id.h"
#include "ns3/vector.h"
#include "EliteHeader.h"
#include <map>
#include <vector>

namespace ns3 {

struct NeighborInfo {
  Vector pos;
  Time lastSeen;
};

class EliteForwarderApp : public Application
{
public:
  static TypeId GetTypeId();
  EliteForwarderApp();
  virtual ~EliteForwarderApp();

  void SetStationId(uint32_t stationId);
  void SetSendPort(uint16_t port);
  void SetReceivePort(uint16_t port);
  void SetAiPort(uint16_t port);           // ns3-ai port
  void SetAiKey(uint16_t key);             // ns3-ai memory block key
  void SetSumoJunctionPos(const std::map<uint32_t, Vector>& jpos); // junction coordinates
  void SetDstPos(Vector pos) { m_dstPos = pos; }

  // Hook this inside CAM Rx callback
  void OnCamReceived(uint32_t srcStationId, Vector srcPos, Time now);

  // Initiate a route request (at source node)
  void RequestRoute(uint32_t dstStationId, uint8_t msgType);

protected:
  virtual void StartApplication() override;
  virtual void StopApplication() override;

private:
  void RecvPacket(Ptr<Socket> socket);
  void ForwardOrDeliver(Ptr<Packet> pkt, const EliteHeader& hdr);
  bool AtCurrentTargetJunction(const EliteHeader& hdr) const;
  uint32_t ChooseNextHop(uint32_t nextJunctionId) const;

  // Metrics
  void BeginSegment();
  void EndSegment(uint32_t segHops, double segCtrlMsgs, double segLen);
  void ReportPathFeedback(bool success);

  // ns3-ai exchange
  void BuildObservation(std::vector<double>& obs, bool isRouteRequest, bool isFeedback);
  bool GetAction(std::vector<double>& act, uint8_t& policyId, std::vector<uint32_t>& pathOut);

private:
  Ptr<Socket> m_rxSocket;
  Ptr<Socket> m_txSocket;
  uint16_t m_rxPort{9999};
  uint16_t m_txPort{9999};
  uint16_t m_aiPort{5555};
  uint16_t m_aiKey{2333};

  uint32_t m_stationId{0};
  std::map<uint32_t, NeighborInfo> m_neighbors;
  std::map<uint32_t, Vector> m_junctionPos;

  // Session info and metrics
  bool m_inSession{false};
  uint32_t m_dstStationId{0};
  uint32_t m_pathIdSeq{0};
  std::vector<double> m_segADca, m_segHCca, m_segRCca, m_segLen;
  double m_ADsd{0}, m_HCsd{0}, m_RCsd{0}, m_PL{0}, m_RO{0};
  Vector m_dstPos{0.0, 0.0, 0.0};
};

} // namespace ns3
