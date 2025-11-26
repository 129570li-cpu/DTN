// ns3-integration/EliteForwarderApp.cc
#include "EliteForwarderApp.h"
#include "ns3/packet-socket.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/node.h"
#include <cmath>
#include "ns3/ns3-ai-module.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("EliteForwarderApp");
NS_OBJECT_ENSURE_REGISTERED(EliteForwarderApp);

TypeId EliteForwarderApp::GetTypeId() {
  static TypeId tid = TypeId("ns3::EliteForwarderApp")
    .SetParent<Application>()
    .AddConstructor<EliteForwarderApp>();
  return tid;
}
EliteForwarderApp::EliteForwarderApp() {}
EliteForwarderApp::~EliteForwarderApp() {}

void EliteForwarderApp::SetStationId(uint32_t id) { m_stationId = id; }
void EliteForwarderApp::SetSendPort(uint16_t p) { m_txPort = p; }
void EliteForwarderApp::SetReceivePort(uint16_t p) { m_rxPort = p; }
void EliteForwarderApp::SetAiPort(uint16_t p) { m_aiPort = p; }
void EliteForwarderApp::SetAiKey(uint16_t k) { m_aiKey = k; }
void EliteForwarderApp::SetSumoJunctionPos(const std::map<uint32_t, Vector>& jpos) { m_junctionPos = jpos; }

void EliteForwarderApp::StartApplication() {
  // Switch to PacketSocket (layer-2), since the example scenarios typically do not install Internet stack.
  TypeId tid = TypeId::LookupByName("ns3::PacketSocketFactory");
  m_rxSocket = Socket::CreateSocket(GetNode(), tid);
  PacketSocketAddress local;
  local.SetSingleDevice(GetNode()->GetDevice(0)->GetIfIndex());
  local.SetPhysicalAddress(GetNode()->GetDevice(0)->GetAddress());
  local.SetProtocol(0x88B6); // experimental ethertype for ELITE data packets
  m_rxSocket->Bind(local);
  m_rxSocket->SetRecvCallback(MakeCallback(&EliteForwarderApp::RecvPacket, this));

  m_txSocket = Socket::CreateSocket(GetNode(), tid);
  PacketSocketAddress remote;
  remote.SetSingleDevice(GetNode()->GetDevice(0)->GetIfIndex());
  remote.SetPhysicalAddress(GetNode()->GetDevice(0)->GetBroadcast());
  remote.SetProtocol(0x88B6);
  m_txSocket->Connect(remote);
}

void EliteForwarderApp::StopApplication() {
  if (m_rxSocket) { m_rxSocket->Close(); m_rxSocket = 0; }
  if (m_txSocket) { m_txSocket->Close(); m_txSocket = 0; }
}

void EliteForwarderApp::OnCamReceived(uint32_t srcStationId, Vector srcPos, Time now) {
  NeighborInfo ni; ni.pos = srcPos; ni.lastSeen = now;
  m_neighbors[srcStationId] = ni;
}

void EliteForwarderApp::RequestRoute(uint32_t dstStationId, uint8_t msgType) {
  m_dstStationId = dstStationId;
  std::vector<double> obs, act;
  BuildObservation(obs, /*isRouteRequest=*/true, /*isFeedback=*/false);
  uint8_t policyId; std::vector<uint32_t> areaPath;
  if (!GetAction(act, policyId, areaPath)) {
    NS_LOG_WARN("No action from controller.");
    return;
  }
  EliteHeader hdr; hdr.SetPolicy((EliteHeader::PolicyId)policyId); hdr.SetPath(areaPath);
  hdr.SetMsgType(msgType); hdr.SetPathId(++m_pathIdSeq);
  Ptr<Packet> p = Create<Packet>(1024); // data 1024B
  p->AddHeader(hdr);
  m_txSocket->Send(p);
  m_inSession = true; m_ADsd = m_HCsd = m_RCsd = m_PL = m_RO = 0.0;
  m_segADca.clear(); m_segHCca.clear(); m_segRCca.clear(); m_segLen.clear();
  BeginSegment();
}

void EliteForwarderApp::RecvPacket(Ptr<Socket> socket) {
  Address from; Ptr<Packet> pkt = socket->RecvFrom(from);
  EliteHeader hdr; if (pkt->PeekHeader(hdr) == 0) { return; }
  ForwardOrDeliver(pkt, hdr);
}

bool EliteForwarderApp::AtCurrentTargetJunction(const EliteHeader& hdr) const {
  auto it = m_junctionPos.find(hdr.GetCurrentTarget());
  if (it == m_junctionPos.end()) return false;
  Vector jpos = it->second;
  Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
  Vector me = mob->GetPosition();
  double dx = me.x - jpos.x, dy = me.y - jpos.y;
  return std::sqrt(dx*dx+dy*dy) <= 25.0; // arrival radius
}

uint32_t EliteForwarderApp::ChooseNextHop(uint32_t nextJunctionId) const {
  auto it = m_junctionPos.find(nextJunctionId);
  if (it == m_junctionPos.end() || m_neighbors.empty()) return 0;
  Vector target = it->second;
  Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
  Vector me = mob->GetPosition();
  double best = 1e18; uint32_t bestId = 0;
  for (auto& kv : m_neighbors) {
    uint32_t nid = kv.first; Vector npos = kv.second.pos;
    double d_cur = std::hypot(me.x - target.x, me.y - target.y);
    double d_n   = std::hypot(npos.x - target.x, npos.y - target.y);
    if (d_n + 1e-6 < d_cur && d_n < best) { best = d_n; bestId = nid; }
  }
  if (bestId == 0) {
    // fallback: nearest neighbor to current
    best = 1e18;
    for (auto& kv : m_neighbors) {
      double d = std::hypot(kv.second.pos.x - me.x, kv.second.pos.y - me.y);
      if (d < best) { best = d; bestId = kv.first; }
    }
  }
  return bestId;
}

void EliteForwarderApp::ForwardOrDeliver(Ptr<Packet> pkt, const EliteHeader& hdrIn) {
  EliteHeader hdr = hdrIn;
  if (AtCurrentTargetJunction(hdr)) hdr.IncNext();
  // NOTE: destination detection should check destination vehicle's junction; omitted here.
  uint32_t nextHopId = ChooseNextHop(hdr.GetCurrentTarget());
  if (nextHopId == 0) {
    ReportPathFeedback(false);
    return;
  }
  // TODO: update per segment stats on hop
  Ptr<Packet> p = Create<Packet>(1024);
  p->AddHeader(hdr);
  m_txSocket->Send(p);
}

void EliteForwarderApp::BeginSegment() {
  // TODO: store start timestamp, ctrl msg counter baseline etc.
}
void EliteForwarderApp::EndSegment(uint32_t segHops, double segCtrlMsgs, double segLen) {
  m_segHCca.push_back(segHops);
  m_segRCca.push_back(segCtrlMsgs);
  m_segLen.push_back(segLen);
  // TODO: push segment delay to m_segADca
}

void EliteForwarderApp::ReportPathFeedback(bool success) {
  // Aggregate and send feedback to controller via ns3-ai
  std::vector<double> obs;
  BuildObservation(obs, /*isRouteRequest=*/false, /*isFeedback=*/true);
  std::vector<double> act; uint8_t pid; std::vector<uint32_t> path;
  GetAction(act, pid, path);
  m_inSession = false;
}

void EliteForwarderApp::BuildObservation(std::vector<double>& obs, bool isRouteRequest, bool isFeedback) {
  // This function must match controller_server.py observation layout.
  // Fill with zeros here; integrate with real values in your environment.
  const uint32_t MAX_NEI = 64;
  const uint32_t MAX_PATH = 32;
  obs.clear();
  obs.reserve(8 + MAX_NEI + MAX_NEI + 1 + 10 + 1 + 4*MAX_PATH + 1 + MAX_PATH);
  // simTime, vehicleId, srcId, dstId, msgTypeId, queueLen, bufferBytes, neighborCount
  double simTime = Simulator::Now().GetSeconds();
  obs.push_back(simTime);
  obs.push_back((double)m_stationId);
  obs.push_back((double)m_stationId); // srcId (example)
  obs.push_back((double)m_dstStationId);
  obs.push_back(1.0); // msgTypeId=efficiency
  obs.push_back(0.0); // queue len
  obs.push_back(0.0); // buffer bytes
  obs.push_back((double)std::min<size_t>(m_neighbors.size(), MAX_NEI));
  // neighbors, distToNext
  for (uint32_t i=0;i<MAX_NEI;i++) obs.push_back(0.0);
  for (uint32_t i=0;i<MAX_NEI;i++) obs.push_back(0.0);
  // flags
  obs.push_back(isRouteRequest? 1.0 : 0.0);
  obs.push_back(isFeedback? 1.0 : 0.0);
  obs.push_back(0.0); // success flag if feedback
  // path-level metrics
  obs.push_back(m_ADsd);
  obs.push_back(m_HCsd);
  obs.push_back(m_RCsd);
  obs.push_back(0.0); // hopCount
  obs.push_back(m_PL);
  obs.push_back(m_RO);
  // When route request, encode src/dst nearest junction ids into path_ids[0..1] via path_len=2
  // Compute nearest junction to current position and to provided destination position (if available)
  uint32_t srcJ = 0, dstJ = 0;
  if (isRouteRequest && !m_junctionPos.empty()) {
    // source current junction
    Ptr<MobilityModel> mob = GetNode()->GetObject<MobilityModel>();
    Vector me = mob->GetPosition();
    double best = 1e18;
    for (auto &kvp : m_junctionPos) {
      Vector jp = kvp.second; double dx = me.x - jp.x, dy = me.y - jp.y; double d = std::hypot(dx, dy);
      if (d < best) { best = d; srcJ = kvp.first; }
    }
    // destination nearest junction (if m_dstPos set)
    best = 1e18;
    for (auto &kvp : m_junctionPos) {
      Vector jp = kvp.second; double dx = m_dstPos.x - jp.x, dy = m_dstPos.y - jp.y; double d = std::hypot(dx, dy);
      if (d < best) { best = d; dstJ = kvp.first; }
    }
  }
  obs.push_back((double)m_segADca.size()); // nsd for feedback; ignored in route request
  // segment arrays
  for (uint32_t i=0;i<MAX_PATH;i++) obs.push_back(i<m_segADca.size()? m_segADca[i]:0.0);
  for (uint32_t i=0;i<MAX_PATH;i++) obs.push_back(i<m_segHCca.size()? m_segHCca[i]:0.0);
  for (uint32_t i=0;i<MAX_PATH;i++) obs.push_back(i<m_segLen.size()? m_segLen[i]:0.0); // treat as l_ca
  for (uint32_t i=0;i<MAX_PATH;i++) obs.push_back(i<m_segRCca.size()? m_segRCca[i]:0.0);
  // path ids
  if (isRouteRequest && srcJ!=0 && dstJ!=0) {
    obs.push_back(2.0);
    obs.push_back((double)srcJ);
    obs.push_back((double)dstJ);
    for (uint32_t i=2;i<MAX_PATH;i++) obs.push_back(0.0);
  } else {
    obs.push_back(0.0);
    for (uint32_t i=0;i<MAX_PATH;i++) obs.push_back(0.0);
  }
}

bool EliteForwarderApp::GetAction(std::vector<double>& act, uint8_t& policyId, std::vector<uint32_t>& pathOut) {
  struct Env {
    double simTime;
    int32_t vehicleId;
    int32_t srcId;
    int32_t dstId;
    int32_t msgTypeId;
    double queueLen;
    double bufferBytes;
    int32_t neighborCount;
    int32_t neighbors[64];
    double distToNext[64];
    int32_t routeRequestFlag;
    int32_t feedbackFlag;
    int32_t success;
    double ADsd;
    double HCsd;
    double RCsd;
    double hopCount;
    double PL;
    double RO;
    int32_t nsd;
    double seg_ADca[32];
    double seg_HCca[32];
    double seg_lca[32];
    double seg_RCca[32];
    int32_t path_len;
    int32_t path_ids[32];
  } __attribute__((packed));
  struct Act {
    int32_t policyId;
    int32_t path_len;
    int32_t path_ids[32];
  } __attribute__((packed));
  static Ptr< Ns3AIRL<Env, Act> > rl = CreateObject< Ns3AIRL<Env, Act> >(m_aiKey);
  rl->SetCond(2, 0); // ns-3 even phase
  Env* e = rl->EnvSetterCond();
  if (!e) return false;
  // Fill env
  e->simTime = Simulator::Now().GetSeconds();
  e->vehicleId = (int32_t)m_stationId;
  e->srcId = (int32_t)m_stationId;
  e->dstId = (int32_t)m_dstStationId;
  e->msgTypeId = 1;
  e->queueLen = 0.0;
  e->bufferBytes = 0.0;
  uint32_t ncnt = std::min<size_t>(m_neighbors.size(), 64);
  e->neighborCount = (int32_t)ncnt;
  uint32_t idx=0;
  for (auto& kv : m_neighbors) {
    if (idx>=ncnt) break;
    e->neighbors[idx] = (int32_t)kv.first;
    e->distToNext[idx] = 0.0;
    idx++;
  }
  for (;idx<64;idx++){ e->neighbors[idx]=0; e->distToNext[idx]=0.0; }
  e->routeRequestFlag = 1;
  e->feedbackFlag = 0;
  e->success = 0;
  e->ADsd = m_ADsd;
  e->HCsd = m_HCsd;
  e->RCsd = m_RCsd;
  e->hopCount = 0.0;
  e->PL = m_PL;
  e->RO = m_RO;
  e->nsd = (int32_t)std::min<size_t>(m_segADca.size(), 32);
  for (uint32_t i=0;i<32;i++){
    e->seg_ADca[i] = (i<m_segADca.size()? m_segADca[i]:0.0);
    e->seg_HCca[i] = (i<m_segHCca.size()? m_segHCca[i]:0.0);
    e->seg_lca[i]  = (i<m_segLen.size()?  m_segLen[i]:0.0);
    e->seg_RCca[i] = (i<m_segRCca.size()? m_segRCca[i]:0.0);
  }
  e->path_len = 0;
  for (uint32_t i=0;i<32;i++) e->path_ids[i]=0;
  rl->SetCompleted();
  // Get action
  Act* a = rl->ActionGetterCond();
  if (!a) return false;
  policyId = (uint8_t) std::max(0, a->policyId);
  int plen = std::max(0, a->path_len);
  plen = std::min(plen, 32);
  pathOut.clear();
  for (int i=0;i<plen;i++){ pathOut.push_back((uint32_t)a->path_ids[i]); }
  rl->GetCompleted();
  return true;
}

} // namespace ns3
