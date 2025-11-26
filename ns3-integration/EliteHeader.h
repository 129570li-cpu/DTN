// ns3-integration/EliteHeader.h
// Custom header for ELITE routing: carries area_path, policyId, nextIndex, etc.
#pragma once
#include "ns3/header.h"
#include <vector>
#include <cstdint>
#include <ostream>

namespace ns3 {

static const uint32_t ELITE_MAX_PATH_LEN = 32;

class EliteHeader : public Header
{
public:
  enum PolicyId : uint8_t { BP=0, HRF=1, LDF=2, LBF=3 };

  EliteHeader() :
    m_policyId(BP), m_routeLen(0), m_nextIndex(0), m_msgType(1), m_ttl(64), m_pathId(0) {
      m_route.assign(ELITE_MAX_PATH_LEN, 0);
    }

  void SetPolicy(PolicyId p) { m_policyId = p; }
  PolicyId GetPolicy() const { return (PolicyId)m_policyId; }

  void SetPath(const std::vector<uint32_t>& path) {
    m_routeLen = std::min<uint32_t>(path.size(), ELITE_MAX_PATH_LEN);
    m_route.assign(ELITE_MAX_PATH_LEN, 0);
    for (uint32_t i=0;i<m_routeLen;i++) m_route[i] = path[i];
    m_nextIndex = 0;
  }
  uint32_t GetPathLen() const { return m_routeLen; }
  const std::vector<uint32_t>& GetPath() const { return m_route; }

  void IncNext() { if (m_nextIndex + 1 < m_routeLen) ++m_nextIndex; }
  uint32_t GetNextIndex() const { return m_nextIndex; }
  uint32_t GetCurrentTarget() const { return (m_nextIndex < m_routeLen)? m_route[m_nextIndex] : 0; }

  void SetMsgType(uint8_t t){ m_msgType = t; }
  uint8_t GetMsgType() const { return m_msgType; }

  void SetTtl(uint8_t ttl) { m_ttl = ttl; }
  uint8_t GetTtl() const { return m_ttl; }

  void SetPathId(uint32_t pid) { m_pathId = pid; }
  uint32_t GetPathId() const { return m_pathId; }

  // ns-3 Header API
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("ns3::EliteHeader")
      .SetParent<Header>()
      .AddConstructor<EliteHeader>();
    return tid;
  }
  virtual TypeId GetInstanceTypeId() const override { return GetTypeId(); }
  virtual void Print(std::ostream &os) const override {
    os<<"EliteHeader[pol="<<(int)m_policyId<<", len="<<m_routeLen
      <<", idx="<<m_nextIndex<<", msg="<<(int)m_msgType
      <<", ttl="<<(int)m_ttl<<", pathId="<<m_pathId<<"]";
  }
  virtual uint32_t GetSerializedSize() const override {
    return 1 + 1 + 1 + 1 + 1 + 4 + 4*ELITE_MAX_PATH_LEN;
  }
  virtual void Serialize(Buffer::Iterator start) const override {
    start.WriteU8(m_policyId);
    start.WriteU8(m_routeLen);
    start.WriteU8(m_nextIndex);
    start.WriteU8(m_msgType);
    start.WriteU8(m_ttl);
    start.WriteU32(m_pathId);
    for (uint32_t i=0;i<ELITE_MAX_PATH_LEN;i++) start.WriteU32(m_route[i]);
  }
  virtual uint32_t Deserialize(Buffer::Iterator start) override {
    m_policyId = start.ReadU8();
    m_routeLen = start.ReadU8();
    m_nextIndex = start.ReadU8();
    m_msgType = start.ReadU8();
    m_ttl = start.ReadU8();
    m_pathId = start.ReadU32();
    m_route.assign(ELITE_MAX_PATH_LEN, 0);
    for (uint32_t i=0;i<ELITE_MAX_PATH_LEN;i++) m_route[i] = start.ReadU32();
    return GetSerializedSize();
  }

private:
  uint8_t m_policyId;
  uint8_t m_routeLen;
  uint8_t m_nextIndex;
  uint8_t m_msgType;
  uint8_t m_ttl;
  uint32_t m_pathId;
  std::vector<uint32_t> m_route;
};

} // namespace ns3

