#pragma once
#include <cstdint>
#include <vector>

// qec_tile native core. 헤더 온리라 nanobind 바인딩과 C++ 단위 테스트가
// 모두 직접 #include 할 수 있다.
namespace qec_tile {

// 툴체인 검증용 스캐폴드 — 디코더 단계에서 실제 구현으로 대체된다.
inline int add(int a, int b) { return a + b; }

// decode 시그니처의 예고편: 비트 벡터를 mod 2로 접는다 (패리티).
inline uint8_t parity(const std::vector<uint8_t>& bits) {
    uint8_t acc = 0;
    for (uint8_t b : bits) acc ^= (b & 1u);
    return acc;
}

}  // namespace qec_tile
