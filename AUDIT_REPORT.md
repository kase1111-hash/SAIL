# SAIL Software Audit Report

> **Note:** This audit was conducted on 2026-01-28 against version 0.1.0. Since then,
> the `src/deployment/` module (provisioning, OTA, health monitoring) has been deleted
> entirely, and mobile/vehicle node code has been removed. Issues referencing those
> modules are no longer applicable. Other findings may still be relevant and should be
> verified against the current codebase.

**Date:** 2026-01-28
**Auditor:** Claude Code
**Version:** 0.1.0 Pre-Alpha
**Repository:** SAIL (Situational Awareness Interaction Layer)

---

## Executive Summary

This audit evaluates the SAIL codebase for **correctness** and **fitness for purpose**. SAIL is a privacy-first, locally-hosted AI companion designed to provide contextual guidance for everyday situations where knowledge gaps could lead to harm.

### Overall Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Architecture** | ✅ Good | Well-designed modular architecture with clear separation of concerns |
| **Code Quality** | ⚠️ Moderate | Solid foundation but numerous bugs and incomplete implementations |
| **Security** | ❌ Critical Issues | Multiple security vulnerabilities requiring immediate attention |
| **Test Coverage** | ⚠️ Moderate | ~65-75% estimated coverage with critical gaps |
| **Fitness for Purpose** | ⚠️ Not Production Ready | Requires significant fixes before deployment |

### Key Statistics

- **Total Issues Identified:** 180+
- **Critical Issues:** 32
- **High Priority Issues:** 48
- **Medium Priority Issues:** 65+
- **Low Priority Issues:** 35+
- **Lines of Code Audited:** ~22,427

---

## Critical Issues Requiring Immediate Attention

### 1. Security Vulnerabilities

#### 1.1 Prompt Injection Vulnerability
**Location:** `src/core/llm/prompts.py:46-69, 116-119`
**Severity:** CRITICAL
**Issue:** No input sanitization before embedding user variables into prompts. Attackers can manipulate SAIL's behavior by providing malicious values for template variables.

```python
# VULNERABLE: User input directly substituted
tmpl = Template(self.template)
return tmpl.safe_substitute(values)  # "safe" only prevents KeyError, not injection
```

**Fix Required:** Implement input sanitization for all template variables.

#### 1.2 SQL Injection in Context Persistence
**Location:** `src/context/persistence.py:518`
**Severity:** CRITICAL
**Issue:** LIMIT clause uses f-string interpolation instead of parameterized query.

```python
# VULNERABLE
if limit:
    query += f" LIMIT {limit}"
```

**Fix Required:** Use parameterized query: `query += " LIMIT ?"; params.append(limit)`

#### 1.3 Pre-Shared Key Exposed in Registration
**Location:** `src/nodes/base.py:329`
**Severity:** CRITICAL
**Issue:** Pre-shared key sent in plaintext during registration before TLS is established.

**Fix Required:** Hash or encrypt PSK before transmission, or ensure TLS is established first.

#### 1.4 User Impersonation via Explicit Name
**Location:** `src/users/manager.py:331-334`
**Severity:** CRITICAL
**Issue:** No validation of `explicit_name` parameter allows attackers to impersonate any user.

```python
# VULNERABLE: Accepts any name with maximum confidence
if explicit_name:
    user = self.get_user_by_name(explicit_name)
    if user:
        await self._start_session(user, IdentificationMethod.EXPLICIT, 1.0)
```

**Fix Required:** Require secondary authentication for explicit identification.

#### 1.5 TLS Verification Disabled
**Location:** `src/nodes/protocol.py:242-244`
**Severity:** CRITICAL
**Issue:** Certificate verification can be completely disabled, enabling MITM attacks.

**Fix Required:** Remove ability to disable certificate verification in production.

### 2. Memory Safety Issues

#### 2.1 Memory Leak in Context Buffer Index
**Location:** `src/context/buffer.py:69, 89-93`
**Severity:** CRITICAL
**Issue:** `_entry_index` accumulates stale references when deque auto-evicts entries.

**Fix Required:** Synchronize `_entry_index` cleanup with deque operations.

#### 2.2 Memory Leak in Guardian Verifications
**Location:** `src/intervention/modes.py:252, 332, 384`
**Severity:** HIGH
**Issue:** `_pending_verifications` dict grows unbounded when verifications expire without response.

**Fix Required:** Implement cleanup of expired verifications.

#### 2.3 Alert Queue Data Loss
**Location:** `src/users/guardian.py:334`
**Severity:** CRITICAL
**Issue:** Pending alerts queue silently drops alerts when full (100 max).

**Fix Required:** Add overflow handling with logging and persistence.

### 3. Logic Bugs

#### 3.1 NameError in ModeTransition.to_dict()
**Location:** `src/intervention/base.py:295`
**Severity:** CRITICAL
**Issue:** Undefined variable `reason` (should be `self.reason`).

```python
# BUG: 'reason' is undefined
return {
    "reason": reason,  # Should be self.reason
}
```

#### 3.2 Timer State Collision in Intervention Engine
**Location:** `src/intervention/engine.py:150, 289-300`
**Severity:** HIGH
**Issue:** Single timer used for both escalation and de-escalation, causing mode transitions to be lost.

**Fix Required:** Use separate timers for escalation and de-escalation.

#### 3.3 GPS Speed Unit Conversion Bug
**Location:** `src/sensors/location.py:152`
**Severity:** HIGH
**Issue:** GPS speed from gpsd is in knots, not m/s. Incorrect conversion affects all speed-based calculations.

**Fix Required:** Convert knots to m/s: `speed_mps = packet.speed * 0.514444`

#### 3.4 Timestamp Comparison Logic Bug
**Location:** `src/hub/context_sync.py:378-385`
**Severity:** CRITICAL
**Issue:** Uses `datetime.now()` instead of actual hub entry timestamp for conflict resolution.

**Fix Required:** Store and compare actual entry timestamps.

---

## Module-by-Module Findings

### Core LLM Integration (`src/core/llm/`)

| Issue | Severity | Location |
|-------|----------|----------|
| Prompt injection vulnerability | CRITICAL | prompts.py:46-69 |
| Constitution `_add_disclaimers_if_needed()` is stub | CRITICAL | constitution.py:356-359 |
| Unhandled exception in validation loop | CRITICAL | constitution.py:144 |
| Race condition in HTTP client initialization | HIGH | ollama.py:60-65, llamacpp.py:62-67 |
| Missing `top_k` parameter in llamacpp | HIGH | llamacpp.py:92-98 |
| Greedy regex for JSON extraction | HIGH | parsing.py:243-254 |
| Broad exception handling masks errors | HIGH | Multiple locations |

**Total Issues:** 22

### Voice I/O Pipeline (`src/voice/`)

| Issue | Severity | Location |
|-------|----------|----------|
| Event loop access in non-async methods | CRITICAL | manager.py:451, capture.py:259,302,334 |
| Audio stream resource leak on error | CRITICAL | capture.py:244-262 |
| Audio output stream never closes on hardware errors | CRITICAL | tts/output.py:416-431 |
| Race condition in audio accumulator (division by zero) | CRITICAL | vad.py:348 |
| Poor resampling quality in wake word detection | HIGH | openwakeword.py:183-188 |
| Memory leak: BytesIO not closed | HIGH | tts/piper.py:291-314 |
| Thread not properly awaited on shutdown | HIGH | tts/output.py:254-255 |

**Total Issues:** 20

### Context Buffer System (`src/context/`)

| Issue | Severity | Location |
|-------|----------|----------|
| SQL injection in LIMIT clause | CRITICAL | persistence.py:518 |
| Memory leak in entry index | CRITICAL | buffer.py:69, 89-93 |
| Crisis retention not implemented | CRITICAL | buffer.py:625-626 |
| Race condition in session recovery | HIGH | manager.py:506-510 |
| Weak encryption salt (static) | HIGH | persistence.py:67 |
| Unsafe database connection configuration | HIGH | persistence.py:265 |
| Inefficient batch operations | MEDIUM | persistence.py:317-325 |

**Total Issues:** 12

### Intervention Engine (`src/intervention/`)

| Issue | Severity | Location |
|-------|----------|----------|
| NameError: undefined variable `reason` | CRITICAL | base.py:295 |
| Timer state collision | HIGH | engine.py:150, 289-300 |
| Memory leak in pending verifications | HIGH | modes.py:252, 332, 384 |
| Walkthrough state validation missing | MEDIUM | modes.py:534-593 |
| Risk level/mode threshold misalignment | LOW | risk.py:349-372 |

**Total Issues:** 8

### User Management (`src/users/`)

| Issue | Severity | Location |
|-------|----------|----------|
| User impersonation vulnerability | CRITICAL | manager.py:331-334 |
| USER role over-privileged | CRITICAL | permissions.py:135-143 |
| Session race condition | CRITICAL | manager.py:360-373 |
| Restriction modification never enforced | CRITICAL | permissions.py:504-520 |
| Alert queue silent data loss | CRITICAL | guardian.py:334 |
| Session hijacking via indefinite extension | CRITICAL | base.py:384-390 |
| Voice ID embedding dimension mismatch | HIGH | voice_id.py:83-98 |
| Weak encryption key derivation | HIGH | isolation.py:34-43 |

**Total Issues:** 24

### Hub Coordinator (`src/hub/`)

| Issue | Severity | Location |
|-------|----------|----------|
| Timestamp comparison logic bug | CRITICAL | context_sync.py:378-385 |
| Race condition in sync state | CRITICAL | context_sync.py:212-216 |
| Unsafe dict iteration in sync loop | CRITICAL | coordinator.py:617 |
| Auth cleanup never removes entries | CRITICAL | coordinator.py:644-647 |
| Orphaned futures in router | HIGH | router.py:329-350 |
| Queue empty() race condition | HIGH | router.py:368-376 |
| Inconsistent lockout key tracking | HIGH | coordinator.py:251-327 |

**Total Issues:** 12

### Sensor Fusion (`src/sensors/`)

| Issue | Severity | Location |
|-------|----------|----------|
| GPS speed unit conversion bug | HIGH | location.py:152 |
| Variance classification conceptually flawed | MEDIUM | motion.py:248-270 |
| Race condition in lock usage | MEDIUM | fusion.py:280-296 |
| Unprotected callback lists | MEDIUM | base.py:498, 524, 529 |
| Acceleration magnitude orientation bug | MEDIUM | motion.py:359-361 |
| Overlapping acceleration ranges | MEDIUM | motion.py:173-176 |

**Total Issues:** 26

### Knowledge Domains (`src/knowledge/`)

| Issue | Severity | Location |
|-------|----------|----------|
| Safety domain context scoring bug | HIGH | safety.py:692 |
| Jurisdiction applicability logic bug | HIGH | base.py:227-228 |
| Missing aspirin contraindications | HIGH | emergency.py:380-383 |
| Incomplete CPR guidance for children | MEDIUM | emergency.py:155-157 |
| Scam detection false positives | MEDIUM | financial.py:611 |
| No jurisdiction filtering in financial/safety | MEDIUM | financial.py:710 |

**Total Issues:** 18

### Deployment & Nodes (`src/deployment/`, `src/nodes/`)

| Issue | Severity | Location |
|-------|----------|----------|
| Firmware integrity verification not implemented | CRITICAL | ota.py:276-278 |
| Installation verification always returns true | CRITICAL | ota.py:413-421 |
| Rollback mechanism non-functional | CRITICAL | ota.py:423-439 |
| Node discovery always fails | CRITICAL | provisioning.py:303-304 |
| Pre-shared key stored in plaintext | CRITICAL | provisioning.py:341 |
| Frame signature verification bypass | HIGH | protocol.py:128-132 |
| Alert retention bug (memory leak) | HIGH | health.py:510-514 |

**Total Issues:** 34

### Configuration (`src/config/`)

| Issue | Severity | Location |
|-------|----------|----------|
| SSRF risk in LLM base_url | CRITICAL | schema.py:55 |
| Guardian validation incomplete | HIGH | schema.py:144-149 |
| Thread-unsafe module-level state | HIGH | loader.py:275-291 |
| Type conversion corrupts string values | HIGH | loader.py:87-116 |
| Missing engine/provider type validation | MEDIUM | schema.py:53,67,76,84 |
| Relative storage path fragility | MEDIUM | schema.py:113-115 |

**Total Issues:** 17

---

## Test Coverage Assessment

### Coverage Statistics

| Module | Estimated Coverage | Status |
|--------|-------------------|--------|
| Config | 75-85% | Good |
| Context | 85-90% | Excellent |
| Core/LLM | 60-70% | Missing provider implementations |
| Deployment | 80-85% | Good |
| Hub | 80-85% | Good |
| Intervention | 85-90% | Excellent |
| Knowledge | 80-85% | Very Good |
| Nodes | 50-60% | Moderate |
| Sensors | 85-90% | Very Good |
| Users | 85-90% | Very Good |
| Voice | 80-85% | Good |
| CLI | 0% | **UNTESTED** |
| Voice/Manager | 30-40% | Poor |

**Overall Estimate:** 65-75% line coverage

### Critical Testing Gaps

1. **CLI module completely untested**
2. **Voice manager orchestrator poorly covered**
3. **No network error handling tests for LLM providers**
4. **No end-to-end integration tests**
5. **Minimal mocking of external services**
6. **No concurrent access/race condition tests**

---

## Fitness for Purpose Assessment

### Positive Findings

1. **Well-architected modular design** - Clear separation of concerns enables maintainability
2. **Comprehensive intervention framework** - Four modes properly implemented with escalation logic
3. **Strong knowledge domain structure** - Jurisdiction-aware guidance system well-designed
4. **Privacy-first approach** - Local processing maintains user privacy
5. **Multi-tier context buffer** - Sophisticated memory management with appropriate retention policies
6. **Async-first design** - Good use of asyncio for concurrent operations

### Areas Requiring Significant Work

1. **Security hardening** - Multiple critical vulnerabilities must be fixed
2. **OTA update system** - Currently non-functional stubs
3. **Node provisioning** - Discovery and deployment mechanisms incomplete
4. **Error handling** - Many silent failures and overly broad exception catching
5. **Thread safety** - Race conditions in multiple async operations
6. **Medical/legal guidance** - Critical information gaps and missing contraindications

---

## Recommendations

### Priority 1 - Security (Immediate)

1. Fix prompt injection vulnerability in `prompts.py`
2. Fix SQL injection in `persistence.py`
3. Secure pre-shared key transmission in `base.py`
4. Remove user impersonation vulnerability in `manager.py`
5. Enforce TLS verification in production

### Priority 2 - Critical Bugs (This Week)

1. Fix `NameError` in `intervention/base.py:295`
2. Fix timestamp comparison in `context_sync.py`
3. Fix GPS speed unit conversion in `location.py`
4. Fix timer collision in intervention engine
5. Fix memory leaks in context buffer and guardian verifications

### Priority 3 - Safety Content (Urgent)

1. Add aspirin contraindications to emergency guidance
2. Complete CPR guidance for children/infants
3. Implement jurisdiction filtering in financial/safety domains
4. Add content version tracking for legal knowledge

### Priority 4 - Test Coverage (Short Term)

1. Add CLI module tests
2. Add voice manager orchestration tests
3. Add network failure tests for LLM providers
4. Add concurrent access tests
5. Add end-to-end integration tests

### Priority 5 - System Completion (Medium Term)

1. Implement OTA firmware verification
2. Complete node discovery mechanism
3. Implement actual rollback functionality
4. Add proper config encryption for provisioning

---

## Conclusion

SAIL demonstrates a well-conceived architecture for a privacy-first AI companion system. The modular design, intervention framework, and knowledge domain structure show thoughtful engineering. However, the codebase contains **critical security vulnerabilities** and **numerous bugs** that make it **unsuitable for production deployment** in its current state.

The most pressing concerns are:
- **Security vulnerabilities** that could allow user impersonation, data injection, and MITM attacks
- **Memory leaks** that would cause long-running systems to fail
- **Critical logic bugs** that would cause silent failures in emergency situations
- **Incomplete implementations** marked as functional (OTA, provisioning, rollback)

With focused remediation of the identified issues, SAIL could fulfill its mission of providing safe, privacy-preserving contextual guidance. The foundation is solid; the implementation needs refinement.

---

**End of Audit Report**
