# Security Policy

## SAIL's Security Philosophy

SAIL is designed with security and privacy as foundational principles. As a privacy-first, locally-hosted AI companion, security is not an afterthought but a core design requirement.

### Core Security Principles

1. **Local-Only Processing**: All data processing occurs on user-controlled hardware
2. **No External Communication**: Core functionality never phones home or sends data externally
3. **Data Sovereignty**: Users maintain complete control over their data
4. **Minimal Attack Surface**: Dependencies are carefully selected and minimized

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report vulnerabilities by:

1. **Email**: Send details to the project maintainers (see repository for contact information)
2. **Private Disclosure**: Use GitHub's private vulnerability reporting feature if available

### What to Include

Please provide:

- **Description**: A clear description of the vulnerability
- **Impact**: What an attacker could achieve by exploiting this vulnerability
- **Reproduction Steps**: Detailed steps to reproduce the issue
- **Affected Components**: Which parts of SAIL are affected
- **Suggested Fix**: If you have ideas for remediation (optional)

### Response Timeline

- **Acknowledgment**: Within 48 hours of report
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity (see below)

### Severity Levels

| Severity | Description | Target Resolution |
|----------|-------------|-------------------|
| Critical | Data exfiltration, remote code execution | 24-72 hours |
| High | Privilege escalation, authentication bypass | 7 days |
| Medium | Information disclosure, denial of service | 30 days |
| Low | Minor issues with limited impact | 90 days |

## Security Best Practices for Users

### Installation

- Verify checksums of downloaded releases
- Use virtual environments to isolate dependencies
- Keep Python and system packages updated

### Configuration

- Store configuration files with appropriate permissions (600 or 640)
- Use strong, unique wake words
- Review family configuration for appropriate access controls

### Network Security

- SAIL is designed for local operation; avoid exposing it to public networks
- If running hub/node architecture, use a trusted local network
- Consider network segmentation for IoT deployments

### Data Protection

- Context data is stored locally in encrypted SQLite databases
- Regularly backup your SAIL data directory
- Secure physical access to devices running SAIL

## Security Features

### Built-in Protections

- **Context Isolation**: Per-user data siloing
- **Role-Based Access**: Admin/User/Minor permission levels
- **Constitutional Constraints**: Behavioral guardrails for the AI
- **Input Validation**: Pydantic-based schema validation for all configuration

### Audit Logging

SAIL maintains local audit logs for security-relevant events. These logs:
- Never leave the local system
- Can be reviewed for suspicious activity
- Are configurable for retention period

## Known Security Considerations

### Voice Interface

- Wake word detection operates continuously on audio
- Audio is processed locally and not stored beyond the context buffer
- Consider physical security of microphone-equipped nodes

### LLM Integration

- Local LLM inference via Ollama or llama.cpp
- No API keys or cloud services required for core functionality
- Model files should be obtained from trusted sources

### Sensor Data

- GPS, accelerometer, and biometric data are processed locally
- Sensor permissions are managed at the OS level
- Users should review which sensors are enabled

## Vulnerability Disclosure Policy

We follow coordinated disclosure:

1. Reporter contacts us privately
2. We acknowledge and assess the report
3. We develop and test a fix
4. We release the fix and publish an advisory
5. Reporter may be credited (with permission)

## Security Updates

Security updates are released as:
- Patch releases for the current version
- Advisories published in GitHub Security Advisories
- Changelog entries noting security fixes

Subscribe to repository notifications to stay informed about security updates.

---

Thank you for helping keep SAIL and its users secure.
