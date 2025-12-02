# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue. Instead, please report it via one of the following methods:

1. **Email**: Send details to `attobop@gmail.com` with the subject line `[SECURITY] rank-refine vulnerability report`
2. **GitHub Security Advisory**: Use GitHub's private vulnerability reporting feature (if enabled)

### What to Include

Please include the following information in your report:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact and severity assessment
- Suggested fix (if any)
- Your contact information

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity, but we aim for:
  - **Critical**: 24-48 hours
  - **High**: 1 week
  - **Medium**: 2-4 weeks
  - **Low**: Next release cycle

### Security Best Practices

When using `rank-refine`:

1. **Keep dependencies updated**: Regularly update to the latest version
2. **Validate inputs**: Always validate vector dimensions and ensure they match expected sizes
3. **Handle edge cases**: Be aware of empty vectors, NaN values, and division by zero scenarios
4. **SIMD considerations**: Ensure your target platform supports the SIMD features used
5. **Review code**: This is open source - review the code for your use case

### Known Security Considerations

- **No network access**: This crate performs no network operations
- **No file system access**: All operations are in-memory
- **Input validation**: Callers are responsible for validating vector dimensions and inputs
- **Memory safety**: Built with Rust's memory safety guarantees
- **SIMD operations**: Uses platform-specific SIMD instructions (safe, but platform-dependent)

## Security Updates

Security updates will be released as patch versions (e.g., 0.7.36 → 0.7.37) and will be announced in:

- GitHub Releases
- Changelog
- Security advisories (for critical issues)

## Acknowledgments

We appreciate responsible disclosure. Security researchers who report vulnerabilities will be credited (with permission) in our security advisories and release notes.

