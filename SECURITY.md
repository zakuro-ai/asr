# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities **privately** to **security@zakuro-ai.com**.
Do not open a public issue for security problems.

Include where possible:
- A description of the vulnerability and its impact
- Steps to reproduce or a proof of concept
- Affected version(s) / commit

We aim to acknowledge reports within 3 business days.

## Supported Versions

Security fixes are applied to the latest released version and the default
(`main`) branch. Older versions are not maintained.

## Secrets

Never commit secrets (`.env`, private keys, credentials, database dumps) to this
repository. Such files must be listed in `.gitignore`. If a secret is committed,
treat it as compromised: rotate it immediately and purge it from git history.
