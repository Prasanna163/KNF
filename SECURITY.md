# Security Policy

## Supported version

Security fixes are applied to the latest release line and `main`. The frozen
paper release remains available for reproducibility, but fixes after that tag
are documented separately.

## Reporting a vulnerability

Please use GitHub's private vulnerability-reporting feature for this
repository. Do not open a public issue containing credentials, exploit details,
private molecular data, or identifying filesystem paths.

Include the affected version or commit, operating system, reproduction steps,
impact, and any suggested mitigation. Maintainers will acknowledge a complete
report as soon as practical and coordinate disclosure after a fix is available.

## Credentials and scientific data

Never commit API tokens, private keys, `.env` files, proprietary molecular
inputs, or unpublished result directories. Assume any credential committed to
Git history is compromised and revoke it even if a later commit removes it.
