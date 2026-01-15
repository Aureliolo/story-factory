# Security Policy

## Supported Versions

Story Factory is a hobby project currently in active development. We support the latest version on the main branch.

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Story Factory, please report it responsibly:

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. Email the maintainer directly or use GitHub's private vulnerability reporting feature
3. Include detailed information about the vulnerability:
   - Description of the issue
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if you have one)

### What to Expect

- **Response Time**: We aim to acknowledge reports within 48 hours
- **Fix Timeline**: Depends on severity and complexity
- **Credit**: Security researchers will be credited (unless they prefer to remain anonymous)

## Security Considerations

Story Factory runs locally on your machine and:

- **Does NOT collect or transmit data** to external servers (except to your local Ollama instance)
- **Does NOT require API keys** or credentials
- **Stores data locally** in your project directory

### Best Practices

When using Story Factory:

1. **Review Generated Content**: AI-generated content should be reviewed before publication
2. **Keep Ollama Updated**: Ensure your Ollama installation is up to date
3. **Use Trusted Models**: Only use models from trusted sources
4. **Local Storage**: Story data is stored locally - ensure proper file system permissions
5. **Network Access**: The web UI binds to localhost by default - be cautious when changing this

## Dependencies

We use Dependabot to monitor dependencies for known vulnerabilities. Updates are reviewed and merged regularly.

## Disclaimer

This is a personal hobby project provided "as is" without warranty. Use at your own risk. See LICENSE for full terms.
