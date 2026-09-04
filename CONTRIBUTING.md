# Contributing to Pedacito

Thanks for your interest in contributing to Pedacito! We appreciate your enthusiasm.

## ⚠️ Project Status

**Pedacito is currently in active development and not yet ready for use.** While the repository is public, the API and architecture are still being designed and may change significantly.

## Before You Contribute

**Please open a GitHub Discussion or Issue first** to discuss major contributions. This helps ensure your work aligns with the project's direction and prevents wasted effort.

## What We're Looking For

At this stage, we welcome:

### ✅ Low-Risk Contributions
- **Documentation improvements** – Better READMEs, docstrings, or inline comments
- **Bug reports** – Issues with existing code (even if it's experimental)
- **Questions and discussions** – Help us understand pain points and use cases
- **Typo fixes** – Grammar and spelling corrections

### ⚠️ Approach With Caution
- **New features** – The architecture is still evolving; discuss your idea first
- **Large refactors** – Core systems are being actively designed
- **Dependencies** – We're keeping the dependency list minimal; propose major additions as issues first

### ❌ Not Accepting Yet
- Pull requests that significantly change the core architecture
- New major features without prior discussion
- Third-party integrations until the core is stable

## How to Contribute (When Ready)

1. **Fork the repository**
2. **Create a branch** for your work: `git checkout -b your-feature-branch`
3. **Make your changes** and test them locally
4. **Commit with clear messages**: `git commit -m "Brief description of changes"`
5. **Push to your fork** and open a pull request
6. **Reference the discussion or issue** that motivated your change

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pedacito.git
cd pedacito

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install development dependencies (when they exist)
pip install pytest black flake8
```

## Code Style

When submitting code, please follow these guidelines:

- Use meaningful variable and function names
- Keep functions focused and reasonably sized
- Add docstrings to public functions and classes
- Format with consistent indentation (4 spaces)

We'll establish more formal style guidelines as the project matures.

## Questions?

- **Open a GitHub Discussion** for design questions and ideas
- **Open an Issue** for bugs or specific problems
- **Check existing issues** to see if your question has been answered

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

---

Thank you for being part of the Pedacito community! We look forward to building something great together. 🎉
