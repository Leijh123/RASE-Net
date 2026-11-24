# Publishing on GitHub

Upload the contents of the repository directory, not the outer ZIP file. Start with README.md, configs, docs, rase, experiments, tests, provenance, pyproject.toml, requirements.txt, .gitignore and THIRD_PARTY_NOTICES.md.

```bash
git init
git add .
git status --short
git commit -m "Organize RASE-Net source and documentation"
```

Create your intended repository in GitHub and follow its remote/push instructions. No repository or remote is created by this package. Keep private data, local tokenizer files, checkpoints and caches outside Git. Confirm upstream redistribution terms and your own license before choosing public visibility.
