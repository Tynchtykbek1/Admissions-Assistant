# Company knowledge scopes

`company_demo.json` contains records with explicit usage scopes:

- `production`: approved company facts permitted in production and demo imports;
- `demo`: preliminary facts permitted only in demo/staging-demo imports;
- `test`: synthetic data reserved for automated tests and rejected by the CLI.

The importer defaults to `production`. A staging demo that intentionally uses
preliminary information must set `DEMO_MODE=true` for that process and import a
separate demo composite document explicitly:

```powershell
python scripts/import_knowledge_pack.py knowledge/company_demo.json `
  --database PATH_TO_STAGING_COPY `
  --document-name company-demo-knowledge `
  --base-document-id BASE_ID `
  --scope demo `
  --apply
```

Configure that demo composite document as the staging system document. Do not
reuse it as a production system document. Production import omits `--scope` or
uses `--scope production`, and `DEMO_MODE` remains false by default.
