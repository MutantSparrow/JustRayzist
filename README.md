# JustRayzist Repository

Top-level layout:

- `JustRayzist/` - application source, scripts, models, tests, launchers.
- `dist/` - packaged portable build artifacts.
- `JustRayzist.md` - persistent roadmap/TODO used for development continuity.

## Working Commands

Run app/dev commands from `JustRayzist/`:

```powershell
cd .\JustRayzist
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_env.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model_assets.ps1
.\StartWeb.bat
```

Build portable artifacts into root `dist/`:

```powershell
powershell -ExecutionPolicy Bypass -File .\JustRayzist\scripts\build_portable.ps1 -Clean
powershell -ExecutionPolicy Bypass -File .\JustRayzist\scripts\package_release.ps1
```
