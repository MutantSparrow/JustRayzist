# Release Packaging

Use these commands from repository root.

## Release Artifact

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\package_release.ps1 -Lane cu128 -Version vX.Y.Z -Clean
```

## Repository Readiness Check

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\verify_repo_readiness.ps1
```

Use this before packaging to confirm the source tree still matches the expected release surface:

- launcher prompts only for public enabled pack selection and local/LAN mode, and skips pack choice entirely when only one pack is enabled
- no normal-user startup path asks for explicit runtime profile selection
- docs/examples still match the current API, CLI, and model-pack behavior
- release packaging should carry tracked pack metadata only and continue relying on the installer for model weights

## Cleanup Legacy Artifacts

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\clean_legacy_artifacts.ps1
```

## CUDA Lane Baseline

- `cu126`: NVIDIA driver `>= 561.17` (20xx/30xx/40xx fallback lane)
- `cu128`: NVIDIA driver `>= 572.61` (preferred lane; required for 50xx)

## Packaging Notes

- packaged releases are bootstrap-only
- hidden packs remain engineering-only and should not appear in the normal launcher flow
- `UpdateApp.bat` is the packaged in-place update path and should preserve local models, outputs, and data