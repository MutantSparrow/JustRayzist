# Release Packaging

Use these commands from repository root.

## Bootstrap Artifact (Recommended)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\package_release.ps1 -Mode bootstrap -Lane cu128 -Version vX.Y.Z -Clean
```

## Bundled Artifact (Large, Optional)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\package_release.ps1 -Mode bundled -Lane cu128 -Version vX.Y.Z -Clean
```

## Repository Readiness Check

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\verify_repo_readiness.ps1
```

## Cleanup Legacy Artifacts

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\clean_legacy_artifacts.ps1
```

## CUDA Lane Baseline

- `cu126`: NVIDIA driver `>= 561.17` (20xx/30xx/40xx fallback lane)
- `cu128`: NVIDIA driver `>= 572.61` (preferred lane; required for 50xx)
