# NCIForge 1.0.9

NCIForge 1.0.9 is the frozen software release evaluated in the accompanying
manuscript, *NCIForge: A Hardware-Aware Interaction-Informatics Platform for
Non-Covalent Interaction Fields, Fingerprints, and Searchable Identities*.

## Scientific protocol

- Strict single-point mode (`--sp`) preserves supplied Cartesian coordinates
  and skips contact seeding, pre-optimisation, and geometry optimisation.
- Contact seeding is an explicit opt-in operation through `--seed-contact`.
- Production `f3` uses parsed interfragment xTB Wiberg bond order. The older
  identity-overlap estimate remains explicitly experimental.
- Missing COSMO information is reported as unavailable rather than as a
  physical zero.
- KNF, SNCI, SCDI, K-UID, batch, and atlas outputs carry protocol and
  availability metadata needed for downstream interpretation.

## Performance and execution

- GPU-routed xTB and Torch NCI share a device lock, preventing device
  contention in mixed-size batches.
- Torch basis evaluation groups identical Cartesian powers to reduce small
  kernel launches.
- Parsed Molden wavefunctions and NCI grid payloads are reused where possible.
- The Windows `xtbx` route includes a compact CPU runtime and supports explicit
  discovery of compatible GPU runtimes.

## Interfaces and compatibility

- `nciforge` is the primary command; `knf` remains a compatibility alias.
- `geoinit`, `xtbx`, and `nciforge-api` are installed as companion commands.
- Existing `knf.json`, `output.txt`, `f1`--`f9`, and K-UID family outputs remain
  supported.
- Python 3.10, 3.11, and 3.12 are supported.

## Validation

- The normally enabled test suite passes on the frozen release.
- Opt-in live-xTB regression tests cover UFF+xTB, GeoInit+xTB, and strict
  single-point coordinate/WBO behavior.
- Source distributions and wheels are checked with `twine`, and installed-wheel
  smoke tests cover the CLI, API health endpoint, GeoInit, and xTB wrapper.

See [CHANGELOG.md](CHANGELOG.md) for detailed changes and
[README.md](README.md) for installation, usage, limitations, and citation.
