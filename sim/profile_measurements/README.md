# Profile Measurement Sources

This directory keeps the measurement scripts and source traces used to derive
the simulator profiles.

- `database/`: AlloyDB latency measurements and helper scripts.
- `network/`: 24-hour GCP/AWS RTT and bandwidth probes, including probe
  scripts, run metadata, provenance files, and raw result CSVs.

The simulator does not read this directory directly. Runtime profiles are kept
under `data/measurement/`, where the code in `src/measurement/` loads the
execution-latency CSVs and network trace categories.
