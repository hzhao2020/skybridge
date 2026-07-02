# Profile Measurement Sources

This directory keeps the measurement scripts and staging locations used to
derive the simulator profiles. Historical traces are intentionally not tracked;
collect fresh results before populating runtime profiles.

- `database/`: AlloyDB latency helper scripts.
- `network/`: RTT and bandwidth probe scripts plus an ignored `results/`
  staging directory for raw outputs, run metadata, and provenance files.

The simulator does not read this directory directly. Runtime profiles are kept
under `data/measurement/`, where the code in `src/measurement/` loads the
execution-latency CSVs and network trace categories after they are generated.
