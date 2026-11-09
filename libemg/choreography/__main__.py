"""``python -m libemg.choreography`` -- inspect, recover or re-cut a choreography recording.

    python -m libemg.choreography show data/
    python -m libemg.choreography recover data/
    python -m libemg.choreography cut-repetitions data/ --out data_repetitions/

``recover`` rebuilds ``_session_choreography.json`` (and the session annotation
stages and per-sample target files) from ``_session_choreography_events.jsonl``
after the application died mid-run. Intervals still open when it died are
closed at the journal's last heartbeat and marked ``interrupted``.

``cut-repetitions`` writes each completed repetition, ramp-in start to ramp-out
end, as a data file and a matching per-sample target file (class, phase,
position, intensity), for file-per-repetition regression loaders. The output
folder must be outside the recording folder.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .annotations import RUN_RECORD_NAME, recover_run


def _show(folder):
    record = json.loads((Path(folder) / RUN_RECORD_NAME).read_text(encoding="utf-8"))
    print(f"run {record['run_id']} ({record.get('run_mode')}): {record['status']}"
          + (f", {record['end_reason']}" if record.get("end_reason") else ""))
    print(f"  order {record.get('order_mode')} seed {record.get('shuffle_seed')}; "
          f"cycles {record['schedule']['cycles']}")
    intensity = record.get("intensity") or {}
    if intensity.get("enabled"):
        levels = ", ".join(f"{100 * level:g}%" for level in intensity.get("levels", []))
        print(f"  intensity targets: {levels} ({intensity.get('order')})")
    summary = record["summary"]
    print(f"  repetitions: {summary['repetitions_completed']} completed of "
          f"{summary['repetitions_requested']} requested; {summary['pauses']} pause(s)")
    phases = Counter(iv["derived_label"] for iv in record["intervals"] if iv["track"] == "phase")
    for label, count in sorted(phases.items()):
        print(f"    {label:<28} {count}")
    for modality, name in sorted((record.get("target_files") or {}).items()):
        print(f"  {modality} targets: {name}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m libemg.choreography",
                                     description="Inspect, recover or re-cut a choreography "
                                                 "recording.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("show", "Summarise a run record."),
                       ("recover", "Rebuild the run record from the event journal.")):
        command = commands.add_parser(name, help=text)
        command.add_argument("folder", help="The recording's output folder.")
    cut = commands.add_parser("cut-repetitions",
                              help="Write each repetition with its per-sample targets.")
    cut.add_argument("folder", help="The recording's output folder.")
    cut.add_argument("--out", required=True,
                     help="Where to write (outside the recording folder).")
    cut.add_argument("--modality", action="append", dest="modalities",
                     help="Only this modality (repeatable). Defaults to all.")
    cut.add_argument("--include-incomplete", action="store_true",
                     help="Also write repetitions that were stopped or interrupted.")
    cut.add_argument("--lead-in", type=float, default=0.0,
                     help="Seconds of the preceding rest to include.")
    cut.add_argument("--lead-out", type=float, default=0.0,
                     help="Seconds of the following rest to include.")
    arguments = parser.parse_args(argv)
    if arguments.command == "cut-repetitions":
        from .targets import cut_repetitions
        written = cut_repetitions(arguments.folder, arguments.out, arguments.modalities,
                                  arguments.include_incomplete, arguments.lead_in,
                                  arguments.lead_out)
        print(f"Wrote {len(written)} files to {arguments.out}.")
        return 0
    if arguments.command == "recover":
        path = recover_run(arguments.folder)
        print(f"Wrote {path}.")
    return _show(arguments.folder)


if __name__ == "__main__":
    sys.exit(main())
