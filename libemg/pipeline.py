"""Build and run LibEMG pipelines, with or without the editor.

The visual editor is a view onto a pipeline, not the pipeline itself. This
module is the public way in: it exposes the same document, registry and
compiler the editor drives, and it can run a saved pipeline from a command
line with no display attached.

That matters for more than convenience. A pipeline that only exists inside a
GUI cannot be scripted, cannot be checked into a repository usefully, and
cannot be run by continuous integration. Here a pipeline is a file, and running
one is a command.

Examples
---------
Build one in Python and run it::

    from libemg.pipeline import PipelineDocument, compile_pipeline

    doc = PipelineDocument()
    source = doc.add_node('source.synthetic_streamer', params={'pattern': 'bursts'})
    window = doc.add_node('window.enframe', params={'window_size': 200,
                                                    'window_increment': 50})
    features = doc.add_node('features.extract', params={'features': ['MAV', 'RMS']})
    doc.connect(source, 'emg', window, 'input')
    doc.connect(window, 'output', features, 'input')
    doc.add_probe(features, 'output')
    doc.save('pipeline.json')

Or run a saved one from a terminal::

    python -m libemg.pipeline train pipeline.json
    python -m libemg.pipeline run pipeline.json --seconds 30
    python -m libemg.pipeline check pipeline.json
    python -m libemg.pipeline blocks

``train`` fits the model a stored-data pipeline names, on the recording that
pipeline reads, and writes it where the model block already points. So the same
file describes the fit and the run, and continuous integration can do both.
"""

import argparse
import json
import sys
import time

from libemg._gui._pipeline.compile import (CompileError, OfflinePipeline,
                                           OnlinePipeline, compile_pipeline,
                                           tag_for)
from libemg._gui._pipeline.document import (Link, Node, PipelineDocument, Probe,
                                            SCHEMA_VERSION, ValidationError)
from libemg._gui._pipeline.registry import (NodeSpec, ParamSpec, PortSpec,
                                            PortType, build_registry,
                                            default_registry)
from libemg._gui._pipeline.synthetic import SyntheticStreamer, synthetic_streamer

__all__ = ["PipelineDocument", "Node", "Link", "Probe", "SCHEMA_VERSION",
           "ValidationError", "compile_pipeline", "CompileError",
           "OnlinePipeline", "OfflinePipeline", "tag_for",
           "NodeSpec", "ParamSpec", "PortSpec", "PortType",
           "build_registry", "default_registry",
           "synthetic_streamer", "SyntheticStreamer", "main"]


def _describe_blocks(registry):
    lines = []
    grouped = {}
    for spec in registry.values():
        grouped.setdefault(spec.category, []).append(spec)
    for category in sorted(grouped):
        lines.append(f"{category}:")
        for spec in sorted(grouped[category], key=lambda s: s.id):
            inputs = ", ".join(f"{p.name}:{p.type}" for p in spec.inputs) or "-"
            outputs = ", ".join(f"{p.name}:{p.type}" for p in spec.outputs) or "-"
            lines.append(f"  {spec.id}")
            lines.append(f"      in  {inputs}")
            lines.append(f"      out {outputs}")
            if spec.params:
                lines.append("      params " + ", ".join(p.name for p in spec.params))
    return "\n".join(lines)


def _check(path, registry):
    document = PipelineDocument.load(path, registry=registry)
    problems = document.validate()
    print(f"{path}: {len(document.nodes)} blocks, {len(document.links)} links, "
          f"{len(document.probes)} probes, mode '{document.mode()}'")
    if problems:
        print("not ready:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("ready to run.")
    return 0


def _train(path, registry, model_path, quiet):
    """Fit the model a saved pipeline names, on the recording it reads."""
    document = PipelineDocument.load(path, registry=registry)
    try:
        pipeline = compile_pipeline(document)
    except CompileError as error:
        print(error, file=sys.stderr)
        return 1
    if not isinstance(pipeline, OfflinePipeline):
        print("Training reads stored data. This pipeline's source is a device.",
              file=sys.stderr)
        return 1

    width = 40

    def show(fraction):
        filled = int(fraction * width)
        print(f"\r[{'#' * filled}{'.' * (width - filled)}] {fraction * 100:5.1f}%",
              end="", flush=True)

    try:
        result = pipeline.train(on_progress=None if quiet else show,
                                model_path=model_path)
    except CompileError as error:
        if not quiet:
            print()
        print(error, file=sys.stderr)
        return 1
    if not quiet:
        print()
    print(json.dumps({k: _jsonable(v) for k, v in result.items()}, indent=2))
    return 0


def _run(path, registry, seconds, quiet):
    from libemg.event_log import EventLog

    document = PipelineDocument.load(path, registry=registry)
    log = None if quiet else EventLog(keep=0, to_stdout=False)
    try:
        pipeline = compile_pipeline(document, log=log)
    except CompileError as error:
        print(error, file=sys.stderr)
        return 1

    if isinstance(pipeline, OfflinePipeline):
        # A run over a recording knows its own total, so it can report a real
        # fraction rather than a spinner.
        width = 40

        def show(fraction):
            filled = int(fraction * width)
            print(f"\r[{'#' * filled}{'.' * (width - filled)}] {fraction * 100:5.1f}%",
                  end="", flush=True)

        results = pipeline.run(on_progress=None if quiet else show)
        if not quiet:
            print()
        if results:
            print(json.dumps({k: _jsonable(v) for k, v in results.items()}, indent=2))
        else:
            print("The run finished. Add an Offline Metrics block to score it.")
        return 0

    if log is not None:
        log.start()
    pipeline.start()
    print(f"running for {seconds:g}s. Ctrl-C to stop early.")
    try:
        deadline = time.time() + seconds
        while time.time() < deadline:
            time.sleep(0.5)
            if not quiet:
                status = pipeline.status()
                parts = [f"{t.replace('pipe_', '')}={s.total_samples}"
                         for t, s in sorted(status.items()) if not t.startswith("probe_")]
                print("\r" + "  ".join(parts)[:150], end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        pipeline.stop()
        if log is not None:
            log.stop()
    return 0


def _jsonable(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
    except ImportError:
        pass
    return value


def main(argv=None):
    """Command-line entry point.

    Parameters
    ----------
    argv: list or None (optional), default=None
        Arguments to parse. Defaults to the process arguments.

    Returns
    ----------
    int
        A process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="python -m libemg.pipeline",
        description="Build and run LibEMG pipelines without the editor.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run a saved pipeline.")
    run.add_argument("path", help="The pipeline file.")
    run.add_argument("--seconds", type=float, default=30.0,
                     help="How long to run a live pipeline. Ignored offline.")
    run.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    check = commands.add_parser("check", help="Validate a saved pipeline.")
    check.add_argument("path", help="The pipeline file.")

    train = commands.add_parser(
        "train", help="Fit the model a stored-data pipeline names.")
    train.add_argument("path", help="The pipeline file.")
    train.add_argument("--model-path", default=None,
                       help="Where to save. Defaults to the model block's own "
                            "Fitted Model path.")
    train.add_argument("--quiet", action="store_true",
                       help="Suppress progress output.")

    commands.add_parser("blocks", help="List the blocks that can be placed.")

    arguments = parser.parse_args(argv)
    registry = default_registry()
    if arguments.command == "blocks":
        print(_describe_blocks(registry))
        return 0
    if arguments.command == "check":
        return _check(arguments.path, registry)
    if arguments.command == "train":
        return _train(arguments.path, registry, arguments.model_path,
                      arguments.quiet)
    return _run(arguments.path, registry, arguments.seconds, arguments.quiet)


if __name__ == "__main__":
    sys.exit(main())
