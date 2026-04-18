# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


def onnx2engine_rtx(
    onnx_file: str,
    output_file: Path | str | None = None,
    dynamic: bool = False,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    metadata: dict | None = None,
    verbose: bool = False,
    prefix: str = "",
) -> str:
    """Export a YOLO ONNX model to a TensorRT for RTX engine (FP32 only for now).

    Unlike classic TensorRT, TRT-RTX engines are portable across RTX-class GPUs: final kernel
    selection happens at runtime on the target device. This keeps engine files small and
    device-agnostic at the cost of a one-time JIT cost on first load (cacheable).

    FP16/INT8 are not yet supported: TRT-RTX builds strongly-typed JIT networks that forbid the
    FP16/INT8 builder flags, so precision must be expressed in the ONNX graph (FP16 cast or QDQ
    nodes) — wiring planned as a follow-up.
    """
    try:
        import tensorrt_rtx as trt
    except ImportError:
        check_requirements("tensorrt-rtx>=1.4.0")  # Linux/Windows x86_64 wheels only
        import tensorrt_rtx as trt

    LOGGER.info(f"{prefix} starting export with TensorRT-RTX {trt.__version__}...")
    output_file = Path(output_file) if output_file else Path(onnx_file).with_suffix(".rtx.engine")

    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_file):
        raise RuntimeError(f"failed to load ONNX file: {onnx_file}")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        profile = builder.create_optimization_profile()
        min_shape = (1, shape[1], 32, 32)
        max_shape = (*shape[:2], *(int(max(2, d) * 2) for d in shape[2:]))
        for inp in inputs:
            profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
        config.add_optimization_profile(profile)

    LOGGER.info(f"{prefix} building FP32 RTX engine as {output_file}")

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT-RTX engine build failed, check logs for errors")

    with open(output_file, "wb") as t:
        if metadata is not None:
            meta = json.dumps(metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
        t.write(engine)
    return str(output_file)
