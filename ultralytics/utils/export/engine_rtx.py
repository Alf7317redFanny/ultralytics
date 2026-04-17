# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


def _onnx_to_fp16(onnx_file: str) -> str:
    """Convert an FP32 ONNX model to FP16 in place and return the new path.

    TensorRT-RTX builds strongly-typed JIT networks; precision is dictated by the ONNX graph
    rather than builder flags, so `half=True` requires an FP16 ONNX input.
    """
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        check_requirements(("onnx", "onnxconverter_common"))
        import onnx
        from onnxconverter_common import float16

    fp16_path = Path(onnx_file).with_suffix(".fp16.onnx")
    model_fp16 = float16.convert_float_to_float16(onnx.load(onnx_file), keep_io_types=True)
    onnx.save(model_fp16, fp16_path)
    return str(fp16_path)


def onnx2engine_rtx(
    onnx_file: str,
    output_file: Path | str | None = None,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dataset=None,
    metadata: dict | None = None,
    verbose: bool = False,
    prefix: str = "",
) -> str:
    """Export a YOLO ONNX model to a TensorRT for RTX engine.

    Unlike classic TensorRT, TRT-RTX engines are portable across RTX-class GPUs: final kernel
    selection happens at runtime on the target device. This keeps engine files small and
    device-agnostic at the cost of a one-time JIT cost on first load (cacheable).

    Args:
        onnx_file: Path to the ONNX file to be converted.
        output_file: Path to save the RTX engine. Defaults to <onnx>.rtx.engine.
        half: Enable FP16 precision.
        int8: Enable INT8 precision. Requires `dataset` for calibration.
        dynamic: Enable dynamic input shapes.
        shape: Input shape (batch, channels, height, width).
        dataset: Calibration dataloader for INT8.
        metadata: Metadata dict serialized into the engine header.
        verbose: Enable verbose builder logging.
        prefix: Log message prefix.

    Returns:
        (str): Path to the exported RTX engine file.
    """
    import tensorrt_rtx as trt  # separate package from classic tensorrt

    output_file = Path(output_file) if output_file else Path(onnx_file).with_suffix(".rtx.engine")

    # Strongly-typed JIT networks derive precision from the ONNX graph, so convert upfront.
    if half:
        onnx_file = _onnx_to_fp16(onnx_file)
        LOGGER.info(f"{prefix} converted input ONNX to FP16: {onnx_file}")

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

    LOGGER.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} RTX engine as {output_file}")
    if int8:
        # TensorRT-RTX strongly-typed JIT networks forbid the INT8 builder flag; INT8 precision
        # is expressed via QDQ nodes in the input ONNX. Leaving as a clear follow-up.
        if dataset is None:
            raise ValueError("INT8 calibration requires a dataset (pass data= during export).")
        raise NotImplementedError(
            "INT8 for tensorrt_rtx requires a QDQ-annotated ONNX; calibration path not yet wired up."
        )

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
