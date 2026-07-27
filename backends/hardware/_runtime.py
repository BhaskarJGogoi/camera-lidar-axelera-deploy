"""Generic axelera.runtime model-loading boilerplate, shared by both
compiled models -- loading doesn't care which model.json it's pointed at."""
from pathlib import Path

import numpy as np
from axelera.runtime import Context


def load_model(ctx: Context, model_json: Path, aipu_cores: int = 1) -> dict:
    model = ctx.load_model(str(model_json))
    input_infos = model.inputs()
    output_infos = model.outputs()
    batch_size = input_infos[0].shape[0]
    conn = ctx.device_connect(device=None, num_sub_devices=batch_size)
    instance = conn.load_model_instance(model, num_sub_devices=batch_size, aipu_cores=aipu_cores)
    inputs = [np.zeros(info.shape, info.dtype) for info in input_infos]
    outputs = [np.zeros(info.shape, info.dtype) for info in output_infos]
    return {"model": model, "input_infos": input_infos, "output_infos": output_infos,
            "instance": instance, "inputs": inputs, "outputs": outputs}
