# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from os.path import join

import pytest
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER

from tvm.testing.aot import run_and_check, AOTTestModel, AOTCompiledTestModel, AOTTestRunner

import tvm
#from tvm.relay.backend.contrib.uma._template.backend import MyAiHwBackend
from test_uma_vanilla_accelerator import VanillaAcceleratorBackend
from tvm import relay, IRModule
from tvm.contrib.download import download_testdata
import numpy as np
from collections import OrderedDict
import tarfile
from pathlib import Path
import onnx

from tvm.contrib import utils


# def import_mnist12() -> [IRModule, dict]:
#     model_url = "".join(
#         ["https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.onnx"])
#     model_path = download_testdata(model_url, "mnist-12.onnx", module="onnx")
#     onnx_model = onnx.load(model_path)
#     input_name = "Input3"
#     shape_dict = {input_name: (1, 1, 28, 28)}
#     mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
#     return mod, params
#
#
# def import_restnet50() -> [IRModule, dict]:
#     model_url = "".join(
#         ["https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx"])
#     model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
#     onnx_model = onnx.load(model_path)
#     input_name = "data"
#     shape_dict = {input_name: (1, 3, 224, 224)}
#     mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
#     return mod, params


def download_and_import_onnx_model(model_url: str) -> [IRModule, dict, dict, dict]:
    """
    Download an ONNX NN model from `url`  and import it using the TVM onnx frontend
    """
    def _get_shape(io):
        shape = []
        dimensions = io.type.tensor_type.shape.dim
        for dim in dimensions:
            shape.append(dim.dim_value)
        return shape

    filename = model_url.split("/")[-1]
    model_url = "".join([model_url])
    model_path = download_testdata(model_url, filename, module="onnx")
    onnx_model = onnx.load(model_path)
    graph_input = onnx_model.graph.input
    assert len(graph_input) == 1
    input_name = graph_input[0].name
    input_shape = _get_shape(graph_input[0])
    graph_output = onnx_model.graph.output
    assert len(graph_output) == 1
    output_name = graph_output[0].name
    output_shape = _get_shape(graph_output[0])
    input_shape_dict = {input_name: input_shape}
    output_shape_dict = {output_name: output_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, input_shape_dict)
    return mod, params, input_shape_dict, output_shape_dict


def _vanilla_accelerator_run(mod, params, input_shapes, output_shapes):
    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    mod = uma_backend.partition(mod)

    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))
    generic_target = tvm.target.Target("c")
    runtime = tvm.relay.backend.Runtime("crt")
    executor = tvm.relay.backend.Executor("aot", {"workspace-byte-alignment": 8})

    with tvm.transform.PassContext(
        opt_level=3,
        config={"tir.disable_vectorize": True,
                "tir.usmp.enable": True,
                "tir.usmp.algorithm": "greedy_by_conflicts"},
        disabled_pass=["AlterOpLayout"]
    ):
        relay_mod = relay.build(mod, target=[generic_target, target], runtime=runtime, executor=executor, params=params)

    def _generate_rt_data(input_shapes: dict, output_shapes: dict) -> [OrderedDict, OrderedDict]:
        assert len(input_shapes) == 1
        assert len(output_shapes) == 1

        iname = list(input_shapes.keys())[0]
        oname = list(output_shapes.keys())[0]
        ishape = input_shapes[iname]
        oshape = output_shapes[oname]
        i_data = np.random.uniform(0, 1, ishape).astype("float32")
        o_data = np.random.uniform(0, 1, oshape).astype("float32")
        inputs = OrderedDict([(iname, i_data)])
        outputs = OrderedDict([(oname, o_data)])
        return inputs, outputs
    input_list, output_list = _generate_rt_data(input_shapes, output_shapes)
    model = AOTTestModel(module=mod, inputs=input_list, outputs=output_list)
    compiled_model = [AOTCompiledTestModel(model=model, executor_factory=relay_mod)]
    runner = AOTTestRunner(pass_config={"tir.usmp.enable": True})
    run_and_check(compiled_model, runner, "packed")

    # temp = utils.tempdir(keep_for_debug=True)
    # model_library_format_tar_path = Path(join(temp.path, "build", "lib.tar"))
    # model_library_format_tar_path.unlink(missing_ok=True)
    # model_library_format_tar_path.parent.mkdir(parents=True, exist_ok=True)
    # print(model_library_format_tar_path.absolute())
    #
    # tvm.micro.export_model_library_format(module, model_library_format_tar_path)
    #
    # print("Built MLF Library: ")
    # with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
    #     print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))
    #     tar_f.extractall(model_library_format_tar_path.parent)



@pytest.mark.parametrize(
    "url",
    ["https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx",
     "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.onnx"
     ]
)
def test_vanilla_accelerator_integration(url: str):
    mod, params, input_shapes, output_shapes = download_and_import_onnx_model(url)
    _vanilla_accelerator_run(mod, params, input_shapes, output_shapes)

# def test_single_layer():
#     x0 = relay.var("input")
#     x1 = relay.var("param0")
#     x1 = relay.var("param0")
#     y = relay.nn.dense(x0, x1, units=256)
#
#     RELAY_MODEL = """
#     #[version = "0.0.5"]
#     def @main(
#         %input_1:   Tensor[(1, 1, 28, 28), float32],
#         %v_param_1: Tensor[(256, 784), float32],
#         %v_param_2: Tensor[(256), float32]
#         )
#     {
#       %0 = nn.batch_flatten(%input_1);
#       %1 = nn.dense(%0, %v_param_1, units=256);
#       %2 = nn.bias_add(%1, %v_param_2);
#       nn.relu(%2)
#     }
#     """
#     mod = tvm.parser.fromtext(RELAY_MODEL)


if __name__ == "__main__":
    test_vanilla_accelerator_integration(
        "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.onnx")



