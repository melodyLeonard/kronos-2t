"""Prints which compute device Core ML assigns to each op type of a model.

usage: python ane/placement.py <model.mlpackage>
"""
import collections, sys
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
from coremltools.models.compute_device import MLNeuralEngineComputeDevice, MLGPUComputeDevice, MLCPUComputeDevice

path = sys.argv[1]
model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)  # keeps the compiled copy alive
compiled = model.get_compiled_model_path()
plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
counts = collections.Counter()
for fn in plan.model_structure.program.functions.values():
    for op in fn.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = usage.preferred_compute_device if usage else None
        name = {MLNeuralEngineComputeDevice: "ANE", MLGPUComputeDevice: "GPU", MLCPUComputeDevice: "CPU"}.get(type(dev), "-")
        counts[(op.operator_name, name)] += 1
for (op, dev), n in sorted(counts.items()):
    print(f"{op:<40} {dev:<4} x{n}")
