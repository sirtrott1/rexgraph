"""Float64 sparse products on Vulkan compute devices.

SparseOperator keeps the matrix on the selected device between products. Dense
inputs and results cross the host boundary on each call. Propagation and solver
recurrences use these products with NumPy vector updates on the host.
"""
from __future__ import annotations

from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
import operator
import struct
import subprocess
import sys
from threading import RLock

import numpy as np


@lru_cache(maxsize=1)
def available():
    """Whether a bounded child process can execute a float64 Vulkan product."""
    code = "import numpy as np; from rexgraph.vulkan_compute import sparse_mm; a=np.array([[2.]]); assert np.array_equal(sparse_mm(a,np.array([3.])),[6.])"
    try:
        result = subprocess.run([sys.executable, "-B", "-I", "-c", code],
                                capture_output=True, timeout=20)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _matrix_arrays(matrix):
    if np.iscomplexobj(matrix):
        raise TypeError("Vulkan sparse products currently require real coefficients")
    from rexgraph.native_sparse import as_native
    matrix = as_native(matrix)
    if max(*matrix.shape, matrix.nnz) > np.iinfo(np.int32).max:
        raise ValueError("Vulkan sparse dimensions and nonzero counts must fit int32")
    data = np.ascontiguousarray(matrix.data, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError("Vulkan sparse coefficients must be finite")
    return matrix.shape, (np.ascontiguousarray(matrix.dual.row_ptr, dtype=np.uint32),
                          np.ascontiguousarray(matrix.dual.col_idx, dtype=np.uint32), data)


def _dense_values(values, columns):
    from rexgraph.linear_operator import _numeric_array
    values = _numeric_array(values, operation="Vulkan sparse product")
    if values.ndim not in (1, 2) or values.shape[0] != columns:
        raise ValueError("Vulkan sparse product requires a matching vector or matrix")
    if np.iscomplexobj(values):
        raise TypeError("Vulkan sparse products currently require real values")
    return np.ascontiguousarray(values, dtype=np.float64)


class SparseOperator:
    """Device resident sparse coefficients with synchronous vector/block products.

    device selects a physical device index as reported by the Vulkan loader.
    None selects the first hardware GPU with float64 compute support.
    """

    def __init__(self, matrix, *, device=None):
        self.shape, arrays = _matrix_arrays(matrix)
        if isinstance(device, bool):
            raise TypeError("Vulkan device index must be an integer")
        device = operator.index(device) if device is not None else None
        if device is not None and device < 0:
            raise ValueError("Vulkan device index must be nonnegative")
        self._resources = ExitStack()
        self._lock = RLock()
        self._closed = False
        self._vectors = None
        try:
            import vulkan as vk
            self.vk = vk
            self.instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
                pApplicationInfo=vk.VkApplicationInfo(pApplicationName="RexGraph", apiVersion=vk.VK_MAKE_VERSION(1, 0, 0))), None)
            self._resources.callback(vk.vkDestroyInstance, self.instance, None)
            devices = vk.vkEnumeratePhysicalDevices(self.instance)
            if device is None:
                for i, candidate in enumerate(devices):
                    properties = vk.vkGetPhysicalDeviceProperties(candidate)
                    features = vk.vkGetPhysicalDeviceFeatures(candidate)
                    queues = vk.vkGetPhysicalDeviceQueueFamilyProperties(candidate)
                    if properties.deviceType in (1, 2, 3) and features.shaderFloat64 and any(
                            q.queueCount and q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT for q in queues):
                        device = i
                        break
                if device is None:
                    raise RuntimeError("No Vulkan hardware GPU with float64 compute support is visible")
            if device >= len(devices):
                raise RuntimeError(f"Vulkan device {device} is not visible")
            physical = devices[device]
            properties = vk.vkGetPhysicalDeviceProperties(physical)
            if properties.deviceType not in (vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                                              vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                                              vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU):
                raise RuntimeError("The selected Vulkan device is not a hardware GPU")
            features = vk.vkGetPhysicalDeviceFeatures(physical)
            if not features.shaderFloat64:
                raise RuntimeError("The selected Vulkan GPU does not support shaderFloat64")
            self.name = properties.deviceName
            self.limits = properties.limits
            if self.limits.maxPerStageDescriptorStorageBuffers < 5:
                raise RuntimeError("Vulkan sparse products require five storage buffer bindings")
            queues = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical)
            queue_index = next((i for i, q in enumerate(queues) if q.queueCount and q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT), None)
            if queue_index is None:
                raise RuntimeError("The selected Vulkan GPU has no compute queue")
            self.device = vk.vkCreateDevice(physical, vk.VkDeviceCreateInfo(
                pQueueCreateInfos=[vk.VkDeviceQueueCreateInfo(queueFamilyIndex=queue_index, queueCount=1, pQueuePriorities=[1.0])],
                pEnabledFeatures=vk.VkPhysicalDeviceFeatures(shaderFloat64=True)), None)
            self._resources.callback(vk.vkDestroyDevice, self.device, None)
            self.queue = vk.vkGetDeviceQueue(self.device, queue_index, 0)
            self.memory = vk.vkGetPhysicalDeviceMemoryProperties(physical)
            self.command_pool = self._own(vk.vkCreateCommandPool, vk.vkDestroyCommandPool,
                                          vk.VkCommandPoolCreateInfo(queueFamilyIndex=queue_index,
                                                                    flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT))
            self.command = vk.vkAllocateCommandBuffers(self.device, vk.VkCommandBufferAllocateInfo(
                commandPool=self.command_pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
            self.fence = self._own(vk.vkCreateFence, vk.vkDestroyFence, vk.VkFenceCreateInfo())
            bindings = [vk.VkDescriptorSetLayoutBinding(binding=i, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT) for i in range(5)]
            self.descriptor_layout = self._own(vk.vkCreateDescriptorSetLayout, vk.vkDestroyDescriptorSetLayout,
                                               vk.VkDescriptorSetLayoutCreateInfo(pBindings=bindings))
            self.pipeline_layout = self._own(vk.vkCreatePipelineLayout, vk.vkDestroyPipelineLayout,
                vk.VkPipelineLayoutCreateInfo(pSetLayouts=[self.descriptor_layout], pPushConstantRanges=[
                    vk.VkPushConstantRange(stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=8)]))
            code = Path(__file__).with_name("shaders").joinpath("csr_f64.spv").read_bytes()
            shader = vk.vkCreateShaderModule(self.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
            try:
                # The binding copies stage by value; retain its entry point storage
                # until pipeline creation has consumed the embedded structure.
                stage = vk.VkPipelineShaderStageCreateInfo(
                    stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=shader, pName="main")
                self.pipeline = vk.vkCreateComputePipelines(self.device, vk.VK_NULL_HANDLE, 1, [
                    vk.VkComputePipelineCreateInfo(layout=self.pipeline_layout, stage=stage)], None)[0]
            finally:
                vk.vkDestroyShaderModule(self.device, shader, None)
            self._resources.callback(vk.vkDestroyPipeline, self.device, self.pipeline, None)
            self.descriptor_pool = self._own(vk.vkCreateDescriptorPool, vk.vkDestroyDescriptorPool,
                vk.VkDescriptorPoolCreateInfo(maxSets=1, pPoolSizes=[vk.VkDescriptorPoolSize(
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=5)]))
            self.descriptors = vk.vkAllocateDescriptorSets(self.device, vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self.descriptor_pool, pSetLayouts=[self.descriptor_layout]))[0]
            self.buffers = []
            for array in arrays:
                buffer = self._buffer(max(8, array.nbytes), self._resources)
                buffer[2][:array.nbytes] = array.tobytes()
                self.buffers.append(buffer)
        except BaseException:
            self._resources.close()
            self._closed = True
            raise

    def _own(self, create, destroy, info):
        handle = create(self.device, info, None)
        self._resources.callback(destroy, self.device, handle, None)
        return handle

    def _buffer(self, size, resources):
        vk = self.vk
        if size > self.limits.maxStorageBufferRange:
            raise MemoryError("Vulkan buffer exceeds the device's maxStorageBufferRange")
        buffer = vk.vkCreateBuffer(self.device, vk.VkBufferCreateInfo(
            size=size, usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
        memory = None
        try:
            needs = vk.vkGetBufferMemoryRequirements(self.device, buffer)
            flags = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            index = next((i for i in range(self.memory.memoryTypeCount) if needs.memoryTypeBits & (1 << i)
                          and self.memory.memoryTypes[i].propertyFlags & flags == flags), None)
            if index is None:
                raise RuntimeError("Vulkan GPU has no coherent memory type for storage buffers")
            memory = vk.vkAllocateMemory(self.device, vk.VkMemoryAllocateInfo(allocationSize=needs.size, memoryTypeIndex=index), None)
            vk.vkBindBufferMemory(self.device, buffer, memory, 0)
            mapped = vk.vkMapMemory(self.device, memory, 0, size, 0)
        except BaseException:
            vk.vkDestroyBuffer(self.device, buffer, None)
            if memory is not None:
                vk.vkFreeMemory(self.device, memory, None)
            raise
        # ExitStack unwinds in reverse order: unmap, destroy buffer, free memory.
        resources.callback(vk.vkFreeMemory, self.device, memory, None)
        resources.callback(vk.vkDestroyBuffer, self.device, buffer, None)
        resources.callback(vk.vkUnmapMemory, self.device, memory)
        return buffer, size, mapped

    def apply(self, values):
        """Return matrix @ values as a float64 array with the corresponding shape."""
        values = _dense_values(values, self.shape[1])
        columns = 1 if values.ndim == 1 else values.shape[1]
        shape = (self.shape[0],) if values.ndim == 1 else (self.shape[0], columns)
        with self._lock:
            if self._closed:
                raise RuntimeError("Vulkan sparse operator is closed")
            if not self.shape[0] or not columns:
                return np.zeros(shape, dtype=np.float64)
            count = self.shape[0] * columns
            if max(count, values.size) > np.iinfo(np.int32).max:
                raise ValueError("Vulkan dense product sizes must fit int32")
            sizes = (max(8, values.nbytes), max(8, count * 8))
            if self._vectors is None or self._sizes != sizes:
                if self._vectors is not None:
                    self._vectors.close()
                self._vectors = ExitStack()
                try:
                    self._input, self._output = [self._buffer(size, self._vectors) for size in sizes]
                except BaseException:
                    self._vectors.close()
                    self._vectors = None
                    raise
                self._sizes = sizes
            vk = self.vk
            self._input[2][:values.nbytes] = values.tobytes()
            buffers = self.buffers + [self._input, self._output]
            writes = [vk.VkWriteDescriptorSet(dstSet=self.descriptors, dstBinding=i,
                      descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                      pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=buffer, offset=0, range=size)])
                      for i, (buffer, size, _) in enumerate(buffers)]
            vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)
            groups = (count + 63) // 64
            x = min(groups, self.limits.maxComputeWorkGroupCount[0])
            y = (groups + x - 1) // x
            if y > self.limits.maxComputeWorkGroupCount[1]:
                raise ValueError("Vulkan dispatch exceeds the device's compute grid limits")
            vk.vkResetCommandBuffer(self.command, 0)
            vk.vkBeginCommandBuffer(self.command, vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            vk.vkCmdBindPipeline(self.command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
            vk.vkCmdBindDescriptorSets(self.command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, [self.descriptors], 0, None)
            dims = vk.ffi.new("char[]", struct.pack("II", self.shape[0], columns))
            vk.vkCmdPushConstants(self.command, self.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, dims)
            vk.vkCmdDispatch(self.command, x, y, 1)
            barrier = vk.VkMemoryBarrier(srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT, dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT)
            vk.vkCmdPipelineBarrier(self.command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   vk.VK_PIPELINE_STAGE_HOST_BIT, 0, 1, [barrier], 0, None, 0, None)
            vk.vkEndCommandBuffer(self.command)
            vk.vkResetFences(self.device, 1, [self.fence])
            vk.vkQueueSubmit(self.queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[self.command])], self.fence)
            vk.vkWaitForFences(self.device, 1, [self.fence], True, 60_000_000_000)
            result = np.frombuffer(self._output[2], dtype=np.float64, count=count).copy().reshape(shape)
            if not np.isfinite(result).all():
                raise FloatingPointError("Vulkan sparse result is outside numerical range")
            return result

    def close(self):
        with self._lock:
            if not self._closed:
                try:
                    self.vk.vkDeviceWaitIdle(self.device)
                finally:
                    try:
                        if self._vectors is not None:
                            self._vectors.close()
                    finally:
                        self._resources.close()
                        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Vulkan sparse operator is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def sparse_mm(matrix, values, *, device=None):
    """Multiply a sparse matrix by a real vector or block on a Vulkan GPU."""
    shape, arrays = _matrix_arrays(matrix)
    values = _dense_values(values, shape[1])
    result_shape = (shape[0],) if values.ndim == 1 else (shape[0], values.shape[1])
    if not np.prod(result_shape) or not arrays[2].size:
        return np.zeros(result_shape, dtype=np.float64)
    with SparseOperator(matrix, device=device) as resident:
        return resident.apply(values)


def matfunc_apply(matrix, values, coefficients, lam_max):
    """Apply supplied Chebyshev coefficients using Vulkan sparse products."""
    values = _dense_values(values, matrix.shape[1])
    if not values.size:
        return np.zeros_like(values)
    with SparseOperator(matrix) as resident:
        previous = values
        result = coefficients[0] * values
        if len(coefficients) < 2:
            return result
        current = (2.0 / lam_max) * resident.apply(values) - values
        result = result + coefficients[1] * current
        for coefficient in coefficients[2:]:
            following = (4.0 / lam_max) * resident.apply(current) - 2.0 * current - previous
            result += coefficient * following
            previous, current = current, following
        return result


def block_cg(matrix, values, dinv, *, tol=1e-10, maxit=1000):
    """Solve a positive definite system with Vulkan sparse products."""
    from rexgraph.sparse_character import _block_cg
    values = _dense_values(values, matrix.shape[1])
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Vulkan conjugate gradient requires a square matrix")
    if not values.size:
        return np.zeros_like(values)
    vector = values.ndim == 1
    with SparseOperator(matrix) as resident:
        result = _block_cg(resident.apply, values[:, None] if vector else values,
                          np.asarray(dinv, dtype=np.float64), tol=tol, maxit=maxit)
    return result[:, 0] if vector else result


def greens_diagonal(matrix, *, tol=1e-10, chunk=512):
    """Return the diagonal of a positive definite inverse using Vulkan products."""
    from rexgraph.native_sparse import as_native
    from rexgraph.sparse_character import _block_cg
    matrix = as_native(matrix)
    n = matrix.shape[0]
    result = np.zeros(n, dtype=np.float64)
    if not n:
        return result
    diagonal = matrix.diagonal()
    dinv = np.where(np.abs(diagonal) > 1e-30, 1.0 / diagonal, 1.0)
    step = max(1, min(n, operator.index(chunk)))
    with SparseOperator(matrix) as resident:
        for start in range(0, n, step):
            stop = min(n, start + step)
            rhs = np.zeros((n, stop - start), dtype=np.float64)
            rows, columns = np.arange(start, stop), np.arange(stop - start)
            rhs[rows, columns] = 1.0
            solved = _block_cg(resident.apply, rhs, dinv, tol=tol)
            result[start:stop] = solved[rows, columns]
    return result


def qualify(size=64):
    """Execute sparse, propagation and solver checks against CPU results."""
    from rexgraph.gpu_preflight import _sparse_spd
    from rexgraph import scale_propagator as spg
    size = operator.index(size)
    if size < 2:
        raise ValueError("Vulkan qualification size must be at least two")
    matrix, values = _sparse_spd(size)
    with SparseOperator(matrix) as resident:
        product = resident.apply(values)
        error = float(np.max(np.abs(product - matrix @ values)))
        name = resident.name
    heat = spg.heat_apply(matrix, values, 0.2, backend="vulkan")
    cpu = spg.heat_apply(matrix, values, 0.2, backend="cpu")
    heat_error = float(np.max(np.abs(heat - cpu)))
    solved = block_cg(matrix, values, 1.0 / matrix.diagonal())
    residual = float(np.max(np.abs(matrix @ solved - values)))
    return {"device": name, "sparse_max_abs_error": error, "heat_max_abs_error": heat_error,
            "solve_max_abs_residual": residual,
            "ok": error < 1e-10 and heat_error < 1e-10 and residual < 1e-7}
