"""GPU visibility in this process, separately from numerical qualification.

Runtime libraries, physical devices, and executable RexGraph kernels are different
claims. Native enumeration runs in bounded subprocesses so a broken driver cannot
crash or indefinitely block hardware discovery. Vulkan needs no ``vulkaninfo`` tool.
"""
from __future__ import annotations

import errno
import json
import os
import platform
import stat
import subprocess
import sys


_VULKAN_PROBE = r'''
import ctypes as C
import ctypes.util
import json

class Application(C.Structure):
    _fields_ = [('sType', C.c_uint32), ('pNext', C.c_void_p),
                ('pApplicationName', C.c_char_p), ('applicationVersion', C.c_uint32),
                ('pEngineName', C.c_char_p), ('engineVersion', C.c_uint32),
                ('apiVersion', C.c_uint32)]
class InstanceCreate(C.Structure):
    _fields_ = [('sType', C.c_uint32), ('pNext', C.c_void_p), ('flags', C.c_uint32),
                ('pApplicationInfo', C.POINTER(Application)),
                ('enabledLayerCount', C.c_uint32), ('ppEnabledLayerNames', C.c_void_p),
                ('enabledExtensionCount', C.c_uint32), ('ppEnabledExtensionNames', C.c_void_p)]
class QueueFamily(C.Structure):
    _fields_ = [('queueFlags', C.c_uint32), ('queueCount', C.c_uint32),
                ('timestampValidBits', C.c_uint32), ('minImageTransferGranularity', C.c_uint32 * 3)]
class MemoryType(C.Structure):
    _fields_ = [('propertyFlags', C.c_uint32), ('heapIndex', C.c_uint32)]
class MemoryHeap(C.Structure):
    _fields_ = [('size', C.c_uint64), ('flags', C.c_uint32)]
class MemoryProperties(C.Structure):
    _fields_ = [('memoryTypeCount', C.c_uint32), ('memoryTypes', MemoryType * 32),
                ('memoryHeapCount', C.c_uint32), ('memoryHeaps', MemoryHeap * 16)]

report = {'runtime_loaded': False, 'available': False, 'devices': [], 'error': None}
instance = C.c_void_p()
try:
    name = C.util.find_library('vulkan')
    if not name:
        raise OSError('Vulkan loader not found')
    lib = C.CDLL(name)
    report['runtime_loaded'] = True
    report['library'] = name
    lib.vkCreateInstance.argtypes = [C.POINTER(InstanceCreate), C.c_void_p, C.POINTER(C.c_void_p)]
    lib.vkCreateInstance.restype = C.c_int32
    lib.vkDestroyInstance.argtypes = [C.c_void_p, C.c_void_p]
    lib.vkEnumeratePhysicalDevices.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.c_void_p]
    lib.vkEnumeratePhysicalDevices.restype = C.c_int32
    lib.vkGetPhysicalDeviceProperties.argtypes = [C.c_void_p, C.c_void_p]
    lib.vkGetPhysicalDeviceProperties.restype = None
    lib.vkGetPhysicalDeviceMemoryProperties.argtypes = [C.c_void_p, C.POINTER(MemoryProperties)]
    lib.vkGetPhysicalDeviceMemoryProperties.restype = None
    lib.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.c_void_p]
    lib.vkGetPhysicalDeviceQueueFamilyProperties.restype = None
    app = Application(sType=0, pApplicationName=b'RexGraph', apiVersion=1 << 22)
    info = InstanceCreate(sType=1, pApplicationInfo=C.pointer(app))
    result = lib.vkCreateInstance(C.byref(info), None, C.byref(instance))
    report['instance_result'] = result
    if result != 0:
        raise RuntimeError(f'vkCreateInstance returned {result}')
    for attempt in range(3):
        count = C.c_uint32()
        result = lib.vkEnumeratePhysicalDevices(instance, C.byref(count), None)
        if result != 0:
            raise RuntimeError(f'vkEnumeratePhysicalDevices returned {result}')
        if not count.value:
            break
        devices = (C.c_void_p * count.value)()
        result = lib.vkEnumeratePhysicalDevices(instance, C.byref(count), devices)
        if result == 5:  # VK_INCOMPLETE: the device list changed between calls.
            continue
        if result != 0:
            raise RuntimeError(f'vkEnumeratePhysicalDevices returned {result}')
        for index, device in enumerate(devices[:count.value]):
            # VkPhysicalDeviceProperties has a fixed Vulkan 1.0 ABI. Keep aligned
            # storage for its entire limits/sparse-properties tail, reading only
            # the five uint32 fields and 256-byte name at the front.
            properties = (C.c_uint64 * 512)()
            lib.vkGetPhysicalDeviceProperties(device, properties)
            raw = C.string_at(C.addressof(properties), C.sizeof(properties))
            prefix = (C.c_uint32 * 5).from_buffer(properties)
            dtype = int(prefix[4])
            queues_count = C.c_uint32()
            lib.vkGetPhysicalDeviceQueueFamilyProperties(device, C.byref(queues_count), None)
            queues = (QueueFamily * queues_count.value)()
            lib.vkGetPhysicalDeviceQueueFamilyProperties(device, C.byref(queues_count), queues)
            compute = any(q.queueCount and q.queueFlags & 2 for q in queues[:queues_count.value])
            memory = MemoryProperties()
            lib.vkGetPhysicalDeviceMemoryProperties(device, C.byref(memory))
            local_bytes = sum(h.size for h in memory.memoryHeaps[:memory.memoryHeapCount] if h.flags & 1)
            report['devices'].append({
                'index': index, 'name': raw[20:276].split(b'\0', 1)[0].decode(errors='replace'),
                'vendor_id': int(prefix[2]), 'device_id': int(prefix[3]),
                'device_type': dtype, 'integrated': dtype == 1,
                'hardware_gpu': dtype in (1, 2, 3), 'compute': bool(compute),
                'device_local_bytes': local_bytes})
        break
    else:
        raise RuntimeError('Vulkan device list repeatedly changed during enumeration')
    report['available'] = any(d['hardware_gpu'] and d['compute'] for d in report['devices'])
except Exception as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
finally:
    if instance.value:
        lib.vkDestroyInstance(instance, None)
print(json.dumps(report))
'''


_HIP_PROBE = r'''
import ctypes as C
import ctypes.util
import importlib.util
import json
from pathlib import Path

report = {'runtime_loaded': False, 'available': False, 'devices': [], 'error': None}
try:
    # Preserve hip_ternary's Torch-first runtime binding when both are installed.
    try:
        import torch
    except ImportError:
        pass
    candidates = []
    sdk = importlib.util.find_spec('_rocm_sdk_core')
    if sdk and sdk.origin:
        candidates.extend(str(p) for p in sorted((Path(sdk.origin).parent / 'lib').glob('libamdhip64.so*')))
    name = C.util.find_library('amdhip64')
    candidates.extend([name] if name else ['libamdhip64.so'])
    lib = None
    errors = []
    for candidate in candidates:
        try:
            lib = C.CDLL(candidate)
            report['library'] = candidate
            break
        except OSError as exc:
            errors.append(str(exc))
    if lib is None:
        raise OSError('; '.join(errors))
    report['runtime_loaded'] = True
    lib.hipInit.argtypes, lib.hipInit.restype = [C.c_uint], C.c_int
    lib.hipGetDeviceCount.argtypes, lib.hipGetDeviceCount.restype = [C.POINTER(C.c_int)], C.c_int
    lib.hipGetErrorString.argtypes, lib.hipGetErrorString.restype = [C.c_int], C.c_char_p
    lib.hipDeviceGetName.argtypes, lib.hipDeviceGetName.restype = [C.c_char_p, C.c_int, C.c_int], C.c_int
    result = lib.hipInit(0)
    report['init_result'] = result
    if result:
        raise RuntimeError(f'hipInit returned {result}: {lib.hipGetErrorString(result).decode()}')
    count = C.c_int()
    result = lib.hipGetDeviceCount(C.byref(count))
    report['enumeration_result'] = result
    if result:
        raise RuntimeError(f'hipGetDeviceCount returned {result}: {lib.hipGetErrorString(result).decode()}')
    for index in range(count.value):
        name = C.create_string_buffer(256)
        result = lib.hipDeviceGetName(name, len(name), index)
        if result:
            raise RuntimeError(f'hipDeviceGetName returned {result}')
        report['devices'].append({'index': index, 'name': name.value.decode(errors='replace')})
    report['available'] = bool(report['devices'])
except Exception as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
print(json.dumps(report))
'''


def _native_probe(source: str, timeout: float) -> dict:
    failure = {'runtime_loaded': False, 'available': False, 'devices': []}
    try:
        process = subprocess.run([sys.executable, '-c', source], capture_output=True,
                                 text=True, timeout=timeout, check=False)
        if process.returncode:
            return {**failure, 'error': f'probe exited {process.returncode}',
                    'stderr': process.stderr[-4000:]}
        report = json.loads(process.stdout)
        if not isinstance(report, dict) or not isinstance(report.get('devices'), list):
            raise ValueError('native probe returned an invalid report')
        if not isinstance(report.get('available'), bool) or not isinstance(report.get('runtime_loaded'), bool):
            raise ValueError('native probe returned invalid availability flags')
        if process.stderr:
            report['stderr'] = process.stderr[-4000:]
        return report
    except subprocess.TimeoutExpired:
        return {**failure, 'error': f'TimeoutExpired: native GPU probe exceeded {timeout:g}s'}
    except (OSError, ValueError) as exc:
        return {**failure, 'error': f'{type(exc).__name__}: {exc}'}


def probe_vulkan(*, timeout: float = 5.0) -> dict:
    """Enumerate Vulkan compute devices; enumeration does not qualify kernels."""
    return _native_probe(_VULKAN_PROBE, timeout)


def probe_hip(*, timeout: float = 15.0) -> dict:
    """Ask the HIP runtime directly, independently of Torch device enumeration."""
    return _native_probe(_HIP_PROBE, timeout)


def _path_status(path: str, *, open_device: bool = False) -> dict:
    result = {'path': path}
    try:
        info = os.stat(path)
        result.update(present=True, mode=oct(stat.S_IMODE(info.st_mode)),
                      uid=info.st_uid, gid=info.st_gid)
        if stat.S_ISCHR(info.st_mode):
            result.update(major=os.major(info.st_rdev), minor=os.minor(info.st_rdev))
            if open_device:
                descriptor = os.open(path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
                os.close(descriptor)
                result['read_write_open'] = True
    except OSError as exc:
        result.update(errno=exc.errno, error=str(exc))
        if 'present' not in result:
            result['present'] = False if exc.errno == errno.ENOENT else None
        elif open_device:
            result['read_write_open'] = False
    return result


def device_access() -> dict:
    """Local device visibility and open errors, never an assertion about the host."""
    report = {'platform': platform.system(), 'paths': [], 'environment': {}}
    keys = ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
            'GPU_DEVICE_ORDINAL', 'HSA_OVERRIDE_GFX_VERSION', 'VK_ICD_FILENAMES',
            'VK_DRIVER_FILES', 'VK_LOADER_DRIVERS_SELECT', 'VK_LOADER_DRIVERS_DISABLE')
    report['environment'] = {key: os.environ[key] for key in keys if key in os.environ}
    if platform.system() != 'Linux':
        return report
    report['paths'] = [_path_status('/dev/dri'), _path_status('/dev/kfd', open_device=True),
                       _path_status('/sys/class/drm'), _path_status('/sys/class/kfd')]
    try:
        for name in sorted(os.listdir('/dev/dri')):
            if name.startswith('renderD'):
                report['paths'].append(_path_status('/dev/dri/' + name, open_device=True))
    except OSError:
        pass
    try:
        with open('/proc/self/mountinfo') as source:
            for line in source:
                before, after = line.rstrip().split(' - ', 1)
                fields = before.split()
                if fields[4] == '/dev':
                    report['dev_mount'] = {'filesystem': after.split()[0], 'options': fields[5]}
    except (OSError, ValueError, IndexError):
        pass
    return report


def diagnose() -> dict:
    """Probe both AMD compute routes without claiming training or throughput."""
    return {'access': device_access(), 'hip': probe_hip(), 'vulkan': probe_vulkan(),
            'numerical_qualification': False}
