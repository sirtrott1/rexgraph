"""Visibility failures cannot become claims about physical hardware or kernel correctness."""
import errno
import json
import subprocess
from types import SimpleNamespace

import pytest

from rexgraph import gpu_access as A


@pytest.fixture
def device_os(monkeypatch):
    device = SimpleNamespace(
        stat=lambda path: SimpleNamespace(
            st_mode=0o20660, st_uid=0, st_gid=42, st_rdev=(226 << 8) | 128),
        major=lambda number: number >> 8,
        minor=lambda number: number & 255,
        O_RDWR=2, O_NONBLOCK=2048, O_CLOEXEC=524288,
    )
    monkeypatch.setattr(A, "os", device)
    return device


def test_missing_path_and_denied_path_remain_distinct(device_os):
    def deny(path):
        raise PermissionError(errno.EACCES, "permission denied", path)
    device_os.stat = deny
    denied = A._path_status('/dev/kfd')
    assert denied['present'] is None and denied['errno'] == errno.EACCES
    def missing(path):
        raise FileNotFoundError(errno.ENOENT, "not visible", path)
    device_os.stat = missing
    hidden = A._path_status('/dev/kfd')
    assert hidden['present'] is False and hidden['errno'] == errno.ENOENT


def test_present_device_with_denied_open_preserves_node_facts(device_os):
    def denied(*args):
        raise PermissionError(errno.EACCES, 'no read/write device access')
    device_os.open = denied
    node = A._path_status('/dev/dri/renderD128', open_device=True)
    assert node['present'] and node['major'] == 226 and node['minor'] == 128
    assert node['read_write_open'] is False and node['errno'] == errno.EACCES


def test_opened_device_handle_is_closed(device_os):
    closed = []
    device_os.open = lambda *args: 73
    device_os.close = closed.append
    assert A._path_status('/dev/kfd', open_device=True)['read_write_open']
    assert closed == [73]


@pytest.mark.parametrize('returncode', [-11, 1])
def test_failed_native_driver_probe_is_a_diagnostic_not_a_process_crash(monkeypatch, returncode):
    monkeypatch.setattr(A.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=returncode, stdout='', stderr='driver error'))
    result = A.probe_vulkan()
    assert not result['available'] and not result['devices']
    assert str(returncode) in result['error'] and result['stderr'] == 'driver error'


def test_hung_driver_is_bounded(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    monkeypatch.setattr(A.subprocess, 'run', timeout)
    result = A.probe_hip(timeout=.1)
    assert not result['available'] and 'TimeoutExpired' in result['error']


@pytest.mark.parametrize('stdout', ['non-json output', '[]', '{}'])
def test_malformed_native_output_cannot_report_a_gpu(monkeypatch, stdout):
    monkeypatch.setattr(A.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=stdout, stderr=''))
    result = A.probe_vulkan()
    assert not result['available'] and result['error']


def test_native_loader_error_and_stderr_are_preserved(monkeypatch):
    native = {'runtime_loaded': True, 'available': False, 'devices': [],
              'error': 'vkEnumeratePhysicalDevices returned -3'}
    monkeypatch.setattr(A.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps(native), stderr='ICD diagnostic'))
    result = A.probe_vulkan()
    assert result['runtime_loaded'] and not result['available']
    assert result['error'] == native['error'] and result['stderr'] == 'ICD diagnostic'


def test_only_visibility_variables_enter_report(monkeypatch):
    monkeypatch.setenv('HIP_VISIBLE_DEVICES', '')
    monkeypatch.setenv('SECRET_API_KEY', 'must-not-enter-report')
    report = A.device_access()
    assert report['environment']['HIP_VISIBLE_DEVICES'] == ''
    assert 'SECRET_API_KEY' not in report['environment']


@pytest.mark.parametrize('device_type,queue_flags,available', [
    (1, 2, True), (2, 2, True), (3, 2, True), (4, 2, False), (1, 1, False)])
def test_vulkan_abi_reading_distinguishes_compute_gpus_from_software_and_graphics_only(
        monkeypatch, capsys, device_type, queue_flags, available):
    """A stub Vulkan ABI exercises decoding; it is not a hardware execution test."""
    import ctypes as C
    import ctypes.util
    namespace, destroyed = {}, []
    lib = SimpleNamespace()
    def set_uint(pointer, value):
        C.cast(pointer, C.POINTER(C.c_uint32))[0] = value
    def create(info, allocator, instance):
        assert C.cast(info, C.POINTER(namespace['InstanceCreate'])).contents.sType == 1
        C.cast(instance, C.POINTER(C.c_void_p))[0] = 101
        return 0
    def enumerate_devices(instance, count, devices):
        set_uint(count, 1)
        if devices is not None:
            devices[0] = 202
        return 0
    def properties(device, buffer):
        assert device == 202
        prefix = C.cast(buffer, C.POINTER(C.c_uint32))
        for i, value in enumerate([1 << 22, 7, 0x1002, 0x1586, device_type]):
            prefix[i] = value
        C.memmove(C.addressof(buffer) + 20, b'ABI fixture\0', 12)
    def queues(device, count, buffer):
        set_uint(count, 1)
        if buffer is not None:
            buffer[0].queueFlags, buffer[0].queueCount = queue_flags, 1
    def memory(device, pointer):
        mem = C.cast(pointer, C.POINTER(namespace['MemoryProperties'])).contents
        mem.memoryHeapCount = 2
        mem.memoryHeaps[0].flags, mem.memoryHeaps[0].size = 1, 32 * 2**30
        mem.memoryHeaps[1].flags, mem.memoryHeaps[1].size = 0, 96 * 2**30
    def destroy(instance, allocator):
        destroyed.append(instance.value)
    for name, fn in [('vkCreateInstance', create), ('vkEnumeratePhysicalDevices', enumerate_devices),
                     ('vkGetPhysicalDeviceProperties', properties),
                     ('vkGetPhysicalDeviceQueueFamilyProperties', queues),
                     ('vkGetPhysicalDeviceMemoryProperties', memory), ('vkDestroyInstance', destroy)]:
        setattr(lib, name, fn)
    monkeypatch.setattr(C, 'CDLL', lambda name: lib)
    monkeypatch.setattr(C.util, 'find_library', lambda name: 'ABI stub')
    exec(A._VULKAN_PROBE, namespace)
    report = json.loads(capsys.readouterr().out)
    assert report['error'] is None and report['available'] is available
    assert report['devices'][0]['vendor_id'] == 0x1002
    assert report['devices'][0]['name'] == 'ABI fixture'
    assert report['devices'][0]['device_local_bytes'] == 32 * 2**30
    assert destroyed == [101]
