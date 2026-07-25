# RexGraph root build definition (meson-python backend).
# Compiles all Cython extensions to shared objects.

# pip install . # standard build
# pip install -e ".[dev]" --no-build-isolation # editable
# python -m pytest tests/ # run tests
project(
    'rexgraph',
    'c', 'cython',
    version: '1.0.0',
    license: 'Apache-2.0',
    meson_version: '>=1.3.0',
    default_options: [
        'c_std=c11',
        'buildtype=release',
        'b_ndebug=if-release',
        'b_lto=true',
        'warning_level=2',
    ],
)

py = import('python').find_installation(pure: false)

cc = meson.get_compiler('c')
c_args = []
link_args = []
# Native/fast-math tuning is OPT-IN. By default the build is portable
# (no -march=native) and IEEE-correct (no -ffast-math), so wheels built on
# one machine run on any compatible CPU and numerics are reproducible.
# For a machine-specific optimized build:
# pip install . -Csetup-args=-Dnative=true
# WARNING: -march=native binaries crash with SIGILL on older CPUs, and
# -ffast-math relaxes NaN/inf handling and float reproducibility.
if cc.get_id() == 'gcc' or cc.get_id() == 'clang'
    if get_option('native')
        c_args += ['-march=native', '-ffast-math']
    endif
    if get_option('buildtype') == 'release'
        c_args += ['-DNDEBUG']
    endif
elif cc.get_id() == 'msvc'
    c_args += ['/O2']
    if get_option('native')
        c_args += ['/fp:fast']
    endif
endif

# Use the ACTIVE interpreter's numpy headers (numpy.get_include()), NOT whatever
# `dependency('numpy')` resolves to globally. Otherwise the C compile can include a
# DIFFERENT numpy's headers than the one Cython generates code against (e.g. a stray
# ~/.local numpy for another Python version), so Cython emits PyDataType_* / datetime
# C-API calls the mismatched headers don't declare -> "implicit declaration" build
# failures. Binding the include to `py`'s own numpy keeps pxd and headers in lockstep.
numpy_inc = run_command(py, '-c', 'import numpy; print(numpy.get_include())',
                        check: true).stdout().strip()
numpy_dep = declare_dependency(include_directories: include_directories(numpy_inc))
omp_dep = dependency('openmp', required: false)
m_dep = cc.find_library('m', required: false)
mvec_dep = cc.find_library('mvec', required: false)

cython_args = [
    '--fast-fail',
    '-X', 'boundscheck=False',
    '-X', 'wraparound=False',
    '-X', 'cdivision=True',
    '-X', 'language_level=3',
]
add_project_arguments(cython_args, language: 'cython')

subdir('rexgraph')
