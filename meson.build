# RexGraph root build definition (meson-python backend).
# Compiles all Cython extensions to shared objects.
#
#   pip install .                              # standard build
#   pip install -e ".[dev]" --no-build-isolation  # editable
#   python -m pytest tests/                    # run tests
project(
    'rexgraph',
    'c', 'cython',
    version: '0.4.0',
    license: 'AGPL-3.0-or-later',
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
if cc.get_id() == 'gcc' or cc.get_id() == 'clang'
    c_args += ['-march=native', '-ffast-math']
    if get_option('buildtype') == 'release'
        c_args += ['-DNDEBUG']
    endif
elif cc.get_id() == 'msvc'
    c_args += ['/O2', '/fp:fast']
endif

numpy_dep = dependency('numpy')
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
