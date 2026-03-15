# rexgraph/core

py.install_sources(
    ['__init__.py', '_common.pxd', '_linalg.pxd'],
    subdir: 'rexgraph/core',
)

# Dependencies shared by all extension modules.
core_deps = [numpy_dep]
if omp_dep.found()
    core_deps += [omp_dep]
endif
if m_dep.found()
    core_deps += [m_dep]
endif
if mvec_dep.found()
    core_deps += [mvec_dep]
endif

# BLAS/LAPACK (needed by _linalg.pxd inline functions: cblas_dgemm, dsyev_, etc.)
# OpenBLAS bundles both CBLAS and LAPACK in a single library.
# Try openblas first (conda-forge), then separate blas/lapack (system).
openblas_dep = dependency('openblas', required: false)
if openblas_dep.found()
    core_deps += [openblas_dep]
else
    blas_dep = dependency('blas', required: false)
    lapack_dep = dependency('lapack', required: false)
    if blas_dep.found()
        core_deps += [blas_dep]
    endif
    if lapack_dep.found()
        core_deps += [lapack_dep]
    endif
endif

_common_c_args = c_args + ['-Wno-unused-function']

# _common is built first so other modules can cimport from _common.pxd.
_common_lib = py.extension_module(
    '_common',
    '_common.pyx',
    dependencies: core_deps,
    c_args: _common_c_args,
    install: true,
    subdir: 'rexgraph/core',
    override_options: ['werror=false'],
)

# All other Cython extension modules.
_cython_modules = [
    '_sparse',
    '_boundary',
    '_cycles',
    '_faces',
    '_laplacians',
    '_overlap',
    '_hodge',
    '_spectral',
    '_standard',
    '_rex',
    '_state',
    '_temporal',
    '_transition',
    '_wave',
    '_persistence',
    '_linalg',
    '_frustration',
    '_relational',
    '_character',
    '_void',
    '_rcfe',
    '_joins',
    '_query',
    '_fiber',
    '_quotient',
    '_field',
    '_signal',
]

foreach mod : _cython_modules
    py.extension_module(
        mod,
        mod + '.pyx',
        dependencies: core_deps,
        c_args: _common_c_args,
        install: true,
        subdir: 'rexgraph/core',
        override_options: ['werror=false'],
    )
endforeach
