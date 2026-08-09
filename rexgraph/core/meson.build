# rexgraph/core

py.install_sources(
    ['__init__.py', '_common.pxd', '_linalg.pxd', '_temporal_entity_py.py'],
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
# OpenBLAS bundles both CBLAS and LAPACK. Try pkg-config (conda-forge / distro
# *-dev packages) first, then a bare library search, then separate blas/lapack.
# If none is found the extensions still compile but import with
# "undefined symbol: cblas_dgemm", so warn loudly at configure time.
blas_found = false
openblas_dep = dependency('openblas', required: false)
if openblas_dep.found()
    core_deps += [openblas_dep]
    blas_found = true
else
    openblas_lib = cc.find_library('openblas', required: false)
    if openblas_lib.found()
        core_deps += [openblas_lib]
        blas_found = true
    else
        blas_dep = dependency('blas', required: false)
        lapack_dep = dependency('lapack', required: false)
        if blas_dep.found() and lapack_dep.found()
            core_deps += [blas_dep, lapack_dep]
            blas_found = true
        endif
    endif
endif

if not blas_found
    # This USED to be a warning, and a warning is not enough. The extensions compile
    # without a BLAS and fail at import, so the build succeeds, the artifact looks
    # right, and it breaks on the first machine that does not already have a BLAS
    # loaded into the process. That is exactly what happened: manylinux wheels were
    # produced whose _laplacians and _linalg raised
    # "undefined symbol: cblas_dgemm" in a clean container, while passing every check
    # on a machine whose conda environment supplied the symbols by accident.
    message = (
        'No BLAS/LAPACK found at build time. The compiled extensions call cblas_* ' +
        'and LAPACK (dsyev_, dgesvd_, ...), so they will build and then fail at ' +
        'IMPORT with "undefined symbol: cblas_dgemm" on any machine that does not ' +
        'already have those symbols loaded. Install the dev files first:\n' +
        '  Debian/Ubuntu   apt install libopenblas-dev\n' +
        '  RHEL/Alma       yum install openblas-devel\n' +
        '  conda           conda install openblas\n' +
        '  macOS           brew install openblas\n' +
        'To build anyway, knowing the symbols arrive some other way at runtime:\n' +
        '  pip install . -Csetup-args=-Dblas=disabled'
    )
    if get_option('blas').disabled()
        warning(message)
    else
        error(message)
    endif
endif

# Target the NumPy 2.0 C-API so the accessor functions Cython 3.2+ generates
# (PyDataType_*, _PyUFuncObject_GET_ITEM_DATA, the datetime metadata macro, ...)
# are declared. Without this the build compiles against NumPy's baseline API and
# those symbols are implicitly-declared -> build failure on NumPy >= 2.x.
_common_c_args = c_args + ['-Wno-unused-function',
                           '-DNPY_TARGET_VERSION=NPY_2_0_API_VERSION']

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
    '_void',
    '_rcfe',
    '_joins',
    '_query',
    '_fiber',
    '_quotient',
    '_field',
    '_signal',
    '_dirac',
    '_hypermanifold',
    '_interfacing',
    '_channels',
    '_cross_complex',
    '_l_gb',
    '_holomorphic',
    '_color',
    '_harmonic',
    '_temporal_entity',
    '_curvature',
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

# Modules that return inf/nan need -ffinite-math-only disabled.
# -ffast-math implies -ffinite-math-only, which makes float('inf')
# undefined behavior at the C level. Adding -fno-finite-math-only
# restores IEEE inf/nan semantics while keeping all other fast-math
# optimizations (reassociation, FMA contraction, no errno).
_finite_safe_args = _common_c_args + ['-fno-finite-math-only']
_finite_safe_modules = ['_character']

foreach mod : _finite_safe_modules
    py.extension_module(
        mod,
        mod + '.pyx',
        dependencies: core_deps,
        c_args: _finite_safe_args,
        install: true,
        subdir: 'rexgraph/core',
        override_options: ['werror=false'],
    )
endforeach
