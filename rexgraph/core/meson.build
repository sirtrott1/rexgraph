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
# 'blas_provider' can name the library outright, which is what the wheels do: Accelerate
# on macOS, and the scipy-openblas32 library on Windows with its scipy_ prefix renamed.
# Otherwise OpenBLAS is tried through pkg-config (conda-forge / distro *-dev packages),
# then as a bare library, then separate blas/lapack. Without any of them the extensions
# still compile but import with "undefined symbol: cblas_dgemm", so this is an error.
blas_found = false
blas_provider = get_option('blas_provider')
if blas_provider == 'accelerate'
    core_deps += [dependency('appleframeworks', modules: ['Accelerate'], required: true)]
    blas_found = true
elif blas_provider == 'scipy-openblas'
    blas_dir = get_option('blas_dir')
    scipy_openblas = cc.find_library(
        host_machine.system() == 'windows' ? 'libscipy_openblas' : 'scipy_openblas',
        dirs: blas_dir == '' ? [] : [blas_dir], required: true)
    renames = []
    foreach symbol : ['cblas_daxpy', 'cblas_dcopy', 'cblas_ddot', 'cblas_dgemm', 'cblas_dgemv',
                      'cblas_dnrm2', 'cblas_dscal', 'cblas_dsymv', 'dgelsd_', 'dgesvd_', 'dlamch_',
                      'dpotrf_', 'dpotrs_', 'dsyev_', 'dsyevr_', 'zgesvd_', 'zheev_',
                      'dgeqrf_', 'dorgqr_', 'dgesv_']
        renames += ['-D' + symbol + '=scipy_' + symbol]
    endforeach
    core_deps += [declare_dependency(compile_args: renames, dependencies: [scipy_openblas])]
    blas_found = true
endif

if not blas_found
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
# Build and runtime requirements therefore both declare numpy>=2.0.
#
# The warning flag is GCC and Clang only; MSVC rejects it outright, so it is kept only
# where the compiler accepts it.
_common_c_args = c_args + cc.get_supported_arguments('-Wno-unused-function') + [
    '-DNPY_TARGET_VERSION=NPY_2_0_API_VERSION']
if not omp_dep.found()
    # A BLAS pkg-config file can add -fopenmp even when our OpenMP option is
    # disabled. Override that compile flag so the serial kernels do not retain
    # unresolved omp_* symbols without a matching link dependency.
    _common_c_args += cc.get_supported_arguments('-fno-openmp')
endif

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
    '_harmonic',
    '_temporal_entity',
    '_curvature',
    '_recordlog',
    '_exact_ratio',
    '_ternary',
    '_channel_tower',
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
_finite_safe_args = _common_c_args + cc.get_supported_arguments('-fno-finite-math-only')
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

# Build the optional native kernels with system ROCm or the ROCm Python SDK.
hip_option = get_option('hip')
if not hip_option.disabled()
    hip_builder = files('../_hip_build.py')
    hip_probe = run_command(py, hip_builder, '--probe', '--arch', get_option('hip_arch'), check: true)
    hip_arch = hip_probe.stdout().strip()
    if hip_arch != ''
        custom_target('lib_ternary_hip',
            input: '_ternary_hip.hip',
            output: 'lib_ternary_hip.so',
            command: [py, hip_builder, '--arch', hip_arch, '--compile', '@INPUT@', '@OUTPUT@'],
            install: true,
            install_dir: py.get_install_dir() / 'rexgraph/core',
            build_by_default: true)
        message('HIP packed kernels: building for ' + hip_arch)
    elif hip_option.enabled()
        error('HIP kernels requested but no compiler and GPU target were found. Run make install-gpu or set HIPCC and HIP_ARCH.')
    else
        message('HIP packed kernels: no compiler or GPU target found')
    endif
endif
