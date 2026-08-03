"""
Test the agent layer logic without requiring compiled rexgraph.

Exercises: input type detection, feature matrix adapter edge construction,
correlation adapter, spectral clustering, auto-threshold, and session basics.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Add the agent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.adapters import EdgeConstruction
from agent.adapters.correlation import CorrelationAdapter
from agent.adapters.feature_matrix import (
    FeatureMatrixAdapter,
    _auto_threshold,
    _compute_correlation,
    _detect_column_families,
    _spectral_cluster_features,
)
from agent.auto import _classify_csv, detect_input_type
from agent.session import Session, list_sessions


def test_detect_input_type():
    """Test input type detection on various data shapes."""
    print("── Input type detection ──")

    # Feature matrix (rectangular)
    X = np.random.randn(50, 20)
    assert detect_input_type(X) == "feature_matrix"
    print("  ✓ rectangular array -> feature_matrix")

    # Correlation matrix (square symmetric, values in [-1,1])
    R = np.corrcoef(X.T)
    assert detect_input_type(R) == "correlation"
    print("  ✓ correlation matrix -> correlation")

    # Adjacency matrix (square, integer-valued, not correlation-like)
    A = np.zeros((10, 10))
    for i in range(9):
        A[i, i+1] = 1
        A[i+1, i] = 1
    assert detect_input_type(A) == "adjacency"
    print("  ✓ adjacency matrix -> adjacency")

    # File paths
    assert detect_input_type("data.rex") == "rex_file"
    assert detect_input_type("data.zarr") == "rex_file"
    assert detect_input_type("graph.json") == "json"
    print("  ✓ file extensions classified correctly")

    # DataFrame with many numeric columns -> feature matrix
    df = pd.DataFrame(np.random.randn(30, 15), columns=[f"feat_{i}" for i in range(15)])
    assert detect_input_type(df) == "feature_matrix"
    print("  ✓ numeric DataFrame -> feature_matrix")

    # DataFrame with string columns -> edge list
    df_edges = pd.DataFrame({"source": ["a", "b", "c"], "target": ["b", "c", "d"], "weight": [1.0, 2.0, 3.0]})
    assert detect_input_type(df_edges) == "edge_csv"
    print("  ✓ source/target DataFrame -> edge_csv")

    print()


def test_csv_detection():
    """Test CSV classification (edge list vs feature matrix)."""
    print("── CSV classification ──")

    # Edge list CSV
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("source,target,weight\na,b,1.0\nb,c,2.0\nc,d,3.0\n")
        edge_csv = f.name
    from pathlib import Path
    assert _classify_csv(Path(edge_csv)) == "edge_csv"
    os.unlink(edge_csv)
    print("  ✓ source/target CSV -> edge_csv")

    # Feature matrix CSV
    df = pd.DataFrame(np.random.randn(20, 10), columns=[f"feat_{i}" for i in range(10)])
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f, index=False)
        feat_csv = f.name
    assert _classify_csv(Path(feat_csv)) == "feature_csv"
    os.unlink(feat_csv)
    print("  ✓ numeric CSV -> feature_csv")

    print()


def test_feature_matrix_adapter():
    """Test the feature matrix adapter produces valid edges."""
    print("── Feature matrix adapter ──")

    np.random.seed(42)
    n_samples, n_features = 100, 30
    X = np.random.randn(n_samples, n_features)

    # Inject some correlation structure
    X[:, 5] = X[:, 0] * 0.9 + np.random.randn(n_samples) * 0.1
    X[:, 10] = X[:, 1] * 0.85 + np.random.randn(n_samples) * 0.15
    X[:, 15] = -X[:, 2] * 0.8 + np.random.randn(n_samples) * 0.2

    adapter = FeatureMatrixAdapter()
    edges = adapter.build(X)

    assert isinstance(edges, EdgeConstruction)
    assert edges.nV == n_features
    assert edges.nE > 0
    assert len(edges.sources) == edges.nE
    assert len(edges.targets) == edges.nE
    assert len(edges.weights) == edges.nE
    assert len(edges.signs) == edges.nE
    assert len(edges.type_labels) == edges.nE
    assert np.all(edges.weights >= 0)
    assert np.all(np.isin(edges.signs, [-1.0, 1.0]))
    print(f"  ✓ Built {edges.nE} edges from {n_features} features")
    print(f"    {edges.n_types} types: {edges.type_names}")

    # Check that highly correlated features have edges
    R = _compute_correlation(X)
    assert abs(R[0, 5]) > 0.8  # should be strongly correlated
    # Find if edge (0,5) or (5,0) exists
    has_edge = False
    for k in range(edges.nE):
        if (edges.sources[k] == 0 and edges.targets[k] == 5) or \
           (edges.sources[k] == 5 and edges.targets[k] == 0):
            has_edge = True
            break
    assert has_edge, "Expected edge between correlated features 0 and 5"
    print("  ✓ Correlated features connected by edges")

    # Check negative signs for anti-correlated features
    for k in range(edges.nE):
        if (edges.sources[k] == 2 and edges.targets[k] == 15) or \
           (edges.sources[k] == 15 and edges.targets[k] == 2):
            assert edges.signs[k] == -1.0, "Anti-correlated edge should have negative sign"
            print("  ✓ Anti-correlated features have negative signs")
            break

    # Test with column family names
    names = [f"shape_{i}" for i in range(10)] + \
            [f"texture_{i}" for i in range(10)] + \
            [f"intensity_{i}" for i in range(10)]
    edges_typed = adapter.build(X, feature_names=names, typing="column_family")
    assert edges_typed.n_types > 1
    assert "cross" in edges_typed.type_names
    print(f"  ✓ Column family typing: {edges_typed.type_names}")

    print()


def test_auto_threshold():
    """Test adaptive threshold selection."""
    print("── Auto threshold ──")

    np.random.seed(42)
    R = np.corrcoef(np.random.randn(50, 20).T)
    np.fill_diagonal(R, 0.0)

    thresh = _auto_threshold(R)
    assert 0.0 < thresh < 1.0

    # Count edges at this threshold
    n = R.shape[0]
    n_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(R[i, j]) > thresh:
                n_edges += 1
    density = n_edges / (n * (n - 1) / 2)
    print(f"  ✓ Threshold={thresh:.3f}, {n_edges} edges, density={density:.3f}")
    assert 0.02 < density < 0.3  # reasonable range

    print()


def test_spectral_clustering():
    """Test spectral feature clustering."""
    print("── Spectral clustering ──")

    np.random.seed(42)
    n = 20
    # Build a block-diagonal correlation matrix (2 clear clusters)
    R = np.zeros((n, n))
    R[:10, :10] = 0.6
    R[10:, 10:] = 0.6
    np.fill_diagonal(R, 1.0)
    # Add noise
    R += np.random.randn(n, n) * 0.05
    R = (R + R.T) / 2
    np.fill_diagonal(R, 0.0)

    labels = _spectral_cluster_features(R, n_clusters=2)
    assert len(labels) == n
    assert labels.dtype == np.int32

    # Check that the two blocks are mostly in different clusters
    block1_label = labels[0]
    block2_label = labels[10]
    assert block1_label != block2_label, "Two blocks should be in different clusters"
    same_block1 = np.sum(labels[:10] == block1_label)
    same_block2 = np.sum(labels[10:] == block2_label)
    assert same_block1 >= 7, f"Block 1 clustering weak: {same_block1}/10"
    assert same_block2 >= 7, f"Block 2 clustering weak: {same_block2}/10"
    print(f"  ✓ Block 1: {same_block1}/10 correct, Block 2: {same_block2}/10 correct")

    print()


def test_column_family_detection():
    """Test column name family detection."""
    print("── Column family detection ──")

    names = ["shape_volume", "shape_area", "shape_sphericity",
             "texture_contrast", "texture_entropy", "texture_energy",
             "intensity_mean", "intensity_std", "intensity_skew"]
    labels = _detect_column_families(names)
    assert labels is not None
    assert len(labels) == 9
    # shape features should share a label
    assert labels[0] == labels[1] == labels[2]
    # texture features should share a different label
    assert labels[3] == labels[4] == labels[5]
    # shape and texture should be different
    assert labels[0] != labels[3]
    print(f"  ✓ Detected families: {sorted(set(labels.tolist()))}")

    # No clear families
    names_flat = ["alpha", "beta", "gamma", "delta"]
    labels_flat = _detect_column_families(names_flat)
    assert labels_flat is None
    print("  ✓ No families detected for flat names")

    print()


def test_correlation_adapter():
    """Test the correlation matrix adapter."""
    print("── Correlation adapter ──")

    np.random.seed(42)
    X = np.random.randn(50, 15)
    R = np.corrcoef(X.T)

    adapter = CorrelationAdapter()
    edges = adapter.build(R, labels=[f"v{i}" for i in range(15)])

    assert edges.nV == 15
    assert edges.nE > 0
    assert np.all(edges.weights >= 0)
    assert np.all(edges.weights <= 1.0 + 1e-6)
    print(f"  ✓ Built {edges.nE} edges from 15×15 correlation matrix")

    print()


def test_agent_package_imports_without_pandas():
    # The platform must be pandas-optional: importing the package (which imports auto) in a fresh
    # interpreter must NOT drag in pandas. pandas is a soft dep, loaded only if a DataFrame/table
    # feature is actually exercised.
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent("""
        import sys
        import agent                  # runs __init__ -> from .auto import ...
        assert 'pandas' not in sys.modules, 'agent import pulled in pandas'
        print('OK')
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"agent import loaded pandas or failed:\n{r.stdout}\n{r.stderr}"


def test_session():
    """Test session creation and persistence."""
    print("── Session management ──")

    with tempfile.TemporaryDirectory() as tmpdir:
        session = Session("test001", tmpdir)
        assert session.session_id == "test001"
        assert len(session.snapshots) == 0
        print("  ✓ Session created")

        # Can't fully test add_snapshot without rexgraph, but test the structure
        info = session.info()
        assert info["session_id"] == "test001"
        assert info["n_steps"] == 0
        print(f"  ✓ Session info: {info['session_id']}, {info['n_steps']} steps")

        # Test list_sessions
        sessions = list_sessions(tmpdir)
        # No index file yet so it won't appear
        print(f"  ✓ Listed {len(sessions)} sessions")

    print()


def test_csv_missing_values_stay_numeric():
    # A numeric feature CSV with blank cells and NA-style tokens must still classify as feature_csv
    # and keep those columns (as float with NaN), matching pandas read_csv/select_dtypes behavior.
    import os
    import tempfile
    from pathlib import Path

    import numpy as np
    from agent.auto import _classify_csv, _read_numeric_csv
    header = ",".join(f"feat_{i}" for i in range(6))
    rows = []
    for r in range(12):
        cells = []
        for c in range(6):
            if r == 3 and c == 2:
                cells.append("")        # blank cell
            elif r == 5 and c == 4:
                cells.append("NA")      # NA token
            else:
                cells.append(f"{r + c * 0.5:.3f}")
        rows.append(",".join(cells))
    text = header + "\n" + "\n".join(rows) + "\n"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write(text)
        path = f.name
    try:
        assert _classify_csv(Path(path)) == "feature_csv"       # missing values do not demote it
        X, names = _read_numeric_csv(path)
        assert X.shape == (12, 6)                                # no column dropped on a blank/NA
        assert len(names) == 6
        assert np.isnan(X).sum() == 2                            # the blank + the NA became NaN
    finally:
        os.unlink(path)


if __name__ == "__main__":
    print("=" * 60)
    print("  RexGraph Agent - Core Logic Tests")
    print("=" * 60)
    print()

    test_detect_input_type()
    test_csv_detection()
    test_auto_threshold()
    test_spectral_clustering()
    test_column_family_detection()
    test_feature_matrix_adapter()
    test_correlation_adapter()
    test_session()
    test_csv_missing_values_stay_numeric()

    print("=" * 60)
    print("  All tests passed.")
    print("=" * 60)
