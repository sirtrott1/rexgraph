"""
agent.rcdb_protected_index: compatibility surface for the sibling rcdb package.

The implementation moved out of the agent so a store can be installed and reasoned about
without the application. Everything public is re-exported here, so the thirty-odd modules
and sixty-odd test files that import `agent.rcdb_protected_index` keep working unchanged.

Re-exported dynamically rather than by name because the surface is large and a hand
written list is a second place to forget something: a name added to the package and not
to the list would simply stop existing here.
"""
from rcdb import protected_index as _source

for _name in dir(_source):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_source, _name)

del _name, _source
