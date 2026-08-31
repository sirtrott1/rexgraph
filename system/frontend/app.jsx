const e = React.createElement;

const VIEWS = [
  "Overview", "Structure", "Hodge", "Character", "Flow", "Green", "Critical",
  "Temporal", "Files", "RCDB", "Models", "Agents", "State", "Queries"
];

function useApi(path) {
  const [data, setData] = React.useState(null);
  const [error, setError] = React.useState(null);
  React.useEffect(() => {
    fetch(path).then(r => r.ok ? r.json() : r.json().then(x => Promise.reject(x.detail)))
      .then(setData).catch(x => setError(String(x)));
  }, [path]);
  return [data, error];
}

function Card({title, children}) {
  return e("div", {className:"card"},
    e("div", {className:"card-header"}, e("h3", null, title)),
    e("div", {className:"card-body"}, children));
}

function Overview({source}) {
  const [data, error] = useApi(source ? `/api/sources/${encodeURIComponent(source)}` : "/api/health");
  if (error) return e("div", {className:"error"}, error);
  if (!data) return e("div", {className:"empty"}, "Loading");
  if (!source) return e(Card, {title:"System"}, e("div", {className:"system-empty"}, "Register a Rex source to inspect it."));
  const cells = data.cells || [];
  return e(React.Fragment, null,
    e("div", {className:"system-grid"},
      e(Card, {title:"Dimension"}, e("div", {className:"stat hero"}, e("div", {className:"value"}, data.dimension ?? 0), e("div", {className:"label"}, "highest grade"))),
      e(Card, {title:"Grades"}, e("div", {className:"stat hero"}, e("div", {className:"value"}, cells.length), e("div", {className:"label"}, "cell spaces"))),
      e(Card, {title:"Cells"}, e("div", {className:"stat hero"}, e("div", {className:"value"}, cells.reduce((a,b)=>a+b,0)), e("div", {className:"label"}, cells.join(" / ")))),
      e(Card, {title:"Betti"}, e("div", {className:"stat hero"}, e("div", {className:"value"}, (data.betti || []).join(" / ") || "n/a"), e("div", {className:"label"}, "by grade")))
    ),
    e(Card, {title:"Boundary tower"}, e("pre", {className:"json"}, JSON.stringify(data.boundaries || [], null, 2))));
}

function QueryBackedPanel({name, source}) {
  const [data, error] = useApi(source ? `/api/panels/${name.toLowerCase()}?source=${encodeURIComponent(source)}` : "/api/health");
  if (!source) return e(Card, {title:name}, e("div", {className:"empty"}, "Select a source."));
  if (error) return e("div", {className:"error"}, error);
  return e(Card, {title:name}, e("pre", {className:"json system-result"}, data ? JSON.stringify(data, null, 2) : "Loading"));
}

function PanelView({name, source}) {
  const queryBacked = new Set(["Structure", "Hodge", "Character", "Flow", "State"]);
  const notes = {
    Green:"Green fields, significance, Gram structure, spread and resolvent actions.",
    Critical:"Sigma deformation, critical symmetry and derived complex structures.",
    Temporal:"Existence, orientation, signing, head identity and temporal lineage.",
    RCDB:"Records, versions, indexes, lineage and bitemporal state.",
    Models:"Model state, relational operators and device placement.",
    Agents:"TurnField, Hive and relational agent state."
  };
  if (queryBacked.has(name)) return e(QueryBackedPanel, {name, source});
  return e(Card, {title:name}, e("p", {className:"system-view-note"}, notes[name] || ""));
}


function FilesView({source}) {
  const [q, setQ] = React.useState("");
  const path = source ? `/api/catalogs/${encodeURIComponent(source)}?q=${encodeURIComponent(q)}` : "/api/health";
  const [data, error] = useApi(path);
  if (!source) return e(Card, {title:"Files"}, e("div", {className:"empty"}, "Select a catalog source."));
  return e(React.Fragment, null,
    e(Card, {title:"Search"}, e("input", {className:"input", value:q, onChange:x=>setQ(x.target.value), placeholder:"literal terms"})),
    error ? e("div", {className:"error"}, error) : null,
    e(Card, {title:"Catalog"}, e("pre", {className:"json system-result"}, data ? JSON.stringify(data, null, 2) : "Loading")));
}

function QueryView({source}) {
  const initial = source ? `EXPLAIN FROM REX("${source}") RETURN DESCRIBE(), BETTI(0)` : "FROM $current RETURN DESCRIBE()";
  const [text, setText] = React.useState(initial);
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState(null);
  React.useEffect(() => { if (source) setText(`EXPLAIN FROM REX("${source}") RETURN DESCRIBE(), BETTI(0)`); }, [source]);
  function run() {
    setError(null);
    fetch("/api/query", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({query:text})})
      .then(r => r.ok ? r.json() : r.json().then(x => Promise.reject(x.detail)))
      .then(setResult).catch(x => setError(String(x)));
  }
  return e(React.Fragment, null,
    e(Card, {title:"RCQL"},
      e("textarea", {className:"input system-query", value:text, onChange:x=>setText(x.target.value), spellCheck:false}),
      e("div", {style:{marginTop:"8px"}}, e("button", {className:"primary", onClick:run}, "Run"))),
    error ? e("div", {className:"error"}, error) : null,
    e(Card, {title:"Result"}, e("pre", {className:"json system-result"}, result ? JSON.stringify(result, null, 2) : "No query run.")));
}

function App() {
  const [view, setView] = React.useState("Overview");
  const [sourcesData] = useApi("/api/sources");
  const names = (sourcesData && sourcesData.sources || []).map(x => x.name);
  const [source, setSource] = React.useState("");
  React.useEffect(() => { if (!source && names.length) setSource(names[0]); }, [names.join("|")]);
  const content = view === "Overview" ? e(Overview, {source}) : view === "Files" ? e(FilesView, {source}) : view === "Queries" ? e(QueryView, {source}) : e(PanelView, {name:view, source});
  return e("div", {className:"app system-shell"},
    e("aside", {className:"sidebar"},
      e("div", {className:"sidebar-brand"}, e("div", {className:"dot"}), e("h1", null, "rexgraph system")),
      e("div", {className:"sidebar-section"}, "Observe"),
      e("nav", null, VIEWS.map(name => e("button", {key:name, className:view===name?"active":"", onClick:()=>setView(name)}, name))),
      e("div", {className:"sidebar-spacer"}),
      e("div", {className:"sidebar-footer"}, "system 0.1.0")),
    e("main", {className:"main system-main"},
      e("div", {className:"mobile-nav"}, VIEWS.map(name => e("button", {key:name, className:view===name?"active":"", onClick:()=>setView(name)}, name))),
      e("div", {className:"content"},
        e("div", {className:"system-toolbar"},
          e("h1", null, view),
          e("div", {style:{flex:1}}),
          e("select", {className:"input", value:source, onChange:x=>setSource(x.target.value)},
            e("option", {value:""}, "No source"), names.map(name => e("option", {key:name, value:name}, name)))),
        content)));
}

ReactDOM.createRoot(document.getElementById("root")).render(e(App));
