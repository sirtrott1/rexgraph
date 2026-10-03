const e = React.createElement;

const VIEWS = [
  "Overview", "Structure", "Hodge", "Character", "Flow", "Files", "RCDB", "State", "Queries"
];
const EMPTY_VIEWS = ["Overview", "Queries"];

function sourceViews(info) {
  if (!info?.accessible || info.error) return EMPTY_VIEWS;
  return VIEWS.filter(name => (info.panels || []).includes(name));
}

function useApi(path) {
  const [data, setData] = React.useState(null);
  const [error, setError] = React.useState(null);
  React.useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setData(null);
    setError(null);
    fetch(path, {signal:controller.signal}).then(r => r.ok ? r.json() : r.json().then(x => Promise.reject(x.detail)))
      .then(x => { if (active) setData(x); })
      .catch(x => { if (active && x?.name !== "AbortError") setError(String(x)); });
    return () => { active = false; controller.abort(); };
  }, [path]);
  return [data, error];
}

function Card({title, children}) {
  return e("div", {className:"card"},
    e("div", {className:"card-header"}, e("h3", null, title)),
    e("div", {className:"card-body"}, children));
}

function Overview({source}) {
  const [data, error] = useApi(source ? `/api/source?name=${encodeURIComponent(source)}` : "/api/health");
  if (error) return e("div", {className:"error"}, error);
  if (!data) return e("div", {className:"empty"}, "Loading");
  if (!source) return e(Card, {title:"System"}, e("div", {className:"system-empty"}, "Register a Rex source to inspect it."));
  if (!data.cells) return e(Card, {title:"Source"}, e("pre", {className:"json"}, JSON.stringify(data, null, 2)));
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
  return e(QueryBackedPanel, {name, source});
}


function FilesView({source}) {
  const [q, setQ] = React.useState("");
  const path = source ? `/api/catalog?name=${encodeURIComponent(source)}&q=${encodeURIComponent(q)}` : "/api/health";
  const [data, error] = useApi(path);
  if (!source) return e(Card, {title:"Files"}, e("div", {className:"empty"}, "Select a catalog source."));
  return e(React.Fragment, null,
    e(Card, {title:"Search"}, e("input", {className:"input", value:q, onChange:x=>setQ(x.target.value), placeholder:"literal terms"})),
    error ? e("div", {className:"error"}, error) : null,
    e(Card, {title:"Catalog"}, e("pre", {className:"json system-result"}, data ? JSON.stringify(data, null, 2) : "Loading")));
}

function QueryView({source, sourceInfo}) {
  const initial = sourceInfo?.default_query || "";
  const [text, setText] = React.useState(initial);
  const [exactness, setExactness] = React.useState("declared");
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState(null);
  const pending = React.useRef(null);
  React.useEffect(() => {
    setText(sourceInfo?.default_query || "");
  }, [source, sourceInfo?.default_query]);
  React.useEffect(() => {
    setResult(null);
    setError(null);
    return () => { pending.current?.abort(); pending.current = null; };
  }, [source, sourceInfo?.default_query, exactness]);
  function run() {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setResult(null);
    setError(null);
    fetch("/api/query", {method:"POST", signal:controller.signal, headers:{"Content-Type":"application/json"}, body:JSON.stringify({query:text, exactness})})
      .then(r => r.ok ? r.json() : r.json().then(x => Promise.reject(x.detail)))
      .then(x => { if (pending.current === controller) setResult(x); })
      .catch(x => { if (pending.current === controller && x?.name !== "AbortError") setError(String(x)); });
  }
  return e(React.Fragment, null,
    e(Card, {title:"RCQL"},
      e("textarea", {className:"input system-query", value:text, onChange:x=>setText(x.target.value), spellCheck:false, placeholder:"Enter an RCQL query."}),
      e("select", {className:"input", value:exactness, onChange:x=>setExactness(x.target.value), "aria-label":"Numeric results"},
        e("option", {value:"declared"}, "Declared numeric results"),
        e("option", {value:"exact"}, "Require exact numeric results")),
      e("div", {style:{marginTop:"8px"}}, e("button", {className:"primary", onClick:run, disabled:!text.trim()}, "Run"))),
    error ? e("div", {className:"error"}, error) : null,
    e(Card, {title:"Result"}, e("pre", {className:"json system-result"}, result ? JSON.stringify(result, null, 2) : "No query run.")));
}

function App() {
  const [view, setView] = React.useState("Overview");
  const [sourcesData, sourcesError] = useApi("/api/sources");
  const rows = sourcesData?.sources || [];
  const names = rows.filter(x => x.accessible && !x.error).map(x => x.name);
  const [source, setSource] = React.useState("");
  const sourceInfo = (sourcesData?.sources || []).find(x => x.name === source);
  const views = sourceViews(sourceInfo);
  const activeView = views.includes(view) ? view : "Overview";
  React.useEffect(() => {
    if (sourcesData && !names.includes(source)) setSource(names[0] || "");
  }, [sourcesData, source]);
  React.useEffect(() => {
    if (!views.includes(view)) setView("Overview");
  }, [source, view, sourceInfo?.panels]);
  const content = activeView === "Overview" ? e(Overview, {source}) : activeView === "Files" ? e(FilesView, {source}) : activeView === "Queries" ? e(QueryView, {source, sourceInfo}) : e(PanelView, {name:activeView, source});
  return e("div", {className:"app system-shell"},
    e("aside", {className:"sidebar"},
      e("div", {className:"sidebar-brand"}, e("div", {className:"dot"}), e("h1", null, "rexgraph system")),
      e("div", {className:"sidebar-section"}, "Observe"),
      e("nav", null, views.map(name => e("button", {key:name, className:activeView===name?"active":"", onClick:()=>setView(name)}, name))),
      e("div", {className:"sidebar-spacer"}),
      e("div", {className:"sidebar-footer"}, `system ${sourcesData?.version || ""}`)),
    e("main", {className:"main system-main"},
      e("div", {className:"mobile-nav"}, views.map(name => e("button", {key:name, className:activeView===name?"active":"", onClick:()=>setView(name)}, name))),
      e("div", {className:"content"},
        e("div", {className:"system-toolbar"},
          e("h1", null, activeView),
          e("div", {style:{flex:1}}),
          e("select", {className:"input", value:source, onChange:x=>setSource(x.target.value)},
            e("option", {value:""}, "No source"), rows.map(row => e("option", {key:row.name, value:row.name, disabled:!row.accessible || !!row.error}, row.name)))),
        sourcesError ? e("div", {className:"error"}, sourcesError) : null,
        content)));
}

ReactDOM.createRoot(document.getElementById("root")).render(e(App));
