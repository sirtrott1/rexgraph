var h=React.createElement,{useState,useEffect,useRef,useCallback}=React;
try{var _th=localStorage.getItem("rexgraph_theme");if(_th)document.documentElement.dataset.theme=_th;}catch(e){}

// ── Auth state (module-level so all components can access) ──
var _authToken=sessionStorage.getItem("rexgraph_token")||"";
var _workspace=sessionStorage.getItem("rexgraph_workspace")||"default";
var _onAuthFail=null; // set by App to trigger login screen
function setAuth(token,ws){
  _authToken=token||"";_workspace=ws||_workspace;
  if(token)sessionStorage.setItem("rexgraph_token",token);else sessionStorage.removeItem("rexgraph_token");
  if(ws)sessionStorage.setItem("rexgraph_workspace",ws)}
function authHeaders(){
  var h={}; if(_authToken)h["Authorization"]="Bearer "+_authToken;
  if(_workspace&&_workspace!=="default")h["X-Workspace"]=_workspace; return h}

// ── API ──
function api(p,o){
  o=o||{};o.headers=Object.assign({},o.headers||{},authHeaders());
  return fetch("/api"+p,o).then(function(r){
    if(r.status===401){if(_onAuthFail)_onAuthFail();throw new Error("Authentication required")}
    if(!r.ok)return r.json().catch(function(){return{}}).then(function(d){throw new Error(d.detail||d.error||r.statusText)});
    return r.json().then(function(d){if(d&&d.error)throw new Error(d.error);return d})})}
function jpost(p,b){return api(p,{method:"POST",headers:Object.assign({"Content-Type":"application/json"},authHeaders()),body:JSON.stringify(b)})}
function jdel(p){return api(p,{method:"DELETE"})}
function fpost(p,fd){
  var hdrs=authHeaders(); // don't set Content-Type for FormData
  return fetch("/api"+p,{method:"POST",headers:hdrs,body:fd}).then(function(r){
    if(r.status===401){if(_onAuthFail)_onAuthFail();throw new Error("Authentication required")}
    if(!r.ok)return r.json().catch(function(){return{}}).then(function(d){throw new Error(d.detail||d.error||r.statusText)});
    return r.json().then(function(d){if(d&&d.error)throw new Error(d.error);return d})})}
function fmt(v,n){return v!=null?(typeof v==="number"?v.toFixed(n||4):String(v)):"-"}
function pct(v){return v!=null?(v*100).toFixed(1)+"%":"-"}
function downloadJSON(d,f){var b=new Blob([JSON.stringify(d,null,2)],{type:"application/json"});var u=URL.createObjectURL(b);var a=document.createElement("a");a.href=u;a.download=f||"export.json";a.click();URL.revokeObjectURL(u)}
function copyJSON(d){navigator.clipboard.writeText(JSON.stringify(d,null,2)).catch(function(){})}
function kc(k){return k>=0.7?"kappa-high":k>=0.4?"kappa-mid":"kappa-low"}

// ── Shared ──
function Table(p){if(!p.rows||!p.rows.length)return h("p",{className:"muted"},"No data.");return h("table",null,h("thead",null,h("tr",null,p.cols.map(function(c,i){return h("th",{key:i},c.l)}))),h("tbody",null,p.rows.map(function(r,i){return h("tr",{key:i},p.cols.map(function(c,j){var v=c.r?c.r(r,i):r[c.k];var cell=(v==null)?"-":(React.isValidElement(v)||Array.isArray(v)||typeof v!=="object")?v:JSON.stringify(v);return h("td",{key:j},cell)}))})))}
function Card(p){return h("div",{className:"card"},h("div",{className:"card-header"},h("h3",null,p.title),p.actions),h("div",{className:"card-body"},p.children))}
function Badge(p){return h("span",{className:"badge "+(p.type||"neutral")},p.children)}
function Err(p){return p.msg?h("div",{className:"error"},p.msg):null}
function XBar(p){if(!p.data)return null;return h("div",{className:"input-row",style:{marginTop:6}},h("button",{className:"sm",onClick:function(){downloadJSON(p.data,(p.name||"export")+".json")}},"JSON"),h("button",{className:"sm",onClick:function(){copyJSON(p.data)}},"Copy"))}
function CBar(p){var t=p.T||0,g=p.G||0,f=p.F||0,c=p.C||0,s=t+g+f+c||1;return h("div",{className:"channel-bar"},h("div",{className:"seg",style:{width:(t/s*100)+"%",background:"var(--color-T)"}}),h("div",{className:"seg",style:{width:(g/s*100)+"%",background:"var(--color-G)"}}),h("div",{className:"seg",style:{width:(f/s*100)+"%",background:"var(--color-F)"}}),h("div",{className:"seg",style:{width:(c/s*100)+"%",background:"var(--color-C)"}}))}
function SubTabs(p){return h("div",{className:"subtabs"},p.tabs.map(function(t){return h("button",{key:t,className:p.active===t?"active":"",onClick:function(){p.onChange(t)}},t)}))}
function Upload(p){var ref=useRef(),s=useState(false),busy=s[0],set=s[1],e=useState(""),err=e[0],setE=e[1];function go(f){if(!f)return;set(true);setE("");var fd=new FormData();fd.append("file",f);fd.append("options","{}");fpost(p.endpoint||"/upload",fd).then(function(d){if(p.onDone)p.onDone(d)}).catch(function(x){setE(x.message)}).finally(function(){set(false)})}return h("div",null,h("div",{className:"upload",onClick:function(){ref.current.click()},onDragOver:function(ev){ev.preventDefault()},onDrop:function(ev){ev.preventDefault();go(ev.dataTransfer.files[0])}},h("input",{ref:ref,type:"file",style:{display:"none"},accept:p.accept||".csv,.json,.tsv,.txt,.pdf,.png,.jpg,.jpeg",onChange:function(ev){go(ev.target.files[0])}}),busy?"Processing…":p.label||"Drop file or click to upload"),h(Err,{msg:err}))}
function Stat(p){return h("div",{className:"stat"},h("div",{className:"value"},p.value),h("div",{className:"label"},p.label))}
// ── information metrics (token perplexity/varentropy, structural, session trends) ──
function ppType(x){return x==null?"neutral":x<10?"good":x<30?"neutral":"warn"}
function Trend(p){var t=p.t;if(!t||t==="insufficient")return null;var a=t==="rising"?"↑":t==="falling"?"↓":"->";return h("span",{className:"badge "+(p.good===t?"good":p.bad===t?"warn":"neutral"),style:{marginLeft:4}},a+" "+t)}
function ReplyMetrics(p){var m=p.m;if(!m||(!m.token&&!m.structural&&!m.advisory))return null;var t=m.token||{},s=m.structural||{};
  return h("div",{className:"reply-metrics",style:{display:"flex",gap:6,flexWrap:"wrap",alignItems:"center",marginTop:5}},
    t.perplexity!=null&&h("span",{className:"badge "+ppType(t.perplexity),title:"token perplexity = exp(cross-entropy) - model fluency/confidence"},"PPL "+fmt(t.perplexity,2)),
    t.varentropy!=null&&h("span",{className:"badge neutral",title:"varentropy - spread of surprisal across tokens"},"vent "+fmt(t.varentropy,3)),
    s.structural_perplexity!=null&&h("span",{className:"badge neutral",title:"structural perplexity = effective modes of the reply's own relational complex"},"struct "+fmt(s.structural_perplexity,1)),
    m.response_coherence!=null&&h("span",{className:"badge "+(m.response_coherence>0.6?"good":"warn"),title:"coherence of the relations the reply asserts"},"κ "+fmt(m.response_coherence,2)),
    m.advisory&&h("span",{className:"badge warn",title:m.advisory},"⚠ fluent-but-hollow"))}
function SessionMetrics(p){var s=p.s;if(!s||!s.session)return null;var se=s.session,c=se.coherence,pp=se.perplexity;
  return h("div",{className:"session-metrics",style:{display:"flex",gap:10,flexWrap:"wrap",alignItems:"center",fontSize:12,padding:"6px 10px",background:"var(--panel-2,rgba(255,255,255,.03))",borderRadius:6,margin:"4px 0"}},
    h("span",{className:"muted"},"Session: "+(se.n_turns||0)+" turns"),
    c&&h("span",null,"coherence ",h("b",null,fmt(c.mean,3)),h(Trend,{t:se.coherence_trend,good:"rising",bad:"falling"})),
    pp&&h("span",null,"perplexity ",h("b",null,fmt(pp.mean,2)),h(Trend,{t:se.perplexity_trend,good:"falling",bad:"rising"})),
    p.onStruct&&h("button",{className:"sm",style:{marginLeft:"auto"},onClick:p.onStruct,title:"compute the structural tier (lazy, ~250ms/msg)"},"+ structure"))}

// ══════════════════════════════════════════
// PIPELINE
// ══════════════════════════════════════════
// Pipeline persistent state (survives tab switches)
var _pl={results:[],phases:[],busy:false,err:"",ontology:null,listeners:[]};
function _plNotify(){_pl.listeners.forEach(function(fn){try{fn()}catch(e){}})}
function _plRun(fd){
  if(_pl.busy)return;_pl.busy=true;_pl.phases=[];_pl.err="";_pl.ontology=null;_plNotify();
  fetch("/api/v1/pipeline/stream",{method:"POST",headers:authHeaders(),body:fd}).then(function(r){
    if(r.status===401){if(_onAuthFail)_onAuthFail();throw new Error("Authentication required")}
    if(!r.ok)return r.json().then(function(d){throw new Error(d.detail||d.error||r.statusText)});
    var reader=r.body.getReader(),dec=new TextDecoder(),buf="";
    function pump(){return reader.read().then(function(result){if(result.done){if(_pl.busy){_pl.busy=false;_plNotify()}return}
      buf+=dec.decode(result.value,{stream:true});var lines=buf.split("\n");buf=lines.pop();var evT="",evD="";
      for(var i=0;i<lines.length;i++){var ln=lines[i];if(ln.indexOf("event:")==0)evT=ln.slice(6).trim();
        else if(ln.indexOf("data:")==0){evD=ln.slice(5).trim();if(evT&&evD){try{var d=JSON.parse(evD)}catch(ex){continue}
          if(evT==="phase"){_plUpsertPhase(d);_plNotify()}
          else if(evT==="done"){if(d.documents)d.documents.forEach(function(doc){var exists=_pl.results.some(function(r){return r.doc_id===doc.doc_id});if(!exists)_pl.results.push(doc)});_pl.ontology=d.ontology||null;_pl.busy=false;_plNotify()}
          else if(evT==="error"){_pl.err=d.error||"Pipeline error";_pl.busy=false;_plNotify()}}evT="";evD=""}}return pump()})}
    return pump()}).catch(function(x){_pl.err=x.message;_pl.busy=false;_plNotify()})}
// Upsert a progress row keyed by (phase, doc) so streamed per-stage
// analysis events collapse into one updating row instead of growing
// without bound (audit A3).
function _plUpsertPhase(d){
  var key=(d.phase||"")+"|"+(d.doc_id||"");
  var arr=_pl.phases;
  for(var i=0;i<arr.length;i++){if(arr[i]._key===key){arr[i]=Object.assign({},d,{_key:key});return}}
  arr.push(Object.assign({},d,{_key:key}))}

function Pipeline(){
  var tick=useState(0),setTick=tick[1];
  useEffect(function(){function u(){setTick(function(n){return n+1})}_pl.listeners.push(u);return function(){_pl.listeners=_pl.listeners.filter(function(f){return f!==u})}},[]);
  var q=useState(""),qry=q[0],setQ=q[1];
  var bk=useState("auto"),backend=bk[0],setBk=bk[1];
  var dp=useState("standard"),depth=dp[0],setDp=dp[1];
  var on=useState(false),onto=on[0],setOn=on[1];
  var fu=useState(false),fuse=fu[0],setFuse=fu[1];
  var ref=useRef();
  var busy=_pl.busy,results=_pl.results,phases=_pl.phases,err=_pl.err,ontology=_pl.ontology;
  function run(){if(!ref.current||!ref.current.files.length)return;
    var fd=new FormData();for(var i=0;i<ref.current.files.length;i++)fd.append("files",ref.current.files[i]);
    if(qry.trim())fd.append("query",qry.trim());fd.append("depth",depth);if(onto)fd.append("ontology","true");if(fuse)fd.append("fusion","true");if(backend&&backend!=="auto")fd.append("backend",backend);fd.append("workspace",_workspace||"default");
    _plRun(fd)}
  function clearResults(){_pl.results=[];_pl.phases=[];_pl.err="";_pl.ontology=null;_plNotify()}
  var pn={ingest:"Ingest",read:"Read",ocr:"OCR",corpus:"Corpus",analysis:"Analysis",chunking:"Chunking",ontology:"Ontology",query:"Query",model:"Model",hallucination:"Verification"};
  return h("div",null,h("h2",null,"Pipeline"),
    h(Card,{title:"Upload & Analyze"},
      h("input",{ref:ref,type:"file",multiple:true,accept:".csv,.json,.tsv,.txt,.pdf,.png,.jpg,.jpeg",style:{marginBottom:8}}),
      h("div",{className:"muted",style:{fontSize:11,marginBottom:6}},"Auto-detects the input - tables (CSV/TSV), records (JSON), edge lists, text, and pre-built complexes (.safetensors/.rex/.parquet) are read by format. OCR runs only for images & PDFs."),
      h("div",{className:"input-row"},
        h("span",{className:"muted",style:{fontSize:11,alignSelf:"center"},title:"OCR backend - used only for image/PDF files; ignored for tables, JSON, and text"},"OCR:"),
        h("select",{className:"input",style:{width:120},value:backend,title:"OCR backend - used only for image/PDF files; ignored for tables, JSON, and text",onChange:function(ev){setBk(ev.target.value)}},
          h("option",{value:"auto"},"Auto"),h("option",{value:"server"},"GPU Server"),h("option",{value:"tesseract"},"Tesseract"),h("option",{value:"got-ocr"},"GOT-OCR")),
        h("select",{className:"input",style:{width:110},value:depth,onChange:function(ev){setDp(ev.target.value)},title:"Analysis depth"},
          h("option",{value:"quick"},"Quick"),h("option",{value:"standard"},"Standard"),h("option",{value:"full"},"Full")),
        h("label",{className:"chk",style:{display:"flex",alignItems:"center",gap:4,fontSize:12},title:"Run TrustGraph ontology enrichment"},
          h("input",{type:"checkbox",checked:onto,onChange:function(ev){setOn(ev.target.checked)}}),"Ontology"),
        h("label",{className:"chk",style:{display:"flex",alignItems:"center",gap:4,fontSize:12},title:"Compare OCR backends and keep the best (needs 2+ backends)"},
          h("input",{type:"checkbox",checked:fuse,onChange:function(ev){setFuse(ev.target.checked)}}),"Fusion"),
        h("input",{className:"input",value:qry,onChange:function(ev){setQ(ev.target.value)},placeholder:"Optional query…"}),
        h("button",{className:"primary",onClick:run,disabled:busy},busy?"Running…":"Run Pipeline"))),
    busy&&phases.length>0&&h(Card,{title:"Progress"},phases.map(function(p){return h("div",{key:p._key||p.phase,className:"status-row"},h("span",{className:"name"},(pn[p.phase]||p.phase)+(p.doc_id?" - "+p.doc_id:"")+(p.stage?" · "+p.stage:"")),h(Badge,{type:p.status==="done"||p.status==="doc_done"?"ok":p.status==="doc_crashed"?"fail":"gold"},p.status))})),
    h(Err,{msg:err}),
    ontology&&h(Card,{title:"Ontology enrichment"},h("div",{className:"status-row"},h("span",{className:"name"},ontology.available?("TrustGraph: "+(ontology.n_triples||0)+" enrichment triples"):("Unavailable"+(ontology.reason?": "+ontology.reason:""))),h(Badge,{type:ontology.available?"ok":"gold"},ontology.available?"ok":"n/a"))),
    results.length>0&&h("div",null,
      h(Card,{title:"Results ("+results.length+" documents)",actions:h("div",{className:"input-row"},h(XBar,{data:results,name:"pipeline-results"}),h("button",{className:"sm",onClick:clearResults},"Clear"))},
        h(Table,{cols:[{l:"ID",k:"doc_id"},{l:"V",k:"nV"},{l:"E",k:"nE"},{l:"F",k:"nF"},{l:"Betti",r:function(x){return x.betti?x.betti.join(", "):""}},
          {l:"κ",r:function(x){return x.kappa_mean!=null?h("span",{className:"kappa-val "+kc(x.kappa_mean)},fmt(x.kappa_mean,3)):"-"}},
          {l:"Hodge",r:function(x){return x.hodge?pct(x.hodge.gradient)+" / "+pct(x.hodge.curl)+" / "+pct(x.hodge.harmonic):""}}],rows:results})),
      results.map(function(doc){return doc.chunks&&doc.chunks.length>0&&h(Card,{key:"ch-"+doc.doc_id,title:"Chunks - "+doc.doc_id+" ("+doc.chunks.length+")",actions:h(XBar,{data:doc.chunks,name:"chunks-"+doc.doc_id})},
        h(Table,{cols:[{l:"#",r:function(x){return x.idx}},{l:"Edges",k:"n_edges"},{l:"κ",r:function(x){return x.kappa!=null?h("span",{className:"kappa-val "+kc(x.kappa)},fmt(x.kappa,3)):"-"}},
          {l:"Channel",r:function(x){return x.dominant_channel?h(Badge,{type:x.dominant_channel},x.dominant_channel):"-"}},
          {l:"G/C/H",r:function(x){return pct(x.hodge_gradient)+"/"+pct(x.hodge_curl)+"/"+pct(x.hodge_harmonic)}},
          {l:"Preview",r:function(x){return x.text_preview?x.text_preview.slice(0,80)+"…":""}}],rows:doc.chunks}))})))}

// ══════════════════════════════════════════
// DOCUMENTS
// ══════════════════════════════════════════
function Documents(){
  var d=useState([]),docs=d[0],setD=d[1],e=useState(""),err=e[0],setE=e[1],sel=useState(null),detail=sel[0],setSel=sel[1];
  var sub=useState("files"),tab=sub[0],setSub=sub[1];
  var ss=useState([]),sessions=ss[0],setSs=ss[1],sd=useState(null),sesDetail=sd[0],setSd=sd[1],sa=useState(null),analysis=sa[0],setSa=sa[1];
  var es=useState(""),expSes=es[0],setEs=es[1],ep=useState("betti"),propName=ep[0],setEp=ep[1],pv=useState(null),propVal=pv[0],setPv=pv[1];
  var ed=useState(0),expDim=ed[0],setEd=ed[1],ei=useState(0),expIdx=ei[0],setEi=ei[1],ex=useState(null),expl=ex[0],setEx=ex[1];
  var cx=useState(null),ctxR=cx[0],setCx=cx[1];
  var hr=useState(null),hodgeR=hr[0],setHr=hr[1],iv=useState(null),iface=iv[0],setIv=iv[1],it=useState(""),iTgt=it[0],setIt=it[1],rc=useState(null),reconf=rc[0],setRc=rc[1];
  var PROPS=["betti","euler_characteristic","chain_valid","structural_character","vertex_character","coherence","nhats","coupling_constants","eigenvalues_L0","per_channel_mixing_times","dimension","nV","nE","nF","void_complex"];
  function refresh(){api("/v1/admin/workspace/files").then(function(r){setD(r.files||r||[])}).catch(function(x){setE(x.message)})}
  function loadDetail(id){api("/v1/admin/workspace/doc/"+id).then(setSel).catch(function(x){setE(x.message)})}
  function del(id){if(!confirm("Delete "+id+"?"))return;api("/v1/admin/workspace/files/"+id,{method:"DELETE"}).then(refresh).catch(function(x){setE(x.message)})}
  function loadSessions(){api("/sessions").then(function(r){setSs(r||[])}).catch(function(){})}
  function loadSession(id){api("/sessions/"+id).then(function(r){setSd(r);setSa(null)}).catch(function(x){setE(x.message)})}
  function delSession(id){if(!confirm("Delete session "+id+"?"))return;api("/sessions/"+id,{method:"DELETE"}).then(function(){setSd(null);loadSessions()}).catch(function(x){setE(x.message)})}
  function gotoStep(id,step){setE("");api("/sessions/"+id+"/goto/"+step,{method:"POST"}).then(function(){loadSession(id)}).catch(function(x){setE(x.message)})}
  function loadAnalysis(id){setE("");api("/analysis/"+id).then(setSa).catch(function(x){setE(x.message)})}
  function exportSession(id,f){window.open("/api/v1/export/session/"+id+"?format="+(f||"json"))}
  function getProp(){if(!expSes)return;setE("");api("/explore/"+expSes+"/property/"+propName).then(setPv).catch(function(x){setE(x.message)})}
  function explainCell(){if(!expSes)return;setE("");setCx(null);api("/explore/"+expSes+"/explain/"+expDim+"/"+expIdx).then(setEx).catch(function(x){setE(x.message)})}
  function agentContext(){if(!expSes)return;setE("");var body=expDim===0?{vertices:[expIdx],max_cells:12,t:1.0}:{edges:[expIdx],max_cells:12,t:1.0};jpost("/explore/"+expSes+"/context",body).then(setCx).catch(function(x){setE(x.message)})}
  function runHodge(){if(!expSes)return;setE("");jpost("/explore/"+expSes+"/hodge",{signal:"uniform"}).then(setHr).catch(function(x){setE(x.message)})}
  function runIface(){if(!expSes||!iTgt.trim())return;setE("");var idx=iTgt.split(",").map(function(s){return parseInt(s.trim())}).filter(function(n){return!isNaN(n)});jpost("/explore/"+expSes+"/interfacing",{target_indices:idx,signal:"uniform"}).then(setIv).catch(function(x){setE(x.message)})}
  function runReconfig(){if(!expSes)return;setE("");jpost("/explore/"+expSes+"/reconfig",{}).then(setRc).catch(function(x){setE(x.message)})}
  useEffect(function(){refresh();loadSessions()},[]);
  return h("div",null,h("h2",null,"Documents"),h(Err,{msg:err}),
    h(SubTabs,{tabs:["files","sessions","explore"],active:tab,onChange:setSub}),
    tab==="files"&&h("div",null,
      h(Upload,{onDone:refresh,label:"Upload document to workspace",endpoint:"/upload"}),
      h(Card,{title:"Workspace ("+docs.length+" files)"},
        h(Table,{cols:[{l:"Name",r:function(x){return h("span",{style:{cursor:"pointer",color:"var(--gold)"},onClick:function(){loadDetail(x.name||x)}},x.name||x)}},{l:"Size",r:function(x){return x.size?Math.round(x.size/1024)+"K":""}},{l:"",r:function(x){return h("button",{className:"sm danger",onClick:function(){del(x.name||x)}},"×")}}],
          rows:docs.map?docs.map(function(f){return typeof f==="string"?{name:f}:f}):docs})),
      detail&&h(Card,{title:"Detail - "+(detail.doc_id||""),actions:h(XBar,{data:detail,name:"doc-"+(detail.doc_id||"")})},
        detail.nV!=null&&h("div",{className:"grid-3"},h(Stat,{value:detail.nV,label:"Vertices"}),h(Stat,{value:detail.nE,label:"Edges"}),h(Stat,{value:detail.nF,label:"Faces"})),
        detail.betti&&h("p",{className:"detail"},"Betti: ("+detail.betti.join(", ")+")"),
        detail.kappa_mean!=null&&h("p",{className:"detail"},"κ: "+fmt(detail.kappa_mean,3)),
        detail.hodge&&h("p",{className:"detail"},"Hodge: gradient "+pct(detail.hodge.gradient)+" · curl "+pct(detail.hodge.curl)+" · harmonic "+pct(detail.hodge.harmonic)))),
    tab==="sessions"&&h("div",null,
      h(Card,{title:"Sessions ("+sessions.length+")",actions:h("button",{className:"sm",onClick:loadSessions},"Refresh")},
        h(Table,{cols:[{l:"ID",r:function(x){return h("span",{style:{cursor:"pointer",color:"var(--gold)"},onClick:function(){loadSession(x.session_id)}},x.session_id)}},{l:"Name",r:function(x){return x.name||"-"}},{l:"Steps",k:"n_steps"},{l:"Created",r:function(x){return x.created?new Date(x.created*1000).toLocaleDateString():"-"}},{l:"",r:function(x){return h("div",{style:{display:"flex",gap:4}},h("button",{className:"sm",onClick:function(){exportSession(x.session_id,"json")}},"↓"),h("button",{className:"sm danger",onClick:function(){delSession(x.session_id)}},"×"))}}],rows:sessions})),
      sesDetail&&h(Card,{title:"Session - "+sesDetail.session_id+(sesDetail.name?" - "+sesDetail.name:""),actions:h(XBar,{data:sesDetail,name:"session-"+sesDetail.session_id})},
        h("div",{className:"grid-3"},h(Stat,{value:sesDetail.n_steps||0,label:"Steps"}),h(Stat,{value:sesDetail.current_step!=null?sesDetail.current_step:"-",label:"Current"}),h(Stat,{value:sesDetail.created?new Date(sesDetail.created*1000).toLocaleDateString():"-",label:"Created"})),
        sesDetail.history&&sesDetail.history.length>0&&h("div",{style:{marginTop:12}},h("h4",null,"Snapshot Timeline"),
          h(Table,{cols:[{l:"Step",r:function(x){return h("span",{style:{cursor:"pointer",color:x.step===sesDetail.current_step?"var(--gold)":"inherit",fontWeight:x.step===sesDetail.current_step?700:400},onClick:function(){gotoStep(sesDetail.session_id,x.step)}},x.step)}},{l:"Action",k:"action"},{l:"Summary",k:"summary"},{l:"Time",r:function(x){return x.timestamp?new Date(x.timestamp*1000).toLocaleTimeString():"-"}},{l:"Rex",r:function(x){return x.has_rex?"✓":"-"}}],rows:sesDetail.history})),
        h("div",{className:"input-row",style:{marginTop:12}},h("button",{className:"primary sm",onClick:function(){loadAnalysis(sesDetail.session_id)}},"Load Analysis"),h("button",{className:"sm",onClick:function(){exportSession(sesDetail.session_id,"json")}},"Export JSON"),h("button",{className:"sm",onClick:function(){exportSession(sesDetail.session_id,"safetensors")}},"⤓ .safetensors"),h("button",{className:"sm",onClick:function(){exportSession(sesDetail.session_id,"hdf5")}},"⤓ .h5"),h("button",{className:"sm",onClick:function(){exportSession(sesDetail.session_id,"rex")}},"Export .rex"))),
      sesDetail&&analysis&&h(Card,{title:"Analysis Results",actions:h(XBar,{data:analysis,name:"analysis-"+sesDetail.session_id})},
        analysis.construction&&h("div",{className:"grid-3"},h(Stat,{value:analysis.construction.nV||"-",label:"Vertices"}),h(Stat,{value:analysis.construction.nE||"-",label:"Edges"}),h(Stat,{value:analysis.construction.nF||"-",label:"Faces"})),
        analysis.topology&&analysis.topology.betti&&h("p",{className:"detail"},"Betti: ("+analysis.topology.betti.join(", ")+")"),
        analysis.relational&&analysis.relational.kappa_mean!=null&&h("p",{className:"detail"},"κ: "+fmt(analysis.relational.kappa_mean,3)),
        analysis.hodge&&h("p",{className:"detail"},"Hodge: gradient "+pct(analysis.hodge.pct_gradient)+" · curl "+pct(analysis.hodge.pct_curl)+" · harmonic "+pct(analysis.hodge.pct_harmonic)),
        analysis.void&&h("p",{className:"detail"},"Voids: "+(analysis.void.n_voids||0)+"/"+(analysis.void.n_potential||0)))),
    tab==="explore"&&h("div",null,
      h("div",{className:"input-row"},h("label",{className:"muted"},"Session:"),
        sessions.length>0?h("select",{className:"input",style:{width:220},value:expSes,onChange:function(ev){setEs(ev.target.value)}},h("option",{value:""},"Select session…"),sessions.map(function(s){return h("option",{key:s.session_id,value:s.session_id},s.session_id+(s.name?" - "+s.name:""))})):h("input",{className:"input",style:{width:200},value:expSes,onChange:function(ev){setEs(ev.target.value)},placeholder:"Session ID"})),
      !expSes&&h("p",{className:"muted",style:{padding:20,textAlign:"center"}},"Select a session to explore."),
      expSes&&h("div",null,
        h("div",{className:"grid-2"},
          h(Card,{title:"Property Browser"},
            h("div",{className:"input-row"},h("select",{className:"input",value:propName,onChange:function(ev){setEp(ev.target.value)}},PROPS.map(function(p){return h("option",{key:p,value:p},p)})),h("button",{className:"primary sm",onClick:getProp},"Get")),
            propVal&&h("div",null,h("p",{className:"detail"},"Property: "+propVal.property),h("pre",{className:"json"},JSON.stringify(propVal.value,null,2)),h(XBar,{data:propVal,name:"prop-"+propVal.property}))),
          h(Card,{title:"Cell Explanation"},
            h("div",{className:"input-row"},h("select",{className:"input",style:{width:90},value:expDim,onChange:function(ev){setEd(parseInt(ev.target.value))}},h("option",{value:0},"Vertex"),h("option",{value:1},"Edge")),h("input",{className:"input",style:{width:80},value:expIdx,onChange:function(ev){setEi(parseInt(ev.target.value)||0)},placeholder:"Index",type:"number"}),h("button",{className:"primary sm",onClick:explainCell},"Explain"),h("button",{className:"sm",onClick:agentContext,title:"forged local context (bounded relevant sub-complex)"},"Context")),
            expl&&h("div",{style:{marginTop:6}},
              h("div",{style:{display:"flex",gap:6,flexWrap:"wrap",alignItems:"center",marginBottom:6}},
                expl.kappa!=null&&h("span",{className:"badge "+kc(expl.kappa)},"κ "+fmt(expl.kappa,3)),
                expl.degree!=null&&h("span",{className:"badge neutral"},"degree "+expl.degree),
                expl.dominant_channel!=null&&h("span",{className:"badge neutral"},"dom ch "+(["T","G","F","C"][expl.dominant_channel]||expl.dominant_channel)),
                expl.effective_resistance!=null&&h("span",{className:"badge "+(expl.effective_resistance>0.9?"warn":"good"),title:"effective resistance - 1 = bridge (critical/no backup), lower = redundant"},(expl.effective_resistance>0.9?"⚠ BRIDGE ":"redundant ")+fmt(expl.effective_resistance,3)),
                expl.n_incident_faces!=null&&h("span",{className:"badge neutral"},expl.n_incident_faces+" faces")),
              h("details",null,h("summary",{className:"muted",style:{fontSize:11,cursor:"pointer"}},"raw"),h("pre",{className:"json"},JSON.stringify(expl,null,2))),h(XBar,{data:expl,name:"explain"})),
            ctxR&&ctxR.neighborhood&&h(Card,{title:"Local Context (forged, bounded)"},
              h("div",{className:"muted",style:{fontSize:12,marginBottom:4}},"Reached "+((ctxR.neighborhood.vertices||[]).length)+" entities · "+((ctxR.neighborhood.edges||[]).length)+" relations from one diffusion"),
              (ctxR.neighborhood.vertex_labels||[]).length>0&&h("div",{style:{marginBottom:4}},h("b",{className:"muted",style:{fontSize:11}},"Entities: "),(ctxR.neighborhood.vertex_labels||[]).slice(0,16).map(function(l,i){var k=(ctxR.neighborhood.vertex_coherence||[])[i];return h("span",{key:i,className:"badge "+(k!=null?kc(k):"neutral"),style:{margin:2},title:k!=null?"κ "+fmt(k,3):""},l)})),
              (ctxR.neighborhood.edge_labels||[]).length>0&&h("div",null,h("b",{className:"muted",style:{fontSize:11}},"Relations: "),(ctxR.neighborhood.edge_labels||[]).slice(0,16).map(function(l,i){return h("span",{key:i,className:"badge neutral",style:{margin:2}},l)})),
              h(XBar,{data:ctxR,name:"context"})))),
        h("div",{className:"grid-2"},
          h(Card,{title:"Custom Hodge Decomposition"},
            h("div",{className:"input-row"},h("select",{className:"input",style:{width:130},value:"uniform"},h("option",{value:"uniform"},"Uniform signal")),h("button",{className:"primary sm",onClick:runHodge},"Decompose")),
            hodgeR&&h("div",null,h("div",{className:"grid-3"},h(Stat,{value:pct(hodgeR.pct_grad),label:"Gradient"}),h(Stat,{value:pct(hodgeR.pct_curl),label:"Curl"}),h(Stat,{value:pct(hodgeR.pct_harm),label:"Harmonic"})),h(XBar,{data:hodgeR,name:"hodge-custom"}))),
          h(Card,{title:"Interfacing Vector"},
            h("div",{className:"input-row"},h("input",{className:"input",value:iTgt,onChange:function(ev){setIt(ev.target.value)},placeholder:"Target vertex indices (0,1,2)"}),h("button",{className:"primary sm",onClick:runIface},"Compute")),
            iface&&h("div",null,h("pre",{className:"json"},JSON.stringify(iface,null,2)),h(XBar,{data:iface,name:"interfacing"})))),
        h(Card,{title:"Reconfigure"},
          h("p",{className:"muted",style:{marginBottom:8}},"Current parameters and configuration."),
          h("button",{className:"primary sm",onClick:runReconfig},"Check Parameters"),
          reconf&&h("div",{style:{marginTop:8}},reconf.message&&h("p",{style:{fontSize:13}},reconf.message),reconf.current_params&&h("pre",{className:"json"},JSON.stringify(reconf.current_params,null,2)),h(XBar,{data:reconf,name:"reconfig"}))))))}

// ══════════════════════════════════════════
// CORPUS
// ══════════════════════════════════════════
function Corpus(){
  var c=useState(null),corp=c[0],setC=c[1],e=useState(""),err=e[0],setE=e[1];
  var q=useState(""),qry=q[0],setQ=q[1],qr=useState(null),qres=qr[0],setQR=qr[1];
  var qm=useState("chi"),qmode=qm[0],setQM=qm[1],dp=useState("standard"),depth=dp[0],setDp=dp[1];
  var bl=useState(false),bld=bl[0],setBl=bl[1],tr=useState(null),temp=tr[0],setTr=tr[1];
  var sub=useState("corpus"),tab=sub[0],setSub=sub[1];
  var da=useState(0),docA=da[0],setDa=da[1],db=useState(1),docB=db[0],setDb=db[1];
  var br=useState(null),bridge=br[0],setBr=br[1],vd=useState(null),voids=vd[0],setVd=vd[1],pd=useState(null),persist=pd[0],setPd=pd[1];
  var fu=useState(null),fusion=fu[0],setFu=fu[1],fl=useState(false),fuBusy=fl[0],setFl=fl[1];
  var cm=useState(null),cmat=cm[0],setCm=cm[1],og=useState(null),onto=og[0],setOg=og[1],ob=useState(false),oBusy=ob[0],setOb=ob[1];
  var mt=useState(null),cmet=mt[0],setMt=mt[1];
  function loadMetrics(){setE("");api("/v1/corpus/metrics").then(setMt).catch(function(x){setE(x.message)})}
  function refresh(){api("/v1/corpus/summary").then(setC).catch(function(x){setE(x.message)})}
  function build(){setBl(true);setE("");jpost("/v1/corpus/build",{depth:depth}).then(refresh).catch(function(x){setE(x.message)}).finally(function(){setBl(false)})}
  function reset(){jpost("/v1/corpus/reset",{}).then(function(){setC(null);setQR(null);setCm(null);setOg(null)}).catch(function(x){setE(x.message)})}
  function compareAll(){setE("");setCm(null);var fd=new FormData();fd.append("metric","bottleneck");fpost("/v1/corpus/compare",fd).then(setCm).catch(function(x){setE(x.message)})}
  function enrich(){setOb(true);setE("");setOg(null);var fd=new FormData();fd.append("depth",depth);fpost("/v1/corpus/trustgraph",fd).then(setOg).catch(function(x){setE(x.message)}).finally(function(){setOb(false)})}
  function search(){if(!qry.trim())return;jpost("/v1/corpus/query",{query:qry.trim(),mode:qmode}).then(setQR).catch(function(x){setE(x.message)})}
  function temporal(){api("/v1/corpus/temporal").then(setTr).catch(function(x){setE(x.message)})}
  function getBridge(){setE("");setBr(null);api("/v1/corpus/bridge/"+docA+"/"+docB).then(setBr).catch(function(x){setE(x.message)})}
  function getVoids(){setE("");setVd(null);api("/v1/corpus/voids/"+docA+"/"+docB).then(setVd).catch(function(x){setE(x.message)})}
  function getPersist(){setE("");setPd(null);api("/v1/corpus/persistence/"+docA+"/"+docB).then(setPd).catch(function(x){setE(x.message)})}
  function runFusion(file){if(!file)return;setFl(true);setE("");setFu(null);var fd=new FormData();fd.append("file",file);fd.append("backends","paddleocr,offline");fpost("/v1/corpus/fusion",fd).then(setFu).catch(function(x){setE(x.message)}).finally(function(){setFl(false)})}
  useEffect(refresh,[]);
  var n=corp?corp.n_documents||0:0;
  var ids=corp&&corp.doc_ids?corp.doc_ids:[];
  return h("div",null,h("h2",null,"Corpus"),h(Err,{msg:err}),
    h(SubTabs,{tabs:["corpus","metrics","compare"],active:tab,onChange:function(t){setSub(t);if(t==="metrics")loadMetrics()}}),
    tab==="metrics"&&h("div",null,
      !corp||!corp.built?h("p",{className:"muted",style:{padding:20,textAlign:"center"}},"Build the corpus to see information metrics."):
      !cmet?h("p",{className:"muted"},"Loading…"):h("div",null,
        h(Card,{title:"Corpus Information Metrics",actions:h("div",null,h("button",{className:"sm",onClick:loadMetrics},"↻"),h(XBar,{data:cmet,name:"corpus_metrics"}))},
          cmet.corpus&&h("div",{className:"grid-2",style:{marginBottom:8}},
            h(Stat,{value:cmet.n_documents||0,label:"Documents"}),
            h(Stat,{value:cmet.corpus.corpus_diversity!=null?fmt(cmet.corpus.corpus_diversity,2):"-",label:"Diversity (effective distinct docs)"}),
            cmet.corpus.structural_perplexity&&h(Stat,{value:fmt(cmet.corpus.structural_perplexity.mean,1),label:"Struct. perplexity (mean)"}),
            cmet.corpus.coherence&&h(Stat,{value:fmt(cmet.corpus.coherence.mean,3),label:"Coherence (mean)"})),
          h(Table,{cols:[
            {l:"Document",k:"doc_id"},
            {l:"Struct. Perplexity",r:function(x){return h("span",{title:"effective modes of this document's relational complex"},fmt(x.structural_perplexity,1))}},
            {l:"Coherence",r:function(x){return h("span",{className:"kappa-val "+kc(x.coherence)},fmt(x.coherence,3))}},
            {l:"Varentropy gap",r:function(x){return fmt(x.varentropy_gap,3)}},
            {l:"H₂ reliable",r:function(x){return x.reliable?h(Badge,{type:"good"},"yes"):h(Badge,{type:"warn"},"no")}}
          ],rows:cmet.per_document||[]}))),
      ),
    tab==="corpus"&&h("div",null,
      h(Upload,{onDone:refresh,label:"Add document to corpus",endpoint:"/v1/corpus/add"}),
      h("div",{className:"input-row"},
        h("select",{className:"input",style:{width:110},value:depth,onChange:function(ev){setDp(ev.target.value)}},h("option",{value:"quick"},"Quick"),h("option",{value:"standard"},"Standard"),h("option",{value:"full"},"Full")),
        h("button",{className:"primary",onClick:build,disabled:n<1||bld},bld?"Building…":"Build ("+n+" docs)"),h("button",{onClick:reset},"Reset"),h("button",{onClick:enrich,disabled:!corp||!corp.built||oBusy,title:"TrustGraph ontology enrichment"},oBusy?"Enriching…":"Ontology")),
      onto&&h(Card,{title:"Ontology Enrichment"},onto.available?h("div",{className:"grid-2"},h(Stat,{value:onto.n_triples||0,label:"Enrichment Triples"}),h(Stat,{value:onto.n_entities||"-",label:"Entities"})):h("p",{className:"muted"},"Unavailable"+(onto.reason?": "+onto.reason:""))),
      corp&&corp.documents&&h(Card,{title:"Documents ("+corp.documents.length+")",actions:h(XBar,{data:corp,name:"corpus"})},
        h(Table,{cols:[{l:"ID",k:"doc_id"},{l:"V",k:"nV"},{l:"E",k:"nE"},{l:"Betti",r:function(x){return x.betti?x.betti.join(", "):"-"}},{l:"κ",r:function(x){return fmt(x.kappa_mean,3)}},{l:"Hodge",r:function(x){return x.hodge?pct(x.hodge.gradient)+"/"+pct(x.hodge.curl)+"/"+pct(x.hodge.harmonic):"-"}}],rows:corp.documents})),
      corp&&corp.built&&h("div",{className:"grid-2"},
        h(Card,{title:"Structural Query"},
          h("div",{className:"input-row"},h("button",{className:qmode==="chi"?"primary sm":"sm",onClick:function(){setQM("chi")}},"Chi"),h("button",{className:qmode==="spectral"?"primary sm":"sm",onClick:function(){setQM("spectral")}},"Spectral"),h("button",{className:qmode==="hybrid"?"primary sm":"sm",onClick:function(){setQM("hybrid")}},"Hybrid")),
          h("div",{className:"input-row"},h("input",{className:"input",value:qry,onChange:function(ev){setQ(ev.target.value)},onKeyDown:function(ev){if(ev.key==="Enter")search()},placeholder:"Search query…"}),h("button",{className:"primary sm",onClick:search},"Search")),
          qres&&qres.ranked&&h("div",null,h(Table,{cols:[{l:"Doc",k:"doc_id"},{l:"Score",r:function(x){return fmt(x.score)}},{l:"Shared",k:"n_shared_entities"},{l:"κ",r:function(x){return fmt(x.kappa_mean)}}],rows:qres.ranked}),h(XBar,{data:qres,name:"query"}))),
        h(Card,{title:"Temporal BIOES",actions:h("button",{className:"sm",onClick:temporal},"Run")},
          temp&&temp.tags?h(Table,{cols:[{l:"Doc",k:"doc_id"},{l:"Phase",k:"phase"},{l:"Tag",k:"tag"}],rows:temp.tags}):h("p",{className:"muted"},"Topological phase transition detection across documents.")))),
    tab==="compare"&&h("div",null,
      !corp||!corp.built?h("p",{className:"muted",style:{padding:20,textAlign:"center"}},"Build the corpus to compare documents."):h("div",null,
        h(Card,{title:"Cross-Document Comparison"},
          h("div",{className:"input-row"},
            h("label",{className:"muted"},"Doc A:"),h("select",{className:"input",style:{width:160},value:docA,onChange:function(ev){setDa(parseInt(ev.target.value))}},ids.map(function(id,i){return h("option",{key:i,value:i},id)})),
            h("label",{className:"muted"},"Doc B:"),h("select",{className:"input",style:{width:160},value:docB,onChange:function(ev){setDb(parseInt(ev.target.value))}},ids.map(function(id,i){return h("option",{key:i,value:i},id)}))),
          h("div",{className:"input-row"},h("button",{className:"primary sm",onClick:getBridge},"Bridge"),h("button",{className:"primary sm",onClick:getVoids},"Voids"),h("button",{className:"primary sm",onClick:getPersist},"Persistence"),h("button",{className:"sm",onClick:compareAll,title:"All-pairs persistence-distance matrix"},"Compare All")),
        cmat&&cmat.distance_matrix&&h(Card,{title:"All-Pairs Distance ("+(cmat.metric||"bottleneck")+")",actions:h(XBar,{data:cmat,name:"compare-all"})},
          h("table",null,h("thead",null,h("tr",null,[h("th",{key:"h"},"")].concat((cmat.doc_ids||[]).map(function(id,i){return h("th",{key:i},id)})))),
            h("tbody",null,(cmat.distance_matrix||[]).map(function(row,i){return h("tr",{key:i},[h("th",{key:"r"},(cmat.doc_ids||[])[i])].concat(row.map(function(v,j){return h("td",{key:j},i===j?"-":fmt(v,3))})))}))))),
        bridge&&h(Card,{title:"Bridge Analysis - "+ids[docA]+" <-> "+ids[docB],actions:h(XBar,{data:bridge,name:"bridge"})},
          bridge.kappa_correlation!=null&&h("div",{className:"grid-3"},h(Stat,{value:fmt(bridge.kappa_correlation,4),label:"κ Correlation"}),h(Stat,{value:bridge.n_shared||0,label:"Shared Vertices"}),h(Stat,{value:fmt(bridge.void_fraction_diff,4),label:"Void Fraction Δ"})),
          bridge.kappa_correlation==null&&h("pre",{className:"json"},JSON.stringify(bridge,null,2))),
        voids&&h(Card,{title:"Void Comparison - "+ids[docA]+" <-> "+ids[docB],actions:h(XBar,{data:voids,name:"voids"})},
          h("pre",{className:"json"},JSON.stringify(voids,null,2))),
        persist&&h(Card,{title:"Persistence Distance - "+ids[docA]+" <-> "+ids[docB],actions:h(XBar,{data:persist,name:"persistence"})},
          h("div",{className:"grid-3"},h(Stat,{value:fmt(persist.bottleneck,4),label:"Bottleneck"}),h(Stat,{value:fmt(persist.wasserstein,4),label:"Wasserstein"}),h(Stat,{value:(persist.dgm_a_size||0)+" / "+(persist.dgm_b_size||0),label:"Diagram Sizes"})),
          persist.entropy_a!=null&&h("div",{className:"grid-3"},h(Stat,{value:fmt(persist.entropy_a,4),label:"Entropy A"}),h(Stat,{value:fmt(persist.entropy_b,4),label:"Entropy B"}),h(Stat,{value:fmt(persist.entropy_delta,4),label:"Entropy Δ"})),
          persist.landscape_distance!=null&&h("p",{className:"detail"},"Landscape distance: "+fmt(persist.landscape_distance,4))),
        h(Card,{title:"OCR Fusion"},
          h("p",{className:"muted",style:{marginBottom:8}},"Compare OCR backends on a single document."),
          h("div",{className:"upload",onClick:function(){var inp=document.createElement("input");inp.type="file";inp.accept=".pdf,.png,.jpg,.jpeg";inp.onchange=function(){runFusion(inp.files[0])};inp.click()}},fuBusy?"Processing…":"Drop file or click to upload for OCR fusion"),
          fusion&&h("div",{style:{marginTop:8}},h("div",{className:"grid-2"},h(Stat,{value:fusion.n_backends||0,label:"Backends"}),h(Stat,{value:fusion.best_coherence||"-",label:"Best Coherence"})),fusion.summary&&h("p",{style:{fontSize:13,marginTop:8}},fusion.summary),h(XBar,{data:fusion,name:"fusion"}))))))}

// ══════════════════════════════════════════
// TRUSTGRAPH
// ══════════════════════════════════════════
function TrustGraph(){
  var e=useState(""),err=e[0],setE=e[1];
  var sub=useState("triples"),tab=sub[0],setSub=sub[1];
  var hs=useState(null),health=hs[0],setHs=hs[1];
  var fl=useState("default"),flow=fl[0],setFl=fl[1];
  var cmp=useState(null),compare=cmp[0],setCmp=cmp[1];
  var evo=useState(null),evol=evo[0],setEvo=evo[1];
  var qa=useState(null),assess=qa[0],setQa=qa[1],ents=useState(""),entities=ents[0],setEnts=ents[1];
  var tri=useState("Alice, knows, Bob\nBob, worksAt, Acme\nCarol, knows, Alice\nCarol, worksAt, Acme"),triText=tri[0],setTri=tri[1],tgr=useState(null),triRes=tgr[0],setTriRes=tgr[1];
  function analyzeTriples(){setE("");var rows=triText.split("\n").map(function(l){return l.split(",").map(function(x){return x.trim()})}).filter(function(r){return r.length===3&&r[0]&&r[1]&&r[2]});if(!rows.length){setE("Enter triples as: subject, predicate, object (one per line)");return}jpost("/v1/trustgraph/analyze",{triples:rows}).then(setTriRes).catch(function(x){setE(x.message)})}
  function getHealth(){setE("");jpost("/v1/trustgraph/health",{flow:flow}).then(setHs).catch(function(x){setE(x.message)})}
  function compareFlows(){setE("");jpost("/v1/trustgraph/compare",{flows:flow.split(",").map(function(f){return f.trim()})}).then(setCmp).catch(function(x){setE(x.message)})}
  function trackEvolution(){setE("");jpost("/v1/trustgraph/evolution",{flow:flow}).then(setEvo).catch(function(x){setE(x.message)})}
  function assessQuery(){if(!entities.trim())return;setE("");jpost("/v1/trustgraph/assess",{entities:entities.split(",").map(function(e){return e.trim()}),flow:flow}).then(setQa).catch(function(x){setE(x.message)})}
  return h("div",null,h("h2",null,"TrustGraph"),h(Err,{msg:err}),
    h("div",{className:"input-row"},h("label",{className:"muted"},"Flow:"),h("input",{className:"input",style:{width:200},value:flow,onChange:function(ev){setFl(ev.target.value)},placeholder:"default"})),
    h(SubTabs,{tabs:["triples","health","compare","evolution","query"],active:tab,onChange:setSub}),
    tab==="triples"&&h("div",null,
      h(Card,{title:"Analyze Knowledge Graph (standalone - no server needed)"},
        h("p",{style:{fontSize:13,marginBottom:8}},"Paste triples as \"subject, predicate, object\" (one per line). RexGraph turns them into a typed relational complex and computes topology, Hodge flow, and voids - the missing edges your graph's structure implies."),
        h("textarea",{className:"input",style:{width:"100%",height:120,fontFamily:"monospace",fontSize:12},value:triText,onChange:function(ev){setTri(ev.target.value)}}),
        h("button",{className:"primary",onClick:analyzeTriples,style:{marginTop:8}},"Analyze Triples"),
        triRes&&h("div",{style:{marginTop:12}},h(XBar,{data:triRes,name:"tg-triples"}),
          h("div",{className:"grid-4"},h(Stat,{value:triRes.n_entities,label:"Entities"}),h(Stat,{value:triRes.n_relations,label:"Relations"}),h(Stat,{value:triRes.betti?JSON.stringify(triRes.betti):"-",label:"Betti"}),h(Stat,{value:triRes.void_complex?triRes.void_complex.n_voids+"/"+triRes.void_complex.n_potential:"-",label:"Voids"})),
          triRes.hodge&&h("div",{style:{marginTop:8}},h("div",{className:"grid-3"},h(Stat,{value:pct(triRes.hodge.gradient),label:"Gradient"}),h(Stat,{value:pct(triRes.hodge.curl),label:"Curl"}),h(Stat,{value:pct(triRes.hodge.harmonic),label:"Harmonic (unresolved)"}))),
          triRes.predicate_types&&h("p",{className:"muted",style:{marginTop:8,fontSize:12}},"Predicate types: "+triRes.predicate_types.join(", ")),
          triRes.interpretation&&h("p",{style:{marginTop:6,fontSize:12}},triRes.interpretation)))),
    tab==="health"&&h("div",null,
      h("button",{className:"primary",onClick:getHealth,style:{marginBottom:12}},"Get Health Snapshot"),
      health&&h(Card,{title:"Health - "+flow,actions:h(XBar,{data:health,name:"tg-health"})},
        h("div",{className:"grid-4"},h(Stat,{value:health.status||"-",label:"Status"}),h(Stat,{value:health.dim_H||0,label:"Oscillatory Modes"}),
          h(Stat,{value:health.health_ratio!=null?fmt(health.health_ratio,3):"-",label:"Health Ratio"}),h(Stat,{value:health.cost_multiplier!=null?fmt(health.cost_multiplier,2)+"×":"-",label:"Cost Multiplier"})),
        h("div",{className:"grid-3",style:{marginTop:8}},h(Stat,{value:health.nV||0,label:"Entities"}),h(Stat,{value:health.nE||0,label:"Relations"}),h(Stat,{value:health.nF||0,label:"Faces"})))),
    tab==="compare"&&h("div",null,
      h("p",{className:"muted",style:{marginBottom:8}},"Enter comma-separated flow names to compare."),
      h("button",{className:"primary",onClick:compareFlows,style:{marginBottom:12}},"Compare Flows"),
      compare&&compare.comparison&&h(Card,{title:"Flow Comparison",actions:h(XBar,{data:compare,name:"tg-compare"})},
        h("div",{className:"grid-2"},h(Stat,{value:compare.comparison.most_stable||"-",label:"Most Stable"}),h(Stat,{value:compare.comparison.most_frustrated||"-",label:"Most Frustrated"})),
        compare.per_flow&&h(Table,{cols:[{l:"Flow",r:function(x){return x[0]}},{l:"V",r:function(x){return x[1].nV||"-"}},{l:"E",r:function(x){return x[1].nE||"-"}},
          {l:"dim(H)",r:function(x){return x[1].dim_H||0}},{l:"Health",r:function(x){return x[1].health_ratio!=null?fmt(x[1].health_ratio,3):"-"}},{l:"Harmonic%",r:function(x){return pct(x[1].pct_harmonic)}}],
          rows:Object.entries(compare.per_flow)}))),
    tab==="evolution"&&h("div",null,
      h("button",{className:"primary",onClick:trackEvolution,style:{marginBottom:12}},"Track Evolution"),
      evol&&h(Card,{title:"Evolution - trend: "+(evol.trend||"unknown"),actions:h(XBar,{data:evol,name:"tg-evolution"})},
        h(Badge,{type:evol.trend==="stabilizing"?"ok":evol.trend==="fragmenting"?"fail":"neutral"},evol.trend),
        evol.steps&&h(Table,{cols:[{l:"Step",r:function(x){return x.step}},{l:"Core",k:"core_id"},{l:"V",k:"nV"},{l:"E",k:"nE"},
          {l:"dim(H)",k:"dim_H"},{l:"Health",r:function(x){return x.health_ratio!=null?fmt(x.health_ratio,3):"-"}}],rows:evol.steps}))),
    tab==="query"&&h("div",null,
      h("div",{className:"input-row"},h("input",{className:"input",value:entities,onChange:function(ev){setEnts(ev.target.value)},placeholder:"Entity names, comma-separated…"}),
        h("button",{className:"primary sm",onClick:assessQuery},"Assess Query")),
      assess&&h(Card,{title:"Query Assessment",actions:h(XBar,{data:assess,name:"tg-query"})},
        h("div",{className:"grid-3"},h(Stat,{value:assess.health_ratio!=null?fmt(assess.health_ratio,3):"-",label:"Health"}),h(Stat,{value:assess.adjusted_tokens||0,label:"Adj. Tokens"}),h(Stat,{value:assess.dim_H||0,label:"Modes"})),
        assess.entities_found&&h("p",{className:"detail"},"Found: "+assess.entities_found.join(", ")),
        assess.entities_missing&&assess.entities_missing.length>0&&h("p",{className:"detail",style:{color:"var(--fail)"}},"Missing: "+assess.entities_missing.join(", ")),
        assess.recommendation&&h("p",{style:{marginTop:8,fontSize:13}},assess.recommendation),
        assess.per_entity&&h(Table,{cols:[{l:"Entity",r:function(x){return x[0]}},{l:"Connections",r:function(x){return x[1].connections}},{l:"Local Harmonic",r:function(x){return pct(x[1].local_harmonic_fraction)}},{l:"Frustration",r:function(x){return fmt(x[1].local_frustration,4)}}],rows:Object.entries(assess.per_entity)}))))}

// ══════════════════════════════════════════
// MODELS (GPU + Registry + Cache + OCR + Generate + HuggingFace)
// ══════════════════════════════════════════
function Models(){
  var s=useState(null),stat=s[0],setS=s[1],e=useState(""),err=e[0],setE=e[1];
  var ml=useState(""),model=ml[0],setM=ml[1];
  var oc=useState(null),ocr=oc[0],setOc=oc[1];
  var sub=useState("gpu"),tab=sub[0],setSub=sub[1];
  var hft=useState(""),hfText=hft[0],setHfT=hft[1],hfm=useState(""),hfModel=hfm[0],setHfM=hfm[1];
  var hfr=useState(null),hfResult=hfr[0],setHfR=hfr[1];
  var rg=useState(null),registry=rg[0],setRg=rg[1],pl=useState(""),pulling=pl[0],setPl=pl[1];
  var cc=useState(null),cache=cc[0],setCc=cc[1];
  var gp=useState(""),genPrompt=gp[0],setGp=gp[1],gs=useState(""),genSes=gs[0],setGs=gs[1];
  var gr=useState(null),genResult=gr[0],setGr=gr[1],gl=useState(false),genBusy=gl[0],setGl=gl[1];
  var or=useState(null),ocrResult=or[0],setOr=or[1],ob=useState(""),ocrBk=ob[0],setOb=ob[1],ol=useState(false),ocrBusy=ol[0],setOl=ol[1];
  function refresh(){api("/v1/status").then(function(r){setS(r.gpu_server||r)}).catch(function(x){setE(x.message)});api("/v1/ocr/status").then(setOc).catch(function(){})}
  useEffect(refresh,[]);
  function start(){setE("");jpost("/v1/models/deploy",{model_id:model||"Unlimited-OCR"}).then(function(){setTimeout(refresh,2000)}).catch(function(x){setE(x.message)})}
  function stop(){jpost("/v1/models/stop",{}).then(function(){setTimeout(refresh,1000)}).catch(function(x){setE(x.message)})}
  function analyzeHF(){if(!hfText.trim())return;setE("");jpost("/v1/huggingface/analyze",{text:hfText,model:hfModel||undefined}).then(setHfR).catch(function(x){setE(x.message)})}
  function loadRegistry(){setE("");api("/v1/models/list").then(setRg).catch(function(x){setE(x.message)})}
  var cpm=useState({model_id:"",path:"",model_type:"transformers"}),cpath=cpm[0],setCpath=cpm[1];
  function registerPath(){if(!cpath.model_id.trim()||!cpath.path.trim())return;setE("");jpost("/v1/models/set-path",cpath).then(function(){loadRegistry();setCpath({model_id:"",path:"",model_type:"transformers"})}).catch(function(x){setE(x.message)})}
  function assignPipeline(mid){var p=prompt("Assign '"+mid+"' to which pipeline stage? (chat, ocr, embedding)","chat");if(p)jpost("/v1/models/set-pipeline",{purpose:p,model_id:mid}).then(function(){setE("")}).catch(function(x){setE(x.message)})}
  function pullModel(id){setE("");setPl(id);jpost("/v1/models/pull",{model_id:id}).then(function(){setPl("");loadRegistry()}).catch(function(x){setE(x.message);setPl("")})}
  function loadCache(){setE("");api("/v1/models/cache").then(setCc).catch(function(x){setE(x.message)})}
  function delCache(name){if(!confirm("Delete cached model "+name+"?"))return;api("/v1/models/cache/"+name,{method:"DELETE"}).then(loadCache).catch(function(x){setE(x.message)})}
  function generate(){if(!genPrompt.trim())return;setGl(true);setE("");setGr(null);jpost("/v1/model/generate",{prompt:genPrompt,session_id:genSes||undefined}).then(setGr).catch(function(x){setE(x.message)}).finally(function(){setGl(false)})}
  function directOCR(file){if(!file)return;setOl(true);setE("");setOr(null);var fd=new FormData();fd.append("file",file);if(ocrBk)fd.append("backend",ocrBk);fpost("/v1/ocr",fd).then(setOr).catch(function(x){setE(x.message)}).finally(function(){setOl(false)})}
  var lrv=useState(null),lrStat=lrv[0],setLr=lrv[1],lpv=useState(""),lpath=lpv[0],setLp=lpv[1],lbv=useState(false),lBusy=lbv[0],setLb=lbv[1];
  function loadLocal(){api("/v1/model/local/status").then(setLr).catch(function(x){setE(x.message)})}
  function startLocal(){if(!lpath.trim())return;setLb(true);setE("");jpost("/v1/model/local/start",{model_path:lpath.trim()}).then(function(){loadLocal()}).catch(function(x){setE(x.message)}).finally(function(){setLb(false)})}
  function stopLocal(){jpost("/v1/model/local/stop",{}).then(function(){loadLocal()}).catch(function(x){setE(x.message)})}
  // Embedder worker - the hive's embedding worker (powers the alignment/hallucination signal).
  var emv=useState(""),epath=emv[0],setEp=emv[1],emb=useState(false),eBusy=emb[0],setEb=emb[1];
  function startEmbedder(){if(!epath.trim())return;setEb(true);setE("");jpost("/v1/model/embedder/start",{model_path:epath.trim()}).then(function(){loadLocal()}).catch(function(x){setE(x.message)}).finally(function(){setEb(false)})}
  function stopEmbedder(){jpost("/v1/model/embedder/stop",{}).then(function(){loadLocal()}).catch(function(x){setE(x.message)})}
  // Live endpoints - probe running inference servers (ollama/vllm/llama.cpp/LM Studio) on this host.
  var epsv=useState(null),eps=epsv[0],setEps=epsv[1],epb=useState(false),epBusy=epb[0],setEpb=epb[1];
  function probeEndpoints(){setEpb(true);setE("");api("/v1/model/local/endpoints").then(setEps).catch(function(x){setE(x.message)}).finally(function(){setEpb(false)})}
  // Introspection: run the RCF relational math on the running model's OWN internals.
  var iet=useState("t-cell, tumor, receptor, antibody, apoptosis, mutation"),iText=iet[0],setIText=iet[1];
  var ier=useState(null),iEmb=ier[0],setIEmb=ier[1];
  var iap=useState("The cat sat on the mat because it was tired."),iPrompt=iap[0],setIPrompt=iap[1];
  var iar=useState(null),iAttn=iar[0],setIAttn=iar[1];
  var iav=useState(null),iAvail=iav[0],setIAvail=iav[1];
  var ibv=useState(false),iBusy=ibv[0],setIBusy=ibv[1];
  function loadIntrospect(){api("/v1/model/introspect/attention/available").then(function(d){setIAvail(d.available)}).catch(function(){setIAvail(false)})}
  function runEmbed(){setIBusy(true);setE("");setIEmb(null);jpost("/v1/model/introspect",{texts:iText.split(",").map(function(s){return s.trim()}).filter(Boolean)}).then(setIEmb).catch(function(x){setE(x.message)}).finally(function(){setIBusy(false)})}
  function runAttn(){setIBusy(true);setE("");setIAttn(null);jpost("/v1/model/introspect/attention",{prompt:iPrompt}).then(setIAttn).catch(function(x){setE(x.message)}).finally(function(){setIBusy(false)})}
  return h("div",null,h("h2",null,"Models & Inference"),h(Err,{msg:err}),
    h(SubTabs,{tabs:["local","introspect","gpu","registry","cache","ocr","generate","huggingface"],active:tab,onChange:function(t){setSub(t);if(t==="registry"&&!registry)loadRegistry();if(t==="cache"&&!cache)loadCache();if(t==="local")loadLocal();if(t==="introspect"){loadLocal();loadIntrospect()}}}),
    tab==="local"&&h("div",null,
      h(Card,{title:"Local Model Runtime (llama.cpp)",actions:h("button",{className:"sm",onClick:loadLocal},"↻")},
        !lrStat?h("p",{className:"muted"},"Loading…"):h("div",null,
          h("div",{style:{display:"flex",gap:8,alignItems:"center",flexWrap:"wrap",marginBottom:8}},
            h(Badge,{type:lrStat.running?"good":"neutral"},lrStat.running?"Running: "+(lrStat.model||"model"):"Stopped"),
            h(Badge,{type:lrStat.binary_found?"good":"warn"},lrStat.binary_found?"llama.cpp found":"llama.cpp not installed"),
            lrStat.running&&lrStat.url&&h("span",{className:"muted",style:{fontSize:12}},lrStat.url+" · ctx "+lrStat.ctx_size+" · ngl "+lrStat.n_gpu_layers),
            lrStat.running&&h("button",{className:"sm danger",onClick:stopLocal},"Stop")),
          lrStat.hardware&&h("p",{className:"muted",style:{fontSize:12}},"Detected: ",h("strong",null,(lrStat.hardware.backends||[]).join(", ")||"cpu")," · ",(lrStat.hardware.gpu&&lrStat.hardware.gpu.name)||"no GPU"," · ",lrStat.hardware.ram_gb," GB RAM · model budget ≈ ",lrStat.hardware.model_budget_gb," GB"),
          lrStat.running&&lrStat.model_summary&&h("p",{className:"muted",style:{fontSize:12}},"Loaded: ",(function(m){var p=m.n_params?(m.n_params>=1e9?(m.n_params/1e9).toFixed(1)+"B":Math.round(m.n_params/1e6)+"M"):null;return [m.arch,p&&(p+" params"),m.embedding_dim&&("dim "+m.embedding_dim),m.quant,m.file_gb&&(m.file_gb+" GB")].filter(Boolean).join(" · ")})(lrStat.model_summary)),
          !lrStat.binary_found&&h("p",{className:"muted",style:{fontSize:12}},"Build a llama.cpp for your GPU (CUDA ",h("code",null,"-DGGML_CUDA=ON"),", ROCm ",h("code",null,"-DGGML_HIP=ON"),", Vulkan ",h("code",null,"-DGGML_VULKAN=ON"),", Metal ",h("code",null,"-DGGML_METAL=ON"),", or CPU) and put ",h("code",null,"llama-server")," on PATH (or set ",h("code",null,"LLAMA_SERVER_BIN"),"). Then paste a .gguf path below."),
          h("div",{className:"input-row"},h("input",{className:"input",style:{flex:1},value:lpath,onChange:function(ev){setLp(ev.target.value)},placeholder:"/path/to/model.gguf"}),h("button",{className:"primary",onClick:startLocal,disabled:lBusy||!lpath.trim()||!lrStat.binary_found},lBusy?"Starting…":"Start & use for chat")))),
      lrStat&&h(Card,{title:"Detected on disk - models already installed on this machine"},
        (lrStat.detected&&lrStat.detected.length)?h("div",null,
          h("p",{className:"muted",style:{fontSize:11,marginBottom:6}},"Auto-scanned ",h("strong",null,String(lrStat.detected.length))," model(s) across the HF cache, ollama, LM Studio, ~/models and ",h("code",null,"REXGRAPH_MODEL_DIRS"),". GGUF -> click ",h("strong",null,"Use"),". Transformers snapshots serve via vLLM/transformers."),
          h(Table,{cols:[
            {l:"Model",k:"name"},
            {l:"Format",r:function(x){return h(Badge,{type:x.format==="gguf"?"good":"neutral"},x.format)}},
            {l:"Size",r:function(x){return x.size_gb+" GB"}},
            {l:"Source",r:function(x){return h("span",{className:"muted",style:{fontSize:11}},x.source)}},
            {l:"",r:function(x){return x.format==="gguf"?h("div",{style:{display:"flex",gap:4}},
              h("button",{className:"sm",onClick:function(){setLp(x.path)},title:x.path},"Use"),
              (/embed|nomic|bge|gte|e5/i.test(x.name)?h("button",{className:"sm",onClick:function(){setEp(x.path)},title:"load as the embedding worker"},"As embedder"):null)
            ):h("span",{className:"muted",style:{fontSize:11}},"vLLM/transformers")}}
          ],rows:lrStat.detected})
        ):h("p",{className:"muted"},"No local models found. Put a .gguf under ~/models (or set ",h("code",null,"REXGRAPH_MODEL_DIRS"),"), or pull one from the model catalog below.")),
      h(Card,{title:"Live inference servers - running endpoints on this host",actions:h("button",{className:"sm",onClick:probeEndpoints,disabled:epBusy},epBusy?"Probing…":"Probe")},
        h("p",{className:"muted",style:{fontSize:11,marginBottom:6}},"Probes Ollama, vLLM, llama.cpp, LM Studio & TGI on their well-known ports (+ ",h("code",null,"REXGRAPH_PROBE_URLS"),") - actual serving endpoints, not files. Any of these is a real backend the hive can wire to."),
        !eps?h("p",{className:"muted"},"Click ",h("strong",null,"Probe")," to scan for running servers."):
          (eps.endpoints&&eps.endpoints.length?h(Table,{cols:[
            {l:"Endpoint",r:function(x){return h("code",{style:{fontSize:11}},x.url)}},
            {l:"Kind",r:function(x){return h(Badge,{type:x.managed?"gold":"good"},x.managed||x.kind)}},
            {l:"Models",r:function(x){return x.n_models?h("span",{title:(x.models||[]).join(", ")},x.n_models+" - "+(x.models||[]).slice(0,2).join(", ")+(x.n_models>2?"…":"")):h("span",{className:"muted"},"none loaded")}}
          ],rows:eps.endpoints}):h("p",{className:"muted"},"No live servers found on ",String((eps.probed||[]).length)," probed ports. Start one (",h("code",null,"ollama serve"),", ",h("code",null,"llama-server"),", ",h("code",null,"vllm serve"),") or set ",h("code",null,"REXGRAPH_PROBE_URLS"),"."))),
      lrStat&&lrStat.binary_found&&h(Card,{title:"Embedding worker - the alignment signal"},
        h("div",{style:{display:"flex",gap:8,alignItems:"center",flexWrap:"wrap",marginBottom:8}},
          h(Badge,{type:(lrStat.embedder&&lrStat.embedder.running)?"good":"neutral"},(lrStat.embedder&&lrStat.embedder.running)?"Running: "+(lrStat.embedder.model||"embedder"):"Stopped"),
          lrStat.embedder&&lrStat.embedder.running&&h("span",{className:"muted",style:{fontSize:12}},lrStat.embedder.url),
          lrStat.embedder&&lrStat.embedder.running&&h("button",{className:"sm danger",onClick:stopEmbedder},"Stop")),
        h("p",{className:"muted",style:{fontSize:11}},"Runs alongside the chat model (e.g. nomic-embed-text) so the agent monitor's semantic alignment / hallucination signal is always live."),
        h("div",{className:"input-row"},h("input",{className:"input",style:{flex:1},value:epath,onChange:function(ev){setEp(ev.target.value)},placeholder:"/path/to/embedding-model.gguf"}),h("button",{className:"primary",onClick:startEmbedder,disabled:eBusy||!epath.trim()},eBusy?"Starting…":"Start embedder"))),
      lrStat&&lrStat.recommended&&h(Card,{title:"Hive stack that fits this machine (coordinator · workers · embedder)"},
        lrStat.recommended.length?h(Table,{cols:[
          {l:"Role",r:function(x){var t=x.role==="queen"?"gold":x.role==="embedder"?"good":"neutral";return h(Badge,{type:t},x.role||"-")}},
          {l:"Model",k:"name"},{l:"Type",r:function(x){return x.kind+" · "+x.active}},
          {l:"~Size",r:function(x){return x.approx_gb+" GB"}},
          {l:"Why",r:function(x){return h("span",{className:"muted",style:{fontSize:11}},x.why)}}
        ],rows:lrStat.recommended}):h("p",{className:"muted"},"No catalog model fits the detected memory budget - a smaller quant may still run.")),
        h("p",{className:"muted",style:{fontSize:11,marginTop:4}},"The ",h("strong",null,"coordinator")," drives; ",h("strong",null,"workers")," are specialists; the ",h("strong",null,"embedder")," powers the alignment/hallucination signal. Run it alongside your chat model below.")),
    tab==="introspect"&&h("div",null,
      (!lrStat||!lrStat.running)?h(Card,{title:"Model Introspection - RCF on the model's own internals"},
        h("p",{className:"muted"},"Start a local model in the Local tab first. Introspection runs the relational-complex math on the running model's own embeddings (Tier-1) and attention (Tier-2).")):h("div",null,
        h(Card,{title:"Embedding geometry (Tier-1) - the representation space as a relational complex"},
          h("div",{className:"input-row"},
            h("input",{className:"input",style:{flex:1},value:iText,onChange:function(ev){setIText(ev.target.value)},placeholder:"comma-separated concepts"}),
            h("button",{className:"primary",onClick:runEmbed,disabled:iBusy},iBusy?"…":"Analyze")),
          iEmb&&h("div",{style:{marginTop:8,fontSize:12}},
            h("p",null,"n=",iEmb.n_items," · betti=",JSON.stringify(iEmb.betti)," · struct perplexity=",(iEmb.structural&&iEmb.structural.structural_perplexity)," · coherence=",iEmb.coherence_mean),
            iEmb.bridges&&iEmb.bridges.length?h("div",null,h("strong",null,"load-bearing concept links (effective resistance):"),
              h("ul",{style:{margin:"4px 0"}},iEmb.bridges.slice(0,6).map(function(b,i){return h("li",{key:i},b.from," <-> ",b.to," (R=",b.effective_resistance,")")}))):null)),
        h(Card,{title:"Attention (Tier-2) - the model's own attention through the Hodge/channel math"},
          iAvail===false&&h("p",{className:"muted",style:{fontSize:12}},"Tier-2 capture host not built. Build llama.cpp, then ",h("code",null,"LLAMA_DIR=~/llama.cpp bash agent/agent/native/build.sh"),"."),
          h("div",{className:"input-row"},
            h("input",{className:"input",style:{flex:1},value:iPrompt,onChange:function(ev){setIPrompt(ev.target.value)},placeholder:"prompt to run the model on"}),
            h("button",{className:"primary",onClick:runAttn,disabled:iBusy||iAvail===false},iBusy?"…":"Capture & analyze")),
          iAttn&&h("div",{style:{marginTop:8}},
            h("p",{className:"muted",style:{fontSize:12}},"n_tokens=",iAttn.n_tokens," · layers analyzed=",(iAttn.per_layer||[]).length),
            h(Table,{cols:[
              {l:"Layer",k:"layer"},
              {l:"hodge g/c/h",r:function(x){return [x.hodge_gradient,x.hodge_curl,x.hodge_harmonic].map(function(v){return v==null?"-":Number(v).toFixed(2)}).join("/")}},
              {l:"χ T/G/F/C",r:function(x){return ["chi_T","chi_G","chi_F","chi_C"].map(function(k){return x[k]==null?"-":Number(x[k]).toFixed(2)}).join("/")}},
              {l:"κ",r:function(x){return x.kappa_mean==null?"-":Number(x.kappa_mean).toFixed(3)}},
              {l:"betti",r:function(x){return JSON.stringify(x.betti||[])}}
            ],rows:iAttn.per_layer||[]}))))),
    tab==="gpu"&&h("div",{className:"grid-2"},
      h(Card,{title:"GPU Inference Server"},
        stat?h("div",null,
          h("div",{className:"status-row"},h("span",{className:"name"},"Status"),h(Badge,{type:stat.status==="running"?"ok":"neutral"},stat.status||"unknown")),
          stat.model&&h("div",{className:"status-row"},h("span",{className:"name"},"Model"),h("span",{className:"detail"},stat.model)),
          stat.port&&h("div",{className:"status-row"},h("span",{className:"name"},"Port"),h("span",{className:"detail"},stat.port)),
          stat.vram&&h("div",{className:"status-row"},h("span",{className:"name"},"VRAM"),h("span",{className:"detail"},stat.vram)),
          stat.backend&&h("div",{className:"status-row"},h("span",{className:"name"},"Backend"),h("span",{className:"detail"},stat.backend)),
          h("div",{className:"input-row",style:{marginTop:12}},
            h("input",{className:"input",value:model,onChange:function(ev){setM(ev.target.value)},placeholder:"Model name (e.g. DeepSeek-OCR-2)"}),
            h("button",{className:"primary sm",onClick:start},"Start"),h("button",{className:"sm danger",onClick:stop},"Stop"))
        ):h("p",{className:"muted"},"Loading…")),
      h(Card,{title:"vLLM Routing"},
        h("p",{style:{fontSize:13}},"Routes inference through vLLM when the server is active."),
        stat&&stat.status==="running"&&h("p",{className:"detail",style:{marginTop:8}},"Active: "+stat.model+" on port "+stat.port))),
    tab==="registry"&&h("div",null,
      h(Card,{title:"Register Custom Model / Weights"},
        h("p",{style:{fontSize:13,marginBottom:8}},"Point the app at your own model weights on the server (a HuggingFace-format dir, a .safetensors file, or a GGUF). Or serve your model anywhere OpenAI-compatible and set its URL in the Chat tab. Then assign it to the chat stage."),
        h("div",{className:"input-row",style:{flexWrap:"wrap",gap:6}},
          h("input",{className:"input",style:{width:160},value:cpath.model_id,onChange:function(ev){setCpath(Object.assign({},cpath,{model_id:ev.target.value}))},placeholder:"model id (my-model)"}),
          h("input",{className:"input",style:{flex:1,minWidth:200},value:cpath.path,onChange:function(ev){setCpath(Object.assign({},cpath,{path:ev.target.value}))},placeholder:"/path/to/weights or .safetensors"}),
          h("select",{className:"input",style:{width:130},value:cpath.model_type,onChange:function(ev){setCpath(Object.assign({},cpath,{model_type:ev.target.value}))}},h("option",{value:"transformers"},"transformers"),h("option",{value:"gguf"},"gguf"),h("option",{value:"vllm"},"vllm")),
          h("button",{className:"primary sm",onClick:registerPath,disabled:!cpath.model_id.trim()||!cpath.path.trim()},"Register"))),
      h(Card,{title:"Model Registry",actions:h("button",{className:"sm",onClick:loadRegistry},"Refresh")},
        registry&&registry.models?h(Table,{cols:[{l:"Model",r:function(x){return x.model_id||x.id}},{l:"Type",k:"type"},{l:"Purpose",r:function(x){return x.purpose||"-"}},{l:"Size",r:function(x){return (x.size_gb||0)+"GB"}},{l:"Status",r:function(x){return h(Badge,{type:x.loaded?"ok":x.downloaded?"gold":"neutral"},x.loaded?"Loaded":x.downloaded?"Local":"Available")}},{l:"",r:function(x){return h("div",{style:{display:"flex",gap:4}},x.downloaded?h("button",{className:"sm",onClick:function(){assignPipeline(x.model_id||x.id)}},"Assign"):h("button",{className:"primary sm",onClick:function(){pullModel(x.model_id||x.id)},disabled:pulling===(x.model_id||x.id)},pulling===(x.model_id||x.id)?"Pulling…":"Pull"))}}],rows:registry.models}):h("p",{className:"muted"},"Loading…"))),
    tab==="cache"&&h(Card,{title:"Model Cache"+(cache?" - "+cache.total_mb+"MB total":""),actions:h("button",{className:"sm",onClick:loadCache},"Refresh")},
      cache&&cache.entries?h(Table,{cols:[{l:"Name",k:"name"},{l:"Size",r:function(x){return x.size_mb+"MB"}},{l:"Path",r:function(x){return h("span",{className:"detail"},x.path)}},{l:"",r:function(x){return h("button",{className:"sm danger",onClick:function(){delCache(x.name)}},"×")}}],rows:cache.entries}):h("p",{className:"muted"},"Loading…")),
    tab==="ocr"&&h("div",null,
      h(Card,{title:"OCR Backends"+(ocr&&ocr.gpu?" ("+ocr.gpu.toUpperCase()+")":"")},
        ocr&&ocr.backends?h("div",null,Object.keys(ocr.backends).map(function(k){var b=ocr.backends[k];
          return h("div",{key:k,style:{marginBottom:10}},
            h("div",{className:"status-row"},h("span",{className:"name"},k),
              h(Badge,{type:b.ready?"ok":b.installed?"gold":"neutral"},b.ready?"Running":b.installed?"Installed":"Not installed")),
            b.vllm!=null&&h("p",{style:{fontSize:12,color:"var(--fg2)",paddingLeft:8,margin:"2px 0"}},
              "vLLM: "+(b.vllm?"yes":"no")+", Model: "+(b.model_downloaded?"downloaded":"not downloaded")+
              (b.server_status?", Server: "+b.server_status:"")),
            b.libraries!=null&&h("p",{style:{fontSize:12,color:"var(--fg2)",paddingLeft:8,margin:"2px 0"}},
              "Libraries: "+(b.libraries?"installed":"missing")+", Model: "+(b.model_downloaded?"downloaded":"not downloaded")),
            b.note&&h("p",{style:{fontSize:12,color:"var(--fg3)",paddingLeft:8,margin:"2px 0"}},b.note),
            b.install&&h("p",{style:{fontSize:12,fontFamily:"var(--mono)",color:"var(--fg2)",background:"var(--bg2)",padding:"4px 8px",borderRadius:4,margin:"4px 0"}},b.install),
            b.start&&b.installed&&h("p",{style:{fontSize:12,fontFamily:"var(--mono)",color:"var(--gold)",background:"var(--gold-bg)",padding:"4px 8px",borderRadius:4,margin:"4px 0"}},b.start))})):h("p",{className:"muted"},"Loading…")),
      h(Card,{title:"Direct OCR"},
        h("p",{className:"muted",style:{marginBottom:8}},"OCR a file directly."),
        h("div",{className:"input-row"},
          h("select",{className:"input",style:{width:140},value:ocrBk,onChange:function(ev){setOb(ev.target.value)}},h("option",{value:""},"Auto"),h("option",{value:"server"},"GPU Server"),h("option",{value:"tesseract"},"Tesseract"),h("option",{value:"got-ocr"},"GOT-OCR")),
          h("div",{className:"upload",style:{flex:1,padding:12,margin:0},onClick:function(){var inp=document.createElement("input");inp.type="file";inp.accept=".pdf,.png,.jpg,.jpeg";inp.onchange=function(){directOCR(inp.files[0])};inp.click()}},ocrBusy?"Processing…":"Select file")),
        ocrResult&&h("div",{style:{marginTop:8}},
          h("div",{className:"grid-3"},h(Stat,{value:ocrResult.backend||"-",label:"Backend"}),h(Stat,{value:ocrResult.elapsed?fmt(ocrResult.elapsed,1)+"s":"-",label:"Elapsed"}),h(Stat,{value:ocrResult.pages?ocrResult.pages.length:0,label:"Pages"})),
          ocrResult.text&&h("pre",{className:"json",style:{maxHeight:200}},ocrResult.text.slice(0,2000)+(ocrResult.text.length>2000?"…":"")),
          h(XBar,{data:ocrResult,name:"ocr-direct"})))),
    tab==="generate"&&h(Card,{title:"Model Generate"},
      h("p",{className:"muted",style:{marginBottom:8}},"Prompt the active model. Attach a session ID for structural context."),
      h("div",{className:"input-row"},h("label",{className:"muted"},"Session:"),h("input",{className:"input",style:{width:140},value:genSes,onChange:function(ev){setGs(ev.target.value)},placeholder:"Optional session ID"})),
      h("div",{className:"input-row"},h("input",{className:"input",value:genPrompt,onChange:function(ev){setGp(ev.target.value)},onKeyDown:function(ev){if(ev.key==="Enter")generate()},placeholder:"Prompt…"}),h("button",{className:"primary",onClick:generate,disabled:genBusy},genBusy?"Generating…":"Generate")),
      genResult&&h("div",{style:{marginTop:12}},
        h("p",{style:{whiteSpace:"pre-wrap",fontSize:13}},genResult.text),
        h("div",{className:"grid-3",style:{marginTop:8}},h(Stat,{value:genResult.model||"-",label:"Model"}),h(Stat,{value:genResult.context_included?"Yes":"No",label:"Context"}),h(Stat,{value:genResult.usage?genResult.usage.completion_tokens||0:0,label:"Tokens"})),
        h(XBar,{data:genResult,name:"generate"}))),
    tab==="huggingface"&&h("div",null,
      h(Card,{title:"Transformer Attention Analysis"},
        h("p",{style:{fontSize:13,marginBottom:12}},"Analyze transformer attention patterns. Builds a relational complex per layer, measuring chain violation, equiweight deviation, and channel specialization."),
        h("div",{className:"input-row"},h("input",{className:"input",style:{width:250},value:hfModel,onChange:function(ev){setHfM(ev.target.value)},placeholder:"Model (e.g. mistralai/Mistral-7B-v0.1)"})),
        h("div",{className:"input-row"},h("input",{className:"input",value:hfText,onChange:function(ev){setHfT(ev.target.value)},placeholder:"Input text to analyze…"}),
          h("button",{className:"primary sm",onClick:analyzeHF},"Analyze")),
        hfResult&&h("div",null,h(XBar,{data:hfResult,name:"hf-analysis"}),
          hfResult.per_layer_chain_violation&&h(Table,{cols:[{l:"Layer",r:function(x,i){return i}},{l:"Chain Violation",r:function(x){return fmt(x,6)}}],rows:hfResult.per_layer_chain_violation.map(function(v){return{v:v}})})))))}

// ══════════════════════════════════════════
// BUILDER (with LangChain + LangGraph)
// ══════════════════════════════════════════
function Builder(){
  var st=useState([]),steps=st[0],setSt=st[1],e=useState(""),err=e[0],setE=e[1];
  var rs=useState(null),result=rs[0],setRs=rs[1],ld=useState(false),busy=ld[0],setLd=ld[1];
  var sub=useState("steps"),tab=sub[0],setSub=sub[1];
  var STEPS=["ocr","text_adapter","corpus_build","corpus_query","chunking","hodge_decompose","void_analysis","hallucination_check","exchange","signal_decompose","training_export","model_inference","pipeline_run"];
  var TEMPLATES={"Document Analysis":["ocr","text_adapter","corpus_build","chunking","hodge_decompose"],"RAG Pipeline":["ocr","text_adapter","corpus_build","corpus_query","model_inference","hallucination_check"],"Training Export":["ocr","text_adapter","corpus_build","chunking","training_export"],"Full Pipeline":["ocr","text_adapter","corpus_build","chunking","corpus_query","model_inference","hallucination_check","exchange"]};
  function addStep(t){setSt(function(s){return s.concat({type:t,params:{}})})}
  function removeStep(i){setSt(function(s){return s.filter(function(_,j){return j!==i})})}
  function moveStep(i,dir){setSt(function(s){var a=s.slice();if(i+dir<0||i+dir>=a.length)return a;var t=a[i];a[i]=a[i+dir];a[i+dir]=t;return a})}
  function loadTemplate(n){if(TEMPLATES[n])setSt(TEMPLATES[n].map(function(t){return{type:t,params:{}}}))}
  function run(){if(!steps.length)return;setE("Export your config as JSON, then use it with the Pipeline tab or CLI: rexgraph-agent run --config agent-config.json")}
  // vLLM structural router
  var rp=useState(""),prompt=rp[0],setPr=rp[1],rr=useState(null),route=rr[0],setRoute=rr[1];
  function doRoute(){if(!prompt.trim())return;setE("");jpost("/v1/vllm/route",{text:prompt}).then(setRoute).catch(function(x){setE(x.message)})}
  // LangChain confidence (session-aware)
  var ses=useState([]),sessions=ses[0],setSes=ses[1],cSel=useState(""),cSes=cSel[0],setCSes=cSel[1];
  var cf=useState(null),conf=cf[0],setConf=cf[1];
  function loadSes(){api("/sessions").then(function(r){setSes(r||[])}).catch(function(){})}
  function checkConf(){setE("");var b=cSes?{session_id:cSes}:{text:prompt||"the quick brown fox"};jpost("/v1/langchain/confidence",b).then(setConf).catch(function(x){setE(x.message)})}
  useEffect(function(){loadSes()},[]);
  // LangChain tools list
  var lc=useState(null),lcTools=lc[0],setLc=lc[1];
  function getLCTools(){jpost("/v1/langchain/tools",{}).then(setLc).catch(function(x){setE(x.message)})}
  // LangGraph
  var lg=useState(null),lgState=lg[0],setLg=lg[1];
  function getLGState(){jpost("/v1/langgraph/state",{}).then(setLg).catch(function(x){setE(x.message)})}
  // Deploy - containerize this agent/pipeline
  var dmo=useState("service"),dmode=dmo[0],setDmode=dmo[1];
  var dmu=useState(""),dmUrl=dmu[0],setDmUrl=dmu[1];
  var dq=useState(""),dQuery=dq[0],setDq=dq[1],dbk=useState(""),dBackend=dbk[0],setDbk=dbk[1];
  var dpv=useState(null),dPrev=dpv[0],setDpv=dpv[1];
  function deploySpec(){return {name:"rexgraph-agent",mode:dmode,model_url:dmUrl.trim(),query:dQuery.trim(),backend:dBackend.trim(),builder_config:{steps:steps}}}
  function deployPreview(){setE("");jpost("/v1/deploy/preview",deploySpec()).then(setDpv).catch(function(x){setE(x.message)})}
  function deployDownload(){setE("");fetch("/api/v1/deploy/bundle",{method:"POST",headers:Object.assign({"Content-Type":"application/json"},authHeaders()),body:JSON.stringify(deploySpec())}).then(function(r){if(!r.ok)throw new Error("bundle failed");return r.blob()}).then(function(b){var u=URL.createObjectURL(b);var a=document.createElement("a");a.href=u;a.download=(dmode==="service"?"rexgraph-app":"rexgraph-agent")+"-deploy.zip";document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(u)}).catch(function(x){setE(x.message)})}
  return h("div",null,h("h2",null,"Agent Builder"),h(Err,{msg:err}),
    h(SubTabs,{tabs:["steps","router","confidence","langchain","langgraph","deploy"],active:tab,onChange:setSub}),
    tab==="steps"&&h("div",null,
      h(Card,{title:"Templates"},h("div",{className:"input-row"},Object.keys(TEMPLATES).map(function(n){return h("button",{key:n,className:"sm",onClick:function(){loadTemplate(n)}},n)}))),
      h(Card,{title:"Pipeline Steps ("+steps.length+")"},
        steps.map(function(s,i){return h("div",{key:i,className:"status-row"},
          h("span",{className:"name"},h(Badge,{type:"gold"},i+1)," ",s.type),
          h("div",{style:{display:"flex",gap:4}},
            h("button",{className:"sm",onClick:function(){moveStep(i,-1)},disabled:i===0},"↑"),
            h("button",{className:"sm",onClick:function(){moveStep(i,1)},disabled:i===steps.length-1},"↓"),
            h("button",{className:"sm danger",onClick:function(){removeStep(i)}},"×")))}),
        h("div",{className:"input-row",style:{marginTop:8}},
          h("select",{className:"input",style:{width:200},value:"",onChange:function(ev){if(ev.target.value)addStep(ev.target.value);ev.target.value=""}},
            h("option",{value:""},"+ Add step…"),STEPS.map(function(t){return h("option",{key:t,value:t},t)})),
          h("button",{className:"primary sm",onClick:run,disabled:busy||!steps.length},busy?"Running…":"Run"),
          h("button",{className:"sm",onClick:function(){downloadJSON({steps:steps},"agent-config.json")},disabled:!steps.length},"Export Config"))),
      result&&h(Card,{title:"Result",actions:h(XBar,{data:result,name:"builder-result"})},h("pre",{className:"json"},JSON.stringify(result,null,2)))),
    tab==="router"&&h("div",null,
      h(Card,{title:"vLLM Structural Router"},
        h("p",{style:{fontSize:13,marginBottom:12}},"Route a prompt to a model class by its structural character - no second LLM. T->reasoning, G->creative, F->analytical, C->multi-hop."),
        h("div",{className:"input-row"},h("input",{className:"input",value:prompt,onChange:function(ev){setPr(ev.target.value)},onKeyDown:function(ev){if(ev.key==="Enter")doRoute()},placeholder:"Enter a prompt to route…"}),h("button",{className:"primary sm",onClick:doRoute},"Route")),
        route&&h("div",{style:{marginTop:12}},h(XBar,{data:route,name:"vllm-route"}),
          h("div",{className:"grid-3"},h(Stat,{value:route.routed_to,label:"Routed to"}),h(Stat,{value:route.dominant_channel,label:"Dominant channel"}),h(Stat,{value:route.confidence,label:"Confidence"})),
          route.character&&h("div",{style:{marginTop:8}},h(CBar,{T:route.character.T,G:route.character.G,F:route.character.F,C:route.character.C})),
          route.reason&&h("p",{className:"muted",style:{marginTop:8,fontSize:12}},route.reason)))),
    tab==="confidence"&&h("div",null,
      h(Card,{title:"Structural Confidence (LangChain tool)"},
        h("p",{style:{fontSize:13,marginBottom:12}},"Exact structural confidence for an agent - void affinity, coherence, chain condition. A theorem, not a probability."),
        h("div",{className:"input-row"},
          h("select",{className:"input",style:{width:240},value:cSes,onChange:function(ev){setCSes(ev.target.value)}},h("option",{value:""},"Use router prompt / sample text"),sessions.map(function(s){return h("option",{key:s.session_id,value:s.session_id},s.session_id+(s.name?" - "+s.name:""))})),
          h("button",{className:"sm",onClick:loadSes},"↻"),
          h("button",{className:"primary sm",onClick:checkConf},"Check")),
        conf&&h("div",{style:{marginTop:12}},h(XBar,{data:conf,name:"confidence"}),
          h("div",{style:{marginBottom:8}},h(Badge,{type:conf.verdict==="supported"?"good":"fail"},conf.verdict)),
          h("div",{className:"grid-4"},h(Stat,{value:conf.kappa_mean!=null?fmt(conf.kappa_mean,3):"-",label:"κ mean"}),h(Stat,{value:conf.void_affinity!=null?fmt(conf.void_affinity,3):"-",label:"Void affinity"}),h(Stat,{value:conf.n_voids!=null?conf.n_voids+"/"+conf.n_potential:"-",label:"Voids"}),h(Stat,{value:conf.chain_valid?"✓":"✗",label:"∂²=0"})),
          conf.guidance&&h("p",{style:{marginTop:8,fontSize:13}},conf.guidance)))),
    tab==="langchain"&&h("div",null,
      h(Card,{title:"LangChain Tools"},
        h("p",{style:{fontSize:13,marginBottom:12}},"Four LangChain tools for structural analysis: Confidence Check, Full Analysis, Hodge Decomposition, Cell Explanation. Run Confidence directly in the tab above."),
        h("button",{className:"primary sm",onClick:getLCTools,style:{marginBottom:12}},"Load Tools"),
        lcTools&&h(Table,{cols:[{l:"Tool",k:"name"},{l:"Description",k:"description"}],rows:lcTools.tools||lcTools||[]}))),
    tab==="langgraph"&&h("div",null,
      h(Card,{title:"LangGraph State Analysis"},
        h("p",{style:{fontSize:13,marginBottom:12}},"Model agent state as a relational complex. Gradient = progress, curl = stable loops, harmonic = unresolved oscillation."),
        h("button",{className:"primary sm",onClick:getLGState,style:{marginBottom:12}},"Analyze Default State Machine"),
        lgState&&h("div",null,h(XBar,{data:lgState,name:"langgraph-state"}),
          lgState.hodge&&h("div",{className:"grid-3"},h(Stat,{value:pct(lgState.hodge.gradient),label:"Gradient (progress)"}),h(Stat,{value:pct(lgState.hodge.curl),label:"Curl (loops)"}),h(Stat,{value:pct(lgState.hodge.harmonic),label:"Harmonic (stuck)"})),
          lgState.cycles&&h("p",{className:"muted",style:{marginTop:6,fontSize:12}},"Independent cycles: "+(lgState.cycles.n_independent_cycles!=null?lgState.cycles.n_independent_cycles:"-")),
          lgState.recommendation&&h("p",{style:{marginTop:8,fontSize:13}},"Recommendation: ",h(Badge,{type:lgState.recommendation==="continue"?"good":"neutral"},lgState.recommendation)," ",lgState.reason)))),
    tab==="deploy"&&h("div",null,
      h(Card,{title:"Containerize this Agent"},
        h("p",{style:{fontSize:13,marginBottom:10}},"Export a self-contained Docker bundle. ",h("strong",null,"Service")," = the full web app + REST API. ",h("strong",null,"Pipeline")," = a headless agent that analyzes mounted documents with your configured settings. Build & run with a single ",h("code",null,"docker compose up"),"."),
        h("div",{className:"input-row"},h("label",{className:"muted",style:{width:70}},"Mode"),h("select",{className:"input",value:dmode,onChange:function(ev){setDmode(ev.target.value)}},h("option",{value:"service"},"Service (web app + API)"),h("option",{value:"pipeline"},"Pipeline (headless agent)"))),
        h("div",{className:"input-row"},h("label",{className:"muted",style:{width:70}},"LLM URL"),h("input",{className:"input",value:dmUrl,onChange:function(ev){setDmUrl(ev.target.value)},placeholder:"http://host:8000 (OpenAI-compatible, optional)"})),
        dmode==="pipeline"&&h("div",{className:"input-row"},h("label",{className:"muted",style:{width:70}},"Query"),h("input",{className:"input",value:dQuery,onChange:function(ev){setDq(ev.target.value)},placeholder:"question the agent answers per document (optional)"})),
        dmode==="pipeline"&&h("div",{className:"input-row"},h("label",{className:"muted",style:{width:70}},"OCR"),h("input",{className:"input",style:{width:160},value:dBackend,onChange:function(ev){setDbk(ev.target.value)},placeholder:"backend (e.g. tesseract)"})),
        h("div",{className:"input-row",style:{marginTop:10}},
          h("button",{className:"primary",onClick:deployDownload},"⤓ Download Container Bundle"),
          h("button",{className:"sm",onClick:deployPreview},"Preview files"),
          steps.length>0&&h("span",{className:"muted",style:{fontSize:12}},steps.length+" pipeline steps embedded"))),
      dPrev&&h(Card,{title:"Dockerfile"},h("pre",{className:"json",style:{maxHeight:200,overflow:"auto"}},dPrev.Dockerfile)),
      dPrev&&h(Card,{title:"docker-compose.yml"},h("pre",{className:"json",style:{maxHeight:160,overflow:"auto"}},dPrev["docker-compose.yml"]))))}

// ══════════════════════════════════════════
// TRAINING
// ══════════════════════════════════════════
function Training(){
  var e=useState(""),err=e[0],setE=e[1],rs=useState(null),result=rs[0],setRs=rs[1];
  var ld=useState(false),busy=ld[0],setLd=ld[1],tgt=useState("summary"),target=tgt[0],setTgt=tgt[1];
  var ef=useState("safetensors"),efmt=ef[0],setFmt=ef[1];
  var FEATURES=["kappa","chi_T","chi_G","chi_F","chi_C","hodge_gradient","hodge_curl","hodge_harmonic","void_fraction","betti_1"];
  function generate(){setLd(true);setE("");jpost("/v1/model/training",{target:target,format:efmt}).then(setRs).catch(function(x){setE(x.message)}).finally(function(){setLd(false)})}
  function download(){window.open("/api/v1/model/training/download?fmt="+encodeURIComponent(efmt)+"&target="+encodeURIComponent(target))}
  return h("div",null,h("h2",null,"Training Data"),h(Err,{msg:err}),
    h("div",{className:"grid-2"},
      h(Card,{title:"Configuration"},
        h("div",{className:"input-row"},h("label",{className:"muted",style:{width:55}},"Target"),h("select",{className:"input",value:target,onChange:function(ev){setTgt(ev.target.value)}},h("option",{value:"summary"},"Summary"),h("option",{value:"channel"},"Channel"),h("option",{value:"kappa"},"Coherence κ"),h("option",{value:"custom"},"Custom"))),
        h("div",{className:"input-row"},h("label",{className:"muted",style:{width:55}},"Format"),h("select",{className:"input",value:efmt,onChange:function(ev){setFmt(ev.target.value)}},h("option",{value:"safetensors"},"SafeTensors"),h("option",{value:"pairs"},"Training Pairs"),h("option",{value:"hf_dataset"},"HF Dataset"),h("option",{value:"rex"},"Rex Bundles"))),
        h("div",{className:"input-row",style:{marginTop:8}},
          h("button",{className:"primary",onClick:generate,disabled:busy},busy?"Generating…":"Generate"),
          (efmt==="safetensors"||efmt==="pairs")&&h("button",{className:"sm",onClick:download,title:"Download the safetensors file"},"⤓ Download"))),
      h(Card,{title:"10 Structural Features"},FEATURES.map(function(f){return h("div",{key:f,className:"status-row"},h("span",{className:"name"},f))}))),
    result&&h(Card,{title:"Export Ready",actions:h(XBar,{data:result,name:"training-"+efmt})},
      result.n_samples!=null&&h("p",{className:"detail"},"Samples: "+result.n_samples+" · Features: "+(result.n_features||"-")),
      (efmt==="safetensors"||efmt==="pairs")&&h("button",{className:"primary sm",onClick:download,style:{marginTop:8}},"⤓ Download "+efmt+" file"),
      result.path&&h("p",{className:"detail",style:{marginTop:6,fontSize:11,opacity:.6}},"Server copy: "+result.path)))}

// ══════════════════════════════════════════
// CHAT
// ══════════════════════════════════════════
function Chat(){
  var ms=useState([]),msgs=ms[0],setMs=ms[1],inp=useState(""),txt=inp[0],setInp=inp[1];
  var e=useState(""),err=e[0],setE=e[1],ld=useState(false),busy=ld[0],setLd=ld[1];
  var ss=useState([]),sessions=ss[0],setSs=ss[1];
  var cs=useState(""),curSes=cs[0],setCs=cs[1],scrollRef=useRef();
  var mo=useState(null),model=mo[0],setMo=mo[1],mu=useState(""),mUrl=mu[0],setMu=mu[1];
  var sm=useState(null),sess=sm[0],setSm=sm[1];   // session metrics (trends)
  function loadSessions(){api("/sessions").then(function(r){var l=r||[];setSs(l);if(!curSes&&l.length)setCs(l[0].session_id)}).catch(function(){})}
  function loadModel(){api("/v1/model/chat-config").then(setMo).catch(function(){})}
  useEffect(function(){loadSessions();loadModel()},[]);
  function saveModel(){jpost("/v1/model/chat-config",{url:mUrl.trim()}).then(function(r){setMo(r.status);setMu("")}).catch(function(x){setE(x.message)})}
  function loadSession(sid,structural){api("/chat/"+sid+"/metrics"+(structural?"?structural=1":"")).then(setSm).catch(function(){})}
  function send(){if(!txt.trim()||busy)return;var msg=txt.trim();var sid=curSes||("s-"+Date.now());if(!curSes)setCs(sid);setInp("");setLd(true);
    setMs(function(m){return m.concat({role:"user",content:msg})});
    jpost("/chat/"+sid,{message:msg}).then(function(r){
      setMs(function(m){return m.concat({role:"assistant",content:r.response||r.text||JSON.stringify(r),
        sections:r.sections,queryComplex:r.query_complex,modelUsed:r.model_used,cached:r.cached,
        drift:r.drift&&r.drift.kappa_drift,kappa:r.exchange&&r.exchange.kappa,metrics:r.metrics})});
      loadSession(sid);
    }).catch(function(x){setE(x.message)}).finally(function(){setLd(false);if(scrollRef.current)scrollRef.current.scrollTop=scrollRef.current.scrollHeight})}
  return h("div",null,h("h2",null,"Chat"),h(Err,{msg:err}),
    h("div",{className:"input-row",style:{marginBottom:8,flexWrap:"wrap",gap:8}},
      h("label",{className:"muted"},"Session:"),
      h("select",{className:"input",style:{width:220},value:curSes,onChange:function(ev){setCs(ev.target.value);setMs([])}},
        h("option",{value:""},"New (no document)"),
        sessions.map(function(s){return h("option",{key:s.session_id,value:s.session_id},s.session_id+(s.name?" - "+s.name:""))})),
      h("button",{className:"sm",onClick:loadSessions},"↻"),
      h("span",{style:{flex:1}}),
      h(Badge,{type:model&&model.available?"good":"neutral"},model?(model.available?"Model: "+(model.source||"on"):"No model (structural)"):"…"),
      h("input",{className:"input",style:{width:180},value:mUrl,onChange:function(ev){setMu(ev.target.value)},placeholder:"model URL (OpenAI-compat)…"}),
      h("button",{className:"sm",onClick:saveModel,disabled:!mUrl.trim()},"Set")),
    h("div",{className:"chat-container"},
      h("div",{ref:scrollRef,className:"chat-scroll"},
        msgs.length===0&&h("p",{className:"muted",style:{padding:20,textAlign:"center"}},curSes?"Ask about the selected document. Each question builds its own relational complex and retrieves the passages that structurally resonate.":"Pick a session with an uploaded document, or ask a general question."),
        msgs.map(function(m,i){return h("div",{key:i,className:"chat-msg "+m.role},h("div",{className:"from"},m.role),
          h("div",{className:"bubble",style:{whiteSpace:"pre-wrap"}},m.content),
          m.role==="assistant"&&m.queryComplex&&m.queryComplex.n_concepts>0&&h("div",{className:"muted",style:{fontSize:11,marginTop:4}},
            "query complex: "+m.queryComplex.n_concepts+" concepts, "+(m.queryComplex.n_relations||0)+" relations"+
            (m.queryComplex.betti?" · β="+JSON.stringify(m.queryComplex.betti):"")+
            (m.modelUsed?" · LLM":" · structural")+(m.cached?" · cached":"")),
          m.role==="assistant"&&m.sections&&m.sections.length>0&&h("details",{style:{marginTop:4}},
            h("summary",{className:"muted",style:{fontSize:11,cursor:"pointer"}},m.sections.length+" retrieved passage(s)"),
            m.sections.map(function(s,j){return h("div",{key:j,className:"muted",style:{fontSize:11,padding:"2px 0",borderLeft:"2px solid var(--border)",paddingLeft:8,margin:"3px 0"}},
              (s.doc_id?"["+s.doc_id+"] ":"")+"("+fmt(s.score,2)+") "+(s.text||"").slice(0,160))})),
          (m.kappa!=null||m.drift!=null)&&h("div",{className:"hall-score"},m.kappa!=null?"κ: "+fmt(m.kappa,3):"",m.drift!=null?" · drift: "+fmt(m.drift,3):""),
          m.role==="assistant"&&h(ReplyMetrics,{m:m.metrics}))}),
        busy&&h("div",{className:"chat-msg assistant"},h("div",{className:"bubble",style:{opacity:.5}},"Thinking…"))),
      sess&&sess.session&&sess.session.n_turns>0&&curSes&&h(SessionMetrics,{s:sess,onStruct:function(){loadSession(curSes,true)}}),
      h("div",{className:"input-row"},h("input",{className:"input",value:txt,onChange:function(ev){setInp(ev.target.value)},onKeyDown:function(ev){if(ev.key==="Enter")send()},placeholder:"Ask about your documents…",disabled:busy}),h("button",{className:"primary",onClick:send,disabled:busy},"Send"))))}

// ══════════════════════════════════════════
// SYSTEM
// ══════════════════════════════════════════
function System(){
  var e=useState(""),err=e[0],setE=e[1],ws=useState(null),wksp=ws[0],setWs=ws[1];
  var tk=useState([]),tokens=tk[0],setTk=tk[1],ov=useState([]),overlaps=ov[0],setOv=ov[1];
  var sub=useState("workspace"),tab=sub[0],setSub=sub[1];
  var wl=useState([]),wsList=wl[0],setWl=wl[1];
  var ac=useState(null),activity=ac[0],setAc=ac[1],ax=useState(null),actComplex=ax[0],setAx=ax[1];
  var qh=useState([]),queries=qh[0],setQh=qh[1];
  var meS=useState(null),me=meS[0],setMe=meS[1],mm=useState([]),members=mm[0],setMembers=mm[1],sw=useState(""),selWs=sw[0],setSelWs=sw[1];
  function refreshWs(){api("/v1/admin/workspace/stats").then(setWs).catch(function(x){setE(x.message)})}
  function refreshTokens(){api("/v1/admin/tokens").then(function(r){setTk(r.tokens||r||[])}).catch(function(){})}
  function loadMe(){api("/v1/admin/whoami").then(function(r){setMe(r);setSelWs(function(w){return w||r.current_workspace||"default"})}).catch(function(){})}
  function refreshMembers(w){var ws=w||selWs||"default";api("/v1/admin/members?workspace="+encodeURIComponent(ws)).then(function(r){setMembers(r.members||[])}).catch(function(){setMembers([])})}
  function refreshOverlaps(){api("/v1/admin/workspace/overlap").then(function(r){setOv(r.overlaps||r||[])}).catch(function(){})}
  function loadWorkspaces(){api("/v1/admin/workspaces").then(function(r){setWl(r.workspaces||[])}).catch(function(x){setE(x.message)})}
  function loadActivity(){api("/v1/admin/workspace/activity").then(setAc).catch(function(x){setE(x.message)})}
  function loadComplex(){api("/v1/admin/workspace/complex").then(setAx).catch(function(x){setE(x.message)})}
  function loadQueries(){api("/v1/export/queries?limit=50").then(function(r){setQh(r.queries||[])}).catch(function(x){setE(x.message)})}
  useEffect(function(){refreshWs();refreshTokens();loadMe();loadWorkspaces()},[]);
  useEffect(function(){if(selWs)refreshMembers(selWs)},[selWs]);
  var nt=useState(""),newLabel=nt[0],setNl=nt[1],nr=useState("user"),newRole=nr[0],setNr=nr[1],gt=useState(""),genTkn=gt[0],setGt=gt[1];
  function addMember(){if(!newLabel.trim())return;setE("");setGt("");jpost("/v1/admin/members?workspace="+encodeURIComponent(selWs||"default"),{user_id:newLabel.trim(),role:newRole}).then(function(r){if(r.token)setGt(r.token);else setE("Updated "+newLabel.trim()+" (existing token kept)");setNl("");refreshMembers()}).catch(function(x){setE(x.message)})}
  function revokeMember(uid){if(!confirm("Revoke "+uid+" from workspace '"+(selWs||"default")+"'?"))return;setE("");jdel("/v1/admin/members/"+encodeURIComponent(uid)+"?workspace="+encodeURIComponent(selWs||"default")).then(function(){refreshMembers()}).catch(function(x){setE(x.message)})}
  function enableAuth(){if(!members.length){setE("Add at least one member before enabling authentication, or you will be locked out.");return}if(!confirm("Enable authentication? Everyone will need their token to access rexgraph after this. Make sure at least one admin has copied their token."))return;setE("");jpost("/v1/admin/auth/enable",{}).then(function(){setAuth(genTkn||_authToken);window.location.reload()}).catch(function(x){setE(x.message)})}
  function disableAuth(){setE("");jpost("/v1/admin/auth/disable",{}).then(function(){window.location.reload()}).catch(function(x){setE(x.message)})}
  function exportWs(f){window.open("/api/v1/export/workspace?format="+(f||"json"))}
  var adminHere=(!me)||me.instance_admin||(me.roles&&me.roles[selWs||"default"]==="admin");
  var availWs=(function(){var s={"default":1};(wsList||[]).forEach(function(w){s[w]=1});((me&&me.workspaces)||[]).forEach(function(w){s[w]=1});if(selWs)s[selWs]=1;return Object.keys(s).sort()})();
  var wsSelect=h("select",{className:"input",value:selWs||"default",onChange:function(ev){setSelWs(ev.target.value)},style:{width:130}},availWs.map(function(w){return h("option",{key:w,value:w},w)}));
  return h("div",null,h("h2",null,"System"),h(Err,{msg:err}),
    h(SubTabs,{tabs:["workspace","workspaces","auth","activity","queries"],active:tab,onChange:function(t){setSub(t);if(t==="workspaces"&&!wsList.length)loadWorkspaces();if(t==="activity"){loadActivity();loadComplex()}if(t==="queries"&&!queries.length)loadQueries()}}),
    tab==="workspace"&&wksp&&h(Card,{title:"Workspace - "+(wksp.name||"default")},
      h("div",{className:"grid-3"},h(Stat,{value:wksp.n_documents||0,label:"Documents"}),h(Stat,{value:wksp.n_queries||0,label:"Queries"}),h(Stat,{value:wksp.n_users||0,label:"Users"})),
      h("div",{className:"input-row",style:{marginTop:12}},h("button",{className:"sm",onClick:function(){exportWs("json")}},"Export JSON"),h("button",{className:"sm",onClick:function(){exportWs("rex")}},"Export .rex"),h("button",{className:"sm",onClick:refreshWs},"Refresh"))),
    tab==="workspaces"&&h(Card,{title:"All Workspaces",actions:h("button",{className:"sm",onClick:loadWorkspaces},"Refresh")},
      wsList.length>0?h(Table,{cols:[{l:"Workspace",r:function(x){return x}},{l:"Status",r:function(){return h(Badge,{type:"ok"},"Active")}}],rows:wsList}):h("p",{className:"muted"},"No workspaces found.")),
    tab==="auth"&&h("div",null,
      h(Card,{title:"Authentication"},
        me&&h("div",{className:"status-row"},h("span",{className:"name"},"You"),h("span",{className:"detail"},(me.user_id||"local")+"  "),h(Badge,{type:me.is_admin?"gold":"neutral"},(me.role||"user")+" @ "+(me.current_workspace||"default")),me.instance_admin?h(Badge,{type:"good"},"instance admin"):null),
        h("div",{className:"status-row"},h("span",{className:"name"},"Status"),h(Badge,{type:_authToken?"ok":"neutral"},_authToken?"Authenticated":"Open access (auth disabled)")),
        h("p",{className:"muted",style:{fontSize:12,marginTop:6}},"Shared workspaces with per-workspace roles. An admin of a workspace manages its members and runs its consequential actions; a user reads and runs build verbs. The 'default' workspace is the instance - its admins enable auth and remove members entirely."),
        h("div",{className:"input-row",style:{marginTop:12}},
          _authToken?h("button",{className:"sm",onClick:disableAuth},"Disable Auth"):
          h("button",{className:"primary sm",onClick:enableAuth,disabled:!members.length},"Enable Auth"),
          !_authToken&&!members.length&&h("span",{className:"muted",style:{fontSize:12}},"Add a member first"))),
      h(Card,{title:"Members",actions:h("div",{className:"input-row",style:{margin:0,gap:6}},h("label",{className:"muted",style:{fontSize:12}},"Workspace"),wsSelect,h("button",{className:"sm",onClick:function(){refreshMembers()}},"Refresh"))},
        adminHere&&h("div",{style:{marginBottom:10}},
          h("div",{className:"input-row"},
            h("input",{className:"input",value:newLabel,onChange:function(ev){setNl(ev.target.value)},placeholder:"user id, e.g. alice",style:{flex:1}}),
            h("select",{className:"input",value:newRole,onChange:function(ev){setNr(ev.target.value)},style:{width:100}},h("option",{value:"user"},"user"),h("option",{value:"admin"},"admin")),
            h("button",{className:"primary sm",onClick:addMember},"Add / Update")),
          h("p",{className:"muted",style:{fontSize:11,marginTop:4}},"Grants the role in workspace '"+(selWs||"default")+"'. A new member gets a token (shown once); an existing member's role updates in place (their token is kept)."),
          genTkn&&h("div",{style:{marginTop:8,padding:12,background:"var(--gold-bg)",border:"1px solid var(--gold-border)",borderRadius:"var(--radius-sm)"}},
            h("p",{style:{fontSize:12,fontWeight:600,color:"var(--gold)",marginBottom:6}},"Give this token to the member. Shown once."),
            h("input",{className:"input",value:genTkn,readOnly:true,onClick:function(ev){ev.target.select()},style:{fontFamily:"var(--mono)",fontSize:12}}),
            h("div",{className:"input-row",style:{marginTop:6}},
              h("button",{className:"primary sm",onClick:function(){navigator.clipboard.writeText(genTkn).then(function(){})}},"Copy Token"),
              h("button",{className:"sm",onClick:function(){setGt("")}},"Done")))),
        members.length?h(Table,{cols:[
          {l:"User",k:"user_id"},
          {l:"Role in "+(selWs||"default"),r:function(x){return h(Badge,{type:x.role==="admin"?"gold":"neutral"},x.role)}},
          {l:"Since",r:function(x){return x.created?new Date(x.created*1000).toLocaleDateString():""}},
          {l:"",r:function(x){return adminHere?h("button",{className:"sm",onClick:function(){revokeMember(x.user_id)}},"Revoke"):null}}
        ],rows:members}):h("p",{className:"muted"},adminHere?("No members in '"+(selWs||"default")+"' yet."):("You are not an admin of '"+(selWs||"default")+"'.")))),
    tab==="activity"&&h("div",null,
      h(Card,{title:"Activity Summary",actions:h("button",{className:"sm",onClick:loadActivity},"Refresh")},
        activity?h("div",null,
          h("div",{className:"grid-4"},h(Stat,{value:activity.n_users||0,label:"Users"}),h(Stat,{value:activity.n_documents||0,label:"Documents"}),h(Stat,{value:activity.n_queries||0,label:"Queries"}),h(Stat,{value:activity.n_events||0,label:"Events"})),
          activity.users&&activity.users.length>0&&h("p",{className:"detail",style:{marginTop:8}},"Active users: "+activity.users.join(", "))
        ):h("p",{className:"muted"},"Loading activity…")),
      h(Card,{title:"Activity Complex",actions:h("button",{className:"sm",onClick:loadComplex},"Build")},
        actComplex&&actComplex.status?h("p",{className:"muted"},actComplex.status):
        actComplex?h("div",null,
          h("div",{className:"grid-4"},h(Stat,{value:actComplex.nV||0,label:"Vertices"}),h(Stat,{value:actComplex.nE||0,label:"Edges"}),h(Stat,{value:actComplex.n_users||0,label:"Users"}),h(Stat,{value:fmt(actComplex.kappa,3),label:"κ"})),
          actComplex.betti&&h("p",{className:"detail"},"Betti: ("+actComplex.betti.join(", ")+")"),
          actComplex.chi_mean&&h("div",{style:{marginTop:8}},h(CBar,{T:actComplex.chi_mean.T||0,G:actComplex.chi_mean.G||0,F:actComplex.chi_mean.F||0,C:actComplex.chi_mean.C||0}),
            actComplex.dominant&&h("p",{className:"detail"},"Dominant channel: "+actComplex.dominant)),
          actComplex.labels&&h("p",{className:"detail",style:{marginTop:4}},"Entities: "+actComplex.labels.join(", ")),
          h(XBar,{data:actComplex,name:"activity-complex"})
        ):h("p",{className:"muted"},"Build a relational complex from workspace activity.")),
      h(Card,{title:"Query Overlap Detection",actions:h("button",{className:"sm",onClick:refreshOverlaps},"Check")},
        overlaps.length>0?h(Table,{cols:[{l:"User A",k:"user_a"},{l:"User B",k:"user_b"},{l:"Shared",r:function(x){return(x.shared_terms||[]).join(", ")}},{l:"Suggestion",k:"suggestion"}],rows:overlaps}):
          h("p",{className:"muted"},"Detects overlapping queries across users."))),
    tab==="queries"&&h(Card,{title:"Query History",actions:h("button",{className:"sm",onClick:loadQueries},"Refresh")},
      queries.length>0?h("div",null,
        h(Table,{cols:[{l:"User",k:"user"},{l:"Query",r:function(x){return x.query?x.query.slice(0,60)+(x.query.length>60?"…":""):"-"}},{l:"Mode",k:"mode"},{l:"Time",r:function(x){return x.timestamp?new Date(x.timestamp*1000).toLocaleString():""}}],rows:queries}),
        h(XBar,{data:queries,name:"query-history"})
      ):h("p",{className:"muted"},"No queries recorded yet.")))}

// ══════════════════════════════════════════
// LOGIN SCREEN
// ══════════════════════════════════════════
function Login(p){
  var tk=useState(""),token=tk[0],setTk=tk[1];
  var e=useState(""),err=e[0],setE=e[1];
  var rc=useState(false),showRecover=rc[0],setRc=rc[1];
  var rk=useState(""),recKey=rk[0],setRkv=rk[1];
  var rl=useState(false),recovering=rl[0],setRl=rl[1];
  var sv=useState(false),showVal=sv[0],setSv=sv[1];
  function submit(){if(!token.trim())return;setE("");
    _authToken=token.trim();
    // validate against whoami: any valid member (admin OR user) gets 200; only a bad token 401s
    fetch("/api/v1/admin/whoami",{headers:{"Authorization":"Bearer "+token.trim()}})
      .then(function(r){if(r.status===401){setE("Invalid token.");_authToken="";return}
        setAuth(token.trim());if(p.onLogin)p.onLogin()})
      .catch(function(x){setE(x.message);_authToken=""})}
  function recover(){if(!recKey.trim())return;setRl(true);setE("");
    fetch("/api/v1/admin/recover",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({recovery_key:recKey.trim()})})
      .then(function(r){return r.json().then(function(d){if(!r.ok)throw new Error(d.detail||d.error||"Recovery failed");return d})})
      .then(function(d){setAuth(d.token);if(p.onLogin)p.onLogin()})
      .catch(function(x){setE(x.message)})
      .finally(function(){setRl(false)})}
  function pwInput(val,onChange,onKey,placeholder){
    return h("div",{style:{position:"relative",marginBottom:8}},
      h("input",{className:"input",type:showVal?"text":"password",value:val,onChange:onChange,
        onKeyDown:onKey,placeholder:placeholder,
        style:{fontFamily:"var(--mono)",fontSize:12,paddingRight:48,width:"100%",boxSizing:"border-box"}}),
      h("button",{onClick:function(){setSv(!showVal)},
        style:{position:"absolute",right:6,top:"50%",transform:"translateY(-50%)",background:"none",border:"none",color:"var(--fg3)",fontSize:11,cursor:"pointer",padding:"2px 6px"}},showVal?"Hide":"Show"))}
  return h("div",{style:{display:"flex",alignItems:"center",justifyContent:"center",minHeight:"100vh",background:"var(--bg2)"}},
    h("div",{style:{width:400,maxWidth:"92vw"}},
      h("div",{style:{textAlign:"center",marginBottom:24}},
        h("div",{style:{display:"inline-block",width:10,height:10,borderRadius:"50%",background:"var(--gold)",boxShadow:"0 0 0 3px rgba(139,105,20,.15)",marginBottom:12}}),
        h("h1",{style:{fontSize:24,fontWeight:700}},"rexgraph"),
        h("p",{className:"muted",style:{marginTop:4}},showRecover?"Use recovery key":"Paste your API token to sign in")),
      h(Err,{msg:err}),
      !showRecover&&h(Card,{title:"Sign In"},
        pwInput(token,function(ev){setTk(ev.target.value)},function(ev){if(ev.key==="Enter")submit()},"Paste API token"),
        h("button",{className:"primary",onClick:submit,style:{width:"100%"}},"Sign In")),
      showRecover&&h(Card,{title:"Recover Access"},
        h("p",{style:{fontSize:13,lineHeight:1.6,marginBottom:10}},"Enter your recovery key to generate a new token and sign in."),
        pwInput(recKey,function(ev){setRkv(ev.target.value)},function(ev){if(ev.key==="Enter")recover()},"rk_xxxxxxxxxxxxxxxxxxxx"),
        h("button",{className:"primary",onClick:recover,disabled:recovering,style:{width:"100%"}},recovering?"Recovering…":"Recover access")),
      h("div",{style:{textAlign:"center",marginTop:12}},
        h("button",{onClick:function(){setRc(!showRecover);setE("")},style:{background:"none",border:"none",color:"var(--fg2)",fontSize:12,cursor:"pointer",textDecoration:"underline"}},showRecover?"Back to sign in":"Lost your token?"))))}

// ══════════════════════════════════════════
// FIRST-RUN SETUP
// ══════════════════════════════════════════
function Setup(p){
  var ph=useState("welcome"),phase=ph[0],setPh=ph[1];
  var tk=useState(""),token=tk[0],setTk=tk[1];
  var rk=useState(""),recKey=rk[0],setRk=rk[1];
  var cf=useState(""),confirm_=cf[0],setCf=cf[1];
  var e=useState(""),err=e[0],setE=e[1];
  var c1=useState(false),copiedTk=c1[0],setCt=c1[1];
  var c2=useState(false),copiedRk=c2[0],setCr=c2[1];
  function createCredentials(){setE("");setPh("creating");
    jpost("/v1/admin/token",{user_id:"admin",workspaces:["default"],role:"admin"})
      .then(function(r){setTk(r.token);setAuth(r.token);sessionStorage.setItem("rexgraph_setup_done","1");
        return jpost("/v1/admin/recovery-key",{})})
      .then(function(r){setRk(r.recovery_key);setPh("secure")})
      .catch(function(x){setE(x.message);setPh("welcome")})}
  function activate(){setE("");
    jpost("/v1/admin/auth/enable",{})
      .then(function(){setAuth(token);sessionStorage.setItem("rexgraph_setup_done","1");if(p.onDone)p.onDone()})
      .catch(function(x){setE(x.message)})}
  function skip(){sessionStorage.setItem("rexgraph_setup_done","1");if(p.onDone)p.onDone()}
  var matched=confirm_.trim()===token.trim()&&token.length>0;
  function copyTk(){navigator.clipboard.writeText(token).then(function(){setCt(true)})}
  function copyRk(){navigator.clipboard.writeText(recKey).then(function(){setCr(true)})}
  return h("div",{style:{display:"flex",alignItems:"center",justifyContent:"center",minHeight:"100vh",background:"var(--bg2)"}},
    h("div",{style:{width:480,maxWidth:"92vw"}},
      h("div",{style:{textAlign:"center",marginBottom:24}},
        h("div",{style:{display:"inline-block",width:12,height:12,borderRadius:"50%",background:"var(--gold)",boxShadow:"0 0 0 3px rgba(139,105,20,.15)",marginBottom:12}}),
        h("h1",{style:{fontSize:26,fontWeight:700}},"rexgraph"),
        h("p",{className:"muted",style:{marginTop:4}},"Relational complex analysis framework")),
      h(Err,{msg:err}),
      phase==="welcome"&&h(Card,{title:"Get Started"},
        h("p",{style:{fontSize:13,lineHeight:1.6,marginBottom:16}},"Authentication is optional. When enabled, API tokens protect workspace access and a recovery key is generated for account recovery."),
        h("div",{style:{display:"flex",flexDirection:"column",gap:8}},
          h("button",{className:"primary",onClick:createCredentials,style:{padding:"12px 16px"}},"Set up authentication"),
          h("button",{onClick:skip,style:{padding:"12px 16px"}},"Continue without auth"))),
      phase==="creating"&&h(Card,{title:"Creating Credentials"},
        h("p",{className:"muted"},"Generating credentials…")),
      phase==="secure"&&h("div",null,
        h(Card,{title:"Step 1 - Save your API token"},
          h("p",{style:{fontSize:13,lineHeight:1.6,marginBottom:10}},"Your admin API token. Copy it now. Shown once."),
          h("div",{style:{padding:12,background:"var(--gold-bg)",border:"1px solid var(--gold-border)",borderRadius:"var(--radius-sm)"}},
            h("input",{className:"input",value:token,readOnly:true,onClick:function(ev){ev.target.select()},style:{fontFamily:"var(--mono)",fontSize:12,marginBottom:8}}),
            h("button",{className:"primary sm",onClick:copyTk},copiedTk?"Copied ✓":"Copy API token"))),
        h(Card,{title:"Step 2 - Save your recovery key"},
          h("p",{style:{fontSize:13,lineHeight:1.6,marginBottom:10}},"Your recovery key for account access. Store it offline. Shown once."),
          h("div",{style:{padding:12,background:"var(--bg3)",border:"1px solid var(--border)",borderRadius:"var(--radius-sm)"}},
            h("input",{className:"input",value:recKey,readOnly:true,onClick:function(ev){ev.target.select()},style:{fontFamily:"var(--mono)",fontSize:12,marginBottom:8}}),
            h("button",{className:"sm",onClick:copyRk},copiedRk?"Copied ✓":"Copy recovery key"))),
        h(Card,{title:"Step 3 - Confirm your API token"},
          h("p",{style:{fontSize:13,lineHeight:1.6,marginBottom:10}},"Paste your API token to confirm."),
          h("input",{className:"input",value:confirm_,onChange:function(ev){setCf(ev.target.value)},placeholder:"Paste your API token here…",style:{fontFamily:"var(--mono)",fontSize:12,marginBottom:12}}),
          matched&&h("p",{style:{fontSize:13,color:"var(--gold)",fontWeight:600,marginBottom:8}},"Token verified ✓"),
          h("div",{style:{display:"flex",gap:8}},
            h("button",{className:"primary",onClick:activate,disabled:!matched,style:{flex:1,padding:"12px 16px"}},matched?"Enable auth & sign in":"Paste token to continue"),
            h("button",{onClick:skip,style:{padding:"12px 16px"}},"Skip"))))))}

// ══════════════════════════════════════════
// APP SHELL
// ══════════════════════════════════════════
function Database(){
  var e=useState(""),err=e[0],setE=e[1],inf=useState(null),info=inf[0],setInf=inf[1];
  var rc=useState([]),recs=rc[0],setRecs=rc[1];
  var ss=useState([]),sessions=ss[0],setSs=ss[1],cs=useState(""),curSes=cs[0],setCs=cs[1];
  var tg=useState(""),tags=tg[0],setTg=tg[1];
  var qb=useState(""),qBetti=qb[0],setQb=qb[1],qk=useState(""),qKappa=qk[0],setQk=qk[1],qt=useState(""),qTags=qt[0],setQt=qt[1];
  function loadInfo(){api("/v1/db/info").then(setInf).catch(function(x){setE(x.message)})}
  function loadList(){api("/v1/db/list").then(function(r){setRecs(r.records||[])}).catch(function(x){setE(x.message)})}
  function loadSessions(){api("/sessions").then(function(r){setSs(r||[]);if(!curSes&&r&&r.length)setCs(r[0].session_id)}).catch(function(){})}
  useEffect(function(){loadInfo();loadList();loadSessions()},[]);
  function store(){if(!curSes)return;jpost("/v1/db/put",{session_id:curSes,tags:tags.split(",").map(function(t){return t.trim()}).filter(Boolean)}).then(function(){setTg("");loadInfo();loadList()}).catch(function(x){setE(x.message)})}
  function query(){var q={};if(qBetti)q.min_betti1=parseInt(qBetti);if(qKappa)q.min_kappa=parseFloat(qKappa);if(qTags)q.tags_any=qTags.split(",").map(function(t){return t.trim()}).filter(Boolean);jpost("/v1/db/query",q).then(function(r){setRecs(r.records||[])}).catch(function(x){setE(x.message)})}
  function del(id){api("/v1/db/"+id,{method:"DELETE"}).then(function(){loadInfo();loadList()}).catch(function(x){setE(x.message)})}
  function exportRec(id){window.open("/api/v1/db/export/"+id)}
  var sm=useState(null),similar=sm[0],setSm=sm[1];
  function findSimilar(id){setE("");setSm(null);jpost("/v1/db/similar",{id:id,top_k:8}).then(function(r){setSm({source:id,matches:r.matches||[]})}).catch(function(x){setE(x.message)})}
  var cl=useState(null),families=cl[0],setCl=cl[1];
  function findFamilies(){setE("");setCl(null);jpost("/v1/db/cluster",{threshold:0.7}).then(setCl).catch(function(x){setE(x.message)})}
  var lg=useState(null),lin=lg[0],setLg=lg[1];
  function loadLineage(id){setE("");setLg(null);api("/v1/db/lineage/"+encodeURIComponent(id)).then(setLg).catch(function(x){setE(x.message)})}
  return h("div",null,h("h2",null,"RCDB Overview"),h(Err,{msg:err}),
    h("p",{className:"muted",style:{marginBottom:10,fontSize:13}},"Every record is a relational complex - documents, schemas, ontologies - queryable by structure (cycles β₁, coherence κ, tags), not just by id. Backend is set per-deployment (SQLite, Postgres, file, …)."),
    info&&h("div",{className:"grid-4",style:{marginBottom:12}},h(Stat,{value:info.backend,label:"Backend"}),h(Stat,{value:info.count,label:"Complexes"}),h(Stat,{value:info.total_edges,label:"Total edges"}),h(Stat,{value:info.mean_kappa!=null?fmt(info.mean_kappa,3):"-",label:"Mean κ"})),
    info&&(info.by_tag||info.by_source)&&h("div",{className:"grid-2",style:{marginBottom:12}},
      info.by_tag&&h(Card,{title:"By tag"},Object.keys(info.by_tag).length?Object.keys(info.by_tag).map(function(k){return h("div",{key:k,className:"status-row"},h("span",{className:"name"},k),h(Badge,{type:k==="schema"?"gold":"neutral"},info.by_tag[k]))}):h("p",{className:"muted"},"No records yet")),
      info.by_source&&h(Card,{title:"By source"},Object.keys(info.by_source).map(function(k){return h("div",{key:k,className:"status-row"},h("span",{className:"name"},k||"-"),h("span",{className:"detail"},info.by_source[k]))}))),
    h(Card,{title:"Store current analysis"},
      h("div",{className:"input-row"},
        h("select",{className:"input",style:{width:220},value:curSes,onChange:function(ev){setCs(ev.target.value)}},h("option",{value:""},"pick a session…"),sessions.map(function(s){return h("option",{key:s.session_id,value:s.session_id},s.session_id)})),
        h("input",{className:"input",value:tags,onChange:function(ev){setTg(ev.target.value)},placeholder:"tags (comma-separated)"}),
        h("button",{className:"primary sm",onClick:store,disabled:!curSes},"Store"),
        h("button",{className:"sm",onClick:loadSessions},"↻"))),
    h(Card,{title:"Structural query"},
      h("div",{className:"input-row",style:{flexWrap:"wrap",gap:6}},
        h("input",{className:"input",style:{width:110},value:qBetti,onChange:function(ev){setQb(ev.target.value)},placeholder:"min β₁"}),
        h("input",{className:"input",style:{width:110},value:qKappa,onChange:function(ev){setQk(ev.target.value)},placeholder:"min κ"}),
        h("input",{className:"input",style:{width:160},value:qTags,onChange:function(ev){setQt(ev.target.value)},placeholder:"tags (any)"}),
        h("button",{className:"primary sm",onClick:query},"Query"),
        h("button",{className:"sm",onClick:loadList},"Show all"))),
    h("div",{className:"input-row",style:{marginBottom:10,gap:6}},h("button",{className:"sm",onClick:findFamilies},"⧉ Structural families"),h("input",{className:"input",style:{width:160},placeholder:"lineage id (e.g. shop)",onKeyDown:function(ev){if(ev.key==="Enter"&&ev.target.value.trim())loadLineage(ev.target.value.trim())}}),h("span",{className:"muted",style:{fontSize:11,alignSelf:"center"}},"↵ to view lineage")),
    families&&h(Card,{title:"Structural families ("+(families.clusters||[]).length+")",actions:h("button",{className:"sm",onClick:function(){setCl(null)}},"Close")},
      (families.clusters||[]).length?families.clusters.map(function(cl,i){return h("div",{key:i,className:"status-row"},h("span",{className:"name"},cl.members.join(" · ")),h("div",{style:{display:"flex",gap:6,alignItems:"center"}},h(Badge,{type:cl.avg_coherence>0.75?"good":"gold"},Math.round(cl.avg_coherence*100)+"% coherent"),h("span",{className:"detail",style:{fontSize:11}},"core: "+cl.centroid)))}):h("p",{className:"muted"},"No structural families (need ≥2 similar complexes)"),
      families.singletons&&families.singletons.length&&h("p",{className:"muted",style:{fontSize:11,marginTop:6}},"Singletons: "+families.singletons.join(", "))),
    lin&&h(Card,{title:"Lineage - "+lin.lineage_id,actions:h("button",{className:"sm",onClick:function(){setLg(null)}},"Close")},
      !lin.versions||!lin.versions.length?h("p",{className:"muted"},"No versions for this lineage id."):
      h("div",null,lin.versions.map(function(v,i){var step=(lin.trajectory||[])[i-1];return h("div",{key:v.id},step&&h("div",{style:{fontSize:11,color:"var(--muted)",margin:"2px 0 2px 12px"}},"↓ ",Math.round(step.match*100),"% match"+(step.added.length?" · +"+step.added.join(","):"")+(step.removed.length?" · -"+step.removed.join(","):"")),h("div",{className:"status-row"},h("span",{className:"name"},v.id),h("span",{className:"detail"},"v"+v.version)))}))),
    similar&&h(Card,{title:"Structurally similar to "+similar.source,actions:h("button",{className:"sm",onClick:function(){setSm(null)}},"Close")},
      similar.matches.length?h(Table,{cols:[{l:"Complex",k:"id"},{l:"Match",r:function(x){return h(Badge,{type:x.match>0.75?"good":x.match>0.5?"gold":"neutral"},Math.round(x.match*100)+"%")}},{l:"Shared",r:function(x){return x.shared+" concepts"}},{l:"Tags",r:function(x){return (x.tags||[]).join(", ")}}],rows:similar.matches}):h("p",{className:"muted"},"No structurally similar complexes found.")),
    h(Card,{title:"Complexes ("+recs.length+")"},
      h(Table,{cols:[{l:"ID",k:"id"},{l:"V",r:function(x){return x.signature.nV}},{l:"E",r:function(x){return x.signature.nE}},{l:"β",r:function(x){return x.signature.betti?JSON.stringify(x.signature.betti):"-"}},{l:"κ",r:function(x){return x.signature.kappa_mean!=null?fmt(x.signature.kappa_mean,3):"-"}},{l:"Tags",r:function(x){return (x.signature.tags||[]).join(", ")}},{l:"",r:function(x){return h("div",{style:{display:"flex",gap:4}},h("button",{className:"sm",title:"Find structurally similar",onClick:function(){findSimilar(x.id)}},"≈"),h("button",{className:"sm",onClick:function(){exportRec(x.id)}},"⤓"),h("button",{className:"sm danger",onClick:function(){del(x.id)}},"×"))}}],rows:recs})))}

function Schema(){
  var e=useState(""),err=e[0],setE=e[1],rs=useState(null),res=rs[0],setRs=rs[1];
  var fm=useState("ddl"),fmt=fm[0],setFmt=fm[1];
  var tx=useState("CREATE TABLE customers (id INT PRIMARY KEY, fav_order INT REFERENCES orders(id));\nCREATE TABLE orders (id INT PRIMARY KEY, customer_id INT REFERENCES customers(id));\nCREATE TABLE line_items (id INT PRIMARY KEY, order_id INT REFERENCES orders(id));"),txt=tx[0],setTx=tx[1];
  var st=useState(""),storeId=st[0],setSt=st[1];
  var dl=useState(""),dialect=dl[0],setDl=dl[1];
  var sr=useState(null),strain=sr[0],setSr=sr[1];
  function schemaBody(){var body={};if(fmt==="ddl"){body.ddl=txt;if(dialect)body.dialect=dialect;}else if(fmt==="connection")body.connection=txt.trim();else if(fmt==="mongo"){try{body.mongo=JSON.parse(txt)}catch(x){setE("Invalid Mongo JSON");return null}}else{try{body.spec=JSON.parse(txt)}catch(x){setE("Invalid JSON spec");return null}}return body}
  function analyze(){setE("");setRs(null);setSr(null);var body=schemaBody();if(!body)return;if(storeId.trim())body.store_id=storeId.trim();jpost("/v1/schema/analyze",body).then(setRs).catch(function(x){setE(x.message)})}
  function computeStrain(){setE("");setSr(null);var body=schemaBody();if(!body)return;jpost("/v1/schema/strain",body).then(setSr).catch(function(x){setE(x.message)})}
  var li=useState(null),lint=li[0],setLi=li[1];
  function computeLint(){setE("");setLi(null);var body=schemaBody();if(!body)return;jpost("/v1/schema/lint",body).then(setLi).catch(function(x){setE(x.message)})}
  var SEVCOLOR={high:"fail",medium:"gold",low:"neutral",info:"neutral"};
  return h("div",null,h("h2",null,"Schema & Ontology Diagnosis"),h(Err,{msg:err}),
    h("p",{className:"muted",style:{marginBottom:10,fontSize:13}},"A schema is a relational complex: tables are cells, foreign keys are typed directional relations. Hodge + void analysis reveal the schema's ",h("em",null,"actual")," topology - circular dependencies, structural tension, and implied-missing relations."),
    h(Card,{title:"Input"},
      h("div",{className:"input-row",style:{marginBottom:6}},h("label",{className:"muted",style:{width:60}},"Format"),h("select",{className:"input",style:{width:220},value:fmt,onChange:function(ev){setFmt(ev.target.value)}},h("option",{value:"ddl"},"SQL DDL (CREATE TABLE…)"),h("option",{value:"json"},"JSON spec"),h("option",{value:"mongo"},"MongoDB (sample docs)"),h("option",{value:"connection"},"Live DB (SQLAlchemy URL)")),
        fmt==="ddl"&&h("select",{className:"input",style:{width:130},value:dialect,onChange:function(ev){setDl(ev.target.value)}},h("option",{value:""},"auto dialect"),["postgres","mysql","oracle","tsql","sqlite","snowflake","bigquery"].map(function(d){return h("option",{key:d,value:d},d)}))),
      h("textarea",{className:"input",style:{width:"100%",height:140,fontFamily:"monospace",fontSize:12},value:txt,onChange:function(ev){setTx(ev.target.value)}}),
      h("div",{className:"input-row",style:{marginTop:8}},h("button",{className:"primary",onClick:analyze},"Diagnose Schema"),h("button",{className:"sm",onClick:computeStrain,title:"Weight by real data magnitudes (needs a live connection) and measure where the data strains the design"},"⚡ Data strain"),h("button",{className:"sm",onClick:computeLint,title:"Character-based lint: label each relation, flag anomalies, surface conflict tables"},"◇ Lint relations"),h("input",{className:"input",style:{width:180},value:storeId,onChange:function(ev){setSt(ev.target.value)},placeholder:"store in RCDB as… (optional)"}))),
    lint&&h(Card,{title:"Relation lint",actions:h("button",{className:"sm",onClick:function(){setLi(null)}},"Close")},
      h(Table,{cols:[{l:"Relation",k:"relation"},{l:"Character",r:function(x){return h(Badge,{type:x.character==="conflicting"?"gold":"neutral"},x.character)}},{l:"Modality",r:function(x){return h("span",{className:"detail",style:{fontSize:11}},x.modality)}},{l:"",r:function(x){return x.anomaly?h(Badge,{type:"fail"},"anomaly"):""}}],rows:lint.relations}),
      lint.conflict_tables&&lint.conflict_tables.length&&h("p",{style:{marginTop:8,fontSize:13}},"Tables pulled in conflicting directions: "+lint.conflict_tables.slice(0,4).map(function(t){return t.table+" ("+fmt(t.frustration,2)+")"}).join(", "))),
    strain&&h(Card,{title:"Data-forced strain",actions:h("button",{className:"sm",onClick:function(){setSr(null)}},"Close")},
      !strain.has_geometry?h("div",null,h("p",{className:"muted"},"No co-participation cycles, so no geometric strain (curvature). But the fan-out load below shows where the data pressures the design."),strain.relation_load&&strain.relation_load.length&&h("div",null,h("div",{className:"stat-label",style:{marginBottom:4}},"Data pressure - fan-out per relation"),strain.relation_load.slice(0,6).map(function(x,i){var mx=strain.relation_load[0].load||1;return h("div",{key:i,className:"status-row"},h("span",{className:"name"},x.relation),h("div",{style:{flex:1,margin:"0 8px",height:8,background:"var(--line)",borderRadius:4,overflow:"hidden"}},h("div",{style:{width:(100*Math.min(x.load/mx,1))+"%",height:"100%",background:x.load/mx>0.66?"var(--fail)":x.load/mx>0.33?"var(--gold)":"var(--accent)"}})),h("span",{className:"detail"},fmt(x.load,1)+"×"))}))):
      strain.total_strain===0?h("p",{className:"muted"},"No strain: with uniform data magnitudes the design and the data agree (flat). Provide a live connection to weight by real cardinality."):
      h("div",null,
        strain.lagrangian_curvature&&strain.lagrangian_curvature.curvature!=null&&h("div",{style:{marginBottom:8,padding:"8px 10px",background:"var(--surface2)",borderRadius:6}},h("span",{style:{fontSize:13}},"Overall imbalance (Lagrangian curvature): "),h("strong",{style:{color:strain.lagrangian_curvature.curvature>3?"var(--fail)":strain.lagrangian_curvature.curvature>1?"var(--gold)":"var(--good)"}},fmt(strain.lagrangian_curvature.curvature,2)),h("span",{className:"muted",style:{fontSize:11}}," - how far the data pushes the schema from topology/geometry balance")),
        h("div",{style:{marginBottom:8}},"Total data-forced strain: ",h("strong",null,fmt(strain.total_strain,1)),strain.effective_root_causes!=null&&h("span",null," · really ",h("strong",null,strain.effective_root_causes),strain.effective_root_causes<1.6?" root cause":" root causes",strain.coupled_relations&&strain.coupled_relations.length?" (some relations are coupled - fixing one helps the other)":"")),
        strain.per_join&&strain.per_join.length&&h("div",{style:{marginBottom:8}},h("div",{className:"stat-label",style:{marginBottom:4}},"Where - strain by join (heat map)"),strain.per_join.slice(0,6).map(function(j,i){var mx=strain.per_join[0].strain||1;return h("div",{key:i,className:"status-row"},h("span",{className:"name"},j.tables.join(" · ")),h("div",{style:{flex:1,margin:"0 8px",height:8,background:"var(--line)",borderRadius:4,overflow:"hidden"}},h("div",{style:{width:(100*j.strain/mx)+"%",height:"100%",background:j.strain/mx>0.66?"var(--fail)":j.strain/mx>0.33?"var(--gold)":"var(--accent)"}})),h("span",{className:"detail"},fmt(j.strain,1)))})),
        strain.table_strain&&strain.table_strain.length&&h("div",{style:{marginTop:8}},h("div",{className:"stat-label",style:{marginBottom:4}},"Which table is the hotspot (star curvature)"),strain.table_strain.slice(0,5).map(function(t,i){var mx=strain.table_strain[0].strain||1;return h("div",{key:i,className:"status-row"},h("span",{className:"name"},t.table),h("div",{style:{flex:1,margin:"0 8px",height:8,background:"var(--line)",borderRadius:4,overflow:"hidden"}},h("div",{style:{width:(100*t.strain/mx)+"%",height:"100%",background:t.strain/mx>0.66?"var(--fail)":t.strain/mx>0.33?"var(--gold)":"var(--accent)"}})),h("span",{className:"detail"},fmt(t.strain,1)))})),
      strain.per_relation&&strain.per_relation.length&&h("div",null,h("div",{className:"stat-label",style:{marginBottom:4}},"Who - fix these relations first"),h(Table,{cols:[{l:"Relation",k:"relation"},{l:"Strain contribution",r:function(x){return h("strong",null,fmt(x.contribution,2))}}],rows:strain.per_relation.slice(0,8)})))),
    res&&h("div",null,
      h("div",{className:"grid-4",style:{marginBottom:10}},
        h(Stat,{value:res.n_tables,label:"Tables"}),h(Stat,{value:res.n_foreign_keys,label:"Foreign keys"}),
        h(Stat,{value:res.ontology_validity!=null?pct(res.ontology_validity):"-",label:"DAG validity"}),
        h("div",{className:"stat"},h("div",{style:{marginBottom:4}},h(Badge,{type:res.verdict==="acyclic"?"good":res.verdict==="cycles_present"?"fail":"gold"},res.verdict)),h("div",{className:"stat-label"},"Verdict"))),
      res.hodge&&h(Card,{title:"Topology (Hodge decomposition of the FK graph)"},
        h("div",{className:"grid-3"},h(Stat,{value:pct(res.hodge.hierarchy_gradient),label:"Hierarchy (gradient)"}),h(Stat,{value:pct(res.hodge.bounded_recursion_curl),label:"Bounded recursion (curl)"}),h(Stat,{value:pct(res.hodge.persistent_circulation_harmonic),label:"Persistent cycle (harmonic)"})),
        res.summary&&h("p",{style:{marginTop:8,fontSize:13}},res.summary)),
      res.order_of_operations&&res.order_of_operations.length>0&&h(Card,{title:"Valid order of operations"},
        h("p",{style:{fontSize:13,fontFamily:"monospace",wordBreak:"break-word"}},res.order_of_operations.join("  ->  ")),
        res.relations_to_cut&&h("p",{style:{marginTop:8,fontSize:13,color:"var(--fail)"}},"Cut these relations to reach a valid DAG: ",h("strong",null,res.relations_to_cut.join(",  ")))),
      res.migration_plan&&h(Card,{title:"Migration plan (deployable)",actions:h(XBar,{data:res.migration_plan,name:"migration-plan"})},
        h("p",{style:{fontSize:12,marginBottom:6}},res.migration_plan.note),
        h("p",{style:{fontSize:12}},h("strong",null,"1. Create in order: "),h("span",{style:{fontFamily:"monospace"}},(res.migration_plan.create_order||[]).join(" -> "))),
        res.migration_plan.post_create_ddl&&res.migration_plan.post_create_ddl.length>0&&h("div",{style:{marginTop:6}},h("p",{style:{fontSize:12}},h("strong",null,"2. Then add the deferred relations:")),h("pre",{className:"json",style:{fontSize:11,maxHeight:140,overflow:"auto"}},res.migration_plan.post_create_ddl.join("\n")))),
      res.findings&&res.findings.length>0&&h(Card,{title:"Findings ("+res.findings.length+")",actions:h(XBar,{data:res,name:"schema-diagnosis"})},
        res.findings.map(function(f,i){return h("div",{key:i,style:{marginBottom:10,borderLeft:"3px solid var(--"+(SEVCOLOR[f.severity]||"neutral")+")",paddingLeft:10}},
          h("div",null,h(Badge,{type:SEVCOLOR[f.severity]||"neutral"},f.severity)," ",f.type&&h("span",{className:"muted",style:{fontSize:11}},"["+f.type+"] "),h("span",{style:{fontSize:13}},f.issue)),
          f.cycles&&h("ul",{style:{margin:"4px 0",fontSize:12,color:"var(--fail)"}},f.cycles.map(function(c,j){return h("li",{key:j},c)})),
          f.tables&&h("p",{className:"muted",style:{fontSize:12,margin:"2px 0"}},"Tables: "+f.tables.join(", ")))})),
      res.central_tables&&h(Card,{title:"Tables - roles & blast radius"},
        h(Table,{cols:[{l:"Table",k:"table"},{l:"Role",r:function(x){return h(Badge,{type:x.role==="hub"?"gold":"neutral"},x.role)}},{l:"Referenced by",k:"referenced_by"},{l:"References",k:"references"},{l:"Impact",r:function(x){return h("strong",null,x.impact)}}],rows:res.central_tables}))))}
function fmt2(v){return v==null?"-":(typeof v==="number"?v.toFixed(3):v)}

function DBManager(){
  var e=useState(""),err=e[0],setE=e[1];
  var cs=useState([]),conns=cs[0],setCs=cs[1];
  var nm=useState(""),cName=nm[0],setNm=nm[1],ur=useState(""),cUri=ur[0],setUr=ur[1],kd=useState("sql"),cKind=kd[0],setKd=kd[1];
  var tb=useState(null),tables=tb[0],setTb=tb[1],ts=useState(null),testR=ts[0],setTs=ts[1];
  var im=useState(null),imp=im[0],setIm=im[1],sel=useState(""),active=sel[0],setSel=sel[1];
  function load(){api("/v1/dbmanager/connections").then(function(r){setCs(r.connections||[])}).catch(function(x){setE(x.message)})}
  useEffect(load,[]);
  function save(){if(!cName.trim()||!cUri.trim())return;jpost("/v1/dbmanager/connections",{name:cName,uri:cUri,kind:cKind}).then(function(){setNm("");setUr("");load()}).catch(function(x){setE(x.message)})}
  function del(n){api("/v1/dbmanager/connections/"+n,{method:"DELETE"}).then(load).catch(function(x){setE(x.message)})}
  function test(n){setTs(null);setE("");jpost("/v1/dbmanager/test",{name:n}).then(function(r){setTs(Object.assign({name:n},r))}).catch(function(x){setE(x.message)})}
  function browse(n){setSel(n);setTb(null);setIm(null);setE("");jpost("/v1/dbmanager/tables",{name:n}).then(function(r){setTb(r.tables||[])}).catch(function(x){setE(x.message)})}
  function importSchema(n){setIm(null);setE("");jpost("/v1/dbmanager/import",{name:n,tags:["prod"]}).then(setIm).catch(function(x){setE(x.message)})}
  var so=useState(null),cstrain=so[0],setSo=so[1];
  function measureStrain(n){setSo(null);setE("");jpost("/v1/dbmanager/strain",{name:n}).then(function(r){setSo(Object.assign({name:n},r))}).catch(function(x){setE(x.message)})}
  return h("div",null,h("h2",null,"Database Manager"),h(Err,{msg:err}),
    h("p",{className:"muted",style:{marginBottom:10,fontSize:13}},"Connect to any SQL database or MongoDB, browse its live schema, and import it into the RCDB as a diagnosed complex. Credentials are stored server-side and masked in responses."),
    h(Card,{title:"Add connection"},
      h("div",{className:"input-row",style:{flexWrap:"wrap",gap:6}},
        h("input",{className:"input",style:{width:130},value:cName,onChange:function(ev){setNm(ev.target.value)},placeholder:"name (prod)"}),
        h("input",{className:"input",style:{flex:1,minWidth:240},value:cUri,onChange:function(ev){setUr(ev.target.value)},placeholder:"postgresql://user:pass@host/db  or  mongodb://…"}),
        h("select",{className:"input",style:{width:100},value:cKind,onChange:function(ev){setKd(ev.target.value)}},h("option",{value:"sql"},"SQL"),h("option",{value:"mongo"},"Mongo")),
        h("button",{className:"primary sm",onClick:save,disabled:!cName.trim()||!cUri.trim()},"Save"))),
    h(Card,{title:"Connections ("+conns.length+")"},
      conns.length?h(Table,{cols:[{l:"Name",k:"name"},{l:"Kind",k:"kind"},{l:"URI (masked)",r:function(x){return h("span",{className:"detail",style:{fontSize:11}},x.uri)}},{l:"",r:function(x){return h("div",{style:{display:"flex",gap:4}},h("button",{className:"sm",onClick:function(){test(x.name)}},"Test"),h("button",{className:"sm",onClick:function(){browse(x.name)}},"Browse"),h("button",{className:"sm",title:"Measure data-forced strain",onClick:function(){measureStrain(x.name)}},"⚡"),h("button",{className:"primary sm",onClick:function(){importSchema(x.name)}},"Import->RCDB"),h("button",{className:"sm danger",onClick:function(){del(x.name)}},"×"))}}],rows:conns}):h("p",{className:"muted"},"No saved connections")),
    cstrain&&h(Card,{title:"Data-forced strain - "+cstrain.name,actions:h("button",{className:"sm",onClick:function(){setSo(null)}},"Close")},
      !cstrain.has_geometry?h("div",null,h("p",{className:"muted"},"No co-participation cycles (curvature 0). Fan-out load shows the data pressure:"),cstrain.relation_load&&cstrain.relation_load.slice(0,6).map(function(x,i){var mx=cstrain.relation_load[0].load||1;return h("div",{key:i,className:"status-row"},h("span",{className:"name"},x.relation),h("div",{style:{flex:1,margin:"0 8px",height:8,background:"var(--line)",borderRadius:4,overflow:"hidden"}},h("div",{style:{width:(100*Math.min(x.load/mx,1))+"%",height:"100%",background:x.load/mx>0.66?"var(--fail)":x.load/mx>0.33?"var(--gold)":"var(--accent)"}})),h("span",{className:"detail"},fmt(x.load,1)+"×"))})):
      h("div",null,
        h("div",{style:{marginBottom:8}},"Total: ",h("strong",null,fmt(cstrain.total_strain,1)),cstrain.effective_root_causes!=null&&h("span",null," · ~",h("strong",null,cstrain.effective_root_causes)," root cause(s)")),
        cstrain.per_join&&cstrain.per_join.length&&h("div",{style:{marginBottom:8}},h("div",{className:"stat-label",style:{marginBottom:4}},"Where - strained joins"),cstrain.per_join.slice(0,5).map(function(j,i){var mx=cstrain.per_join[0].strain||1;return h("div",{key:i,className:"status-row"},h("span",{className:"name"},j.tables.join(" · ")),h("div",{style:{flex:1,margin:"0 8px",height:8,background:"var(--line)",borderRadius:4,overflow:"hidden"}},h("div",{style:{width:(100*j.strain/mx)+"%",height:"100%",background:j.strain/mx>0.66?"var(--fail)":j.strain/mx>0.33?"var(--gold)":"var(--accent)"}})),h("span",{className:"detail"},fmt(j.strain,1)))})),
        cstrain.per_relation&&cstrain.per_relation.length&&h("div",null,h("div",{className:"stat-label",style:{marginBottom:4}},"Who - fix first"),h(Table,{cols:[{l:"Relation",k:"relation"},{l:"Contribution",r:function(x){return fmt(x.contribution,2)}}],rows:cstrain.per_relation.slice(0,6)})))),
    testR&&h("div",{style:{marginBottom:10}},h(Badge,{type:testR.ok?"good":"fail"},testR.ok?"Connection OK - "+testR.name:"Failed - "+(testR.error||""))),
    tables&&h(Card,{title:"Tables in "+active+" ("+tables.length+")",actions:h("div",{style:{display:"flex",gap:4}},h("button",{className:"sm",title:"Measure data-forced strain",onClick:function(){measureStrain(active)}},"⚡ Strain"),h("button",{className:"primary sm",onClick:function(){importSchema(active)}},"Import->RCDB"))},
      h(Table,{cols:[{l:"Table",k:"table"},{l:"Rows",r:function(x){return x.rows!=null?fmt(x.rows,0):"-"}},{l:"Columns",k:"columns"},{l:"FKs",k:"foreign_keys"},{l:"PK",r:function(x){return (x.primary_key||[]).join(", ")||"-"}}],rows:tables})),
    imp&&h(Card,{title:"Imported & diagnosed",actions:h(XBar,{data:imp,name:"import-diag"})},
      h("div",{style:{marginBottom:6}},h(Badge,{type:imp.verdict==="acyclic"?"good":imp.verdict==="cycles_present"?"fail":"gold"},imp.verdict)," stored as ",h("strong",null,imp.stored_as)),
      imp.relations_to_cut&&h("p",{style:{fontSize:13,color:"var(--fail)"}},"Cut to fix: "+imp.relations_to_cut.join(", ")),
      h("p",{className:"muted",style:{fontSize:12}},"Now a queryable RCDB record - see the RCDB Overview.")))}

function SchemaBuilder(){
  var e=useState(""),err=e[0],setE=e[1];
  var tb=useState([{name:"users",pk:"id"},{name:"orders",pk:"id"}]),tables=tb[0],setTb=tb[1];
  var fk=useState([{from:"orders",to:"users",col:"user_id"}]),fks=fk[0],setFk=fk[1];
  var rs=useState(null),diag=rs[0],setRs=rs[1],dd=useState(""),ddl=dd[0],setDd=dd[1];
  function spec(){return {tables:tables.map(function(t){return {name:t.name,primary_key:t.pk?[t.pk]:[],foreign_keys:fks.filter(function(f){return f.from===t.name}).map(function(f){return {columns:[f.col||"ref_id"],references:f.to}})}})}}
  function addTable(){setTb(tables.concat([{name:"table"+(tables.length+1),pk:"id"}]))}
  function rmTable(i){var t=tables.slice();var nm=t[i].name;t.splice(i,1);setTb(t);setFk(fks.filter(function(f){return f.from!==nm&&f.to!==nm}))}
  function setTName(i,v){var t=tables.slice();t[i]=Object.assign({},t[i],{name:v});setTb(t)}
  function addFK(){if(tables.length<2)return;setFk(fks.concat([{from:tables[0].name,to:tables[1].name,col:"ref_id"}]))}
  function setFK(i,k,v){var f=fks.slice();f[i]=Object.assign({},f[i],{[k]:v});setFk(f)}
  function rmFK(i){var f=fks.slice();f.splice(i,1);setFk(f)}
  function diagnose(){setE("");jpost("/v1/schema/analyze",{spec:spec()}).then(setRs).catch(function(x){setE(x.message)})}
  function genDDL(){setE("");jpost("/v1/dbmanager/ddl",{spec:spec()}).then(function(r){setDd(r.ddl)}).catch(function(x){setE(x.message)})}
  function store(){var id=prompt("Store schema in RCDB as:","my-schema");if(id)jpost("/v1/schema/analyze",{spec:spec(),store_id:id,tags:["draft"]}).then(function(){setE("")}).catch(function(x){setE(x.message)})}
  return h("div",null,h("h2",null,"Schema Builder"),h(Err,{msg:err}),
    h("p",{className:"muted",style:{marginBottom:10,fontSize:13}},"Compose a schema and validate its topology live. The goal isn't drag-and-drop wiring - it's building mathematically valid dependencies (a DAG), so the diagnosis updates as you add relations."),
    h("div",{className:"grid-2"},
      h(Card,{title:"Tables",actions:h("button",{className:"sm",onClick:addTable},"+ Table")},
        tables.map(function(t,i){return h("div",{key:i,className:"input-row",style:{marginBottom:4}},h("input",{className:"input",style:{flex:1},value:t.name,onChange:function(ev){setTName(i,ev.target.value)}}),h("button",{className:"sm danger",onClick:function(){rmTable(i)}},"×"))})),
      h(Card,{title:"Foreign keys",actions:h("button",{className:"sm",onClick:addFK},"+ FK")},
        fks.map(function(f,i){return h("div",{key:i,className:"input-row",style:{marginBottom:4,gap:4}},
          h("select",{className:"input",style:{flex:1},value:f.from,onChange:function(ev){setFK(i,"from",ev.target.value)}},tables.map(function(t){return h("option",{key:t.name,value:t.name},t.name)})),
          h("span",{style:{alignSelf:"center"}},"->"),
          h("select",{className:"input",style:{flex:1},value:f.to,onChange:function(ev){setFK(i,"to",ev.target.value)}},tables.map(function(t){return h("option",{key:t.name,value:t.name},t.name)})),
          h("button",{className:"sm danger",onClick:function(){rmFK(i)}},"×"))}))),
    h("div",{className:"input-row",style:{marginTop:10}},h("button",{className:"primary",onClick:diagnose},"Diagnose"),h("button",{className:"sm",onClick:genDDL},"Generate DDL"),h("button",{className:"sm",onClick:store},"Store in RCDB")),
    diag&&h("div",{style:{marginTop:12}},
      h("div",{style:{marginBottom:8}},h(Badge,{type:diag.verdict==="acyclic"?"good":diag.verdict==="cycles_present"?"fail":"gold"},diag.verdict)," ",diag.summary),
      diag.hodge&&h("div",{className:"grid-3"},h(Stat,{value:pct(diag.hodge.hierarchy_gradient),label:"Hierarchy"}),h(Stat,{value:pct(diag.hodge.bounded_recursion_curl),label:"Bounded (curl)"}),h(Stat,{value:pct(diag.hodge.persistent_circulation_harmonic),label:"Broken (harmonic)"})),
      diag.order_of_operations&&h("p",{style:{marginTop:8,fontSize:13,fontFamily:"monospace"}},"Order: "+diag.order_of_operations.join(" -> ")),
      diag.relations_to_cut&&h("p",{style:{fontSize:13,color:"var(--fail)"}},"Cut: "+diag.relations_to_cut.join(", "))),
    ddl&&h(Card,{title:"Generated DDL"},h("pre",{className:"json",style:{fontSize:11,maxHeight:220,overflow:"auto"}},ddl)))}

function Ontology(){
  var e=useState(""),err=e[0],setE=e[1];
  var tx=useState("Dog subClassOf Mammal\nCat subClassOf Mammal\nMammal subClassOf Animal\nHuman equivalentClass Person\nHuman subClassOf Mammal"),txt=tx[0],setTx=tx[1];
  var rs=useState(null),res=rs[0],setRs=rs[1],si=useState(""),sid=si[0],setSi=si[1];
  function analyze(){setE("");var triples=txt.split("\n").map(function(l){return l.trim().split(/\s+/)}).filter(function(t){return t.length>=3}).map(function(t){return [t[0],t[1],t.slice(2).join(" ")]});if(!triples.length){setE("Enter triples as: Subject predicate Object (one per line)");return}var body={triples:triples};if(sid.trim())body.store_id=sid.trim();jpost("/v1/ontology/analyze",body).then(setRs).catch(function(x){setE(x.message)})}
  return h("div",null,h("h2",null,"Ontology"),h(Err,{msg:err}),
    h("p",{className:"muted",style:{marginBottom:10,fontSize:13}},"An ontology is a typed complex: subClassOf is the hierarchy (gradient), equivalent/symmetric/intersection are definitions (bounded faces), and a subsumption cycle is an inconsistency (harmonic). Enter RDF triples - one \"Subject predicate Object\" per line."),
    h("textarea",{className:"input",style:{width:"100%",minHeight:130,fontFamily:"monospace",fontSize:12},value:txt,onChange:function(ev){setTx(ev.target.value)}}),
    h("div",{className:"input-row",style:{marginTop:8}},h("button",{className:"primary",onClick:analyze},"Analyze Ontology"),h("input",{className:"input",style:{width:180},value:sid,onChange:function(ev){setSi(ev.target.value)},placeholder:"store in RCDB as… (optional)"})),
    res&&h("div",{style:{marginTop:12}},
      h("div",{style:{marginBottom:8}},h(Badge,{type:res.state==="acyclic_hierarchy"?"good":res.state==="inconsistent"?"fail":"gold"},res.state)," ",res.summary),
      res.hodge&&h("div",{className:"grid-3"},h(Stat,{value:pct(res.hodge.subsumption_hierarchy),label:"Hierarchy (subsumption)"}),h(Stat,{value:pct(res.hodge.bounded_definitions),label:"Definitions (bounded)"}),h(Stat,{value:pct(res.hodge.inconsistencies),label:"Inconsistencies"})),
      res.definitions&&res.definitions.length&&h("p",{style:{marginTop:8,fontSize:13}},"Definition faces: "+res.definitions.map(function(d){return d.join(" ≡ ")}).join(", ")),
      res.findings&&res.findings.length&&h(Card,{title:"Findings",style:{marginTop:8}},res.findings.map(function(f,i){return h("div",{key:i,className:"status-row"},h(Badge,{type:f.severity==="high"?"fail":f.severity==="info"?"neutral":"gold"},f.severity),h("span",{style:{marginLeft:8,fontSize:13}},f.issue))}))))}

function Connectors(){
  var e=useState(""),err=e[0],setE=e[1];
  var c=useState(null),cat=c[0],setCat=c[1];
  var cn=useState([]),conns=cn[0],setConns=cn[1];
  var u=useState(""),uri=u[0],setUri=u[1];
  var nm=useState(""),cname=nm[0],setCname=nm[1];
  var w=useState(false),weights=w[0],setW=w[1];
  var id=useState(""),recId=id[0],setRecId=id[1];
  var rd=useState(null),readR=rd[0],setRd=rd[1];
  var vl=useState(null),valR=vl[0],setVl=vl[1];
  var ig=useState(null),ingR=ig[0],setIg=ig[1];
  function loadCat(){api("/v1/connectors").then(function(r){setCat(r.connectors||[])}).catch(function(x){setE(x.message)})}
  function loadConns(){api("/v1/dbmanager/connections").then(function(r){setConns(r.connections||[])}).catch(function(){})}
  useEffect(function(){loadCat();loadConns()},[]);
  function reqBody(extra){var b={weights:weights};if(cname)b.name=cname;else if(uri.trim())b.uri=uri.trim();if(extra)Object.assign(b,extra);return b}
  function haveSource(){return !!(cname||uri.trim())}
  function doRead(){setE("");setRd(null);setVl(null);setIg(null);if(!haveSource()){setE("Pick a saved connection or enter a URI");return}jpost("/v1/connectors/read",reqBody()).then(setRd).catch(function(x){setE(x.message)})}
  function doValidate(){setE("");setVl(null);setRd(null);setIg(null);if(!haveSource()){setE("Pick a saved connection or enter a URI");return}jpost("/v1/connectors/validate",reqBody()).then(setVl).catch(function(x){setE(x.message)})}
  function doIngest(){setE("");setIg(null);if(!haveSource()){setE("Pick a saved connection or enter a URI");return}if(!recId.trim()){setE("Provide an id to store as");return}jpost("/v1/connectors/ingest",reqBody({id:recId.trim(),tags:["connector"]})).then(setIg).catch(function(x){setE(x.message)})}
  return h("div",null,h("h2",null,"Connectors"),h(Err,{msg:err}),
    h("p",{className:"muted",style:{marginBottom:10,fontSize:13}},"Every source - SQL, cloud warehouses, MongoDB, ontologies, property graphs, streams - becomes a relational complex through one seam. Validate a source before you trust it, read its structure, or ingest it into the RCDB. Warehouse/graph/stream schemes need their driver installed; the catalog shows which are ready."),
    cat&&h(Card,{title:"Available connectors ("+cat.length+")"},
      cat.map(function(co,i){return h("div",{key:i,className:"status-row",style:{alignItems:"flex-start"}},
        h("span",{className:"name",style:{minWidth:150}},co.connector),
        h("div",{style:{display:"flex",gap:6,flexWrap:"wrap",flex:1}},
          co.schemes.map(function(s){return h(Badge,{key:s.scheme,type:s.driver_available?"neutral":"gold"},s.scheme+(s.driver_available?"":" ⚠"))})),
        h("span",{className:"detail",style:{fontSize:11}},co.capabilities))})),
    cat&&(function(){var hints=[];cat.forEach(function(co){co.schemes.forEach(function(s){if(!s.driver_available&&s.driver_hint)hints.push(s.scheme+" - "+s.driver_hint)})});return hints.length?h(Card,{title:"Not configured - install to enable"},hints.map(function(hn,i){return h("div",{key:i,className:"status-row"},h("span",{className:"detail",style:{fontSize:12,fontFamily:"monospace"}},hn))})):null})(),
    h(Card,{title:"Onboard a source"},
      h("div",{className:"input-row",style:{flexWrap:"wrap",gap:6}},
        h("select",{className:"input",style:{width:180},value:cname,onChange:function(ev){setCname(ev.target.value)}},h("option",{value:""},"saved connection…"),conns.map(function(cc){return h("option",{key:cc.name,value:cc.name},cc.name+" ("+cc.kind+")")})),
        h("span",{className:"muted",style:{alignSelf:"center"}},"or"),
        h("input",{className:"input",style:{flex:1,minWidth:220},value:uri,onChange:function(ev){setUri(ev.target.value);if(ev.target.value)setCname("")},placeholder:"snowflake://user:pass@acct/db   ·   ontology   ·   edges"}),
        h("label",{className:"muted",style:{display:"flex",alignItems:"center",gap:4}},h("input",{type:"checkbox",checked:weights,onChange:function(ev){setW(ev.target.checked)}}),"weights")),
      h("div",{className:"input-row",style:{marginTop:8,gap:6}},
        h("button",{className:"sm",onClick:doValidate},"◇ Validate"),
        h("button",{className:"primary sm",onClick:doRead},"Read"),
        h("input",{className:"input",style:{width:160},value:recId,onChange:function(ev){setRecId(ev.target.value)},placeholder:"store as id…"}),
        h("button",{className:"sm",onClick:doIngest},"Ingest->RCDB"))),
    valR&&h(Card,{title:"Validation - "+valR.connector,actions:h("div",{style:{display:"flex",gap:6,alignItems:"center"}},h(Badge,{type:valR.ok?"good":"fail"},valR.ok?"PASS":"FAIL"),h("button",{className:"sm",onClick:function(){setVl(null)}},"Close"))},
      h(Table,{cols:[{l:"Check",k:"name"},{l:"",r:function(x){return h(Badge,{type:x.passed?"good":"fail"},x.passed?"pass":"fail")}},{l:"Detail",r:function(x){return h("span",{className:"detail",style:{fontSize:11}},x.detail)}}],rows:valR.checks})),
    readR&&h(Card,{title:"Read - "+readR.source,actions:h("button",{className:"sm",onClick:function(){setRd(null)}},"Close")},
      h("div",{className:"grid-4"},h(Stat,{value:readR.nV,label:"Vertices"}),h(Stat,{value:readR.nE,label:"Edges"}),h(Stat,{value:readR.betti?JSON.stringify(readR.betti):"-",label:"Betti"}),h(Stat,{value:readR.chain_valid?"✓":"✗",label:"∂²=0"})),
      h("p",{className:"muted",style:{marginTop:8,fontSize:12}},(readR.weighted?"weighted · ":"")+(readR.modality?"modality-tagged · ":"")+readR.nF+" faces")),
    ingR&&h(Card,{title:"Ingested -> RCDB",actions:h("button",{className:"sm",onClick:function(){setIg(null)}},"Close")},
      h("div",{style:{marginBottom:6}},h(Badge,{type:"good"},"stored")," as ",h("strong",null,ingR.stored_as)),
      h("p",{className:"muted",style:{fontSize:12}},"nV="+ingR.nV+" nE="+ingR.nE+" betti="+JSON.stringify(ingR.betti)+" - now a queryable RCDB record (see RCDB Overview).")))
}

function Setups(){
  var pv=useState(null),profs=pv[0],setProfs=pv[1],av=useState(null),active=av[0],setActive=av[1];
  var sv=useState(null),sel=sv[0],setSel=sv[1],dv=useState(null),draft=dv[0],setDraft=dv[1];
  var ev=useState(""),err=ev[0],setE=ev[1],bv=useState(false),busy=bv[0],setB=bv[1],rv=useState(null),res=rv[0],setRes=rv[1];
  var iv=useState(null),inv=iv[0],setInv=iv[1],cv=useState(null),cinv=cv[0],setCinv=cv[1];
  function load(){api("/v1/hive/profiles").then(function(d){setProfs(d.profiles);setActive(d.active)}).catch(function(x){setE(x.message)});api("/v1/ops/inventory").then(setInv).catch(function(){});api("/v1/ops/compute").then(setCinv).catch(function(){})}
  useEffect(function(){load()},[]);
  function optsFor(kind,fallback){var c=inv&&inv.components&&inv.components[kind];if(!c||!c.length)return fallback;return c.map(function(o){return [o.name,o.name+(o.native?" ✦":"")+(o.available?"":" (unavailable)"),!o.available]})}
  function pick(p){setSel(p.id);setRes(null);setDraft(JSON.parse(JSON.stringify(p)))}
  function edit(k,v){var d=Object.assign({},draft);d[k]=v;setDraft(d)}
  function editCompute(k,v){var c=Object.assign({},draft.compute||{});c[k]=v;var d=Object.assign({},draft);d.compute=c;setDraft(d)}
  function apply(id){setB(true);setE("");setRes(null);jpost("/v1/hive/profiles/"+id+"/apply",{}).then(function(r){setRes(r);setActive(r.profile);load()}).catch(function(x){setE(x.message)}).finally(function(){setB(false)})}
  function saveAs(){var body=Object.assign({},draft,{name:(draft.name||"My setup")+(draft.builtin?" (copy)":""),base:draft.builtin?draft.id:undefined,id:draft.builtin?undefined:draft.id});jpost("/v1/hive/profiles",body).then(function(r){setSel(r.profile.id);setDraft(r.profile);load()}).catch(function(x){setE(x.message)})}
  function del(id){jdel("/v1/hive/profiles/"+id).then(function(){setSel(null);setDraft(null);load()}).catch(function(x){setE(x.message)})}
  var composeOpts=[["auto","auto-compose from disk"],["attach-live","attach running servers"],["auto+attach","auto + attach both"],["manual","manual worker list"]];
  var field=function(label,ctl){return h("div",{style:{marginBottom:8}},h("label",{style:{fontSize:11,color:"var(--muted)",display:"block",marginBottom:3}},label),ctl)};
  return h("div",null,h("h2",null,"Setups - pick, tune, and switch your hive"),h(Err,{msg:err}),
    h("p",{className:"muted",style:{fontSize:12,marginTop:-4}},"A setup is a whole configuration: how the hive is composed, the memory budget, the optimizer & attention, and whether the monitor runs the semantic signal. Start from a preset, tune it, save it as your own, and switch freely - one click brings the whole hive up."),
    h("div",{style:{display:"flex",gap:16,alignItems:"flex-start",flexWrap:"wrap"}},
      h("div",{style:{flex:"1 1 340px",minWidth:300}},
        (profs||[]).map(function(p){var on=p.id===sel,act=p.id===active;
          return h("div",{key:p.id,onClick:function(){pick(p)},style:{cursor:"pointer",border:"1px solid "+(on?"var(--accent,#4a7fe0)":"var(--border,#3333)"),background:on?"var(--accent-soft,#4a7fe015)":"var(--panel,transparent)",borderRadius:12,padding:"11px 13px",marginBottom:8}},
            h("div",{style:{display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}},
              h("strong",null,p.name),
              act&&h(Badge,{type:"good"},"active"),
              h(Badge,{type:p.builtin?"neutral":"gold"},p.builtin?"preset":"saved"),
              h("span",{className:"muted",style:{fontSize:11}},p.compose)),
            h("p",{className:"muted",style:{fontSize:11.5,margin:"5px 0 0"}},p.description),
            (p.tags&&p.tags.length)?h("div",{style:{marginTop:5,display:"flex",gap:4,flexWrap:"wrap"}},p.tags.map(function(t,i){return h("span",{key:i,style:{fontSize:10,color:"var(--muted)",border:"1px solid var(--border,#3333)",borderRadius:6,padding:"1px 6px"}},t)})):null)})),
      h("div",{style:{flex:"1 1 320px",minWidth:300}},
        !draft?h(Card,{title:"Configure"},h("p",{className:"muted"},"Select a setup on the left to view, tune, apply, or clone it.")):
        h(Card,{title:draft.builtin?("Preset - "+draft.name+" (clone to edit)"):("Edit - "+draft.name)},
          field("Name",h("input",{className:"input",value:draft.name||"",onChange:function(e){edit("name",e.target.value)}})),
          field("Compose",h("select",{className:"input",value:draft.compose,onChange:function(e){edit("compose",e.target.value)}},composeOpts.map(function(o){return h("option",{key:o[0],value:o[0]},o[1])}))),
          h("div",{style:{display:"flex",gap:8}},
            field("Budget GB (blank=auto)",h("input",{className:"input",type:"number",value:draft.budget_gb==null?"":draft.budget_gb,onChange:function(e){edit("budget_gb",e.target.value===""?null:parseFloat(e.target.value))}})),
            field("Max workers",h("input",{className:"input",type:"number",value:draft.max_workers,onChange:function(e){edit("max_workers",parseInt(e.target.value||"0"))}}))),
          h("div",{style:{display:"flex",gap:8}},
            field("Optimizer",h("select",{className:"input",value:draft.optimizer,onChange:function(e){edit("optimizer",e.target.value)}},optsFor("optimizer",[["hodge","hodge ✦"],["adam","adam"]]).map(function(o){return h("option",{key:o[0],value:o[0],disabled:o[2]},o[1])}))),
            field("Attention",h("select",{className:"input",value:draft.attention,onChange:function(e){edit("attention",e.target.value)}},optsFor("attention",[["relational","relational ✦"],["standard","standard"]]).map(function(o){return h("option",{key:o[0],value:o[0],disabled:o[2]},o[1])})))),
          h("p",{className:"muted",style:{fontSize:10.5,marginTop:-4}},"✦ = your RexGraph-native component (the default). Switch to a standard PyTorch option any time - the Operations ▸ train phase uses whatever this setup selects."),
          field("Monitor",h("label",{style:{fontSize:12}},h("input",{type:"checkbox",checked:!!draft.monitor_embed,onChange:function(e){edit("monitor_embed",e.target.checked)}})," use the embedder for the semantic alignment signal")),
          h("div",{style:{marginTop:8,borderTop:"1px solid var(--border,#3333)",paddingTop:8}},
            h("div",{style:{fontSize:11,color:"var(--muted)",marginBottom:4}},"Compute (execution layer) - honored by every operation"),
            h("div",{style:{display:"flex",gap:8}},
              field("Threads (blank=all cores)",h("input",{className:"input",type:"number",min:1,value:(draft.compute&&draft.compute.threads!=null)?draft.compute.threads:"",onChange:function(e){editCompute("threads",e.target.value===""?null:parseInt(e.target.value))}})),
              field("Backend",h("select",{className:"input",value:(draft.compute&&draft.compute.backend)||"auto",onChange:function(e){editCompute("backend",e.target.value)}},
                [["auto","auto (best available)",false]].concat(((cinv&&cinv.inventory&&cinv.inventory.backends)||[]).map(function(b){return [b.name,b.name+" - "+b.kind+(b.available?"":" (unavailable)"),!b.available]})).map(function(o){return h("option",{key:o[0],value:o[0],disabled:o[2]},o[1])})))),
            (cinv&&cinv.inventory)?h("p",{className:"muted",style:{fontSize:10.5,margin:"2px 0 0"}},"live: threads="+((cinv.inventory.threads)||"all")+" backend="+cinv.inventory.backend+" · available: "+((cinv.inventory.backends||[]).filter(function(b){return b.available}).map(function(b){return b.name}).join(", "))):null),
          h("div",{style:{display:"flex",gap:6,flexWrap:"wrap",marginTop:6}},
            h("button",{className:"primary",onClick:function(){apply(draft.id)},disabled:busy||draft.builtin&&false},busy?"Applying…":"Apply & switch"),
            h("button",{className:"sm",onClick:saveAs},draft.builtin?"Save as my setup":"Save as new"),
            !draft.builtin&&h("button",{className:"sm",onClick:function(){jpost("/v1/hive/profiles",draft).then(load).catch(function(x){setE(x.message)})}},"Save"),
            !draft.builtin&&h("button",{className:"sm danger",onClick:function(){del(draft.id)}},"Delete")),
          res&&h("div",{style:{marginTop:10,fontSize:12}},
            h("p",null,"Applied ",h(Badge,{type:"good"},res.profile),
              (res.spawned&&res.spawned.length)?(" · spawned "+res.spawned.filter(function(s){return s.ok}).length):"",
              (res.attached&&res.attached.length)?(" · attached "+res.attached.length):"",
              res.status?(" -> "+res.status.n_bees+" worker(s). Open the Hive tab to watch it."):""),
            (res.spawned||[]).filter(function(s){return s&&!s.ok}).map(function(s,i){return h("p",{key:i,className:"muted",style:{color:"var(--fail,#e5534b)"}},"failed: "+s.name+" - "+s.error)})
          )))));
}

function Operations(){
  var pv=useState([]),phases=pv[0],setPhases=pv[1],rv=useState([]),runs=rv[0],setRuns=rv[1];
  var av=useState(null),active=av[0],setActive=av[1],sv=useState(null),detail=sv[0],setDetail=sv[1];
  var ev=useState(""),err=ev[0],setE=ev[1],bv=useState(""),busy=bv[0],setB=bv[1];
  function loadRuns(){api("/v1/ops/runs").then(function(d){setRuns(d.runs)}).catch(function(x){setE(x.message)})}
  function load(){api("/v1/ops/phases").then(function(d){setPhases(d.phases)}).catch(function(x){setE(x.message)});api("/v1/hive/profiles").then(function(d){setActive(d.active)}).catch(function(){});loadRuns()}
  useEffect(function(){load()},[]);
  function poll(id){api("/v1/ops/runs/"+id).then(function(r){if(!r)return;setDetail(r);if(r.status==="running"){setTimeout(function(){poll(id)},700)}else{setB("");loadRuns()}}).catch(function(){setTimeout(function(){poll(id)},700)})}
  function runPhase(name){setB(name);setE("");setDetail(null);jpost("/v1/ops/run",{phase:name,background:true}).then(function(r){setDetail(r);poll(r.id)}).catch(function(x){setE(x.message);setB("")})}
  function openRun(id){api("/v1/ops/runs/"+id).then(setDetail).catch(function(x){setE(x.message)})}
  function spark(tr){if(!tr||tr.length<2)return null;var w=180,hh=34,mn=Math.min.apply(null,tr),mx=Math.max.apply(null,tr),rng=(mx-mn)||1;var pts=tr.map(function(v,i){return (i/(tr.length-1)*w).toFixed(1)+","+(hh-((v-mn)/rng)*hh).toFixed(1)}).join(" ");return h("svg",{viewBox:"0 0 "+w+" "+hh,style:{width:w,height:hh,display:"block"}},h("polyline",{points:pts,fill:"none",stroke:"var(--accent,#4a7fe0)","stroke-width":1.5}))}
  function sparkAB(runs,key){key=key||"eval_trajectory";var all=[];runs.forEach(function(r){((r[key]||r.trajectory)||[]).forEach(function(v){all.push(v)})});if(all.length<2)return null;var w=280,hh=90,mn=Math.min.apply(null,all),mx=Math.max.apply(null,all),rng=(mx-mn)||1;var cols=["var(--accent,#4a7fe0)","var(--gold,#d4a72c)","var(--fail,#e5534b)"];
    return h("svg",{viewBox:"0 0 "+w+" "+hh,style:{width:"100%",maxWidth:w,height:"auto",display:"block"}},runs.map(function(r,ri){var tr=(r[key]||r.trajectory)||[];if(tr.length<2)return null;var pts=tr.map(function(v,i){return (i/(tr.length-1)*w).toFixed(1)+","+(hh-((v-mn)/rng)*(hh-4)-2).toFixed(1)}).join(" ");return h("polyline",{key:ri,points:pts,fill:"none",stroke:cols[ri%3],"stroke-width":1.6})}))}
  var stBadge=function(s){return h(Badge,{type:s==="ok"?"good":s==="error"?"warn":"neutral"},s)};
  var ICON={serve:"❋",train:"◇",finetune:"✦",build:"⚙",deploy:"⛴",test:"✓"};
  return h("div",null,h("h2",null,"Operations - one interface for the whole lifecycle"),h(Err,{msg:err}),
    h("p",{className:"muted",style:{fontSize:12,marginTop:-4}},"Serve, train, build, deploy, and test - every phase driven by your ",h("strong",null,"active setup")," (",active?h(Badge,{type:"gold"},active):h("span",{className:"muted"},"none - pick one in Setups"),"), each run logged. The same actions run from the ",h("code",null,"rexgraph-ops")," CLI and the API."),
    h(Card,{title:"Run a phase"},
      h("div",{style:{display:"flex",gap:8,flexWrap:"wrap"}},(phases||[]).map(function(p){
        return h("button",{key:p.name,className:"sm",style:{flex:"1 1 150px",minWidth:140,textAlign:"left",padding:"9px 11px",opacity:busy&&busy!==p.name?0.6:1},onClick:function(){runPhase(p.name)},disabled:!!busy,title:p.description},
          h("div",{style:{fontWeight:600}},(ICON[p.name]||"•")+" "+p.name+(busy===p.name?" …":"")),
          h("div",{className:"muted",style:{fontSize:10.5,marginTop:2,whiteSpace:"normal"}},p.description));
      }))),
    detail&&h(Card,{title:"Run "+detail.id,actions:stBadge(detail.status)},
      h("p",{className:"muted",style:{fontSize:11}},"phase ",h("strong",null,detail.phase)," · setup ",detail.profile||"none"," · ",detail.started+(detail.ended?(" -> "+detail.ended):"")),
      detail.error&&h("p",{style:{color:"var(--fail,#e5534b)",fontSize:12}},detail.error),
      h("div",{className:"mono",style:{fontSize:11,whiteSpace:"pre-wrap",background:"var(--panel-2,#0002)",padding:8,borderRadius:8,maxHeight:180,overflow:"auto"}},(detail.steps||[]).map(function(s){return s.msg}).join("\n")),
      (detail.result&&detail.result.ab&&detail.result.ab.length)?h("div",{style:{marginTop:8}},
        h("p",{className:"muted",style:{fontSize:11,margin:"0 0 4px"}},"A/B held-out eval loss · ",h("strong",null,detail.result.model_id)),
        sparkAB(detail.result.ab,"eval_trajectory"),
        h("div",{style:{display:"flex",gap:14,marginTop:4,flexWrap:"wrap"}},detail.result.ab.map(function(r,i){var cols=["var(--accent,#4a7fe0)","var(--gold,#d4a72c)","var(--fail,#e5534b)"];return h("span",{key:i,style:{fontSize:11,display:"inline-flex",alignItems:"center",gap:5}},h("span",{style:{width:16,height:3,background:cols[i%3],display:"inline-block",borderRadius:2}}),h("strong",null,r.optimizer_class||r.optimizer),": eval ",r.eval_start," -> ",r.eval_final,h("span",{className:"muted"}," (train ",r.loss_final,")"))})),
        detail.result.verdict?h("p",{style:{fontSize:12,marginTop:4}},"-> ",detail.result.best?h(Badge,{type:"good"},detail.result.best):null," ",detail.result.verdict):null,
        detail.result.ab[0]&&detail.result.ab[0].adapter?h("p",{className:"muted",style:{fontSize:11}},"adapters saved under ",h("code",null,"runs/artifacts/")):null):
      (detail.result&&detail.result.trajectory)?h("div",{style:{marginTop:8}},
        h("p",{className:"muted",style:{fontSize:11,margin:"0 0 2px"}},"training loss · ",h("strong",null,detail.result.optimizer_class||detail.result.optimizer)," + ",h("strong",null,detail.result.attention)," on ",detail.result.device," · ",detail.result.loss_start," -> ",detail.result.loss_final,detail.result.improved?h(Badge,{type:"good"},"improving"):null),
        spark(detail.result.trajectory)):null,
      detail.result&&h("details",{style:{marginTop:8}},h("summary",{className:"muted",style:{fontSize:11,cursor:"pointer"}},"result"),h("div",{className:"mono",style:{fontSize:11,whiteSpace:"pre-wrap",background:"var(--panel-2,#0002)",padding:8,borderRadius:8,marginTop:4}},JSON.stringify(detail.result,null,2)))),
    h(Card,{title:"Recent runs",actions:h("button",{className:"sm",onClick:loadRuns},"↻")},
      (runs&&runs.length)?h(Table,{cols:[
        {l:"Run",r:function(x){return h("a",{href:"#",onClick:function(e){e.preventDefault();openRun(x.id)},style:{fontFamily:"var(--mono,monospace)",fontSize:11}},x.id)}},
        {l:"Phase",r:function(x){return (ICON[x.phase]||"•")+" "+x.phase}},
        {l:"Setup",r:function(x){return h("span",{className:"muted",style:{fontSize:11}},x.profile||"-")}},
        {l:"Status",r:function(x){return stBadge(x.status)}},
        {l:"Started",r:function(x){return h("span",{className:"muted",style:{fontSize:11}},x.started)}}
      ],rows:runs}):h("p",{className:"muted"},"No runs yet - run a phase above.")));
}

function SwarmGraph(props){
  var mon=props.mon,ags=(mon.agents||[]),n=ags.length;
  if(!n)return null;
  var W=520,H=360,cx=260,cy=178,R=122;
  var loads=ags.map(function(a){return a.load_bearing||0});
  var maxLoad=Math.max.apply(null,loads.concat([1e-6]));
  var pos={};
  ags.forEach(function(a,i){
    var ang=2*Math.PI*i/n-Math.PI/2;
    pos[a.agent]={x:cx+R*Math.cos(ang),y:cy+R*Math.sin(ang),
      r:11+17*((a.load_bearing||0)/maxLoad),divergent:a.flag==="divergent",
      load:a.load_bearing,align:a.alignment,coh:a.coherence,msgs:a.messages,hub:i===0};
  });
  var edges=mon.edges||[];
  var maxW=Math.max.apply(null,edges.map(function(e){return e.weight}).concat([1]));
  var edgeEls=edges.map(function(e,i){
    var a=pos[e.from],b=pos[e.to];if(!a||!b)return null;
    var dx=b.x-a.x,dy=b.y-a.y,len=Math.sqrt(dx*dx+dy*dy)||1,ux=dx/len,uy=dy/len;
    var sx=a.x+ux*a.r,sy=a.y+uy*a.r,ex=b.x-ux*(b.r+7),ey=b.y-uy*(b.r+7);
    var mx=(sx+ex)/2+(-uy)*15,my=(sy+ey)/2+ux*15;
    return h("path",{key:"e"+i,d:"M"+sx+","+sy+" Q"+mx+","+my+" "+ex+","+ey,fill:"none",
      stroke:"var(--line,#8888884d)",strokeWidth:1+2.4*(e.weight/maxW),markerEnd:"url(#swarm-arrow)",opacity:0.7});
  });
  var nodeEls=ags.map(function(a,i){
    var p=pos[a.agent],fill=p.divergent?"var(--fail,#e5534b)":(p.hub?"var(--gold,#d4a72c)":"var(--accent,#4a7fe0)");
    return h("g",{key:"n"+i},
      h("title",null,a.agent+"  ·  load-bearing "+p.load+"  ·  alignment "+p.align+"  ·  coherence "+p.coh+"  ·  "+p.msgs+" msgs"+(p.divergent?"  ·  DIVERGENT":"")),
      h("circle",{cx:p.x,cy:p.y,r:p.r,fill:fill,stroke:"var(--bg,#fff)",strokeWidth:2,opacity:0.92}),
      h("text",{x:p.x,y:p.y+p.r+13,textAnchor:"middle",fontSize:11,fill:"var(--fg,#222)"},a.agent));
  });
  return h("div",{style:{overflowX:"auto",margin:"6px 0"}},
    h("svg",{viewBox:"0 0 "+W+" "+H,style:{width:"100%",maxWidth:W,height:"auto",display:"block",margin:"0 auto"}},
      h("defs",null,h("marker",{id:"swarm-arrow",viewBox:"0 0 10 10",refX:"8",refY:"5",markerWidth:"6",markerHeight:"6",orient:"auto-start-reverse"},
        h("path",{d:"M0,0 L10,5 L0,10 z",fill:"var(--line,#88888899)"}))),
      edgeEls,nodeEls),
    h("p",{className:"muted",style:{fontSize:11}},"Node size = load-bearing (effective resistance); ",
      h("span",{style:{color:"var(--gold,#d4a72c)",fontWeight:600}},"gold")," = the hub the hive leans on, ",
      h("span",{style:{color:"var(--fail,#e5534b)",fontWeight:600}},"red")," = divergent (possible hallucination / off-topic). Arrows = message flow, width = volume. Hover a node for its metrics."));
}

function Swarm(){
  var mv=useState(null),mon=mv[0],setMon=mv[1],e=useState(""),err=e[0],setE=e[1];
  var em=useState(false),useEmb=em[0],setEmb=em[1];
  var fv=useState("queen"),ff=fv[0],setFf=fv[1],tv=useState("bio"),tt=tv[0],setTt=tv[1],xv=useState(""),xt=xv[0],setXt=xv[1];
  var qv=useState(""),qq=qv[0],setQq=qv[1],rv=useState(null),rt=rv[0],setRt=rv[1];
  // Hive control surface - the swarm of managed/attached bees the monitor reads.
  var hv=useState(null),hive=hv[0],setHive=hv[1],hb=useState(false),hBusy=hb[0],setHb=hb[1];
  var dqv=useState(""),dq=dqv[0],setDq=dqv[1],drv=useState(null),dres=drv[0],setDres=drv[1];
  var plv=useState(null),plan=plv[0],setPlan=plv[1];
  var ccv=useState(""),ccmd=ccv[0],setCcmd=ccv[1],csv=useState("hive"),cscope=csv[0],setCscope=csv[1],crv=useState(null),cres=crv[0],setCres=crv[1];
  var dv=useState(null),dash=dv[0],setDash=dv[1];
  var nv=useState(null),net=nv[0],setNet=nv[1],lgv=useState([]),logs=lgv[0],setLogs=lgv[1],usv=useState({}),usage=usv[0],setUsage=usv[1];
  var trv=useState("hive"),tier=trv[0],setTier=trv[1],shv=useState("default"),selHive=shv[0],setSelHive=shv[1],swv=useState(""),selWorker=swv[0],setSelWorker=swv[1];
  var liv=useState(false),live=liv[0],setLive=liv[1];
  function refreshAll(){loadMon();loadHive();loadNet();loadUsage();loadLogs()}
  // the display is its own complex: an event touches only the cells that depend on it. The activity
  // log gets the event directly (O(1) append); the derived panels (monitor/roster/network/usage)
  // are marked dirty and refetched ONCE per burst (debounced), not per event.
  var dbT=useRef(null);
  function scheduleRefetch(){if(dbT.current)clearTimeout(dbT.current);dbT.current=setTimeout(function(){loadMon();loadHive();loadNet();loadUsage()},250)}
  function onEvent(ev){
    var show=(tier!=="worker")||(ev.entity==="worker:"+selWorker);
    if(show)setLogs(function(p){return [ev].concat(p).slice(0,60)});
    scheduleRefetch();
  }
  function scopeStr(){return tier==="network"?"network":tier==="worker"?("worker:"+(selWorker||"")):("hive:"+selHive)}
  function loadNet(){api("/v1/agents/network").then(setNet).catch(function(){})}
  function loadUsage(){api("/v1/agents/usage").then(function(r){setUsage(r.usage||{})}).catch(function(){})}
  function loadLogs(){var q="/v1/agents/activity?limit=60";if(tier==="worker"&&selWorker)q+="&entity=worker:"+encodeURIComponent(selWorker);api(q).then(function(r){setLogs(r.events||[])}).catch(function(){})}
  function runCmd(confirm){if(!ccmd.trim())return;jpost("/v1/agents/command",{command:ccmd,scope:cscope,confirm:!!confirm}).then(function(r){setCres(r);refreshAll()}).catch(function(x){setE(x.message)})}
  function loadHive(){api("/v1/hive/status").then(setHive).catch(function(x){setE(x.message)})}
  function planHive(){setE("");api("/v1/hive/plan").then(setPlan).catch(function(x){setE(x.message)})}
  function autoCompose(){setHb(true);setE("");setPlan(null);jpost("/v1/hive/auto",{}).then(function(r){setHive(r.status);loadMon()}).catch(function(x){setE(x.message)}).finally(function(){setHb(false)})}
  function attachLive(){setHb(true);setE("");jpost("/v1/hive/attach-live",{}).then(function(r){setHive(r.status)}).catch(function(x){setE(x.message)}).finally(function(){setHb(false)})}
  function removeBee(n){jpost("/v1/hive/remove",{name:n}).then(function(r){setHive(r.status);loadMon()}).catch(function(x){setE(x.message)})}
  function dispatch(){if(!dq.trim())return;setDres(null);jpost("/v1/hive/dispatch",{query:dq}).then(function(r){setDres(r);loadMon()}).catch(function(x){setE(x.message)})}
  function loadMon(){api("/v1/agents/monitor"+(useEmb?"?embed=true":"")).then(setMon).catch(function(x){setE(x.message)});api("/v1/agents/dashboard").then(setDash).catch(function(){})}
  function sendMsg(){if(!ff.trim()||!tt.trim()||!xt.trim())return;jpost("/v1/agents/message",{from:ff,to:tt,text:xt}).then(function(){setXt("");loadMon()}).catch(function(x){setE(x.message)})}
  function doRoute(){if(!qq.trim())return;jpost("/v1/agents/route",{query:qq}).then(setRt).catch(function(x){setE(x.message)})}
  function reset(){jpost("/v1/agents/reset",{}).then(function(){setMon(null);setRt(null);loadMon()}).catch(function(x){setE(x.message)})}
  useEffect(function(){loadMon();loadHive();loadNet();loadUsage()},[]);
  useEffect(function(){setCscope(scopeStr());loadLogs()},[tier,selHive,selWorker]);
  // live = a push stream (SSE), not a poll. Events arrive the instant they happen; each routes to
  // only the panels it touches. On stream error it reconnects; toggling live off / changing scope
  // aborts and reopens. No periodic polling - idle traffic is just the 15s heartbeat.
  useEffect(function(){
    if(!live)return;
    var stopped=false,ctrl;
    function open(){
      if(stopped)return;
      ctrl=new AbortController();var buf="";
      fetch("/api/v1/agents/events",{headers:authHeaders(),signal:ctrl.signal}).then(function(resp){
        var reader=resp.body.getReader(),dec=new TextDecoder();
        (function pump(){
          reader.read().then(function(res){
            if(res.done||stopped)return;
            buf+=dec.decode(res.value,{stream:true});
            var parts=buf.split("\n\n");buf=parts.pop();
            parts.forEach(function(b){
              var dl=b.split("\n").filter(function(l){return l.indexOf("data:")===0})[0];
              if(dl){try{onEvent(JSON.parse(dl.slice(5).trim()))}catch(e){}}
            });
            pump();
          }).catch(function(){if(!stopped)setTimeout(open,2000)});   // reconnect on a dropped read
        })();
      }).catch(function(){if(!stopped)setTimeout(open,2000)});       // reconnect on a failed connect
    }
    open();
    return function(){stopped=true;if(ctrl)ctrl.abort()};
  },[live,tier,selHive,selWorker,useEmb]);
  var roleBadge=function(r){return h(Badge,{type:r==="queen"?"gold":r==="embedder"?"good":"neutral"},r)};
  return h("div",null,h("h2",null,"Hive Network"),h(Err,{msg:err}),
    h("p",{className:"muted",style:{fontSize:12,marginTop:-4}},"Workers coordinate through one relational complex. The monitor reads its structure: load-bearing workers, circulation, deadlock cycles, alignment."),
    h("div",{style:{display:"flex",justifyContent:"flex-end",gap:6,marginBottom:8}},
      h("button",{className:"sm",onClick:refreshAll},"refresh"),
      h("button",{className:live?"sm primary":"sm",onClick:function(){setLive(!live)},title:"auto-refresh every 2s - watch it update as commands run"},live?"● live":"○ live")),
    h(Card,{title:"Command console",actions:h("div",{style:{display:"flex",gap:6}},
      h("select",{className:"input",style:{width:96,fontSize:11},value:tier,onChange:function(ev){setTier(ev.target.value)}},["network","hive","worker"].map(function(o){return h("option",{key:o,value:o},o)})),
      tier!=="network"?h("select",{className:"input",style:{width:112,fontSize:11},value:selHive,onChange:function(ev){setSelHive(ev.target.value)}},((net&&net.hives)||[{name:"default"}]).map(function(hv){return h("option",{key:hv.name,value:hv.name},hv.name)})):null,
      tier==="worker"?h("select",{className:"input",style:{width:112,fontSize:11},value:selWorker,onChange:function(ev){setSelWorker(ev.target.value)}},[h("option",{key:"_",value:""},"—")].concat(((hive&&hive.bees)||[]).map(function(b){return h("option",{key:b.name,value:b.name},b.name)}))):null)},
      h("p",{className:"muted",style:{fontSize:11,marginTop:-2,marginBottom:8}},"Command the hive. Inspect verbs (",h("code",null,"status"),", ",h("code",null,"monitor"),", ",h("code",null,"dashboard"),") run freely; build verbs (",h("code",null,"require review test"),", ",h("code",null,"forge net mlp"),", ",h("code",null,"chat …"),") act; consequential verbs (",h("code",null,"kill"),") return a proposal until you confirm - you are the governor."),
      h("div",{className:"input-row"},
        h("input",{className:"input",style:{flex:1,fontFamily:"var(--mono)"},value:ccmd,onChange:function(ev){setCcmd(ev.target.value)},placeholder:"require review test   ·   chat how do you handle a 503?   ·   kill rogue",onKeyDown:function(ev){if(ev.key==="Enter")runCmd(false)}}),
        h("button",{className:"primary",onClick:function(){runCmd(false)}},"Run")),
      cres&&(cres.governed?h("div",{style:{marginTop:8,padding:"10px 12px",background:"var(--gold-bg)",border:"1px solid var(--gold-border)",borderRadius:"var(--radius-sm)"}},
          h("span",{style:{fontSize:13}},h(Badge,{type:"gold"},"needs confirm")," ",cres.proposed),
          h("button",{className:"sm danger",style:{marginLeft:10},onClick:function(){runCmd(true)}},"Confirm")):
        h("div",{className:"json",style:{marginTop:8}},JSON.stringify(cres,null,2)))),
    h(Card,{title:"Hive · workers",actions:h("div",{style:{display:"flex",gap:6}},h("button",{className:"sm",onClick:planHive,disabled:hBusy},"Plan from disk"),h("button",{className:"sm",onClick:autoCompose,disabled:hBusy},hBusy?"Composing…":"Auto-compose"),h("button",{className:"sm",onClick:attachLive,disabled:hBusy},hBusy?"…":"Attach live"))},
      plan&&h("div",{style:{marginBottom:10,padding:10,background:"var(--panel-2,#0002)",borderRadius:8}},
        h("p",{style:{fontSize:12,margin:"0 0 6px"}},"Plan from disk - ",h("strong",null,"budget "+plan.budget_gb+" GB")," (usable "+plan.usable_gb+"), would run ",h("strong",null,plan.planned_gb+" GB")," across ",String(plan.n)," worker(s):"),
        plan.plan&&plan.plan.length?h(Table,{cols:[
          {l:"Role",r:function(x){return roleBadge(x.role)}},{l:"Worker",k:"name"},
          {l:"~Size",r:function(x){return x.size_gb+" GB"}},{l:"Model",r:function(x){return h("span",{className:"muted",style:{fontSize:11}},x.model)}},
          {l:"Specialties",r:function(x){return h("span",{className:"muted",style:{fontSize:11}},(x.specialties||[]).join(", ")||"-")}}
        ],rows:plan.plan}):h("p",{className:"muted",style:{fontSize:12,margin:0}},plan.note||"Nothing on disk fits - pull a model (Models ▸ Local)."),
        plan.plan&&plan.plan.length?h("button",{className:"sm primary",style:{marginTop:8},onClick:autoCompose,disabled:hBusy},"Compose this plan"):null),
      (!hive||!hive.n_bees)?h("p",{className:"muted"},"No workers yet. ",h("strong",null,"Auto-compose")," brings up a coordinator, workers, and an embedder that fit this machine from your local models; ",h("strong",null,"Attach live")," enrolls running endpoints; or use ",h("code",null,"python -m agent.hive up --auto"),"."):h("div",null,
        h("div",{style:{display:"flex",gap:12,marginBottom:8,flexWrap:"wrap"}},
          h(Stat,{value:hive.n_bees,label:"members"}),h(Stat,{value:hive.queen||"-",label:"coordinator"}),h(Stat,{value:(hive.workers||[]).length,label:"workers"}),h(Stat,{value:hive.embedder||"-",label:"embedder"})),
        h(Table,{cols:[
          {l:"Worker",k:"name"},
          {l:"Role",r:function(x){return roleBadge(x.role)}},
          {l:"Model",r:function(x){return h("span",{className:"muted",style:{fontSize:11}},x.model||"-")}},
          {l:"Endpoint",r:function(x){return h("code",{style:{fontSize:11}},x.url)}},
          {l:"Kind",r:function(x){return h(Badge,{type:x.managed?"good":"neutral"},x.managed?"managed":"attached")}},
          {l:"",r:function(x){return h("button",{className:"sm danger",onClick:function(){removeBee(x.name)}},"×")}}
        ],rows:hive.bees||[]}),
        h("div",{className:"input-row",style:{marginTop:8}},h("input",{className:"input",style:{flex:1},value:dq,onChange:function(ev){setDq(ev.target.value)},placeholder:"dispatch a query - routes to the best worker and asks it"}),h("button",{className:"primary",onClick:dispatch,disabled:!dq.trim()},"Dispatch")),
        dres&&h("div",{style:{marginTop:8}},
          dres.bee?h("p",{style:{fontSize:12}},"-> routed to ",h(Badge,{type:"gold"},dres.bee),dres.routed&&dres.routed[0]?h("span",{className:"muted"}," (score "+dres.routed[0].score+")"):null):h("p",{className:"muted",style:{fontSize:12}},dres.note||"no worker available"),
          dres.reply?h("div",{className:"mono",style:{fontSize:12,whiteSpace:"pre-wrap",background:"var(--panel-2,#0002)",padding:8,borderRadius:8}},dres.reply):(dres.bee?h("p",{className:"muted",style:{fontSize:11}},"(worker reachable? no reply text - check the endpoint)"):null)))),
    h(Card,{title:"Feed an interaction",actions:h("button",{className:"sm",onClick:reset},"Reset")},
      h("div",{className:"input-row"},
        h("input",{className:"input",style:{width:110},value:ff,onChange:function(ev){setFf(ev.target.value)},placeholder:"from"}),
        h("input",{className:"input",style:{width:110},value:tt,onChange:function(ev){setTt(ev.target.value)},placeholder:"to"}),
        h("input",{className:"input",style:{flex:1},value:xt,onChange:function(ev){setXt(ev.target.value)},placeholder:"message text"}),
        h("button",{className:"primary",onClick:sendMsg},"Send"))),
    h(Card,{title:"Monitor",actions:h("div",{style:{display:"flex",gap:8,alignItems:"center"}},h("label",{style:{fontSize:11}},h("input",{type:"checkbox",checked:useEmb,onChange:function(ev){setEmb(ev.target.checked)}})," semantic"),h("button",{className:"sm",onClick:loadMon},"↻"))},
      (!mon||!mon.n_agents)?h("p",{className:"muted"},"No inter-agent interactions yet - feed some above, or the runtime feeds them live as agents/models message each other."):h("div",null,
        h("div",{style:{display:"flex",gap:12,marginBottom:8,flexWrap:"wrap"}},
          h(Stat,{value:mon.n_agents,label:"agents"}),h(Stat,{value:mon.n_interactions,label:"interactions"}),
          h(Stat,{value:mon.deadlock_cycles,label:"deadlock cycles"}),
          mon.interaction_hodge&&h(Stat,{value:Math.round(mon.interaction_hodge.circulating*100)+"%",label:"disagreement (curl)"})),
        h("p",{className:"muted",style:{fontSize:11}},"alignment: ",h("strong",null,mon.alignment_mode),mon.alignment_mode==="lexical"?" - start the embedder (Models ▸ Local) for the semantic signal that separates hallucination from a distinct specialist":""),
        h(SwarmGraph,{mon:mon}),
        dash&&dash.information_flow&&h("div",{style:{margin:"2px 0 10px"}},
          h("div",{style:{display:"flex",gap:12,marginBottom:6,flexWrap:"wrap"}},
            h(Stat,{value:Math.round((dash.information_flow.draining||0)*100)+"%",label:"draining"}),
            h(Stat,{value:Math.round((dash.information_flow.circulating||0)*100)+"%",label:"circulating"}),
            dash.information_flow.health_ratio!=null?h(Stat,{value:dash.information_flow.health_ratio,label:"frust / copart"}):null),
          (dash.information_flow.stuck_loops||[]).map(function(l,i){return h("div",{key:i,style:{fontSize:12,marginBottom:4}},
            h(Badge,{type:l.kind==="irreducible"?"F":"G"},l.kind),
            h("span",{style:{fontFamily:"var(--mono)",marginLeft:8}},(l.services||[]).join(" → ")+" → "+(l.services||[])[0]),
            h("span",{className:"muted",style:{marginLeft:8}},"circulating "+l.circulating))})),
        h(Table,{cols:[
          {l:"Agent",k:"agent"},
          {l:"Load-bearing",r:function(x){return x.load_bearing}},
          {l:"Coherence",r:function(x){return x.coherence}},
          {l:"Alignment",r:function(x){return x.alignment}},
          {l:"Msgs",r:function(x){return x.messages}},
          {l:"",r:function(x){return x.flag==="divergent"?h(Badge,{type:"warn"},"divergent"):h(Badge,{type:"good"},"ok")}}
        ],rows:mon.agents||[]}))),
    dash&&dash.networks&&dash.networks.length?h(Card,{title:"Forged networks - the NNs the LMs built and control"},
      h(Table,{cols:[
        {l:"Network",k:"name"},
        {l:"Type",r:function(x){return h(Badge,{type:"gold"},(x.type||"").replace("model:",""))}}
      ],rows:dash.networks})):null,
    h(Card,{title:"Activity log"+(tier==="worker"&&selWorker?" · worker:"+selWorker:""),actions:h("button",{className:"sm",onClick:loadLogs},"↻")},
      logs.length?h(Table,{cols:[
        {l:"Entity",r:function(e){return h("span",{style:{fontFamily:"var(--mono)"}},e.entity)}},
        {l:"Action",r:function(e){return h(Badge,{type:(e.action||"").indexOf("remove")>=0?"warn":"neutral"},e.action)}},
        {l:"Detail",r:function(e){return h("span",{className:"muted"},Object.keys(e.detail||{}).map(function(k){return k+"="+e.detail[k]}).join(" "))}}
      ],rows:logs}):h("p",{className:"muted"},"No activity yet.")),
    h(Card,{title:"Model usage",actions:h("button",{className:"sm",onClick:loadUsage},"↻")},
      Object.keys(usage).length?h(Table,{cols:[
        {l:"Model",r:function(r){return h("span",{style:{fontFamily:"var(--mono)"}},r[0])}},
        {l:"Runtime",r:function(r){return r[1].runtime_s+"s"}},
        {l:"Concurrent",r:function(r){return h(Badge,{type:r[1].concurrent>0?"gold":"neutral"},r[1].concurrent)}},
        {l:"Total",r:function(r){return r[1].total_uses}},
        {l:"In use for",r:function(r){return h("span",{className:"muted"},(r[1].active_uses||[]).map(function(a){return a.purpose}).join(", ")||"—")}}
      ],rows:Object.keys(usage).map(function(m){return [m,usage[m]]})}):h("p",{className:"muted"},"No models used yet.")),
    h(Card,{title:"Route a query (= reweight the complex)"},
      h("div",{className:"input-row"},h("input",{className:"input",style:{flex:1},value:qq,onChange:function(ev){setQq(ev.target.value)},placeholder:"query - which agent should handle this?"}),h("button",{className:"primary",onClick:doRoute},"Route")),
      rt&&rt.agents&&(rt.agents.length?h("div",{style:{marginTop:8,display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}},rt.agents.map(function(a,i){return h("span",{key:i},h(Badge,{type:i===0?"gold":"neutral"},a.agent)," ",a.relevance)})):h("p",{className:"muted",style:{marginTop:8}},"no matching agent"))))
}

function ModelStudio(){
  var av=useState([]),arcs=av[0],setArcs=av[1],sv=useState(null),sel=sv[0],setSel=sv[1];
  var pv=useState({}),prm=pv[0],setPrm=pv[1],dv=useState(""),dpath=dv[0],setDp=dv[1];
  var ov=useState("hodge"),opt=ov[0],setOpt=ov[1],mv=useState("single"),mode=mv[0],setMode=mv[1];
  var stv=useState(150),steps=stv[0],setSt=stv[1],rv=useState(null),res=rv[0],setRes=rv[1];
  var bv=useState(false),busy=bv[0],setB=bv[1],ev=useState(""),err=ev[0],setE=ev[1];
  var tv=useState("Metformin, treats, Diabetes\nMetformin, activates, AMPK\nAMPK, regulates, Glucose"),tri=tv[0],setTri=tv[1];
  var lv=useState(""),labs=lv[0],setLabs=lv[1],iv=useState(null),ires=iv[0],setIres=iv[1],ibv=useState(false),iBusy=ibv[0],setIb=ibv[1];
  function load(){api("/v1/ml/archetypes").then(function(d){setArcs(d.archetypes)}).catch(function(x){setE(x.message)})}
  useEffect(function(){load()},[]);
  function pick(a){setSel(a.name);setPrm(Object.assign({},a.params));setRes(null)}
  function edit(k,v){var p=Object.assign({},prm);p[k]=v;setPrm(p)}
  function runTrain(){setB(true);setE("");setRes(null);jpost("/v1/ml/run",{archetype:sel,params:prm,data:dpath||undefined,mode:mode,optimizer:opt,steps:steps}).then(setRes).catch(function(x){setE(x.message)}).finally(function(){setB(false)})}
  function ingest(){setIb(true);setE("");setIres(null);
    var triples=tri.split("\n").map(function(l){return l.split(",").map(function(x){return x.trim()})}).filter(function(r){return r.length===3&&r[0]});
    var labels=null;if(labs.trim()){labels={};labs.split("\n").forEach(function(l){var p=l.split(",").map(function(x){return x.trim()});if(p.length>=2)labels[p[0]]=p[1]})}
    jpost("/v1/ml/ingest",{triples:triples,labels:labels,train:true,archetype:"hgnn",steps:120}).then(setIres).catch(function(x){setE(x.message)}).finally(function(){setIb(false)})}
  function traj(r){var t=r&&(r.trajectory||(r.final!=null?[r.final]:null));if(!t||t.length<2)return null;var w=200,hh=32,mn=Math.min.apply(null,t),mx=Math.max.apply(null,t),rg=(mx-mn)||1;return h("svg",{viewBox:"0 0 "+w+" "+hh,style:{width:w,height:hh}},h("polyline",{points:t.map(function(v,i){return (i/(t.length-1)*w).toFixed(1)+","+(hh-((v-mn)/rg)*hh).toFixed(1)}).join(" "),fill:"none",stroke:"var(--accent,#4a7fe0)","stroke-width":1.5}))}
  var selA=arcs.filter(function(a){return a.name===sel})[0];
  return h("div",null,h("h2",null,"Model Studio - build & train on the substrate"),h(Err,{msg:err}),
    h("p",{className:"muted",style:{fontSize:12,marginTop:-4}},"Pick an archetype, tune it, point it at your data, and train with your optimizer (HodgeAdam by default) - built on ",h("code",null,"rexgraph.nn"),", persisted through the rexgraph IO layer. Modular: archetypes come from the registry."),
    h(Card,{title:"Build & train"},
      h("div",{style:{display:"flex",gap:6,flexWrap:"wrap",marginBottom:8}},arcs.map(function(a){return h("button",{key:a.name,className:a.name===sel?"sm primary":"sm",onClick:function(){pick(a)},title:a.use_case},a.name)})),
      selA&&h("p",{className:"muted",style:{fontSize:11,margin:"0 0 8px"}},selA.use_case," · data: ",selA.data_kind),
      selA&&h("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(150px,1fr))",gap:8}},Object.keys(prm).map(function(k){var v=prm[k];var isB=typeof v==="boolean";return h("div",{key:k},h("label",{style:{fontSize:11,color:"var(--muted)",display:"block"}},k),isB?h("label",{style:{fontSize:12}},h("input",{type:"checkbox",checked:v,onChange:function(e){edit(k,e.target.checked)}})," ",String(v)):h("input",{className:"input",value:v,onChange:function(e){var nv=e.target.value;edit(k,(typeof v==="number"&&nv!==""&&!isNaN(nv))?Number(nv):nv)}}))})),
      selA&&h("div",{className:"input-row",style:{marginTop:10}},
        h("input",{className:"input",style:{flex:2},value:dpath,onChange:function(e){setDp(e.target.value)},placeholder:"data path (parquet/.rex/csv/… - blank = synthetic)"}),
        h("select",{className:"input",value:opt,onChange:function(e){setOpt(e.target.value)}},["hodge","hodge-arch","adam","sgd","adamw"].map(function(o){return h("option",{key:o,value:o},o)})),
        h("select",{className:"input",value:mode,onChange:function(e){setMode(e.target.value)}},["single","multistep","fusion"].map(function(o){return h("option",{key:o,value:o},o)})),
        h("input",{className:"input",style:{width:80},type:"number",value:steps,onChange:function(e){setSt(Number(e.target.value)||150)}}),
        h("button",{className:"primary",onClick:runTrain,disabled:busy||!sel},busy?"Training…":"Train")),
      res&&h("div",{style:{marginTop:10}},
        h("p",{style:{fontSize:13}},h(Badge,{type:"good"},res.archetype||sel)," ",res.metric_name||"metric",": ",h("strong",null,res.metric!=null?res.metric:(res.fused_metric!=null?res.fused_metric:"-")),res.saved?h("span",{className:"muted"}," · saved "+res.saved):null),
        res.diagnosis&&h("p",{style:{fontSize:12,marginTop:2}},h(Badge,{type:(res.diagnosis.status==="healthy"||res.diagnosis.status==="converged")?"good":"warn"},"training: "+res.diagnosis.status),res.diagnosis.cause?h("span",{className:"muted"}," · "+res.diagnosis.cause):null),
        traj(res),
        res.base_models?h("p",{className:"muted",style:{fontSize:11}},"fusion bases: "+res.base_models.map(function(b){return b.archetype+" "+b.final}).join(", ")):null)),
    h(Card,{title:"Ingest - TrustGraph knowledge core -> complex -> train"},
      h("p",{className:"muted",style:{fontSize:11}},"Paste triples (subject, predicate, object - one per line). Optional labels (entity, class per line). Ingests into a relational complex and trains an HGNN over it."),
      h("div",{style:{display:"flex",gap:8,flexWrap:"wrap"}},
        h("textarea",{className:"input",style:{flex:2,minWidth:240,minHeight:80,fontFamily:"var(--mono,monospace)",fontSize:12},value:tri,onChange:function(e){setTri(e.target.value)}}),
        h("textarea",{className:"input",style:{flex:1,minWidth:140,minHeight:80,fontFamily:"var(--mono,monospace)",fontSize:12},value:labs,onChange:function(e){setLabs(e.target.value)},placeholder:"Metformin, drug\nDiabetes, disease"})),
      h("button",{className:"primary",style:{marginTop:8},onClick:ingest,disabled:iBusy},iBusy?"Ingesting…":"Ingest & train"),
      ires&&h("div",{style:{marginTop:8,fontSize:12}},
        h("p",null,"complex: ",h("strong",null,ires.n_nodes)," entities, ",ires.n_classes," classes",ires.train?h("span",null," · trained HGNN -> ",h(Badge,{type:"good"},ires.train.metric)):null),
        (ires.entities&&ires.entities.length)?h("p",{className:"muted",style:{fontSize:11}},"entities: "+ires.entities.join(", ")):null)));
}

var SECTIONS=[
  {label:"ANALYZE",tabs:[{id:"pipeline",label:"Pipeline",icon:"▶"},{id:"documents",label:"Documents",icon:"◫"},{id:"corpus",label:"Corpus",icon:"◈"}]},
  {label:"DATABASE",tabs:[{id:"database",label:"RCDB Overview",icon:"⛁"},{id:"dbmanager",label:"DB Manager",icon:"⌸"},{id:"connectors",label:"Connectors",icon:"⇄"},{id:"schema",label:"Schema Diagnosis",icon:"⧉"},{id:"schemabuilder",label:"Schema Builder",icon:"⊹"},{id:"ontology",label:"Ontology",icon:"❈"}]},
  {label:"CONNECT",tabs:[{id:"trustgraph",label:"TrustGraph",icon:"⬡"},{id:"models",label:"Models",icon:"⊞"}]},
  {label:"BUILD",tabs:[{id:"builder",label:"Agent Builder",icon:"⚙"},{id:"training",label:"Training",icon:"◇"},{id:"chat",label:"Chat",icon:"◬"},{id:"setups",label:"Setups",icon:"◆"},{id:"mlstudio",label:"Model Studio",icon:"◱"},{id:"operations",label:"Operations",icon:"◎"},{id:"swarm",label:"Hive",icon:"❋"}]},
  {label:"ADMIN",tabs:[{id:"system",label:"System",icon:"⊡"}]}];
var ALL_TABS=[];SECTIONS.forEach(function(s){s.tabs.forEach(function(t){ALL_TABS.push(t)})});
var TAB_MAP={pipeline:Pipeline,documents:Documents,corpus:Corpus,database:Database,dbmanager:DBManager,connectors:Connectors,schema:Schema,schemabuilder:SchemaBuilder,ontology:Ontology,trustgraph:TrustGraph,models:Models,builder:Builder,training:Training,chat:Chat,setups:Setups,mlstudio:ModelStudio,operations:Operations,swarm:Swarm,system:System};

function App(){
  var t=useState("pipeline"),tab=t[0],setTab=t[1];
  var au=useState(null),authState=au[0],setAu=au[1]; // null=loading, "setup"=first run, "login"=need token, "ok"=ready
  var ws=useState(_workspace),workspace=ws[0],setWs=ws[1];
  var wsl=useState(["default"]),workspaces=wsl[0],setWsl=wsl[1];

  useEffect(function(){
    fetch("/api/health").then(function(r){return r.json()}).then(function(d){
      if(d.workspaces)setWsl(d.workspaces);
      if(!d.auth_enabled){
        // Auth disabled - check if first run (no tokens created yet)
        fetch("/api/v1/admin/tokens").then(function(r){
          if(!r.ok){setAu("login");return null}
          return r.json()
        }).then(function(td){
          if(!td)return;
          var tks=td.tokens||td||[];
          if(tks.length>0||sessionStorage.getItem("rexgraph_setup_done")){setAu("ok")}else{setAu("setup")}
        }).catch(function(){setAu("ok")});return}
      // Auth enabled - verify stored token
      if(!_authToken){setAu("login");return}
      fetch("/api/v1/admin/tokens",{headers:{"Authorization":"Bearer "+_authToken}})
        .then(function(r){if(r.status===401){setAuth("");setAu("login")}else{setAu("ok")}})
        .catch(function(){setAu("login")})
    }).catch(function(){setAu("ok")})
  },[]);

  useEffect(function(){_onAuthFail=function(){setAu("login")}},[]);

  function switchWorkspace(name){setWs(name);setAuth(_authToken,name);window.location.reload()}
  function logout(){setAuth("");setAu("login")}

  if(authState===null)return h("div",{style:{display:"flex",alignItems:"center",justifyContent:"center",minHeight:"100vh"}},h("p",{className:"muted"},"Loading…"));
  if(authState==="setup")return h(Setup,{onDone:function(){setAu("ok")}});
  if(authState==="login")return h(Login,{onLogin:function(){setAu("ok")}});

  var Comp=TAB_MAP[tab]||Pipeline;
  return h("div",{className:"app"},
    h("aside",{className:"sidebar"},
      h("div",{className:"sidebar-brand"},h("span",{className:"dot"}),h("h1",null,"rexgraph")),
      workspaces.length>1&&h("div",{style:{padding:"4px 8px"}},
        h("select",{className:"input",value:workspace,onChange:function(ev){switchWorkspace(ev.target.value)},style:{fontSize:11}},
          workspaces.map(function(w){return h("option",{key:w,value:w},w)}))),
      SECTIONS.map(function(sec){return h("div",{key:sec.label},
        h("div",{className:"sidebar-section"},sec.label),
        h("nav",null,sec.tabs.map(function(tb){return h("button",{key:tb.id,className:tab===tb.id?"active":"",onClick:function(){setTab(tb.id)}},
          h("span",{className:"icon"},tb.icon),tb.label)})))}),
      h("div",{className:"sidebar-spacer"}),
      _authToken&&h("div",{style:{padding:"4px 8px"}},h("button",{className:"sm",style:{width:"100%"},onClick:logout},"Sign Out")),
      h("div",{className:"sidebar-footer",style:{display:"flex",justifyContent:"space-between",alignItems:"center"}},"rexgraph agent",
        h("button",{className:"sm",onClick:function(){var r=document.documentElement;var d=r.dataset.theme==="dark"?"":"dark";r.dataset.theme=d;try{localStorage.setItem("rexgraph_theme",d)}catch(e){}}},"◐"))),
    h("div",{className:"mobile-nav"},ALL_TABS.map(function(tb){return h("button",{key:tb.id,className:tab===tb.id?"active":"",onClick:function(){setTab(tb.id)}},tb.label)})),
    h("div",{className:"main"},h("div",{className:"content"},h(Comp,null))))}

ReactDOM.render(h(App,null),document.getElementById("root"));
