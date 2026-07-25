var { useState, useMemo } = React;

var BG="#060810",C1="#0c0e18",C2="#10121e",BD="#181c2c",TX="#b8c8e0",SB="#7888a8",DM="#404860";
var AC="#4878d0",GD="#d0a030",TL="#30b878",RD="#d05858",PU="#9068c8",PK="#c070a0",CY="#40b0c0",OR="#e07030";
var MF="'IBM Plex Mono',monospace",HF="'Cormorant Garamond',serif",BF="'Source Sans 3',sans-serif";
var TYPECOLORS=["#30b878","#d0a030","#4878d0","#d05858","#9068c8","#c070a0","#40b0c0","#80b860","#e07030","#5588cc","#cc6644","#88aa44","#aa66aa","#44aaaa","#ccaa33","#7766cc"];
var RL={"source":"#9898b0","sink":"#6aba90","hub":"#e0a030","mediator":"#5b8af0","peripheral":"#6a6a80"};
var _D=/*__REX_DATA__*/null;

// Build type->color map dynamically from attributes
var TC=useMemo?{}:{};
(function(){
  var ta=(_D.meta.attributes||[]).find(function(a){return a.kind==="categorical";});
  if(ta)(ta.values||[]).forEach(function(v,i){TC[v]=TYPECOLORS[i%TYPECOLORS.length];});
  if(!Object.keys(TC).length)TC["edge"]=TYPECOLORS[0];
})();

function Ck({ok}){return <span style={{color:ok?TL:RD,fontWeight:700,fontSize:8}}>{ok?"[pass]":"[fail]"}</span>;}

var ZC="#e8e8f0";
function Bars({vals,color,label}){
  var mx=0;(vals||[]).forEach(function(v){if(Math.abs(v)>mx)mx=Math.abs(v);});if(!mx)mx=0.1;
  var bw=Math.max(3,Math.min(16,Math.floor(340/((vals||[]).length||1))));
  return(<div style={{marginBottom:6}}>
    <div style={{fontSize:7.5,color:SB,marginBottom:3,fontFamily:BF,fontWeight:600}}>{label}</div>
    <div style={{display:"flex",gap:1,alignItems:"flex-end",height:24}}>
      {(vals||[]).map(function(v,i){var h=Math.max(2,Math.abs(v)/mx*20);var z=Math.abs(v)<1e-4;
        return <div key={i} style={{width:bw,height:z?4:h,borderRadius:"1px 1px 0 0",background:z?ZC:color+"50",border:"1px solid "+(z?ZC+"90":color+"30")}}/>;
      })}
    </div>
    <div style={{fontSize:6,color:DM,marginTop:2}}>zero eigenvalues: {(vals||[]).filter(function(v){return Math.abs(v)<1e-4;}).length} of {(vals||[]).length}</div>
  </div>);
}

function Tb({active,color,onClick,children}){
  return <button onClick={onClick} style={{padding:"2px 6px",borderRadius:3,cursor:"pointer",fontSize:7,fontFamily:MF,fontWeight:active?700:400,background:active?(color||AC)+"12":"transparent",border:"1px solid "+(active?(color||AC)+"40":BD),color:active?(color||AC):DM,whiteSpace:"nowrap"}}>{children}</button>;
}

function Info({title,children}){
  var [open,setOpen]=useState(false);
  return(<div style={{marginBottom:3}}>
    <div onClick={function(){setOpen(!open);}} style={{cursor:"pointer",fontSize:7,color:AC,fontFamily:MF,display:"flex",alignItems:"center",gap:3,userSelect:"none"}}>
      <span style={{fontFamily:"monospace",fontSize:9,width:12}}>{open?"-":"+"}</span>{title}
    </div>
    {open&&<div style={{fontSize:7,color:SB,lineHeight:1.55,padding:"4px 8px 4px 20px",borderLeft:"2px solid "+BD,fontFamily:BF}}>{children}</div>}
  </div>);
}

function Stat({label,value,color}){
  return <div style={{flex:1,padding:2,borderRadius:3,textAlign:"center",background:(color||GD)+"06",border:"1px solid "+(color||GD)+"12",minWidth:50}}>
    <div style={{fontSize:12,fontWeight:700,color:color||GD,fontFamily:HF}}>{value}</div>
    <div style={{fontSize:5.5,color:DM}}>{label}</div>
  </div>;
}

function App(){
  var [tab,setTab]=useState("graph");
  var [sel,setSel]=useState(null);
  var [hov,setHov]=useState(null);
  var [view,setView]=useState("structure");
  var [part,setPart]=useState("type");
  var [hodge,setHodge]=useState("flow");
  var [dim,setDim]=useState(2);
  var [search,setSearch]=useState("");
  var [rexView,setRexView]=useState("full");
  var [specPage,setSpecPage]=useState({});
  var [showExport,setShowExport]=useState(false);

  function doExport(mode){
    var out;
    if(mode==="full"){
      out=JSON.stringify(_D,null,2);
    } else if(mode==="compact"){
      out=JSON.stringify(_D);
    } else if(mode==="summary"){
      // Analysis + spectra + topology only, no per-element arrays
      var s={meta:_D.meta,topology:_D.topology,coupling:_D.coupling,spectra:_D.spectra,hodge:_D.hodge,analysis:_D.analysis};
      s.meta=Object.assign({},s.meta);delete s.meta.attributes;
      out=JSON.stringify(s,null,2);
    }
    var blob=new Blob([out],{type:"application/json"});
    var url=URL.createObjectURL(blob);
    var a=document.createElement("a");
    a.href=url;a.download="rex_"+mode+"_"+new Date().toISOString().slice(0,10)+".json";
    document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url);
    setShowExport(false);
  }

  var D=_D,T=D.topology,H=D.hodge,S=D.spectra,K=D.coupling,AN=D.analysis;
  var W=D.meta.svgW,HT=D.meta.svgH;
  var catAttr=(D.meta.attributes||[]).find(function(a){return a.kind==="categorical";});
  var numAttr=(D.meta.attributes||[]).find(function(a){return a.kind==="numeric";});
  var typeVals=catAttr?catAttr.values:["edge"];

  var VP=useMemo(function(){var m={};D.vertices.forEach(function(v){m[v.id]=[v.x,v.y];});return m;},[]);
  var curvature=useMemo(function(){
    var cv={};var pairs={};
    D.edges.forEach(function(e){var key=[e.source,e.target].sort().join("-");if(!pairs[key])pairs[key]=[];pairs[key].push(e.id);});
    Object.keys(pairs).forEach(function(key){var ids=pairs[key];if(ids.length>=2)ids.forEach(function(id,i){cv[id]=(i%2===0?14+i*4:-(14+i*4));});});
    D.edges.forEach(function(e){if(cv[e.id])return;var rev=D.edges.find(function(e2){return e2.source===e.target&&e2.target===e.source&&!cv[e2.id];});if(rev){cv[e.id]=12;cv[rev.id]=-12;}});
    return cv;
  },[]);

  var selE=D.edges.find(function(e){return e.id===sel;})||null;
  var selF=D.faces.find(function(f){return f.id===sel;})||null;
  var selV=D.vertices.find(function(v){return v.id===sel;})||null;
  var hEdges=[];
  if(selF)hEdges=Object.keys(selF.boundary);
  if(selV)hEdges=D.edges.filter(function(e){return e.source===selV.id||e.target===selV.id;}).map(function(e){return e.id;});
  function isHi(eid){return sel===eid||hEdges.indexOf(eid)>=0;}

  var CC_PAL=["#4878d0","#30b878","#d0a030","#d05858","#9068c8","#c070a0","#40b0c0","#e07030","#68a840","#b04878","#50a0d0","#d8a060","#8058b0","#38c098","#c86048","#7090d0","#a0c040","#d070d0","#4890a0","#e0a080","#6070c0","#90c070","#c05888","#d09030"];
  var maxPR=0;D.vertices.forEach(function(v2){if(v2.pagerank>maxPR)maxPR=v2.pagerank;});

  function pCol(item,isV){
    if(part==="none")return DM+"60";
    if(part==="fiedler-v"){if(isV)return item.partL0==="A"?AC:PU;var sv=D.vertices.find(function(v2){return v2.id===item.source;});var tv=D.vertices.find(function(v2){return v2.id===item.target;});if(sv&&tv&&sv.partL0===tv.partL0)return sv.partL0==="A"?AC+"90":PU+"90";return DM+"60";}
    if(part==="fiedler-e"){if(isV)return item.partLO==="A"?AC:PU;return item.partLO==="A"?AC:PU;}
    if(part==="composite"){if(isV){var inc=D.edges.filter(function(e2){return e2.source===item.id||e2.target===item.id;});if(!inc.length)return DM;var aa=inc.filter(function(e2){return e2.partL1a==="A";}).length;return aa>=inc.length-aa?OR:CY;}return item.partL1a==="A"?OR:CY;}
    if(part==="rho"){if(!isV){var r=item.rho||0;if(r>0.3)return PU;if(r>0.15)return PK;return DM+"50";}return null;}
    if(part==="role"){if(isV)return RL[item.role]||DM;return null;}
    if(part==="type"){if(!isV)return TC[item.type]||DM;return null;}
    if(part==="community"){if(isV)return CC_PAL[item.community%CC_PAL.length];var sv2=D.vertices.find(function(v2){return v2.id===item.source;});var tv2=D.vertices.find(function(v2){return v2.id===item.target;});if(sv2&&tv2&&sv2.community===tv2.community)return CC_PAL[sv2.community%CC_PAL.length]+"90";return DM+"40";}
    if(part==="pagerank"){if(isV){var t2=maxPR>0?item.pagerank/maxPR:0;if(t2>0.5)return RD;if(t2>0.2)return OR;if(t2>0.08)return GD;return DM+"50";}return null;}
    return null;
  }

  var showFlow=view!=="structure";
  function eVal(e){
    if(view==="gradient")return e.gradN;
    if(view==="divergence")return e.flowN;
    if(view==="hodge"){if(hodge==="flow")return e.flowN;if(hodge==="grad")return e.gradN;if(hodge==="curl")return e.curlN;if(hodge==="harm")return e.harmN;}
    return 0;
  }

  var fV=D.vertices,fE=D.edges;
  if(search){var q=search.toLowerCase();fV=D.vertices.filter(function(v){return v.id.toLowerCase().indexOf(q)>=0;});
    var vs=new Set(fV.map(function(v){return v.id;}));fE=D.edges.filter(function(e){return vs.has(e.source)||vs.has(e.target)||(e.type||"").toLowerCase().indexOf(q)>=0;});}

  function mkEdge(e){
    var s=VP[e.source],t=VP[e.target];if(!s||!t)return null;
    var hi=isHi(e.id),ih=hov===e.id;
    var col=pCol(e,false)||TC[e.type]||DM;
    var w2=0.8,op=0.35;
    if(showFlow){var val=eVal(e);w2=Math.max(0.5,Math.min(4.5,Math.abs(val)*3.5));op=Math.max(0.15,Math.min(0.9,0.15+Math.abs(val)*0.7));if(val<0)col=RD;else if(val>0)col=view==="hodge"&&hodge==="curl"?TL:(view==="hodge"&&hodge==="harm"?PU:AC);}
    if(part==="rho"&&!showFlow){var rr=e.rho||0;w2=Math.max(0.3,0.3+rr*5);op=Math.max(0.12,0.12+rr*1.5);col=pCol(e,false)||DM;}
    if(hi){w2=Math.max(w2,2.5);op=Math.max(op,0.9);}if(ih){w2=Math.max(w2,1.5);op=Math.max(op,0.65);}
    var cv=curvature[e.id]||0;var dx=t[0]-s[0],dy=t[1]-s[1],len=Math.sqrt(dx*dx+dy*dy);if(len<1)return null;
    var ux=dx/len,uy=dy/len;var x1=s[0]+ux*10,y1=s[1]+uy*10,x2=t[0]-ux*10,y2=t[1]-uy*10;
    return(<g key={e.id} onClick={function(){setSel(sel===e.id?null:e.id);}} onMouseEnter={function(){setHov(e.id);}} onMouseLeave={function(){setHov(null);}} style={{cursor:"pointer"}}>
      {cv===0?<line x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={w2} opacity={op}/>:
        <path d={"M "+x1+" "+y1+" Q "+((x1+x2)/2-uy*cv)+" "+((y1+y2)/2+ux*cv)+" "+x2+" "+y2} stroke={col} strokeWidth={w2} fill="none" opacity={op}/>}
      <polygon points={(x2)+","+(y2)+" "+(x2-ux*5-uy*2.5)+","+(y2-uy*5+ux*2.5)+" "+(x2-ux*5+uy*2.5)+","+(y2-uy*5-ux*2.5)} fill={col} opacity={op}/>
    </g>);
  }

  function mkVert(v){
    var is0=dim===0;
    var isBnd=is0&&rexView==="boundary";
    var isHyp=is0&&rexView==="hyperedge";
    var hi=sel===v.id||(selE&&(selE.source===v.id||selE.target===v.id));var ih=hov===v.id;
    var r=isBnd?Math.max(3,2+v.degree*0.35):(isHyp?Math.max(2,1.5+v.degree*0.2):Math.max(3,2+v.degree*0.45));
    var fc=BG,sc=DM,tc=SB;
    var pc=pCol(v,true);
    // In 0-rex, vertex coloring should always work for vertex-based modes
    if(is0){
      if(part==="fiedler-v"){sc=v.partL0==="A"?AC:PU;tc=sc;}
      else if(part==="fiedler-e"){sc=v.partLO==="A"?AC:PU;tc=sc;}
      else if(part==="composite"){var inc2=D.edges.filter(function(e2){return e2.source===v.id||e2.target===v.id;});if(inc2.length){var aa2=inc2.filter(function(e2){return e2.partL1a==="A";}).length;sc=aa2>=inc2.length-aa2?OR:CY;tc=sc;}else{sc=DM;tc=DM;}}
      else if(part==="rho"){var incR=D.edges.filter(function(e2){return e2.source===v.id||e2.target===v.id;});if(incR.length){var maxR=0;incR.forEach(function(e2){if(e2.rho>maxR)maxR=e2.rho;});if(maxR>0.3){sc=PU;tc=PU;}else if(maxR>0.15){sc=PK;tc=PK;}else{sc=DM+"80";tc=DM;}}else{sc=DM;tc=DM;}}
      else if(part==="role"){sc=RL[v.role]||DM;tc=sc;}
      else if(part==="type"){var incT=D.edges.filter(function(e2){return e2.source===v.id||e2.target===v.id;});if(incT.length){var tc2={};incT.forEach(function(e2){tc2[e2.type]=(tc2[e2.type]||0)+1;});var topType=Object.entries(tc2).sort(function(a,b){return b[1]-a[1];})[0];sc=TC[topType[0]]||DM;tc=sc;}else{sc=DM;tc=DM;}}
      else if(part==="none"){sc=DM+"60";tc=DM;}
      else if(part==="community"){sc=CC_PAL[v.community%CC_PAL.length];tc=sc;}
      else if(part==="pagerank"){var t3=maxPR>0?v.pagerank/maxPR:0;sc=t3>0.5?RD:(t3>0.2?OR:(t3>0.08?GD:DM+"50"));tc=sc;}
      if(isBnd){fc=sc+"12";}
    } else {
      if(pc&&pc!==DM&&pc!==(DM+"60")){sc=pc;tc=pc;}
    }
    if(showFlow&&view==="divergence"){var dv=v.divNorm||0;if(dv<-0.3){fc=AC+"25";sc=AC;tc=AC;}else if(dv>0.3){fc=RD+"25";sc=RD;tc=RD;}}
    if(hi||ih){fc=(sc!==DM?sc:AC)+"18";tc=sc!==DM?sc:AC;sc=tc;}
    var showLabel=hi||ih||v.degree>=5||isBnd;
    return(<g key={v.id} onClick={function(){setSel(sel===v.id?null:v.id);}} onMouseEnter={function(){setHov(v.id);}} onMouseLeave={function(){setHov(null);}} style={{cursor:"pointer"}}>
      <circle cx={v.x} cy={v.y} r={r} fill={fc} stroke={sc} strokeWidth={hi?1.8:(isBnd?0.8:0.6)}/>
      {showLabel&&<text x={v.x} y={v.y-r-2.5} textAnchor="middle" fill={tc} fontSize={hi?7:(isBnd?6:5.5)} fontWeight={700} fontFamily={MF}>{v.id}</text>}
    </g>);
  }

  function mkHyperedge(e){
    var coords=[[e.source,VP[e.source]],[e.target,VP[e.target]]].filter(function(p){return p[1];});
    if(coords.length<2)return null;
    var cx2=0,cy2=0;coords.forEach(function(p){cx2+=p[1][0];cy2+=p[1][1];});cx2/=coords.length;cy2/=coords.length;
    var hi=isHi(e.id);var ih=hov===e.id;
    // Color based on mode
    var col=DM;
    if(part==="type")col=TC[e.type]||DM;
    else if(part==="fiedler-e")col=e.partLO==="A"?AC:PU;
    else if(part==="composite")col=e.partL1a==="A"?OR:CY;
    else if(part==="rho"){var rr=e.rho||0;col=rr>0.3?PU:(rr>0.15?PK:DM+"80");}
    else if(part==="none")col=DM+"60";
    else if(part==="fiedler-v"){var sv2=D.vertices.find(function(v2){return v2.id===e.source;});var tv2=D.vertices.find(function(v2){return v2.id===e.target;});if(sv2&&tv2&&sv2.partL0===tv2.partL0)col=sv2.partL0==="A"?AC:PU;else col=DM+"60";}
    else col=TC[e.type]||DM;
    var baseOp=hi?0.7:(ih?0.5:0.25);
    var baseW=hi?1.8:(ih?1.2:0.6);
    return(<g key={e.id+"h"} onClick={function(){setSel(sel===e.id?null:e.id);}} onMouseEnter={function(){setHov(e.id);}} onMouseLeave={function(){setHov(null);}} style={{cursor:"pointer"}}>
      {coords.map(function(p,i){return <line key={i} x1={cx2} y1={cy2} x2={p[1][0]} y2={p[1][1]} stroke={col} strokeWidth={baseW} opacity={baseOp} strokeDasharray="2 2"/>;})
      }
      <circle cx={cx2} cy={cy2} r={hi?4.5:3} fill={col} fillOpacity={hi?0.35:0.12} stroke={col} strokeWidth={hi?1:0.5} opacity={hi?0.8:0.5}/>
    </g>);
  }

  var svg=<svg width="100%" viewBox={"0 0 "+W+" "+HT} style={{display:"block",background:C1,borderRadius:6}}>
    {dim>=2&&D.faces.map(function(f){
      var hi2=sel===f.id;
      var pts=f.vertices.map(function(vid){return VP[vid];}).filter(Boolean);
      if(pts.length===0)return null;
      if(pts.length===1){
        return <circle key={f.id} cx={pts[0][0]} cy={pts[0][1]} r={hi2?6:3} fill={hi2?GD:"none"} fillOpacity={hi2?0.15:0} stroke={hi2?GD:BD} strokeWidth={hi2?1.2:0.5} strokeOpacity={hi2?0.6:0.25} strokeDasharray="2 2" onClick={function(){setSel(hi2?null:f.id);}} style={{cursor:"pointer"}}/>;
      }
      if(pts.length===2){
        return <line key={f.id} x1={pts[0][0]} y1={pts[0][1]} x2={pts[1][0]} y2={pts[1][1]} stroke={hi2?GD:BD} strokeWidth={hi2?2.5:1.5} strokeOpacity={hi2?0.6:0.25} strokeDasharray="4 2" onClick={function(){setSel(hi2?null:f.id);}} style={{cursor:"pointer"}}/>;
      }
      var cx2=0,cy2=0;pts.forEach(function(p){cx2+=p[0];cy2+=p[1];});cx2/=pts.length;cy2/=pts.length;
      pts.sort(function(a,b){return Math.atan2(a[1]-cy2,a[0]-cx2)-Math.atan2(b[1]-cy2,b[0]-cx2);});
      return <polygon key={f.id} points={pts.map(function(p){return p[0]+","+p[1];}).join(" ")} fill={hi2?GD:GD} fillOpacity={hi2?0.12:0.03} stroke={hi2?GD:GD} strokeWidth={hi2?1.2:0.6} strokeOpacity={hi2?0.7:0.25} strokeDasharray="3 2" onClick={function(){setSel(hi2?null:f.id);}} style={{cursor:"pointer"}}/>;
    })}
    {dim===0&&rexView==="hyperedge"&&fE.map(mkHyperedge)}
    {dim>=1&&fE.map(mkEdge)}
    {fV.map(mkVert)}
  </svg>;

  var detail=null;
  if(selE){detail=<div style={{padding:"5px 8px",borderRadius:4,background:C2,border:"1px solid "+BD,marginBottom:4,fontSize:6.5,fontFamily:MF,color:TX,lineHeight:1.7}}>
    <div><span style={{fontWeight:700,color:TC[selE.type]||AC}}>{selE.id}</span>{"  "}{selE.source}{" -> "}{selE.target}{"  "}<span style={{color:DM}}>({selE.type}, w={selE.w})</span></div>
    <div>Partitions: L0={D.vertices.find(function(v2){return v2.id===selE.source;})?D.vertices.find(function(v2){return v2.id===selE.source;}).partL0:"?"} L_O={selE.partLO} L1(a)={selE.partL1a} | rho={selE.rho}</div>
    <div>L1 coupling: down={selE.L1down} (shared vertices) up={selE.L1up} (shared faces)</div>
    {selE.L1up>0&&<div>Face: avgContrib={selE.faceAvgC} totalContrib={selE.faceTotC} avgSize={selE.faceAvgSz} | bndAsym={selE.bndAsym}</div>}
    <div>Edge betweenness: {selE.eBetw} (norm={selE.eBtwNorm})</div>
    {showFlow&&<div>Hodge: <span style={{color:GD}}>grad={selE.gradRaw}</span> + <span style={{color:TL}}>curl={selE.curlRaw}</span> + <span style={{color:PU}}>harm={selE.harmRaw}</span></div>}
    {Object.keys(selE).filter(function(k){return k.indexOf("meta_")===0;}).length>0&&<div style={{marginTop:2,borderTop:"1px solid "+BD,paddingTop:2}}>
      {Object.keys(selE).filter(function(k){return k.indexOf("meta_")===0;}).map(function(k){return <div key={k} style={{color:SB}}><span style={{color:DM}}>{k.slice(5)}:</span> {selE[k]}</div>;})}
    </div>}
  </div>;}
  if(selF){var bk=Object.keys(selF.boundary);detail=<div style={{padding:"5px 8px",borderRadius:4,background:C2,border:"1px solid "+BD,marginBottom:4,fontSize:6.5,fontFamily:MF,color:TX,lineHeight:1.7}}>
    <div><span style={{fontWeight:700,color:GD}}>{selF.id}</span>{"  "}{selF.size}-cycle | vertices: {selF.vertices.join(", ")}</div>
    <div>Boundary: [{bk.map(function(k2){return(selF.boundary[k2]>0?"+":"-")+k2;}).join(", ")}] | curl={selF.curl}</div>
  </div>;}
  if(selV){detail=<div style={{padding:"5px 8px",borderRadius:4,background:C2,border:"1px solid "+BD,marginBottom:4,fontSize:6.5,fontFamily:MF,color:TX,lineHeight:1.7}}>
    <div><span style={{fontWeight:700,color:RL[selV.role]||AC}}>{selV.id}</span>{"  "}{selV.role} | degree {selV.degree} ({selV.inDeg}in {selV.outDeg}out) | in {selV.faceCount} faces</div>
    <div>L0={selV.partL0} L_O={selV.partLO} Fiedler={selV.fiedler} | div={selV.divergence}</div>
    {selV.faceCount>0&&<div>Face structure: avgContrib={selV.faceAvgC} totalContrib={selV.faceTotC} avgFaceSize={selV.faceAvgSz}</div>}
    <div>Standard: PR={selV.pagerank} btw={selV.betweenness} clust={selV.clustering} community=C{selV.community}</div>
  </div>;}

  // Build analysis text from data
  // ── Automated analysis text (data-adaptive) ──

  var anTopology=(function(){
    var c=AN.topology.components, cy=AN.topology.cycles_total, cf=AN.topology.cycles_filled, cu=AN.topology.cycles_unfilled, vo=AN.topology.voids;
    var s=c+" connected component"+(c!==1?"s":"")+", "+D.meta.nV+" vertices, "+D.meta.nE+" edges. ";
    if(cy===0) s+="Acyclic graph (tree"+(c===1?"":"-forest")+") - no independent cycles.";
    else{
      s+=cy+" independent cycle"+(cy!==1?"s":"")+". ";
      if(cf===cy) s+="All "+cf+" promoted to faces (fully filled).";
      else if(cf===0) s+="No cycles filled - pure 1-complex.";
      else s+=cf+" of "+cy+" filled ("+AN.topology.fill_rate+"%), "+cu+" remain unfilled.";
      if(vo>0) s+=" "+vo+" void"+(vo!==1?"s":"")+" detected (higher-dimensional holes).";
    }
    return s;
  })();

  var anHodge=(function(){
    var g=AN.hodge.gradient_pct, c=AN.hodge.curl_pct, h=AN.hodge.harmonic_pct;
    var s=AN.hodge.dominant+"-dominated signal: "+g+"% gradient, "+c+"% curl, "+h+"% harmonic. ";
    if(g>80) s+="Strongly feed-forward - signal flows along vertex potentials with little recirculation.";
    else if(g>60) s+="Primarily feed-forward with moderate feedback loops.";
    else if(Math.abs(g-c)<15) s+="Near-balanced gradient/curl - signal both flows forward and recirculates through cycles.";
    else if(c>60) s+="Curl-dominated - most signal recirculates through feedback loops.";
    else s+="Mixed signal structure.";
    if(h>10) s+=" Substantial harmonic component ("+h+"%) indicates topologically protected signal on "+AN.topology.cycles_unfilled+" unfilled cycle"+(AN.topology.cycles_unfilled!==1?"s":"")+".";
    else if(h>0&&h<=10) s+=" Small harmonic component ("+h+"%) on "+AN.topology.cycles_unfilled+" unfilled cycle"+(AN.topology.cycles_unfilled!==1?"s":".")+".";
    else if(T.b1_filled===0) s+=" No harmonic component: all cycles are filled, so no topologically sustained signal exists.";
    return s;
  })();

  var anCoupling=(function(){
    var aG=AN.coupling.alpha_G, aT=AN.coupling.alpha_T;
    var s="aG = "+aG.toFixed(3);
    if(aG>4) s+=" (strongly geometry-dominated - overlap structure dominates chain complex)";
    else if(aG>2) s+=" (geometry-dominated - edge overlap couples more strongly than topology)";
    else if(aG>0.5) s+=" (near-balanced - geometric and topological coupling are comparable)";
    else if(aG>0.25) s+=" (topology-dominated - chain complex structure dominates overlap)";
    else s+=" (strongly topology-dominated)";
    s+=". aT = "+aT.toFixed(3);
    if(aT===0) s+=" (no unfilled cycles - all topological structure is captured by faces).";
    else s+=" ("+Math.round(aT*100)+"% of edge space remains in unfilled cycles).";
    return s;
  })();

  var anPart=(function(){
    var L0A=AN.partitions.L0_split[0], L0B=AN.partitions.L0_split[1];
    var LOA=AN.partitions.LO_split[0], LOB=AN.partitions.LO_split[1];
    var agree=AN.partitions.vertex_agreement, nV=D.meta.nV;
    var pct=nV>0?Math.round(agree/nV*100):0;
    var s="L0 Fiedler splits vertices "+L0A+":"+L0B+". L_O Fiedler splits edges "+LOA+":"+LOB+". ";
    s+="Projected to vertices, these agree on "+agree+"/"+nV+" ("+pct+"%). ";
    if(pct>=95) s+="Near-identical partitions: vertex connectivity and edge overlap see essentially the same large-scale structure.";
    else if(pct>=80) s+="High agreement: the two spectral views are broadly consistent with moderate local differences.";
    else if(pct>=60) s+="Partial agreement: vertex connectivity and edge overlap reveal partially independent structure.";
    else s+="Low agreement: L0 and L_O capture substantially different structural features - geometry and topology partition the graph differently.";
    return s;
  })();

  var FS=AN.face_structure||{};
  var anFaces=(function(){
    if(D.meta.nF===0) return "No faces detected. The complex is a pure 1-skeleton (graph only).";
    var s=D.meta.nF+" faces, average size "+FS.mean_face_size+" edges. ";
    s+="Mean boundary asymmetry "+FS.mean_asym.toFixed(2);
    if(FS.mean_asym>0.7) s+=" (high - faces tend to connect hub vertices to peripheral ones)";
    else if(FS.mean_asym>0.4) s+=" (moderate - mixed hub-peripheral and balanced boundaries)";
    else s+=" (low - face boundaries connect equally-embedded vertices)";
    s+=". ";
    if(T.b1_filled>0){
      s+="Asym-rho correlation r="+FS.asym_rho_corr.toFixed(3);
      if(FS.asym_rho_corr<-0.2) s+=" (negative: balanced edges carry more harmonic signal, as expected).";
      else if(FS.asym_rho_corr>0.2) s+=" (positive: unexpected - asymmetric edges carry harmonic signal).";
      else s+=" (weak: boundary asymmetry does not strongly predict harmonic content).";
    } else {
      s+="With B1=0, rho is identically zero on all edges (no harmonic signal to correlate).";
    }
    var top=FS.top_v_by_total_contrib;
    if(top&&top[0]) s+=" Top face contributor: "+top[0].id+" ("+top[0].faceCount+" faces, totC="+top[0].faceTotC+").";
    return s;
  })();

  var SM=AN.standard_metrics||{};
  var anStandard=(function(){
    var s="";
    if(SM.top_pagerank&&SM.top_pagerank[0]){
      var pr=SM.top_pagerank[0];
      s+="PageRank top: "+pr.id+" ("+((pr.pr||0)*100).toFixed(2)+"%, deg "+pr.degree+")";
      if(SM.pr_deg_corr>0.9) s+=" - strong PR-degree correlation (r="+SM.pr_deg_corr+") indicates scale-free flow accumulation at hubs.";
      else if(SM.pr_deg_corr>0.7) s+=" - moderate PR-degree correlation (r="+SM.pr_deg_corr+").";
      else s+=" - weak PR-degree correlation (r="+SM.pr_deg_corr+") - flow accumulates independently of degree.";
    }
    if(SM.top_betweenness&&SM.top_betweenness[0]){
      var btw=SM.top_betweenness[0];
      s+=" Betweenness top: "+btw.id+" (btw="+btw.btw+")";
      if(btw.id!==(SM.top_pagerank&&SM.top_pagerank[0]?SM.top_pagerank[0].id:"")) s+=" - different from PR top, indicating distinct flow vs. bridging roles.";
      else s+=".";
    }
    if(SM.top_clustering&&SM.top_clustering[0]){
      var cl=SM.top_clustering[0];
      s+=" Clustering top: "+cl.id+" (CC="+cl.cc+", deg "+cl.degree+")";
      if(cl.degree<=3) s+=" - high clustering with low degree indicates locally dense triadic closure.";
      else s+=" - high clustering at a hub vertex indicates a densely connected neighborhood.";
    }
    s+=" Louvain: "+SM.n_communities+" communities (Q="+SM.modularity+").";
    if(SM.modularity>0.5) s+=" Strong modular structure.";
    else if(SM.modularity>0.3) s+=" Moderate modularity.";
    else s+=" Weak modular structure.";
    return s;
  })();

  return(
    <div style={{background:BG,color:TX,padding:"14px 12px",fontFamily:BF,minHeight:"100vh",maxWidth:740,margin:"0 auto"}}>
      <div style={{marginBottom:10,borderBottom:"1px solid "+BD,paddingBottom:8,display:"flex",alignItems:"flex-end",justifyContent:"space-between"}}>
        <div>
          <div style={{fontSize:16,fontWeight:700,fontFamily:HF,color:"#d0d8e8",letterSpacing:"0.02em"}}>Relational Complex Visualization Suite</div>
          <div style={{fontSize:9,color:SB,marginTop:2}}>{D.meta.nV} vertices, {D.meta.nE} edges, {D.meta.nF} faces</div>
        </div>
        <div style={{position:"relative"}}>
          <button onClick={function(){setShowExport(!showExport);}} style={{padding:"3px 8px",borderRadius:3,fontSize:7,fontFamily:MF,cursor:"pointer",background:C2,border:"1px solid "+BD,color:SB,whiteSpace:"nowrap"}}>Export ▾</button>
          {showExport&&<div style={{position:"absolute",right:0,top:20,background:C1,border:"1px solid "+BD,borderRadius:4,padding:4,zIndex:10,minWidth:160}}>
            <div onClick={function(){doExport("summary");}} style={{padding:"3px 6px",fontSize:7,fontFamily:MF,color:TL,cursor:"pointer",borderRadius:2}} onMouseEnter={function(e){e.target.style.background=BD;}} onMouseLeave={function(e){e.target.style.background="transparent";}}>Summary JSON<br/><span style={{color:DM,fontSize:6}}>Analysis + spectra only (smallest)</span></div>
            <div onClick={function(){doExport("compact");}} style={{padding:"3px 6px",fontSize:7,fontFamily:MF,color:AC,cursor:"pointer",borderRadius:2,marginTop:2}} onMouseEnter={function(e){e.target.style.background=BD;}} onMouseLeave={function(e){e.target.style.background="transparent";}}>Compact JSON<br/><span style={{color:DM,fontSize:6}}>All data, minified</span></div>
            <div onClick={function(){doExport("full");}} style={{padding:"3px 6px",fontSize:7,fontFamily:MF,color:GD,cursor:"pointer",borderRadius:2,marginTop:2}} onMouseEnter={function(e){e.target.style.background=BD;}} onMouseLeave={function(e){e.target.style.background="transparent";}}>Full JSON<br/><span style={{color:DM,fontSize:6}}>All data, formatted (largest)</span></div>
          </div>}
        </div>
      </div>

      <div style={{display:"flex",gap:3,marginBottom:8}}>
        {["graph","calculus","faces","spectra","analysis","verify"].map(function(t){return <Tb key={t} active={tab===t} onClick={function(){setTab(t);setSel(null);setView("structure");}}>{t.charAt(0).toUpperCase()+t.slice(1)}</Tb>;})}
      </div>

      {tab==="graph"&&<div>
        <Info title="About this view"><span>
          A relational complex (rex) built from {D.meta.nE} directed edges. Vertices are derived from edge boundaries. Three layers: 0-rex (boundary vertices only), 1-rex (+edges), 2-rex (+faces: auto-discovered cycles shown as dashed polygons). Click any element for details.
        </span></Info>
        <div style={{display:"flex",gap:2,marginBottom:4,alignItems:"center"}}>
          <Tb active={dim===2} color={GD} onClick={function(){setDim(2);setRexView("full");}}>2-rex</Tb>
          <Tb active={dim===1} color={TL} onClick={function(){setDim(1);setRexView("full");}}>1-rex</Tb>
          <Tb active={dim===0} color={PU} onClick={function(){setDim(0);setRexView("boundary");}}>0-rex</Tb>
          {dim===0&&<><Tb active={rexView==="boundary"} color={PU} onClick={function(){setRexView("boundary");}}>Boundary</Tb>
            <Tb active={rexView==="hyperedge"} color={PU} onClick={function(){setRexView("hyperedge");}}>Hyperedge</Tb></>}
          <div style={{flex:1}}/>
          <input value={search} onChange={function(e){setSearch(e.target.value);}} placeholder="search..." style={{background:C2,border:"1px solid "+BD,borderRadius:3,padding:"2px 5px",color:TX,fontSize:7,fontFamily:MF,width:90}}/>
        </div>
        <div style={{display:"flex",gap:2,marginBottom:3,flexWrap:"wrap",alignItems:"center"}}>
          <span style={{fontSize:6,color:DM,fontFamily:MF}}>Color:</span>
          {[["none","Uniform"],["fiedler-v","Fiedler(L0)"],["fiedler-e","Fiedler(L_O)"],["composite","Composite"],["rho","Harmonic"],["role","Role"],["type","Type"],["community","Community"],["pagerank","PageRank"]].map(function(p){
            return <Tb key={p[0]} active={part===p[0]} onClick={function(){setPart(p[0]);}}>{p[1]}</Tb>;
          })}
        </div>
        <Info title="Color mode reference"><span>
          <b>Uniform</b>: neutral gray, no structural information.<br/>
          <b>Fiedler(L0)</b>: bipartition of the vertex graph Laplacian. Groups vertices by connectivity. Edges inherit when both endpoints match.<br/>
          <b>Fiedler(L_O)</b>: bipartition of the overlap Laplacian. Groups edges by shared boundary vertices (Jaccard). Detects modules of tightly-coupled interactions.<br/>
          <b>Composite</b>: bipartition of L1 + aG*L_O, combining topological (chain complex) and geometric (overlap) structure.<br/>
          <b>Harmonic</b>: per-edge rho = |harmonic|/|total|. Bright = high rho (irreducible topological signal). Dim = low rho (explainable by local gradients).<br/>
          <b>Role</b>: vertex classification. Source (no in-edges), sink (no out-edges), hub (many faces), mediator (some faces), peripheral.<br/>
          <b>Type</b>: edge attribute "{catAttr?catAttr.name:"type"}". Each value gets a distinct color.<br/>
          <b>Community</b>: Louvain modularity communities ({SM.n_communities} detected, Q={SM.modularity}). Vertices colored by community. Edges colored when both endpoints share a community, gray when cross-community.<br/>
          <b>PageRank</b>: directed flow accumulation. Red=highest PR (flow converges here), orange=moderate, gold=above average, gray=low. Note: high PR does not imply high degree - it captures where directed signal accumulates at sinks.
        </span></Info>
        <div style={{display:"flex",gap:2,marginBottom:4,flexWrap:"wrap"}}>
          <Stat label="B0" value={T.b0}/>
          <Stat label="B1" value={T.b1_filled}/>
          <Stat label="B2" value={T.b2}/>
          <Stat label="Euler" value={T.euler}/>
          <Stat label="aG" value={K.alpha_G.toFixed(2)} color={OR}/>
        </div>
        {detail}{svg}
        <Info title={"Edge types ("+typeVals.length+")"}>
          <div style={{display:"flex",gap:5,flexWrap:"wrap"}}>
            {typeVals.map(function(tv){return <div key={tv} style={{display:"flex",alignItems:"center",gap:2}}>
              <div style={{width:6,height:6,borderRadius:1,background:TC[tv]||DM}}/><span style={{fontSize:6.5,color:SB}}>{tv}{catAttr?" ("+catAttr.counts[tv]+")":""}</span>
            </div>;})}
          </div>
        </Info>
        {numAttr&&<Info title={"Weight: "+numAttr.name}>
          <span style={{fontSize:6.5}}>Range [{numAttr.min}, {numAttr.max}], mean {numAttr.mean}. Used as edge signal magnitude.</span>
        </Info>}
        <Info title={"Vertex roles ("+D.meta.nV+" total)"}>
          <div style={{display:"flex",gap:5,flexWrap:"wrap"}}>
            {Object.entries(AN.structure.roles).map(function(r2){return <div key={r2[0]} style={{display:"flex",alignItems:"center",gap:2}}>
              <div style={{width:6,height:6,borderRadius:"50%",background:RL[r2[0]]||DM}}/><span style={{fontSize:6.5,color:SB}}>{r2[0]} ({r2[1]})</span>
            </div>;})}
          </div>
        </Info>
      </div>}

      {tab==="calculus"&&<div>
        <Info title="About discrete calculus"><span>
          Three operations defined via boundary maps B1 (vertex-edge) and B2 (edge-face):<br/>
          <b>Gradient</b>: grad(f) = B1^T f. Each edge gets the difference in vertex potential. Feed-forward signal.<br/>
          <b>Divergence</b>: div(w) = B1*w. Net flow at each vertex. Blue = source, red = sink.<br/>
          <b>Hodge decomposition</b>: any edge signal = gradient + curl + harmonic (orthogonal). Gradient is from vertex potentials. Curl circulates around faces. Harmonic lives on unfilled cycles.
        </span></Info>
        <div style={{display:"flex",gap:2,marginBottom:4}}>
          <Tb active={view==="gradient"} color={GD} onClick={function(){setView("gradient");}}>Gradient</Tb>
          <Tb active={view==="divergence"} color={AC} onClick={function(){setView("divergence");}}>Divergence</Tb>
          <Tb active={view==="hodge"} color={PU} onClick={function(){setView("hodge");}}>Hodge</Tb>
        </div>
        {view==="hodge"&&<div>
          <div style={{display:"flex",gap:2,marginBottom:3}}>
            <Tb active={hodge==="flow"} color={TX} onClick={function(){setHodge("flow");}}>Full</Tb>
            <Tb active={hodge==="grad"} color={GD} onClick={function(){setHodge("grad");}}>Gradient</Tb>
            <Tb active={hodge==="curl"} color={TL} onClick={function(){setHodge("curl");}}>Curl</Tb>
            <Tb active={hodge==="harm"} color={H.harmPct<1?DM:PU} onClick={function(){setHodge("harm");}}>{H.harmPct<1?"Harmonic (0%)":"Harmonic"}</Tb>
          </div>
          <Info title="Hodge component details"><span>
            Each component is normalized to [-1,1] relative to its own maximum for visibility. Actual energy proportions are shown in the bars below. {hodge==="harm"?(H.harmPct<1?"Harmonic energy is 0% (B1="+T.b1_filled+"). All cycles are filled, so there is no harmonic signal. The view is blank because the raw harmonic component is zero (floating-point noise has been suppressed).":"The harmonic component ("+H.harmPct+"% of energy) is now fully visible because it is self-normalized. Bright purple = strong harmonic signal on that edge."):""}
          </span></Info>
          <div style={{display:"flex",gap:2,marginBottom:4}}>
            <Stat label="Gradient" value={H.gradPct+"%"} color={GD}/>
            <Stat label="Curl" value={H.curlPct+"%"} color={TL}/>
            <Stat label="Harmonic" value={H.harmPct+"%"} color={PU}/>
          </div>
          <Info title={"Edge Hodge data ("+D.meta.nE+" edges)"}>
            <div style={{display:"grid",gridTemplateColumns:"2fr repeat(5,1fr)",gap:"1px 4px",fontSize:6,fontFamily:MF}}>
              <div style={{color:DM,fontWeight:600}}>Edge</div><div style={{color:GD,fontWeight:600}}>Grad</div><div style={{color:TL,fontWeight:600}}>Curl</div><div style={{color:PU,fontWeight:600}}>Harm</div><div style={{color:PK,fontWeight:600}}>Rho</div><div style={{color:TX,fontWeight:600}}>Flow</div>
              {D.edges.slice().sort(function(a,b){return hodge==="grad"?Math.abs(b.gradN)-Math.abs(a.gradN):(hodge==="curl"?Math.abs(b.curlN)-Math.abs(a.curlN):(hodge==="harm"?b.rho-a.rho:Math.abs(b.flowN)-Math.abs(a.flowN)));}).map(function(e2){return [
                <div key={e2.id+"a"} style={{color:SB,cursor:"pointer"}} onClick={function(){setSel(e2.id);setTab("graph");}}>{e2.source+">"+e2.target}</div>,
                <div key={e2.id+"b"} style={{color:GD}}>{e2.gradN.toFixed(3)}</div>,
                <div key={e2.id+"c"} style={{color:TL}}>{e2.curlN.toFixed(3)}</div>,
                <div key={e2.id+"d"} style={{color:PU}}>{e2.harmN.toFixed(3)}</div>,
                <div key={e2.id+"e"} style={{color:e2.rho>0.3?PU:(e2.rho>0.15?PK:DM)}}>{e2.rho.toFixed(3)}</div>,
                <div key={e2.id+"f"} style={{color:TX}}>{e2.flow.toFixed(3)}</div>
              ];})}
            </div>
          </Info>
          {D.meta.nF>0&&<Info title={"Face curl values ("+D.meta.nF+" faces)"}>
            <div style={{display:"grid",gridTemplateColumns:"auto 1fr auto auto",gap:"1px 4px",fontSize:6,fontFamily:MF}}>
              <div style={{color:DM,fontWeight:600}}>Face</div><div style={{color:DM,fontWeight:600}}>Vertices</div><div style={{color:TL,fontWeight:600}}>Curl</div><div style={{color:OR,fontWeight:600}}>CV</div>
              {D.faces.slice().sort(function(a,b){return Math.abs(b.curl)-Math.abs(a.curl);}).map(function(f2){return [
                <div key={f2.id+"a"} style={{color:GD}}>{f2.id}</div>,
                <div key={f2.id+"b"} style={{color:SB}}>{f2.vertices.join(", ")}</div>,
                <div key={f2.id+"c"} style={{color:f2.curl>0?TL:RD}}>{f2.curl.toFixed(4)}</div>,
                <div key={f2.id+"d"} style={{color:f2.conc>0.5?OR:CY}}>{f2.conc.toFixed(3)}</div>
              ];})}
            </div>
          </Info>}
        </div>}
        {view==="gradient"&&<div>
          <div style={{fontSize:7,color:SB,marginBottom:4,padding:"3px 6px",borderRadius:3,background:GD+"06",border:"1px solid "+GD+"10"}}>
            Edge color/width shows normalized gradient component. Blue=positive, red=negative. Self-normalized to max gradient value.
          </div>
          <Info title={"Gradient values ("+D.meta.nE+" edges)"}>
            <div style={{display:"grid",gridTemplateColumns:"2fr 1fr 1fr 1fr",gap:"1px 4px",fontSize:6,fontFamily:MF}}>
              <div style={{color:DM,fontWeight:600}}>Edge</div><div style={{color:GD,fontWeight:600}}>GradRaw</div><div style={{color:GD,fontWeight:600}}>GradNorm</div><div style={{color:TX,fontWeight:600}}>Flow</div>
              {D.edges.slice().sort(function(a,b){return Math.abs(b.gradN)-Math.abs(a.gradN);}).map(function(e2){return [
                <div key={e2.id+"a"} style={{color:SB,cursor:"pointer"}} onClick={function(){setSel(e2.id);setTab("graph");}}>{e2.source+">"+e2.target}</div>,
                <div key={e2.id+"b"} style={{color:GD}}>{e2.gradRaw.toFixed(4)}</div>,
                <div key={e2.id+"c"} style={{color:GD}}>{e2.gradN.toFixed(3)}</div>,
                <div key={e2.id+"d"} style={{color:TX}}>{e2.flow.toFixed(3)}</div>
              ];})}
            </div>
          </Info>
        </div>}
        {view==="divergence"&&<div>
          <div style={{fontSize:7,color:SB,marginBottom:4,padding:"3px 6px",borderRadius:3,background:AC+"06",border:"1px solid "+AC+"10"}}>
            Vertex color shows normalized divergence. Blue=net source, red=net sink. Edge width shows flow magnitude.
          </div>
          <Info title={"Divergence values ("+D.meta.nV+" vertices)"}>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:"1px 4px",fontSize:6,fontFamily:MF}}>
              <div style={{color:DM,fontWeight:600}}>Vertex</div><div style={{color:AC,fontWeight:600}}>Div</div><div style={{color:AC,fontWeight:600}}>DivNorm</div><div style={{color:DM,fontWeight:600}}>Role</div>
              {D.vertices.slice().sort(function(a,b){return Math.abs(b.divergence)-Math.abs(a.divergence);}).map(function(v2){return [
                <div key={v2.id+"a"} style={{color:SB,cursor:"pointer"}} onClick={function(){setSel(v2.id);setTab("graph");}}>{v2.id}</div>,
                <div key={v2.id+"b"} style={{color:v2.divergence>0?AC:RD}}>{v2.divergence.toFixed(4)}</div>,
                <div key={v2.id+"c"} style={{color:v2.divNorm>0?AC:RD}}>{v2.divNorm.toFixed(3)}</div>,
                <div key={v2.id+"d"} style={{color:RL[v2.role]||DM}}>{v2.role}</div>
              ];})}
            </div>
          </Info>
          <Info title={"Edge flow magnitudes ("+D.meta.nE+" edges)"}>
            <div style={{display:"grid",gridTemplateColumns:"2fr 1fr 1fr 1fr",gap:"1px 4px",fontSize:6,fontFamily:MF}}>
              <div style={{color:DM,fontWeight:600}}>Edge</div><div style={{color:TX,fontWeight:600}}>Flow</div><div style={{color:TX,fontWeight:600}}>FlowNorm</div><div style={{color:DM,fontWeight:600}}>Type</div>
              {D.edges.slice().sort(function(a,b){return Math.abs(b.flow)-Math.abs(a.flow);}).map(function(e2){return [
                <div key={e2.id+"a"} style={{color:SB,cursor:"pointer"}} onClick={function(){setSel(e2.id);setTab("graph");}}>{e2.source+">"+e2.target}</div>,
                <div key={e2.id+"b"} style={{color:TX}}>{e2.flow.toFixed(4)}</div>,
                <div key={e2.id+"c"} style={{color:TX}}>{e2.flowN.toFixed(3)}</div>,
                <div key={e2.id+"d"} style={{color:TC[e2.type]||DM}}>{e2.type}</div>
              ];})}
            </div>
          </Info>
        </div>}
        {detail}{svg}
      </div>}

      {tab==="faces"&&<div style={{fontSize:7.5,fontFamily:MF,lineHeight:1.7}}>
        <div style={{fontSize:13,fontWeight:700,color:TX,marginBottom:4,fontFamily:HF,letterSpacing:"0.02em"}}>Face Structure</div>
        <Info title="About face metrics"><span>
          Faces are auto-detected cycles promoted to 2-cells. Each face has a boundary of edges and vertices. These metrics measure how vertices and edges contribute to the face complex, and how balanced those contributions are.
          <br/><b>Avg contribution</b>: mean of 1/|f.boundary| across faces containing v or e. High = small tight cycles. Low = large sprawling cycles.
          <br/><b>Total contribution</b>: sum of per-face contributions. Verified: sums to |F| across all vertices and all edges independently (partition of unity).
          <br/><b>Boundary asymmetry</b>: |fp(src)-fp(tgt)| / max(fp(src),fp(tgt)) for each edge. Measures whether both endpoints participate equally in the face complex. Negatively correlated with harmonic fraction (r={AN.face_structure.asym_rho_corr.toFixed(3)}).
          <br/><b>Face concentration</b>: coefficient of variation of boundary vertex face counts within each face. High = face spans vertices with very different face participation levels.
        </span></Info>
        <div style={{display:"flex",gap:2,marginBottom:4}}>
          <Stat label="Faces" value={D.meta.nF} color={GD}/>
          <Stat label="Avg Size" value={AN.face_structure.mean_face_size} color={GD}/>
          <Stat label="Mean Asym" value={AN.face_structure.mean_asym.toFixed(2)} color={PK}/>
          <Stat label="Asym-rho r" value={AN.face_structure.asym_rho_corr.toFixed(3)} color={PU}/>
        </div>

        <div style={{fontSize:7.5,color:SB,fontWeight:600,marginBottom:3}}>Top Vertices by Total Face Contribution</div>
        <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:1,marginBottom:6,fontSize:6.5,fontFamily:MF}}>
          <div style={{color:DM,fontWeight:600}}>Vertex</div><div style={{color:DM,fontWeight:600}}>Faces</div><div style={{color:DM,fontWeight:600}}>TotC</div><div style={{color:DM,fontWeight:600}}>AvgC</div><div style={{color:DM,fontWeight:600}}>AvgSz</div>
          {AN.face_structure.top_v_by_total_contrib.map(function(v2){return [
            <div key={v2.id+"a"} style={{color:TX,cursor:"pointer"}} onClick={function(){setSel(v2.id);setTab("graph");}}>{v2.id}</div>,
            <div key={v2.id+"b"} style={{color:GD}}>{v2.faceCount}</div>,
            <div key={v2.id+"c"} style={{color:TL}}>{v2.faceTotC}</div>,
            <div key={v2.id+"d"} style={{color:SB}}>{v2.avgContrib}</div>,
            <div key={v2.id+"e"} style={{color:SB}}>{v2.avgFaceSize}</div>
          ];})}
        </div>

        <div style={{fontSize:7.5,color:SB,fontWeight:600,marginBottom:3}}>Top Edges by Total Face Contribution</div>
        <div style={{display:"grid",gridTemplateColumns:"2fr repeat(4,1fr)",gap:1,marginBottom:6,fontSize:6.5,fontFamily:MF}}>
          <div style={{color:DM,fontWeight:600}}>Edge</div><div style={{color:DM,fontWeight:600}}>Faces</div><div style={{color:DM,fontWeight:600}}>TotC</div><div style={{color:DM,fontWeight:600}}>AvgC</div><div style={{color:DM,fontWeight:600}}>AvgSz</div>
          {AN.face_structure.top_e_by_total_contrib.map(function(e2){return [
            <div key={e2.id+"a"} style={{color:TX,cursor:"pointer"}} onClick={function(){setSel(e2.id);setTab("graph");}}>{e2.source+">"+e2.target}</div>,
            <div key={e2.id+"b"} style={{color:GD}}>{e2.faceCount}</div>,
            <div key={e2.id+"c"} style={{color:TL}}>{e2.faceTotC}</div>,
            <div key={e2.id+"d"} style={{color:SB}}>{e2.avgContrib}</div>,
            <div key={e2.id+"e"} style={{color:SB}}>{e2.avgFaceSize}</div>
          ];})}
        </div>

        <div style={{fontSize:7.5,color:SB,fontWeight:600,marginBottom:3}}>Boundary Asymmetry</div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4,marginBottom:6}}>
          <div>
            <div style={{fontSize:6.5,color:TL,fontWeight:600,marginBottom:2}}>Most Balanced (low asym, high rho)</div>
            {AN.face_structure.most_balanced.map(function(e2){return <div key={e2.id} style={{fontSize:6.5,color:SB,marginBottom:1,cursor:"pointer"}} onClick={function(){setSel(e2.id);setTab("graph");}}>
              {e2.source+"->"+e2.target} <span style={{color:DM}}>asym={e2.asym}</span> <span style={{color:PU}}>rho={e2.rho}</span> <span style={{color:DM}}>({e2.srcFaces}:{e2.tgtFaces})</span>
            </div>;})}
          </div>
          <div>
            <div style={{fontSize:6.5,color:RD,fontWeight:600,marginBottom:2}}>Most Asymmetric (hub-to-peripheral)</div>
            {AN.face_structure.most_asymmetric.map(function(e2){return <div key={e2.id} style={{fontSize:6.5,color:SB,marginBottom:1,cursor:"pointer"}} onClick={function(){setSel(e2.id);setTab("graph");}}>
              {e2.source+"->"+e2.target} <span style={{color:DM}}>asym={e2.asym}</span> <span style={{color:PU}}>rho={e2.rho}</span> <span style={{color:DM}}>({e2.srcFaces}:{e2.tgtFaces})</span>
            </div>;})}
          </div>
        </div>

        <div style={{fontSize:7.5,color:SB,fontWeight:600,marginBottom:3}}>Face Concentration (boundary heterogeneity)</div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4,marginBottom:6}}>
          <div>
            <div style={{fontSize:6.5,color:OR,fontWeight:600,marginBottom:2}}>Most Concentrated</div>
            {AN.face_structure.most_concentrated_faces.map(function(f2){return <div key={f2.id} style={{fontSize:6.5,color:SB,marginBottom:1,cursor:"pointer"}} onClick={function(){setSel(f2.id);setTab("graph");}}>
              {f2.id} <span style={{color:DM}}>{f2.size}-cycle</span> <span style={{color:OR}}>CV={f2.conc}</span> <span style={{color:DM}}>{f2.vertices.join(",")}</span>
            </div>;})}
          </div>
          <div>
            <div style={{fontSize:6.5,color:CY,fontWeight:600,marginBottom:2}}>Most Uniform</div>
            {AN.face_structure.most_uniform_faces.map(function(f2){return <div key={f2.id} style={{fontSize:6.5,color:SB,marginBottom:1,cursor:"pointer"}} onClick={function(){setSel(f2.id);setTab("graph");}}>
              {f2.id} <span style={{color:DM}}>{f2.size}-cycle</span> <span style={{color:CY}}>CV={f2.conc}</span> <span style={{color:DM}}>{f2.vertices.join(",")}</span>
            </div>;})}
          </div>
        </div>

        <Info title={"All faces ("+D.meta.nF+")"}>
          <div style={{display:"grid",gridTemplateColumns:"auto 1fr auto auto auto",gap:"1px 6px",fontSize:6.5,fontFamily:MF}}>
            <div style={{color:DM,fontWeight:600}}>Face</div><div style={{color:DM,fontWeight:600}}>Vertices</div><div style={{color:DM,fontWeight:600}}>Size</div><div style={{color:DM,fontWeight:600}}>Curl</div><div style={{color:DM,fontWeight:600}}>CV</div>
            {D.faces.map(function(f2){return [
              <div key={f2.id+"a"} style={{color:GD,cursor:"pointer"}} onClick={function(){setSel(f2.id);setTab("graph");}}>{f2.id}</div>,
              <div key={f2.id+"b"} style={{color:SB}}>{f2.vertices.join(", ")}</div>,
              <div key={f2.id+"c"} style={{color:TX}}>{f2.size}</div>,
              <div key={f2.id+"d"} style={{color:f2.curl>0?TL:RD}}>{f2.curl}</div>,
              <div key={f2.id+"e"} style={{color:f2.conc>0.5?OR:CY}}>{f2.conc}</div>
            ];})}
          </div>
        </Info>
      </div>}

      {tab==="spectra"&&<div>
        <div style={{fontSize:13,fontWeight:700,color:TX,marginBottom:4,fontFamily:HF,letterSpacing:"0.02em"}}>Spectral Fingerprint</div>
        <Info title="What are Laplacian spectra?"><span>
          Each Laplacian has eigenvalues (spectrum). Zero eigenvalues count topological features: B0 for L0, B1 for L1, B2 for L2. The spectral gap (smallest positive eigenvalue) measures connectivity. Expand each panel below to see exact values. Use the page controls to navigate all computed eigenvalues.
        </span></Info>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4}}>
          {[
            ["L0","L0 vertex",PU,S.L0,"B0="+T.b0+", gap="+S.fiedler_L0,"Vertex graph Laplacian (B1*B1^T). Eigenvalues measure vertex connectivity. Zero count = connected components."],
            ["LO","L_O overlap",PK,S.LO,"gap="+S.fiedler_LO,"Jaccard overlap Laplacian on edges. Groups edges by shared boundary vertices. Independent of face structure."],
            ["L1_down","L1 down-only",TL,S.L1_down,"B1(raw)="+T.b1_raw,"Down-Laplacian B1^T*B1. Captures edge coupling through shared vertices only. No face information. Zeros = all independent cycles."],
            ["L1_full","L1 full Hodge","#c09840",S.L1_full,"B1="+T.b1_filled,"Full Hodge 1-Laplacian (B1^T*B1 + B2*B2^T). Down + up coupling. Zeros = unfilled cycles only."],
            ["L2","L2 face",CY,S.L2,"B2="+T.b2,"Face Laplacian (B2^T*B2). Coupling between faces sharing edges. Zeros = independent voids."],
            ["L1_alpha","L1(a) composite",OR,S.L1_alpha,"a="+K.alpha_G.toFixed(3),"Topology-geometry composite: L1 + aG*L_O. Combines chain complex structure with overlap geometry."],
          ].map(function(sp){
            var vals=sp[3]||[];var key=sp[0];var pg=specPage[key]||0;var ps=15;var nPages=Math.max(1,Math.ceil(vals.length/ps));
            var pageVals=vals.slice(pg*ps,pg*ps+ps);var offset=pg*ps;
            return <div key={key} style={{padding:5,borderRadius:4,background:C1,border:"1px solid "+BD}}>
            <Bars vals={pageVals} color={sp[2]} label={sp[1]+" | "+sp[4]}/>
            <Info title={"Eigenvalues ("+vals.length+" total)"}>
              <div style={{fontSize:6.5,color:SB,fontFamily:MF,marginBottom:3}}>{sp[5]}</div>
              {nPages>1&&<div style={{display:"flex",gap:2,marginBottom:3,alignItems:"center",flexWrap:"wrap"}}>
                {Array.from({length:nPages},function(_,p){return <button key={p} onClick={function(){setSpecPage(Object.assign({},specPage,function(){var o={};o[key]=p;return o;}()));}} style={{padding:"1px 4px",borderRadius:2,fontSize:6,fontFamily:MF,cursor:"pointer",background:pg===p?(sp[2]+"20"):"transparent",border:"1px solid "+(pg===p?(sp[2]+"50"):BD),color:pg===p?sp[2]:DM}}>{(p*ps)+"-"+Math.min((p+1)*ps-1,vals.length-1)}</button>;})
              }</div>}
              <div style={{display:"flex",flexWrap:"wrap",gap:"1px 4px"}}>
                {pageVals.map(function(v2,i2){var idx=offset+i2;var z=Math.abs(v2)<1e-4;return <span key={idx} style={{fontSize:6,fontFamily:MF,color:z?ZC:sp[2],opacity:z?1:0.7}}>{"["+idx+"] "+v2.toFixed(6)}</span>;})}
              </div>
            </Info>
          </div>;})}
          <div style={{padding:5,borderRadius:4,background:C1,border:"1px solid "+BD,gridColumn:"1/3"}}>
            {function(){var vals=S.Lambda||[];var pg=specPage["Lambda"]||0;var ps=15;var nPages=Math.max(1,Math.ceil(vals.length/ps));var pageVals=vals.slice(pg*ps,pg*ps+ps);var offset=pg*ps;return <div>
            <Bars vals={pageVals} color={AC} label={"Lambda = B1*L_O*B1^T"}/>
            <Info title={"Eigenvalues ("+vals.length+" total)"}>
              <div style={{fontSize:6.5,color:SB,fontFamily:MF,marginBottom:3}}>Cross-dimension operator. Projects edge overlap geometry to vertices via boundary maps. Generalizes L0.</div>
              {nPages>1&&<div style={{display:"flex",gap:2,marginBottom:3,alignItems:"center",flexWrap:"wrap"}}>
                {Array.from({length:nPages},function(_,p){return <button key={p} onClick={function(){setSpecPage(Object.assign({},specPage,{Lambda:p}));}} style={{padding:"1px 4px",borderRadius:2,fontSize:6,fontFamily:MF,cursor:"pointer",background:pg===p?(AC+"20"):"transparent",border:"1px solid "+(pg===p?(AC+"50"):BD),color:pg===p?AC:DM}}>{(p*ps)+"-"+Math.min((p+1)*ps-1,vals.length-1)}</button>;})
              }</div>}
              <div style={{display:"flex",flexWrap:"wrap",gap:"1px 4px"}}>
                {pageVals.map(function(v2,i2){var idx=offset+i2;var z=Math.abs(v2)<1e-4;return <span key={idx} style={{fontSize:6,fontFamily:MF,color:z?ZC:AC,opacity:z?1:0.7}}>{"["+idx+"] "+v2.toFixed(6)}</span>;})}
              </div>
            </Info></div>;}()}
          </div>
        </div>
      </div>}

      {tab==="analysis"&&<div style={{fontSize:7.5,fontFamily:MF,lineHeight:1.7}}>
        <div style={{fontSize:13,fontWeight:700,color:TX,marginBottom:6,fontFamily:HF,letterSpacing:"0.02em"}}>Structural Analysis</div>
        {[
          ["Topology",GD,anTopology],
          ["Signal Decomposition",PU,anHodge],
          ["Coupling Constants",OR,anCoupling],
          ["Partition Structure",AC,anPart],
          ["Face Structure",PK,anFaces],
        ].map(function(sec){return <div key={sec[0]} style={{padding:6,borderRadius:4,background:C1,border:"1px solid "+BD,marginBottom:4}}>
          <div style={{fontWeight:700,color:sec[1],marginBottom:2,fontSize:8}}>{sec[0]}</div>
          <div style={{color:SB,fontSize:7}}>{sec[2]}</div>
        </div>;})}
        <div style={{padding:6,borderRadius:4,background:C1,border:"1px solid "+BD,marginBottom:4}}>
          <div style={{fontWeight:700,color:TL,marginBottom:2,fontSize:8}}>Key Structures</div>
          <div style={{color:SB,fontSize:7}}>
            <div style={{marginBottom:3}}><b>Highest-degree vertices:</b> {AN.structure.hubs.map(function(h2){return h2.id+" (deg "+h2.degree+", "+h2.faces+" faces)";}).join(", ")}</div>
            <div style={{marginBottom:3}}><b>Sources</b> ({AN.structure.sources.length}): {AN.structure.sources.join(", ")}</div>
            <div style={{marginBottom:3}}><b>Sinks</b> ({AN.structure.sinks.length}): {AN.structure.sinks.join(", ")}</div>
            <div style={{marginBottom:3}}><b>Highest harmonic fraction (rho):</b> {AN.structure.high_rho.map(function(r2){return r2.source+"->"+r2.target+" ("+r2.rho+")";}).join(", ")}</div>
            <div><b>Most cyclic edges (L1-up):</b> {AN.structure.most_cyclic.map(function(c2){return c2.source+"->"+c2.target+" ("+c2.up+" faces)";}).join(", ")}</div>
          </div>
        </div>
        {AN.negative_types.length>0&&<div style={{padding:6,borderRadius:4,background:C1,border:"1px solid "+BD,marginBottom:4}}>
          <div style={{fontWeight:700,color:RD,marginBottom:2,fontSize:8}}>Auto-detected Negative Types</div>
          <div style={{color:SB,fontSize:7}}>These edge types were assigned negative flow based on name pattern matching: {AN.negative_types.join(", ")}. All other types are positive.</div>
        </div>}
        {Object.keys(AN.partitions.LO_A_types).length>0&&<div style={{padding:6,borderRadius:4,background:C1,border:"1px solid "+BD,marginBottom:4}}>
          <div style={{fontWeight:700,color:PK,marginBottom:2,fontSize:8}}>L_O Partition Composition</div>
          <div style={{color:SB,fontSize:7}}>
            <div>Group A ({AN.partitions.LO_split[0]} edges): {Object.entries(AN.partitions.LO_A_types).map(function(e2){return e2[0]+":"+e2[1];}).join(", ")}</div>
            <div>Group B ({AN.partitions.LO_split[1]} edges): {Object.entries(AN.partitions.LO_B_types).map(function(e2){return e2[0]+":"+e2[1];}).join(", ")}</div>
          </div>
        </div>}
        <div style={{padding:6,borderRadius:4,background:C1,border:"1px solid "+BD,marginBottom:4}}>
          <div style={{fontWeight:700,color:CY,marginBottom:3,fontSize:8}}>Standard Graph Metrics</div>
          <div style={{color:SB,fontSize:7,marginBottom:5}}>{anStandard}</div>
          <div style={{fontSize:7,color:SB,marginBottom:3}}>
            <b>PageRank</b> <span style={{color:DM}}>(directed, damping=0.85, sums to 1.0)</span>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:"1px 4px",fontSize:6.5,fontFamily:MF,marginBottom:5}}>
            <div style={{color:DM,fontWeight:600}}>Vertex</div><div style={{color:RD,fontWeight:600}}>PR</div><div style={{color:DM,fontWeight:600}}>Degree</div><div style={{color:DM,fontWeight:600}}>Role</div>
            {SM.top_pagerank.map(function(p2){return [
              <div key={p2.id+"a"} style={{color:TX,cursor:"pointer"}} onClick={function(){setSel(p2.id);setTab("graph");}}>{p2.id}</div>,
              <div key={p2.id+"b"} style={{color:RD}}>{(p2.pr*100).toFixed(2)+"%"}</div>,
              <div key={p2.id+"c"} style={{color:SB}}>{p2.degree}</div>,
              <div key={p2.id+"d"} style={{color:RL[p2.role]||DM}}>{p2.role}</div>
            ];})}
          </div>
          <div style={{fontSize:7,color:SB,marginBottom:3}}>
            <b>Betweenness centrality</b> <span style={{color:DM}}>(PR-degree r={SM.pr_deg_corr}, btw-degree r={SM.btw_deg_corr})</span>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:"1px 4px",fontSize:6.5,fontFamily:MF,marginBottom:5}}>
            <div style={{color:DM,fontWeight:600}}>Vertex</div><div style={{color:OR,fontWeight:600}}>Btw</div><div style={{color:DM,fontWeight:600}}>Norm</div><div style={{color:DM,fontWeight:600}}>Degree</div>
            {SM.top_betweenness.map(function(b2){return [
              <div key={b2.id+"a"} style={{color:TX,cursor:"pointer"}} onClick={function(){setSel(b2.id);setTab("graph");}}>{b2.id}</div>,
              <div key={b2.id+"b"} style={{color:OR}}>{b2.btw.toFixed(1)}</div>,
              <div key={b2.id+"c"} style={{color:SB}}>{b2.btwNorm.toFixed(3)}</div>,
              <div key={b2.id+"d"} style={{color:SB}}>{b2.degree}</div>
            ];})}
          </div>
          <div style={{fontSize:7,color:SB,marginBottom:3}}>
            <b>Clustering coefficient</b> <span style={{color:DM}}>(clust-faceCount r={SM.clust_fc_corr})</span>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:"1px 4px",fontSize:6.5,fontFamily:MF,marginBottom:5}}>
            <div style={{color:DM,fontWeight:600}}>Vertex</div><div style={{color:PK,fontWeight:600}}>CC</div><div style={{color:DM,fontWeight:600}}>Degree</div>
            {SM.top_clustering.map(function(c2){return [
              <div key={c2.id+"a"} style={{color:TX,cursor:"pointer"}} onClick={function(){setSel(c2.id);setTab("graph");}}>{c2.id}</div>,
              <div key={c2.id+"b"} style={{color:PK}}>{c2.cc.toFixed(4)}</div>,
              <div key={c2.id+"c"} style={{color:SB}}>{c2.degree}</div>
            ];})}
          </div>
          <div style={{fontSize:7,color:SB,marginBottom:3}}>
            <b>Edge betweenness</b> <span style={{color:DM}}>(eBtw-rho r={SM.ebtw_rho_corr}, eBtw-L1up r={SM.ebtw_l1up_corr})</span>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"2fr 1fr 1fr 1fr",gap:"1px 4px",fontSize:6.5,fontFamily:MF,marginBottom:3}}>
            <div style={{color:DM,fontWeight:600}}>Edge</div><div style={{color:OR,fontWeight:600}}>eBtw</div><div style={{color:DM,fontWeight:600}}>Norm</div><div style={{color:PU,fontWeight:600}}>Rho</div>
            {SM.top_edge_betw.map(function(e2){return [
              <div key={e2.id+"a"} style={{color:TX,cursor:"pointer"}} onClick={function(){setSel(e2.id);setTab("graph");}}>{e2.source+">"+e2.target}</div>,
              <div key={e2.id+"b"} style={{color:OR}}>{e2.ebtw.toFixed(1)}</div>,
              <div key={e2.id+"c"} style={{color:SB}}>{e2.ebtwNorm.toFixed(3)}</div>,
              <div key={e2.id+"d"} style={{color:e2.rho>0.3?PU:(e2.rho>0.15?PK:DM)}}>{e2.rho.toFixed(3)}</div>
            ];})}
          </div>
        </div>
        {SM.n_communities>0&&<div style={{padding:6,borderRadius:4,background:C1,border:"1px solid "+BD,marginBottom:4}}>
          <div style={{fontWeight:700,color:CY,marginBottom:2,fontSize:8}}>Louvain Communities ({SM.n_communities}, Q={SM.modularity})</div>
          <div style={{color:SB,fontSize:7,marginBottom:3}}>Fiedler bipartition splits {SM.fiedler_splits} of {SM.n_communities} communities - Louvain resolves {SM.n_communities/2|0}x finer structure than spectral bisection.</div>
          {Object.entries(SM.communities).map(function(c2){return <div key={c2[0]} style={{fontSize:6.5,color:SB,marginBottom:1}}>
            <span style={{color:CC_PAL[parseInt(c2[0])%CC_PAL.length],fontWeight:600}}>C{c2[0]}</span>{" ("+c2[1].length+"): "+c2[1].join(", ")}
          </div>;})}
        </div>}
      </div>}

      {tab==="verify"&&<div style={{fontSize:7.5,fontFamily:MF,lineHeight:1.7}}>
        <div style={{fontSize:13,fontWeight:700,color:TX,marginBottom:6,fontFamily:HF,letterSpacing:"0.02em"}}>Verification</div>
        <Info title="About verification"><span>
          Each check compares independently computed values. Chain complex verified by matrix multiplication. Betti numbers verified both algebraically (rank) and spectrally (zero eigenvalue count). Hodge orthogonality verified by energy partition. Correlation checks are structural predictions from the spectral geometry paper, conditioned on the topology of the complex.
        </span></Info>
        <div style={{fontSize:7,color:SB,fontWeight:600,marginBottom:3,marginTop:4}}>Chain Complex</div>
        {[
          [T.chainOk,"B1 * B2 = 0   (boundary of boundary is zero)"],
        ].map(function(c,i){return <div key={"a"+i} style={{padding:"3px 6px",borderRadius:3,background:C1,border:"1px solid "+BD,marginBottom:2}}><Ck ok={c[0]}/>{" "}{c[1]}</div>;})}
        <div style={{fontSize:7,color:SB,fontWeight:600,marginBottom:3,marginTop:6}}>Betti Numbers</div>
        {[
          [T.b0===D.meta.nV-T.rankB1,"B0 = |V| - rank(B1) = "+D.meta.nV+" - "+T.rankB1+" = "+T.b0],
          [T.b1_raw===D.meta.nE-T.rankB1,"B1(raw) = |E| - rank(B1) = "+D.meta.nE+" - "+T.rankB1+" = "+T.b1_raw],
          [T.b1_filled===D.meta.nE-T.rankB1-T.rankB2,"B1 = |E| - rank(B1) - rank(B2) = "+D.meta.nE+" - "+T.rankB1+" - "+T.rankB2+" = "+T.b1_filled],
          [T.b2===D.meta.nF-T.rankB2,"B2 = |F| - rank(B2) = "+D.meta.nF+" - "+T.rankB2+" = "+T.b2],
          [T.b0-(T.b1_filled)+T.b2===T.euler,"Euler: B0 - B1 + B2 = "+T.b0+" - "+T.b1_filled+" + "+T.b2+" = "+T.euler],
        ].map(function(c,i){return <div key={"b"+i} style={{padding:"3px 6px",borderRadius:3,background:C1,border:"1px solid "+BD,marginBottom:2}}><Ck ok={c[0]}/>{" "}{c[1]}</div>;})}
        <div style={{fontSize:7,color:SB,fontWeight:600,marginBottom:3,marginTop:6}}>Hodge Decomposition</div>
        {[
          [Math.abs(H.gradPct+H.curlPct+H.harmPct-100)<2,"Energy conservation: "+H.gradPct+" + "+H.curlPct+" + "+H.harmPct+" = "+(H.gradPct+H.curlPct+H.harmPct).toFixed(1)+"%"],
          [T.b1_filled===0?H.harmPct<1:true,"B1="+T.b1_filled+(T.b1_filled===0?" => harmonic must vanish ("+H.harmPct+"%)":(" => "+T.b1_filled+"-dim harmonic space"))],
        ].map(function(c,i){return <div key={"c"+i} style={{padding:"3px 6px",borderRadius:3,background:C1,border:"1px solid "+BD,marginBottom:2}}><Ck ok={c[0]}/>{" "}{c[1]}</div>;})}
        <div style={{fontSize:7,color:SB,fontWeight:600,marginBottom:3,marginTop:6}}>Spectral Gaps</div>
        {[
          [K.fiedler_LO>0,"L_O gap = "+K.fiedler_LO+" > 0   (overlap graph connected)"],
          [S.fiedler_L0>0,"L0 gap = "+S.fiedler_L0+" > 0   (vertex graph connected)"],
        ].map(function(c,i){return <div key={"d"+i} style={{padding:"3px 6px",borderRadius:3,background:C1,border:"1px solid "+BD,marginBottom:2}}><Ck ok={c[0]}/>{" "}{c[1]}</div>;})}
        <div style={{fontSize:7,color:SB,fontWeight:600,marginBottom:3,marginTop:6}}>Derived Quantities</div>
        <div style={{padding:"4px 6px",borderRadius:3,background:C1,border:"1px solid "+BD,marginBottom:2,fontSize:7,color:DM,lineHeight:1.6}}>
          aG = Fiedler(L1) / Fiedler(L_O) = {K.fiedler_L1} / {K.fiedler_LO} = {K.alpha_G.toFixed(4)}{K.alpha_G>4?" (strongly geometry-dominated)":(K.alpha_G>2?" (geometry-dominated)":(K.alpha_G>0.5?" (near-balanced)":(K.alpha_G>0.25?" (topology-dominated)":" (strongly topology-dominated)")))}<br/>
          aT = B1 / |E| = {T.b1_filled} / {D.meta.nE} = {K.alpha_T.toFixed(4)}{" ("}{Math.round(K.alpha_T*100)}{"% unfilled cycle density)"}<br/>
          L1(a) = L1 + {K.alpha_G.toFixed(3)} * L_O<br/>
          Lambda = B1 * L_O * B1^T   ({D.meta.nV}x{D.meta.nV})<br/>
          rho(e) = |harm(w)_e| / |w_e|   (per-edge resistance constant)
        </div>
        <div style={{fontSize:7,color:SB,fontWeight:600,marginBottom:3,marginTop:6}}>Face Structure</div>
        {[
          [Math.abs(FS.v_tc_sum-D.meta.nF)<0.01,"Partition of unity (vertices): totalContrib sums to |F|: "+FS.v_tc_sum.toFixed(1)+" = "+D.meta.nF],
          [Math.abs(FS.e_tc_sum-D.meta.nF)<0.01,"Partition of unity (edges): totalContrib sums to |F|: "+FS.e_tc_sum.toFixed(1)+" = "+D.meta.nF],
          [T.b1_filled>0?(FS.asym_rho_corr<0):true,T.b1_filled>0?("Asymmetry-rho negative correlation: r="+FS.asym_rho_corr.toFixed(4)+" (balanced edges carry harmonic signal)"):("Asymmetry-rho: r="+FS.asym_rho_corr.toFixed(4)+" (B1=0, rho identically zero, correlation is noise)")],
        ].map(function(c,i){return <div key={"f"+i} style={{padding:"3px 6px",borderRadius:3,background:C1,border:"1px solid "+BD,marginBottom:2}}><Ck ok={c[0]}/>{" "}{c[1]}</div>;})}
        <div style={{padding:"4px 6px",borderRadius:3,background:C1,border:"1px solid "+BD,marginBottom:2,fontSize:7,color:DM,lineHeight:1.6}}>
          contrib(v,f) = 1/|f.vertices|, contrib(e,f) = 1/|f.boundary|<br/>
          totalContrib(v) = sum over faces containing v. Sums to |F| (partition of unity).<br/>
          bndAsym(e) = |fp(src) - fp(tgt)| / max(fp(src), fp(tgt)). Low = both endpoints equally embedded in face complex.<br/>
          When B1{">"} 0: edges on unfilled cycles have high rho and low asymmetry (endpoints equally marginal) => negative correlation.<br/>
          When B1 = 0: all cycles filled, rho = 0 identically, correlation is floating-point noise.
        </div>
        <div style={{fontSize:7,color:SB,fontWeight:600,marginBottom:3,marginTop:6}}>Standard Graph Metrics</div>
        {[
          [Math.abs(SM.pr_sum-1.0)<0.001,"PageRank sums to 1.0: "+SM.pr_sum.toFixed(6)],
          [SM.modularity>0,"Louvain modularity Q="+SM.modularity.toFixed(4)+" > 0 ("+SM.n_communities+" communities)"],
          [Math.abs(SM.clust_fc_corr)<0.3,"Clustering-faceCount independence: r="+SM.clust_fc_corr.toFixed(4)+" (triangle fraction vs face participation are independent measures)"],
        ].map(function(c,i){return <div key={"g"+i} style={{padding:"3px 6px",borderRadius:3,background:C1,border:"1px solid "+BD,marginBottom:2}}><Ck ok={c[0]}/>{" "}{c[1]}</div>;})}
        <div style={{padding:"4px 6px",borderRadius:3,background:C1,border:"1px solid "+BD,marginBottom:2,fontSize:7,color:DM,lineHeight:1.6}}>
          PageRank: damping=0.85, directed (PR-degree r={SM.pr_deg_corr.toFixed(3)}{SM.pr_deg_corr>0.8?", high: scale-free structure":", moderate: mixed structure"})<br/>
          Betweenness: BFS shortest paths, undirected (btw-degree r={SM.btw_deg_corr.toFixed(3)})<br/>
          Clustering: triangle fraction, undirected<br/>
          Louvain: greedy modularity, Fiedler splits {SM.fiedler_splits}/{SM.n_communities} communities<br/>
          Edge betweenness: undirected (eBtw-rho r={SM.ebtw_rho_corr.toFixed(3)}, eBtw-L1up r={SM.ebtw_l1up_corr.toFixed(3)})
        </div>
        <div style={{padding:5,borderRadius:4,background:C2,border:"1px solid "+BD,marginTop:6,fontSize:6.5,color:DM,lineHeight:1.6}}>
          CSV ({D.meta.nE} edges, {D.meta.attributes.length} attributes) {"->"} {D.meta.nV} vertices {"->"} B1 ({D.meta.nV}x{D.meta.nE}) {"->"} {D.meta.nF} faces {"->"} B2 ({D.meta.nE}x{D.meta.nF}) {"->"} 7 Laplacians {"->"} Hodge decomposition {"->"} standard metrics {"->"} spectral layout
        </div>
      </div>}
    </div>
  );
}
