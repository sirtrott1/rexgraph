// Exercise the shipped frontend's hooks with delayed responses, without a DOM.
// This qualifies request/display behavior, not browser layout or React internals.
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");

class Hooks {
  constructor() { this.slots = []; this.index = 0; this.effects = []; }
  state(initial) {
    const i = this.index++;
    if (!this.slots[i]) this.slots[i] = {value:initial};
    return [this.slots[i].value, value => { this.slots[i].value = value; }];
  }
  ref(initial) {
    const i = this.index++;
    if (!this.slots[i]) this.slots[i] = {current:initial};
    return this.slots[i];
  }
  effect(fn, deps) {
    const i = this.index++;
    const old = this.slots[i];
    if (!old || deps.some((value, j) => !Object.is(value, old.deps[j]))) {
      this.effects.push(() => {
        old?.cleanup?.();
        this.slots[i] = {deps, cleanup:fn()};
      });
    }
  }
  render(fn) {
    active = this; this.index = 0;
    const tree = fn();
    this.effects.splice(0).forEach(effect => effect());
    return tree;
  }
  close() { this.slots.forEach(slot => slot.cleanup?.()); }
}

let active;
const pending = [];
const context = vm.createContext({
  AbortController,
  React: {
    Fragment:"fragment",
    createElement:(type, props, ...children) => ({type, props:props || {}, children:children.flat()}),
    useState:value => active.state(value),
    useRef:value => active.ref(value),
    useEffect:(fn, deps) => active.effect(fn, deps)
  },
  ReactDOM:{createRoot:() => ({render:() => {}})},
  document:{getElementById:() => ({})},
  fetch:(path, options) => new Promise((resolve, reject) => pending.push({path, options, resolve, reject}))
});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context);
const evaluate = expression => vm.runInContext(expression, context);
const tick = () => new Promise(resolve => setImmediate(resolve));
const succeed = (request, data) => request.resolve({ok:true, json:() => Promise.resolve(data)});
function find(tree, type) {
  if (!tree || typeof tree !== "object") return null;
  if (tree.type === type) return tree;
  for (const child of tree.children || []) { const item = find(child, type); if (item) return item; }
  return null;
}

async function main() {
  const get = new Hooks();
  get.render(() => evaluate('useApi("/first")'));
  const first = pending.shift();
  get.render(() => evaluate('useApi("/second")'));
  const second = pending.shift();
  assert.equal(first.options.signal.aborted, true);
  succeed(second, {source:"second"}); await tick();
  succeed(first, {source:"first"}); await tick();
  assert.equal(get.slots[0].value.source, "second");
  assert.equal(get.slots[1].value, null);

  get.render(() => evaluate('useApi("/failed")'));
  const failed = pending.shift();
  failed.reject(new Error("failed")); await tick();
  assert.equal(get.slots[0].value, null);
  assert.match(get.slots[1].value, /failed/);
  get.render(() => evaluate('useApi("/recovered")'));
  const recovered = pending.shift();
  assert.equal(get.slots[1].value, null);
  succeed(recovered, {source:"recovered"}); await tick();
  assert.equal(get.slots[0].value.source, "recovered");
  get.close();

  const query = new Hooks();
  context.input = {source:"a/b", sourceInfo:{default_query:'FROM DATASET("a/b") RETURN BETTI(0)'}};
  const renderQuery = () => query.render(() => evaluate("QueryView(input)"));
  renderQuery();
  let tree = renderQuery();
  assert.equal(find(tree, "textarea").props.value, context.input.sourceInfo.default_query);
  find(tree, "textarea").props.onChange({target:{value:'FROM DATASET("a/b") RETURN 1/7'}});
  tree = renderQuery();
  find(tree, "select").props.onChange({target:{value:"exact"}});
  tree = renderQuery();
  assert.equal(find(tree, "textarea").props.value, 'FROM DATASET("a/b") RETURN 1/7');
  find(tree, "button").props.onClick();
  const oldQuery = pending.shift();
  assert.equal(JSON.parse(oldQuery.options.body).exactness, "exact");
  assert.equal(JSON.parse(oldQuery.options.body).query, 'FROM DATASET("a/b") RETURN 1/7');

  context.input = {source:"next", sourceInfo:{default_query:'FROM REX("next") RETURN BETTI(0)'}};
  renderQuery(); tree = renderQuery();
  assert.equal(oldQuery.options.signal.aborted, true);
  succeed(oldQuery, {source:"wrong"}); await tick();
  assert.equal(query.slots[2].value, null);
  assert.equal(find(tree, "textarea").props.value, context.input.sourceInfo.default_query);
  find(tree, "button").props.onClick();
  const currentQuery = pending.shift();
  succeed(currentQuery, {source:"next"}); await tick();
  assert.equal(query.slots[2].value.source, "next");

  tree = renderQuery(); find(tree, "button").props.onClick();
  const stalePolicy = pending.shift();
  find(tree, "select").props.onChange({target:{value:"declared"}});
  renderQuery();
  assert.equal(stalePolicy.options.signal.aborted, true);
  succeed(stalePolicy, {source:"stale-policy"}); await tick();
  assert.equal(query.slots[2].value, null);

  context.input = {source:"", sourceInfo:null};
  renderQuery(); tree = renderQuery();
  assert.equal(find(tree, "textarea").props.value, "");
  assert.equal(find(tree, "button").props.disabled, true);
  query.close();
  assert.equal(pending.length, 0);

  const app = new Hooks();
  let appTree = app.render(() => evaluate("App()"));
  const listing = pending.shift();
  assert.equal(listing.path, "/api/sources");
  succeed(listing, {sources:[
    {name:"denied", accessible:false, panels:[]},
    {name:"invalid", accessible:true, error:"invalid state", panels:[]},
    {name:"graph", accessible:true, panels:["Overview","Structure","Queries"]},
    {name:"store", accessible:true, panels:["Overview","RCDB","Queries"]}
  ]}); await tick();
  app.render(() => evaluate("App()"));
  appTree = app.render(() => evaluate("App()"));
  const navigation = find(appTree, "nav");
  assert.deepEqual(Array.from(navigation.children, b => b.children[0]), ["Overview","Structure","Queries"]);
  const sourceSelect = find(appTree, "select");
  assert.equal(sourceSelect.props.value, "graph");
  assert.equal(sourceSelect.children.find(x => x.props.value === "denied").props.disabled, true);
  assert.equal(sourceSelect.children.find(x => x.props.value === "invalid").props.disabled, true);
  navigation.children.find(b => b.children[0] === "Structure").props.onClick();
  appTree = app.render(() => evaluate("App()"));
  assert.equal(find(appTree, "nav").children.find(b => b.children[0] === "Structure").props.className, "active");
  find(appTree, "select").props.onChange({target:{value:"store"}});
  appTree = app.render(() => evaluate("App()"));
  assert.deepEqual(Array.from(find(appTree,"nav").children, b => b.children[0]), ["Overview","RCDB","Queries"]);
  assert.equal(find(appTree,"nav").children[0].props.className,"active");
  const panel = evaluate('PanelView({name:"RCDB",source:"store"})');
  assert.equal(panel.type,evaluate("QueryBackedPanel"));
  app.close();
  console.log(JSON.stringify({get_stale_response:"passed", get_error_recovery:"passed",
    source_query_and_exact_policy:"passed", query_stale_response:"passed",
    policy_change_response:"passed", empty_selection:"passed", source_navigation:"passed", rcdb_panel_wired:"passed"}));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
