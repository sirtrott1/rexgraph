/* Shallow-render every screen and every component it reaches.

   app.jsx ships without a build step or a test renderer, so a screen that throws on
   mount only shows up in a browser. This calls each component function with stub
   hooks and walks the tree it returns, calling nested components as it goes. That is
   enough to catch the failures that actually happen here: a name resolved from the
   wrong scope, a prop shape a primitive does not accept, a component that does not
   exist.

   Effects are not run. Anything that needs the network or a real DOM stays stubbed,
   so this proves a screen renders, not that it works. */
const fs = require("fs");
const path = require("path");

const noop = () => {};
const el = (type, props, ...children) => ({ $$el: true, type, props: props || {}, children });

function stubHooks() {
  return {
    createElement: el,
    Fragment: "Fragment",
    useState: (init) => [typeof init === "function" ? init() : init, noop],
    useEffect: noop,
    useRef: (v) => ({ current: v === undefined ? null : v }),
    useCallback: (fn) => fn,
    useMemo: (fn) => fn(),
    isValidElement: (v) => !!(v && v.$$el),
  };
}

const store = () => ({ getItem: () => null, setItem: noop, removeItem: noop });
const noEl = { classList: { toggle: noop, add: noop, remove: noop }, style: {},
               appendChild: noop, removeChild: noop, click: noop, remove: noop,
               addEventListener: noop, removeEventListener: noop, focus: noop,
               getBoundingClientRect: () => ({ top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0 }) };

const sandbox = {
  React: stubHooks(),
  ReactDOM: { render: noop, createRoot: () => ({ render: noop }) },
  console,
  localStorage: store(),
  sessionStorage: store(),
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
  location: { pathname: "/", search: "", href: "http://localhost/" },
  history: { pushState: noop, replaceState: noop },
  fetch: () => new Promise(noop),
  EventSource: function () { return { close: noop, addEventListener: noop, onmessage: null, onerror: null }; },
  FormData: function () { return { append: noop }; },
  Blob: function () { return {}; },
  URL: { createObjectURL: () => "blob:x", revokeObjectURL: noop },
  setTimeout: noop, clearTimeout: noop, setInterval: noop, clearInterval: noop,
  requestAnimationFrame: noop,
  alert: noop, confirm: () => false, prompt: () => null,
  TextDecoder: function () { return { decode: () => "" }; },
  document: Object.assign(Object.create(noEl), {
    body: noEl, documentElement: Object.assign(Object.create(noEl), { dataset: {} }),
    createElement: () => Object.create(noEl),
    getElementById: () => Object.create(noEl),
    querySelector: () => null, querySelectorAll: () => [],
    addEventListener: noop, removeEventListener: noop,
  }),
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

const vm = require("vm");
const ctx = vm.createContext(sandbox);
const src = fs.readFileSync(path.join(__dirname, "..", "frontend", "app.jsx"), "utf8");
try {
  vm.runInContext(src, ctx, { filename: "app.jsx" });
} catch (e) {
  console.log("LOAD_FAIL " + (e && e.message));
  process.exit(1);
}

const MAX = 4000;
let calls = 0;
const failures = [];

function walk(node, trail, seen) {
  if (calls > MAX || node == null || node === false || node === true) return;
  if (Array.isArray(node)) { node.forEach((n) => walk(n, trail, seen)); return; }
  if (typeof node !== "object" || !node.$$el) return;

  const t = node.type;
  if (typeof t === "function") {
    const name = t.name || "anonymous";
    const key = name + "|" + trail;
    if (!seen.has(key)) {
      seen.add(key);
      calls++;
      let out;
      try {
        out = t(Object.assign({}, node.props, { children: node.children }));
      } catch (e) {
        failures.push(trail + " > " + name + ": " + (e && e.message));
        return;
      }
      walk(out, trail + " > " + name, seen);
    }
  }
  walk(node.children, trail, seen);
  // props carry elements too: title, actions, panel, ctx, list, legend, tools
  for (const k of Object.keys(node.props || {})) {
    const v = node.props[k];
    if (v && (v.$$el || Array.isArray(v))) walk(v, trail, seen);
  }
}

const screens = ctx.TAB_MAP || {};
const names = Object.keys(screens);
if (!names.length) { console.log("LOAD_FAIL TAB_MAP is empty"); process.exit(1); }

/* Screens read their opening sub-tab through takeSub, which routeSub fills. That is
   the only handle on sub-tab state here, since useState is stubbed to its initial
   value, and it means every sub-tab gets rendered rather than only the default. */
const SUBTABS = ctx.SUBTABS || {};
let rendered = 0;

for (const name of names) {
  const Comp = screens[name];
  if (typeof Comp !== "function") { failures.push(name + ": not a component"); continue; }
  const subs = SUBTABS[name] && SUBTABS[name].length ? SUBTABS[name] : [null];
  for (const sub of subs) {
    if (sub && typeof ctx.routeSub === "function") ctx.routeSub(name, sub);
    const where = sub ? name + ":" + sub : name;
    let out;
    try {
      out = Comp({});
    } catch (e) {
      failures.push(where + ": " + (e && e.message));
      continue;
    }
    rendered++;
    walk(out, where, new Set());
  }
}

// The gates render outside the frame, so TAB_MAP does not reach them.
for (const extra of ["Login", "Setup"]) {
  const Comp = ctx[extra];
  if (typeof Comp !== "function") continue;
  try {
    walk(Comp({}), extra, new Set());
  } catch (e) {
    failures.push(extra + ": " + (e && e.message));
  }
}

console.log("SCREENS " + names.length);
console.log("VIEWS " + rendered);
console.log("COMPONENT_CALLS " + calls);
for (const f of failures) console.log("FAIL " + f);
if (!failures.length) console.log("OK");
