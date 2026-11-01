'use strict';
// Minimal DOM shim for headless checks of simple inline-script pages.
const fs = require('fs');
const vm = require('vm');

class ClassList {
  constructor(el) { this.el = el; }
  _set() { return new Set((this.el.className || '').split(/\s+/).filter(Boolean)); }
  _store(s) { this.el.className = [...s].join(' '); }
  add(...cs) { const s = this._set(); cs.forEach((c) => s.add(c)); this._store(s); }
  remove(...cs) { const s = this._set(); cs.forEach((c) => s.delete(c)); this._store(s); }
  toggle(c, force) { const s = this._set(); const has = s.has(c); const want = force === undefined ? !has : !!force; if (want) s.add(c); else s.delete(c); this._store(s); return want; }
  contains(c) { return this._set().has(c); }
}

class Element {
  constructor(tag, doc) {
    this.tagName = tag.toUpperCase(); this.nodeType = 1; this.children = []; this.childNodes = this.children;
    this.parentNode = null; this.attributes = {}; this.className = ''; this._text = ''; this.value = '';
    this.listeners = {}; this.style = {}; this.dataset = {}; this.classList = new ClassList(this); this._doc = doc; this.checked = false;
  }
  get id() { return this.attributes.id || ''; }
  set id(v) { this.attributes.id = v; }
  get textContent() { return this.children.length ? this.children.map((c) => c.textContent).join('') : this._text; }
  set textContent(v) { this.children = []; this.childNodes = this.children; this._text = String(v); }
  get innerText() { return this.textContent; }
  set innerText(v) { this.textContent = v; }
  get innerHTML() { return this.children.length ? this.children.map((c) => c.outerHTML).join('') : this._text; }
  set innerHTML(v) { this.children = []; this.childNodes = this.children; this._text = ''; parseInto(this, String(v), this._doc); }
  get outerHTML() { return `<${this.tagName.toLowerCase()}>${this.innerHTML}</${this.tagName.toLowerCase()}>`; }
  get firstChild() { return this.children[0] || null; }
  get lastChild() { return this.children[this.children.length - 1] || null; }
  get firstElementChild() { return this.firstChild; }
  get parentElement() { return this.parentNode; }
  get nextSibling() { if (!this.parentNode) return null; const i = this.parentNode.children.indexOf(this); return this.parentNode.children[i + 1] || null; }
  getAttribute(k) { return k in this.attributes ? this.attributes[k] : null; }
  setAttribute(k, v) { this.attributes[k] = String(v); if (k === 'class') this.className = String(v); if (k === 'value') this.value = String(v); }
  removeAttribute(k) { delete this.attributes[k]; }
  hasAttribute(k) { return k in this.attributes; }
  appendChild(c) { if (c.parentNode) c.parentNode.removeChild(c); c.parentNode = this; this.children.push(c); return c; }
  append(...cs) { cs.forEach((c) => this.appendChild(typeof c === 'string' ? this._doc.createTextNode(c) : c)); }
  prepend(c) { if (c.parentNode) c.parentNode.removeChild(c); c.parentNode = this; this.children.unshift(c); }
  insertBefore(c, ref) { if (c.parentNode) c.parentNode.removeChild(c); const i = ref ? this.children.indexOf(ref) : -1; c.parentNode = this; if (i < 0) this.children.push(c); else this.children.splice(i, 0, c); return c; }
  removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parentNode = null; return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  replaceChildren(...cs) { this.children = []; this.childNodes = this.children; cs.forEach((c) => this.appendChild(c)); }
  contains(n) { return n === this || this.children.some((c) => c.contains && c.contains(n)); }
  closest(sel) { let e = this; while (e && e.nodeType === 1) { if (matches(e, sel)) return e; e = e.parentNode; } return null; }
  matches(sel) { return matches(this, sel); }
  addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); }
  removeEventListener(t, fn) { this.listeners[t] = (this.listeners[t] || []).filter((f) => f !== fn); }
  dispatchEvent(ev) {
    ev.target = ev.target || this; let node = this;
    while (node) {
      ev.currentTarget = node;
      for (const fn of (node.listeners[ev.type] || []).slice()) fn.call(node, ev);
      const h = node['on' + ev.type]; if (typeof h === 'function') h.call(node, ev);
      if (ev._stop || !ev.bubbles) break; node = node.parentNode;
    }
    return !ev.defaultPrevented;
  }
  click() { this.dispatchEvent(new Event('click', { bubbles: true })); }
  focus() {} blur() {} select() {}
  querySelectorAll(sel) { const out = []; walk(this, (e) => { if (matches(e, sel)) out.push(e); }); return out; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  getElementsByTagName(t) { return this.querySelectorAll(t); }
  getElementsByClassName(c) { return this.querySelectorAll('.' + c); }
}
class Text { constructor(t) { this.nodeType = 3; this._t = String(t); this.parentNode = null; } get textContent() { return this._t; } set textContent(v) { this._t = String(v); } get outerHTML() { return this._t; } get innerHTML() { return this._t; } get children() { return []; } }
class Event { constructor(type, init = {}) { this.type = type; this.bubbles = !!init.bubbles; this.defaultPrevented = false; this.key = init.key; this.keyCode = init.keyCode; this.code = init.code; this.which = init.keyCode; } preventDefault() { this.defaultPrevented = true; } stopPropagation() { this._stop = true; } }
class KeyboardEvent extends Event {}
class CustomEvent extends Event {}
function walk(n, f) { for (const c of n.children || []) { if (c.nodeType === 1) { f(c); walk(c, f); } } }
function matches(e, sel) {
  return sel.split(',').some((s) => {
    const parts = s.trim().split(/\s+/); const last = parts[parts.length - 1];
    if (!matchSimple(e, last)) return false;
    let anc = e.parentNode;
    for (let i = parts.length - 2; i >= 0; i--) { while (anc && anc.nodeType === 1 && !matchSimple(anc, parts[i])) anc = anc.parentNode; if (!anc || anc.nodeType !== 1) return false; anc = anc.parentNode; }
    return true;
  });
}
function matchSimple(e, s) {
  const notM = s.match(/^(.*?):not\(([^)]+)\)$/);
  if (notM) return (notM[1] === '' || matchSimple(e, notM[1])) && !matchSimple(e, notM[2]);
  s = s.replace(/:(first-child|last-child|checked|hover|focus)$/, (_, p) => { if (p === 'first-child' && e.parentNode && e.parentNode.children[0] !== e) return ':no'; if (p === 'last-child' && e.parentNode && e.parentNode.children[e.parentNode.children.length - 1] !== e) return ':no'; if (p === 'checked' && !e.checked) return ':no'; return ''; });
  if (s.endsWith(':no')) return false;
  if (s === '*') return true;
  const m = s.match(/^([a-zA-Z][\w-]*)?(#[\w-]+)?((?:\.[\w-]+)*)(\[[^\]]+\])?$/); if (!m) return false;
  if (m[1] && e.tagName !== m[1].toUpperCase()) return false;
  if (m[2] && e.id !== m[2].slice(1)) return false;
  for (const c of (m[3] || '').split('.').filter(Boolean)) if (!e.classList.contains(c)) return false;
  if (m[4]) { const a = m[4].slice(1, -1).split('='); const k = a[0]; const v = a[1] ? a[1].replace(/^["']|["']$/g, '') : null; if (!(k in e.attributes) || (v !== null && e.attributes[k] !== v)) return false; }
  return true;
}
const VOID = new Set(['br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'source', 'track', 'wbr']);
function parseInto(parent, html, doc) {
  const re = /<!--[\s\S]*?-->|<!doctype[^>]*>|<\/([a-zA-Z][\w-]*)\s*>|<([a-zA-Z][\w-]*)((?:\s+[^\s=>\/]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?)*)\s*(\/?)>|([^<]+)/gi;
  let stack = [parent]; let m;
  while ((m = re.exec(html))) {
    const cur = stack[stack.length - 1];
    if (m[0].startsWith('<!')) continue;
    if (m[1]) { const t = m[1].toLowerCase(); for (let i = stack.length - 1; i > 0; i--) if (stack[i].tagName.toLowerCase() === t) { stack.length = i; break; } continue; }
    if (m[2]) {
      const tag = m[2].toLowerCase(); const el = doc.createElement(tag);
      const attrRe = /([^\s=\/]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g; let a;
      while ((a = attrRe.exec(m[3] || ''))) { const v = a[2] !== undefined ? a[2] : a[3] !== undefined ? a[3] : a[4] !== undefined ? a[4] : ''; el.setAttribute(a[1], decode(v)); }
      cur.appendChild(el);
      if (tag === 'script' || tag === 'style') {
        const close = new RegExp(`</${tag}\\s*>`, 'i'); const rest = html.slice(re.lastIndex); const idx = rest.search(close);
        const body = idx >= 0 ? rest.slice(0, idx) : rest; el._text = body; re.lastIndex += idx >= 0 ? idx + rest.match(close)[0].length : rest.length;
        if (tag === 'script') doc._scripts.push(el);
        continue;
      }
      if (!m[4] && !VOID.has(tag)) stack.push(el);
      continue;
    }
    if (m[5]) { const t = decode(m[5]); if (t.trim() || cur.tagName !== 'HTML') cur.appendChild(new Text(t)); }
  }
}
function decode(s) { return s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&'); }

function loadPage(htmlPath) {
  const html = fs.readFileSync(htmlPath, 'utf8');
  const doc = { _scripts: [], listeners: {}, readyState: 'loading', nodeType: 9, parentNode: null };
  doc.createElement = (t) => new Element(t, doc);
  doc.createTextNode = (t) => new Text(t);
  doc.createDocumentFragment = () => new Element('fragment', doc);
  doc.documentElement = new Element('html', doc); doc.documentElement.parentNode = null;
  doc.children = [doc.documentElement];
  parseInto(doc.documentElement, html, doc);
  const findTag = (t) => doc.documentElement.querySelector(t) || doc.documentElement;
  doc.head = findTag('head'); doc.body = findTag('body');
  doc.getElementById = (id) => doc.documentElement.querySelector('#' + id) || (doc.documentElement.id === id ? doc.documentElement : null);
  doc.querySelector = (s) => doc.documentElement.querySelector(s);
  doc.querySelectorAll = (s) => doc.documentElement.querySelectorAll(s);
  doc.getElementsByTagName = (t) => doc.documentElement.getElementsByTagName(t);
  doc.getElementsByClassName = (c) => doc.documentElement.getElementsByClassName(c);
  doc.addEventListener = (t, fn) => (doc.listeners[t] = doc.listeners[t] || []).push(fn);
  doc.removeEventListener = () => {};
  doc.dispatchEvent = (ev) => { for (const fn of (doc.listeners[ev.type] || []).slice()) fn.call(doc, ev); return true; };
  doc.activeElement = doc.body;
  const storage = () => { const m = new Map(); return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k), clear: () => m.clear(), key: (i) => [...m.keys()][i] || null, get length() { return m.size; } }; };
  const win = { document: doc, console, setTimeout, clearTimeout, setInterval, clearInterval, Event, KeyboardEvent, CustomEvent, localStorage: storage(), sessionStorage: storage(), alert() {}, confirm: () => true, prompt: () => '', requestAnimationFrame: (f) => setTimeout(f, 0), navigator: { userAgent: 'minidom' }, location: { href: 'file://' + htmlPath, hash: '' }, Math, JSON, Date, Array, Object, String, Number, Boolean, Map, Set, Promise, parseInt, parseFloat, isNaN, Node: Element, HTMLElement: Element, Element, getComputedStyle: () => ({}), crypto: { randomUUID: () => 'id-' + Math.random().toString(16).slice(2) } };
  win.window = win; win.self = win; win.globalThis = win; win.listeners = {};
  win.addEventListener = (t, fn) => (win.listeners[t] = win.listeners[t] || []).push(fn);
  win.removeEventListener = () => {};
  win.dispatchEvent = (ev) => { for (const fn of (win.listeners[ev.type] || []).slice()) fn.call(win, ev); return true; };
  doc.defaultView = win;
  const ctx = vm.createContext(win);
  const errors = [];
  for (const s of doc._scripts) {
    if (s.getAttribute('src')) { errors.push('external script not allowed: ' + s.getAttribute('src')); continue; }
    try { vm.runInContext(s._text, ctx, { filename: 'inline-script' }); } catch (e) { errors.push(String(e && e.stack || e)); }
  }
  doc.readyState = 'complete';
  try { doc.dispatchEvent(new Event('DOMContentLoaded')); win.dispatchEvent(new Event('DOMContentLoaded')); win.dispatchEvent(new Event('load')); if (typeof win.onload === 'function') win.onload(new Event('load')); } catch (e) { errors.push(String(e && e.stack || e)); }
  return { win, doc, errors, Event, KeyboardEvent };
}
module.exports = { loadPage, Event, KeyboardEvent };
