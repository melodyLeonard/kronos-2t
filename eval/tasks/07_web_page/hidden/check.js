const path = require('path');
const fs = require('fs');
const { loadPage, Event, KeyboardEvent } = require('./minidom.js');
const wd = process.argv[2];
const fails = [];
const htmlPath = path.join(wd, 'index.html');
if (!fs.existsSync(htmlPath)) { console.log('index.html missing'); process.exit(1); }
const html = fs.readFileSync(htmlPath, 'utf8');
if (!/<!doctype html>/i.test(html)) fails.push('missing <!doctype html>');
const { doc, errors } = loadPage(htmlPath);
if (errors.length) fails.push('script errors: ' + errors.join(' | ').slice(0, 600));
const input = doc.getElementById('new-todo'), add = doc.getElementById('add'), list = doc.getElementById('todos'), count = doc.getElementById('count');
if (!input || !add || !list || !count) { fails.push(`missing elements: new-todo=${!!input} add=${!!add} todos=${!!list} count=${!!count}`); }
else {
  const items = () => list.querySelectorAll('li');
  const num = () => parseInt((count.textContent || '').trim(), 10);
  if (num() !== 0) fails.push(`initial count ${JSON.stringify(count.textContent)} want 0`);
  input.value = '  buy milk '; add.click();
  if (items().length !== 1) fails.push(`after add: ${items().length} li`);
  else if (!items()[0].textContent.includes('buy milk')) fails.push(`li text ${JSON.stringify(items()[0].textContent)}`);
  if (input.value !== '') fails.push(`input not cleared: ${JSON.stringify(input.value)}`);
  input.value = '   '; add.click();
  if (items().length !== 1) fails.push(`empty input added an item (${items().length})`);
  input.value = 'walk dog';
  input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 13, bubbles: true }));
  if (items().length !== 2) { input.dispatchEvent(new KeyboardEvent('keyup', { key: 'Enter', keyCode: 13, bubbles: true })); input.dispatchEvent(new KeyboardEvent('keypress', { key: 'Enter', keyCode: 13, bubbles: true })); }
  if (items().length !== 2) fails.push(`Enter did not add (${items().length} li)`);
  if (num() !== items().length) fails.push(`count ${num()} want ${items().length}`);
  const first = items()[0]; first.click();
  if (!first.classList.contains('done')) fails.push('click did not toggle class done');
  if (num() !== items().length - 1) fails.push(`count after done ${num()} want ${items().length - 1}`);
  first.click();
  if (first.classList.contains('done')) fails.push('second click did not untoggle');
  if (num() !== items().length) fails.push(`count after untoggle ${num()}`);
  if (!/\.done\b[^{]*\{[^}]*line-through/i.test(html)) fails.push('no .done line-through style');
}
console.log(fails.length ? fails.join('\n') : 'all checks passed');
process.exit(fails.length ? 1 : 0);
