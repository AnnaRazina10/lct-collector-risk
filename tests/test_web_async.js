#!/usr/bin/env node
'use strict';

// Run: node tests/test_web_async.js
// Executes the actual inline application script with controlled fetch completion
// order. The small DOM double supports application events and text, not layout;
// visual/browser behavior and the real API have separate checks.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const webPath = path.join(root, 'web/index.html');
const webBytes = fs.readFileSync(webPath);
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const scripts = [...webBytes.toString('utf8').matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1, 'Expected one inline application script');
const bootstrap = /loadMode\(mode,runId,targetKind\);\s*$/;
assert.match(scripts[0][1], bootstrap, 'Application bootstrap changed; update test startup');
// Each case controls its own initial navigation; application functions, handlers,
// generation guards and rendering code are otherwise executed without changes.
const source = scripts[0][1].replace(bootstrap, '');
const ONSET = 'registered_episode_start_g1';

class Element {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.value = '';
    this.checked = false;
    this.disabled = false;
    this.style = {};
    this.attributes = {};
    this.ownText = '';
    const classes = new Set();
    this.classList = {
      add: name => classes.add(name),
      remove: name => classes.delete(name),
      toggle: (name, force) => {
        const enabled = force === undefined ? !classes.has(name) : force;
        if (enabled) classes.add(name); else classes.delete(name);
        return enabled;
      },
    };
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; this.ownText = ''; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  setCustomValidity(value) { this.validationMessage = value; }
  reportValidity() { return !this.validationMessage; }
  set textContent(value) { this.children = []; this.ownText = String(value); }
  get textContent() {
    return this.ownText + this.children.map(child => child.textContent ?? String(child)).join('');
  }
  get options() {
    return this.children.flatMap(child => child.tagName === 'option' ? [child] : child.options || []);
  }
}

function walk(element) {
  return [element, ...(element.children || []).flatMap(walk)];
}

function harness() {
  const elements = new Map();
  const pending = [];
  const requests = [];
  const document = {
    getElementById(id) {
      for (const element of elements.values()) {
        const found = walk(element).find(child => child.id === id);
        if (found) return found;
      }
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement: tag => new Element(tag),
    createTextNode: text => ({textContent: text}),
    querySelector() { return this.getElementById('badge'); },
  };
  const context = {
    document, URLSearchParams, URL, Intl, Date, Map, Number, Math, JSON, Promise,
    location: {search: '', href: 'http://test.invalid/'},
    history: {replaceState(_state, _title, url) { context.location.href = String(url); }},
    fetch(url, options) {
      return new Promise(resolve => {
        const request = {url, options, resolve};
        pending.push(request);
        requests.push(request);
      });
    },
  };
  vm.createContext(context);
  vm.runInContext(source, context, {filename: 'web/index.html'});
  return {
    context, pending, requests,
    run(code) { return vm.runInContext(code, context); },
    element(id) { return document.getElementById(id); },
    take(expectedUrl) {
      const request = pending.shift();
      assert(request, 'Expected a pending request: ' + expectedUrl);
      assert.equal(request.url, expectedUrl);
      return request;
    },
  };
}

function reply(request, body, status = 200) {
  request.resolve({ok: status < 400, json: async () => body});
}
const tick = () => new Promise(resolve => setImmediate(resolve));
const historyUrl = '/api/forecast-runs?mode=object&limit=500';
const riskUrl = id => '/api/risks?mode=object&run_id=' + id;
const journalUrl = '/api/journal?mode=object';

function metadata(id, day, target) {
  return {
    run_id: id, issue_time: day + 'T00:00:00+03:00',
    forecast_start: day + 'T00:00:00+03:00',
    ...(target ? {target_kind: target} : {}),
  };
}
function history() {
  // Deliberately unordered: latest onset must be chosen by issue time.
  return {runs: [
    metadata('onset-old', '2026-06-28', ONSET),
    metadata('any', '2026-06-29'),
    metadata('onset-new', '2026-06-29', ONSET),
  ]};
}
function payload(id, target) {
  const ids = Array.from({length: 12}, (_, index) => String(index + 1)).sort();
  return {
    run_id: id, mode: 'historical_replay', entity_mode: 'object',
    ...(target ? {target_kind: target, warning_policy: {
      kind: 'daily_top_k', k: 10, tie_break: 'score_desc_object_id_asc',
    }} : {}),
    feature_date: '2026-06-28', issue_time: '2026-06-29T00:00:00+03:00',
    forecast_start: '2026-06-30T00:00:00+03:00',
    forecast_end: '2026-07-01T00:00:00+03:00', minimum_lead_hours: 24,
    threshold: 0.5, archive_metadata: {created_at: '2026-09-29T00:00:00Z'},
    cards: ids.map((objectId, index) => ({
      id: id + '_' + objectId, object_id: objectId, object_name: 'Object ' + id + '/' + objectId,
      rank: index + 1, score: 0.5, warning: target ? index < 10 : true,
      observed_channels: 1, catalog_channels: 1, events_today: 2,
      alarm_days_7d: 0, alarm_channels: 0, explanation: [],
    })).reverse(),
  };
}
async function finishLoad(h, loading, data) {
  reply(h.take(riskUrl(data.run_id)), data);
  await tick();
  reply(h.take(journalUrl), {entries: []});
  await loading;
}
async function open(h, id, target = 'TARGET_ANY') {
  const loading = h.run(`loadMode('object',${JSON.stringify(id)},${target})`);
  reply(h.take(historyUrl), history());
  await tick();
  await finishLoad(h, loading, payload(id, target === 'TARGET_ONSET' ? ONSET : undefined));
}

const cases = [
  ['late_metadata', 'Late metadata cannot replace a newer selection', async () => {
    const h = harness();
    const old = h.run("loadMode('object',null,TARGET_ONSET)");
    const oldRequest = h.take(historyUrl);
    const current = h.run("loadMode('object','any',TARGET_ANY)");
    const currentRequest = h.take(historyUrl);
    reply(oldRequest, history());
    await tick();
    assert.equal(h.pending.length, 0, 'Superseded metadata must not start a card request');
    reply(currentRequest, history());
    await tick();
    await finishLoad(h, current, payload('any'));
    await old;
    assert.equal(h.run('data.run_id'), 'any');
    assert(h.element('detail').textContent.includes('Object any/'));
  }],
  ['late_cards', 'Late card payload cannot replace a newer selection', async () => {
    const h = harness();
    const old = h.run("loadMode('object','onset-old',TARGET_ONSET)");
    reply(h.take(historyUrl), history());
    await tick();
    const oldRequest = h.take(riskUrl('onset-old'));
    await open(h, 'any');
    reply(oldRequest, payload('onset-old', ONSET));
    await old;
    assert.equal(h.run('data.run_id'), 'any');
    assert.equal(h.run('targetKind'), 'any_alarm');
    assert(h.element('detail').textContent.includes('Object any/'));
    assert.equal(h.pending.length, 0);
  }],
  ['latest_onset', 'Latest onset comes from metadata and preserves saved ranks and ties', async () => {
    const h = harness();
    h.element('target-select').value = ONSET;
    const loading = h.element('target-select').onchange();
    reply(h.take(historyUrl), history());
    await tick();
    await finishLoad(h, loading, payload('onset-new', ONSET));
    assert.equal(h.run('runId'), 'onset-new');
    assert.equal(h.run('data.cards[0].rank'), 1);
    assert.equal(h.run('data.cards.filter(card=>card.warning).length'), 10);
    h.element('warnings').checked = true;
    h.element('warnings').onchange();
    assert.equal(h.element('list').children.length, 10, 'Ties must not expand the stored top-10 queue');
    assert.equal(h.run('data.cards.length'), 12, 'Display filter must not mutate the saved queue');
  }],
  ['invalid_run', 'An unknown explicit run stays an error without legacy fallback', async () => {
    const h = harness();
    const loading = h.run("loadMode('object','missing',TARGET_ONSET)");
    reply(h.take(historyUrl), history());
    await tick();
    reply(h.take(riskUrl('missing')), {detail: 'Выпуск не найден'}, 404);
    await loading;
    assert.equal(h.run('data'), null);
    assert.equal(h.run('runId'), 'missing');
    assert.equal(h.pending.length, 0);
    assert(h.element('notice').textContent.includes('Выпуск не найден'));
    assert.equal(h.requests.filter(request => request.url.startsWith('/api/risks')).length, 1);
  }],
  ['metadata_failure', 'Implicit onset never loads legacy when metadata fails', async () => {
    const h = harness();
    const loading = h.run("loadMode('object',null,TARGET_ONSET)");
    reply(h.take(historyUrl), {detail: 'Metadata unavailable'}, 503);
    await tick();
    reply(h.take(journalUrl), {entries: []});
    await loading;
    assert.equal(h.run('data'), null);
    assert.equal(h.run('targetKind'), ONSET);
    assert.equal(h.requests.filter(request => request.url.startsWith('/api/risks')).length, 0);
    assert(h.element('notice').textContent.includes('Не удалось получить выпуски'));
  }],
  ['explicit_target', 'An explicit onset run synchronizes the target using the actual response', async () => {
    const h = harness();
    const loading = h.run("loadMode('object','onset-new',TARGET_ANY)");
    reply(h.take(historyUrl), history());
    await tick();
    await finishLoad(h, loading, payload('onset-new', ONSET));
    assert.equal(h.run('targetKind'), ONSET);
    assert.equal(h.element('target-select').value, ONSET);
    const url = new URL(h.context.location.href);
    assert.equal(url.searchParams.get('run_id'), 'onset-new');
    assert.equal(url.searchParams.get('target'), ONSET);
    assert(h.element('detail').textContent.includes('Цель: Начало серии'));
  }],
  ['feedback_switch', 'In-flight feedback retains the original card/run after switching releases', async () => {
    const h = harness();
    await open(h, 'onset-old', 'TARGET_ONSET');
    const oldNodes = walk(h.element('detail'));
    const save = oldNodes.find(element => element.tagName === 'button' && element.textContent === 'Сохранить решение');
    assert(save, 'Expected the rendered dispatcher save action');
    h.element('operator-name').value = 'TEST REVIEW';
    h.element('decision').value = 'inspect';
    h.element('decision-reason').value = 'needs_inspection';
    h.element('decision-note').value = 'Decision for the original onset card';
    const saving = save.onclick();
    const post = h.take('/api/feedback');
    const body = JSON.parse(post.options.body);
    assert.equal(body.run_id, 'onset-old');
    assert.equal(body.risk_id, 'onset-old_1');
    assert.equal(body.entity_mode, 'object');
    assert.equal(body.note, 'Decision for the original onset card');
    await open(h, 'any');
    const currentDetail = h.element('detail').textContent;
    reply(post, {id: 'old-decision'});
    await saving;
    assert.equal(h.pending.length, 0, 'Old success must not start a request for the new view');
    assert.equal(h.run('data.run_id'), 'any');
    assert.equal(h.element('detail').textContent, currentDetail);
    assert.equal(h.element('feedback-message').textContent, '');
    assert.equal(h.requests.filter(request => request.options?.method === 'POST').length, 1);
  }],
];

async function main() {
  const results = [];
  let failure;
  for (const [id, description, test] of cases) {
    let timer;
    try {
      await Promise.race([
        test(),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error('Async scenario did not settle within 2 seconds')), 2000);
        }),
      ]);
      results.push({id, description, passed: true});
    } catch (error) {
      results.push({id, description, passed: false, error: error.stack});
      failure = error;
      break;
    } finally {
      clearTimeout(timer);
    }
  }
  const unchanged = webBytes.equals(fs.readFileSync(webPath));
  const report = {
    checked_at: new Date().toISOString(),
    command: 'node tests/test_web_async.js',
    node_version: process.version,
    web_file: 'web/index.html', web_sha256: sha256(webBytes),
    harness_file: 'tests/test_web_async.js', harness_sha256: sha256(fs.readFileSync(__filename)),
    web_unchanged_during_check: unchanged,
    scope: 'Actual inline UI script; minimal DOM double; controlled asynchronous fetch responses; no real network or database writes; not a visual/browser check.',
    cases: results,
    passed: !failure && unchanged && results.length === cases.length,
  };
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) process.exitCode = 1;
}

main().catch(error => { console.error(error); process.exitCode = 1; });
