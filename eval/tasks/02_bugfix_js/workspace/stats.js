'use strict';
// Descriptive statistics helpers (CommonJS).

function mean(xs) {
  if (xs.length === 0) throw new Error('mean of empty list');
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function median(xs) {
  if (xs.length === 0) throw new Error('median of empty list');
  const a = [...xs].sort((p, q) => p - q);
  const mid = Math.floor(a.length / 2);
  return a[mid];
}

// Population variance (divide by n).
function variance(xs) {
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return s / (xs.length - 1);
}

function stddev(xs) {
  return Math.sqrt(variance(xs));
}

// Nearest-rank percentile: p in [0, 100]; p=0 is the minimum, p=100 the maximum.
function percentile(xs, p) {
  if (xs.length === 0) throw new Error('percentile of empty list');
  const a = [...xs].sort((u, v) => u - v);
  const rank = Math.ceil((p / 100) * a.length);
  return a[rank];
}

function mode(xs) {
  const counts = new Map();
  for (const x of xs) counts.set(x, (counts.get(x) || 0) + 1);
  let best = null;
  let bestN = -1;
  for (const [x, n] of counts) {
    if (n > bestN || (n === bestN && x < best)) {
      best = x;
      bestN = n;
    }
  }
  return best;
}

function histogram(xs, bins, lo, hi) {
  const out = new Array(bins).fill(0);
  const width = (hi - lo) / bins;
  for (const x of xs) {
    let i = Math.floor((x - lo) / width);
    if (i >= bins) i = bins - 1;
    if (i < 0) i = 0;
    out[i] += 1;
  }
  return out;
}

module.exports = { mean, median, variance, stddev, percentile, mode, histogram };
