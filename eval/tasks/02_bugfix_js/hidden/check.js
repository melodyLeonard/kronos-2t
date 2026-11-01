const path = require('path');
const s = require(path.resolve(process.argv[2], 'stats.js'));
const fails = [];
function eq(name, got, want) {
  const ok = typeof want === 'number' ? Math.abs(got - want) < 1e-9 : JSON.stringify(got) === JSON.stringify(want);
  if (!ok) fails.push(`${name}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`);
}
eq('median_even', s.median([1, 2, 3, 4]), 2.5);
eq('median_odd', s.median([5, 1, 3]), 3);
eq('median_two', s.median([10, 20]), 15);
eq('variance', s.variance([2, 4, 4, 4, 5, 5, 7, 9]), 4);
eq('stddev', s.stddev([2, 4, 4, 4, 5, 5, 7, 9]), 2);
eq('pct100', s.percentile([1, 2, 3, 4, 5], 100), 5);
eq('pct0', s.percentile([1, 2, 3, 4, 5], 0), 1);
eq('pct50', s.percentile([1, 2, 3, 4, 5], 50), 3);
eq('pct90', s.percentile([15, 20, 35, 40, 50], 90), 50);
eq('pct30', s.percentile([15, 20, 35, 40, 50], 30), 20);
eq('mode', s.mode([1, 2, 2, 3, 3]), 2);
eq('hist', s.histogram([0, 1, 2, 3, 9, 10], 2, 0, 10), [4, 2]);
console.log(fails.length ? fails.join('\n') : 'all checks passed');
process.exit(fails.length ? 1 : 0);
