import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const docsRoot = fileURLToPath(new URL('../..', import.meta.url));
const require = createRequire(import.meta.url);
const fern = resolve(dirname(require.resolve('fern-api/package.json')), 'cli.cjs');

// Fern's telemetry requests can race process.exit() in Node 24 on Windows
// (nodejs/node#56645). These local checks need no CLI telemetry. Scope the
// opt-out to child processes; keep every validator and its actual exit code.
const env = { ...process.env, FERN_DISABLE_TELEMETRY: 'true' };
const steps = [
  [resolve(docsRoot, 'fern/scripts/check.mjs')],
  [fern, 'check'],
  [fern, 'docs', 'md', 'check'],
];

for (const args of steps) {
  const result = spawnSync(process.execPath, args, {
    cwd: docsRoot,
    env,
    stdio: 'inherit',
  });
  if (result.error) {
    console.error(result.error.message);
  }
  if (result.status !== 0) {
    if (result.signal) console.error(`Documentation validator stopped by ${result.signal}`);
    process.exitCode = result.status ?? 1;
    break;
  }
}
