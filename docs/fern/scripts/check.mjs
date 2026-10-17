import {readFileSync, readdirSync, existsSync} from 'node:fs';
import {dirname, extname, relative, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {parse} from 'yaml';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repository = resolve(root, '../..');
const errors = [];
const report = (file, message) => errors.push(`${relative(root, file)}: ${message}`);
const inside = (parent, child) => child === parent || child.startsWith(parent + sep);
const walk = directory => readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
  const path = resolve(directory, entry.name);
  return entry.isDirectory() ? walk(path) : [path];
});
const pages = walk(resolve(root, 'pages')).filter(path => /\.mdx?$/.test(path));
if (existsSync(resolve(repository, '.git'))) {
  const ignored = spawnSync('git', ['check-ignore', '--stdin'], {
    cwd: repository, input: pages.join('\n') + '\n', encoding: 'utf8',
  });
  if (ignored.error || ![0, 1].includes(ignored.status)) report(root, 'Cannot verify Git ignore rules');
  for (const file of ignored.stdout?.trim().split('\n').filter(Boolean) ?? []) report(file, 'Public page is excluded by Git');
}
const navigation = new Map();
function visit(value) {
  if (Array.isArray(value)) return value.forEach(visit);
  if (!value || typeof value !== 'object') return;
  if (value.page && value.path) {
    const path = resolve(root, value.path);
    if (!inside(resolve(root, 'pages'), path)) report(path, 'Navigation must point into pages/');
    if (!existsSync(path)) report(path, 'Navigation target does not exist');
    navigation.set(path, (navigation.get(path) ?? 0) + 1);
  }
  Object.values(value).forEach(visit);
}
visit(parse(readFileSync(resolve(root, 'index.yml'), 'utf8')).navigation);

// Strip code before inspecting prose: examples may intentionally contain headings and links.
function prose(file, body) {
  let fence;
  const lines = [];
  for (const line of body.split('\n')) {
    const marker = line.match(/^\s*(`{3,}|~{3,})(.*)$/);
    if (marker) {
      if (!fence) {
        fence = marker[1];
        if (!marker[2].trim()) report(file, 'Code fence needs a language');
      } else if (marker[1][0] === fence[0] && marker[1].length >= fence.length && !marker[2].trim()) {
        fence = undefined;
      }
      lines.push('');
    } else lines.push(fence ? '' : line);
  }
  if (fence) report(file, 'Unclosed code fence');
  return lines.join('\n');
}
const slug = title => title.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^\p{L}\p{N}\p{M}_\-\s]/gu, '').replace(/\s/g, '-');
const documents = new Map();
for (const file of pages) {
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*\.mdx?$/.test(file.split(sep).at(-1))) report(file, 'Use kebab-case filenames');
  if ((navigation.get(file) ?? 0) !== 1) report(file, 'Page must appear exactly once in index.yml');
  const source = readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
  const match = source.match(/^---\n([\s\S]*?)\n---\n/);
  if (!match) { report(file, 'Missing frontmatter'); continue; }
  const metadata = parse(match[1]);
  for (const key of ['title', 'subtitle']) {
    if (typeof metadata?.[key] !== 'string' || !metadata[key].trim()) report(file, `Missing ${key}`);
  }
  if (!match[1].includes('SPDX-License-Identifier: MIT')) report(file, 'Missing project SPDX license');
  const body = prose(file, source.slice(match[0].length));
  const anchors = new Set();
  const repeats = new Map();
  let level = 1;
  for (const heading of body.matchAll(/^\s{0,3}(#{1,6})\s+(.+)$/gm)) {
    const next = heading[1].length;
    if (next === 1) report(file, 'Fern supplies H1; start body headings at H2');
    if (next > level + 1) report(file, 'Heading skips a level');
    level = next;
    const base = slug(heading[2]);
    const occurrence = repeats.get(base) ?? 0;
    anchors.add(occurrence ? `${base}-${occurrence}` : base);
    repeats.set(base, occurrence + 1);
  }
  if (extname(file) === '.md' && /<(Steps|Step|Tabs|Tab|Card|CardGroup|Note|Tip|Info|Warning|Error)\b/.test(body)) report(file, 'Fern components require .mdx');
  documents.set(file, {body, anchors});
}
for (const [file, {body}] of documents) {
  const targets = [
    ...Array.from(body.matchAll(/\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)/g), match => match[1]),
    ...Array.from(body.matchAll(/\bhref=["']([^"']+)["']/g), match => match[1]),
  ];
  for (const target of targets) {
    const source = target.match(/^https:\/\/github\.com\/BGWH123\/Ember\/(?:blob|tree)\/main\/([^#?]+)/);
    if (source && !existsSync(resolve(repository, decodeURIComponent(source[1])))) report(file, `Source target missing locally: ${target}`);
    if (/^[a-z][a-z0-9+.-]*:/i.test(target) || target.startsWith('//')) continue;
    if (target.startsWith('/')) { report(file, `Use a relative source-file link: ${target}`); continue; }
    const [path, fragment] = target.split('#');
    const destination = path ? resolve(dirname(file), decodeURIComponent(path)) : file;
    if (!inside(root, destination)) { report(file, `Link escapes Fern source: ${target}`); continue; }
    if (!existsSync(destination)) { report(file, `Broken link: ${target}`); continue; }
    if (fragment && documents.has(destination) && !documents.get(destination).anchors.has(decodeURIComponent(fragment))) report(file, `Unknown heading: ${target}`);
  }
}
const config = parse(readFileSync(resolve(root, 'docs.yml'), 'utf8'));
for (const asset of [config.logo?.light, config.logo?.dark, config.favicon, ...(config.css ?? [])]) {
  if (asset && !existsSync(resolve(root, asset))) report(resolve(root, 'docs.yml'), `Missing asset: ${asset}`);
}
if (errors.length) {
  process.stderr.write(errors.join('\n') + '\n');
  process.exitCode = 1;
} else process.stdout.write(`Checked ${pages.length} pages: navigation, metadata, headings, fences, links and assets.\n`);
