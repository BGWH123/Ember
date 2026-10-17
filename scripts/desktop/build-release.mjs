import {writeFileSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../../',import.meta.url));
const required=['EMBER_EVALUATOR_PUBLIC_KEY','EMBER_SHELL_UPDATE_URL','EMBER_SHELL_UPDATE_PUBLIC_KEY','TAURI_SIGNING_PRIVATE_KEY'];
for(const name of required)if(!process.env[name])throw new Error(`Missing ${name}`);
if(!/^[a-fA-F0-9]{64}$/.test(process.env.EMBER_EVALUATOR_PUBLIC_KEY))throw new Error('Invalid evaluator Ed25519 public key');
if(!process.env.EMBER_SHELL_UPDATE_URL.startsWith('https://'))throw new Error('Shell updates require HTTPS');
mkdirSync(resolve(root,'artifacts/desktop'),{recursive:true});
const config=resolve(root,'artifacts/desktop/release-tauri.conf.json');
writeFileSync(config,JSON.stringify({bundle:{createUpdaterArtifacts:true},plugins:{updater:{pubkey:process.env.EMBER_SHELL_UPDATE_PUBLIC_KEY,endpoints:[process.env.EMBER_SHELL_UPDATE_URL]}}}));
// Launch the installed JS entry with Node; spawning npm.cmd directly is not
// portable on Windows, and a shell would misinterpret spaces/metacharacters.
const cli=resolve(root,'apps/desktop/node_modules/@tauri-apps/cli/tauri.js');
const result=spawnSync(process.execPath,[cli,'build','--config',config,...process.argv.slice(2)],{cwd:resolve(root,'apps/desktop'),stdio:'inherit',env:process.env});
if(result.error)throw result.error;
process.exit(result.status??1);
