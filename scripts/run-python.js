#!/usr/bin/env node

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const projectRoot = path.resolve(__dirname, '..');
const defaultErrorMessage =
  '\nERROR: Python environment not found. Run ./setup.sh or activate your conda env.\n';

function resolvePython() {
  const candidates = [
    path.join(projectRoot, '.venv', 'Scripts', 'python.exe'),
    path.join(projectRoot, '.venv', 'bin', 'python'),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return process.env.PYTHON || 'python';
}

function parseArgs(argv) {
  let errorMessage = defaultErrorMessage;
  let printExecutable = false;
  let args = argv.slice();

  while (args.length > 0) {
    if (args[0] === '--error-message') {
      if (args.length < 2) {
        throw new Error('Missing value for --error-message');
      }
      errorMessage = args[1];
      args = args.slice(2);
      continue;
    }

    if (args[0] === '--print-executable') {
      printExecutable = true;
      args = args.slice(1);
      continue;
    }

    break;
  }

  return { errorMessage, printExecutable, args };
}

function buildPythonArgs(args) {
  if (args[0] === '--check-import') {
    const moduleName = args[1];
    if (!moduleName || args.length !== 2) {
      throw new Error('Usage: --check-import <module>');
    }
    return ['-c', 'import importlib, sys; importlib.import_module(sys.argv[1])', moduleName];
  }

  if (args.length === 0) {
    throw new Error('No Python arguments provided');
  }

  return args;
}

function run() {
  let parsed;
  try {
    parsed = parseArgs(process.argv.slice(2));
  } catch (error) {
    console.error(String(error.message || error));
    process.exit(1);
  }

  const python = resolvePython();

  if (parsed.printExecutable) {
    console.log(python);
    if (parsed.args.length === 0) {
      return;
    }
  }

  let pythonArgs;
  try {
    pythonArgs = buildPythonArgs(parsed.args);
  } catch (error) {
    console.error(String(error.message || error));
    process.exit(1);
  }

  const child = spawn(python, pythonArgs, {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: false,
  });

  child.on('error', () => {
    process.stderr.write(parsed.errorMessage);
    process.exit(1);
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }

    if (code === 0) {
      process.exit(0);
      return;
    }

    if (parsed.args[0] === '--check-import') {
      process.stderr.write(parsed.errorMessage);
    }
    process.exit(code ?? 1);
  });
}

run();
