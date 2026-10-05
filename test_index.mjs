// Tests for index.ts -- all offline.
//
// api.e2b.app is not reachable from a locked-down sandbox, so the real Sandbox
// cannot be exercised here. This runs the *actual* index.ts in a throwaway
// directory whose node_modules/@e2b/code-interpreter is a stub, which proves the
// script's own sequencing (create -> runCode -> logs -> files.list -> kill) and
// its error handling. dotenv and tsx are symlinked from the repo, so the real
// loader and the real TypeScript runtime are used.
//
//     npm test

import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, copyFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const TSX = join(HERE, 'node_modules', 'tsx', 'dist', 'cli.mjs')

const STUB = `
export class Sandbox {
  static async create() {
    if (process.env.STUB_MODE === 'fail') {
      throw new TypeError('fetch failed')
    }
    if (process.env.STUB_MODE === 'cell-error') {
      return new Sandbox()
    }
    return new Sandbox()
  }
  async runCode(code) {
    if (process.env.STUB_MODE === 'cell-error') {
      return { logs: { stdout: [], stderr: ['ValueError: boom\\n'] },
               error: { name: 'ValueError', value: 'boom' }, results: [] }
    }
    return { logs: { stdout: ['hello world\\n'], stderr: [] }, error: undefined, results: [] }
  }
  get files() {
    return { list: async (path) => [
      { path: '/tmp', type: 'dir' },
      { path: '/home/user', type: 'dir' },
    ] }
  }
  async kill() { console.log('[stub] killed the sandbox') }
}
`

function makeProject() {
  const dir = mkdtempSync(join(tmpdir(), 'e2b-ts-test-'))
  // index.ts uses top-level await, which needs ESM: without this package.json
  // Node treats it as CommonJS and tsx fails with "top-level await is currently
  // not supported". (Same reason the repo root needs one.)
  writeFileSync(join(dir, 'package.json'), JSON.stringify({
    name: 'tmp-e2b-ts', private: true, type: 'module',
  }))
  copyFileSync(join(HERE, 'index.ts'), join(dir, 'index.ts'))
  const pkg = join(dir, 'node_modules', '@e2b', 'code-interpreter')
  mkdirSync(pkg, { recursive: true })
  writeFileSync(join(pkg, 'package.json'), JSON.stringify({
    name: '@e2b/code-interpreter', version: '0.0.0-stub', type: 'module',
    main: 'index.js', exports: { '.': './index.js' },
  }))
  writeFileSync(join(pkg, 'index.js'), STUB)
  for (const dep of ['dotenv']) {
    symlinkSync(join(HERE, 'node_modules', dep), join(dir, 'node_modules', dep), 'dir')
  }
  return dir
}

function run(dir, env = {}, args = ['./index.ts']) {
  const result = spawnSync(process.execPath, [TSX, ...args], {
    cwd: dir,
    env: { ...process.env, ...env },
    encoding: 'utf8',
  })
  return { code: result.status, out: result.stdout ?? '', err: result.stderr ?? '' }
}

let passed = 0
let failed = 0
function check(label, ok, detail = '') {
  if (ok) {
    passed++
    console.log(`  OK   ${label}`)
  } else {
    failed++
    console.log(`  FAIL ${label}${detail ? `\n        ${detail}` : ''}`)
  }
  return ok
}

const dir = makeProject()
try {
  console.log('the happy path (stub sandbox, real index.ts, real tsx)')
  {
    const { code, out, err } = run(dir, { E2B_API_KEY: 'e2b_fake', KEEP_SANDBOX: '' })
    check('exit code 0', code === 0, `got ${code}; stderr=${err.slice(0, 200)}`)
    check('prints the execution logs', out.includes('hello world'), out.slice(0, 200))
    check('prints the file list', out.includes('/home/user'), out.slice(0, 200))
    check('kills the sandbox by default', out.includes('sandbox killed'), out.slice(0, 300))
    check('never prints the key', !out.includes('e2b_fake') && !err.includes('e2b_fake'))
  }

  console.log('\nKEEP_SANDBOX=1 leaves it running')
  {
    const { out } = run(dir, { E2B_API_KEY: 'e2b_fake', KEEP_SANDBOX: '1' })
    check('no kill message', !out.includes('sandbox killed'), out.slice(0, 200))
  }

  console.log('\nmissing key')
  {
    const { code, err } = run(dir, { E2B_API_KEY: '', KEEP_SANDBOX: '' })
    check('exit code 2', code === 2, `got ${code}`)
    check('names the variable', err.includes('E2B_API_KEY'), err.slice(0, 200))
    check('says where to get one', err.includes('e2b.dev/dashboard'), err.slice(0, 200))
  }

  console.log('\nthe key comes from .env too (real dotenv)')
  {
    const dir2 = makeProject()
    writeFileSync(join(dir2, '.env'), 'E2B_API_KEY=e2b_from_dotenv\n')
    const { code, out } = run(dir2, { E2B_API_KEY: '', KEEP_SANDBOX: '' })
    check('exit code 0', code === 0, `got ${code}`)
    check('ran using the .env key', out.includes('hello world'), out.slice(0, 200))
    check('an empty variable does not shadow .env', code === 0, `got ${code}`)
    check('key not echoed', !out.includes('e2b_from_dotenv'))
    rmSync(dir2, { recursive: true, force: true })
  }

  console.log('\na network failure is diagnosed as one')
  {
    const { code, err } = run(dir, { E2B_API_KEY: 'e2b_fake', STUB_MODE: 'fail' })
    check('exit code 3', code === 3, `got ${code}`)
    check('recognises the network cause',
      err.includes('network problem rather than a bad key'), err.slice(0, 300))
  }

  console.log('\na failing cell is reported, not thrown')
  {
    const { code, out, err } = run(dir, { E2B_API_KEY: 'e2b_fake', STUB_MODE: 'cell-error' })
    check('exit code 1', code === 1, `got ${code}`)
    check('names the exception', err.includes('ValueError: boom'), err.slice(0, 200))
    check('still lists files afterwards', out.includes('/home/user'), out.slice(0, 200))
  }
} finally {
  rmSync(dir, { recursive: true, force: true })
  if (!existsSync(TSX)) console.log('(warning: tsx binary not found where expected)')
}

console.log(`\n${passed}/${passed + failed} checks passed`)
process.exit(failed === 0 ? 0 : 1)
