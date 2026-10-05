// index.ts
//
// E2B quickstart: create a sandbox, run Python inside it, then look at the
// filesystem. Run it with:
//
//     npm install          # @e2b/code-interpreter + dotenv (+ tsx)
//     npx tsx ./index.ts
//
// Needs E2B_API_KEY. Set it in the environment, or put it in a .env file next to
// this file (gitignored -- never commit it):
//
//     E2B_API_KEY=e2b_...
//
// This runs on E2B's servers, so it needs outbound network access to
// api.e2b.app. If the sandbox you are running in blocks that, the error below
// says so explicitly instead of failing with a bare TLS stack trace.

import dotenv from 'dotenv'
import { Sandbox } from '@e2b/code-interpreter'

// dotenv does not override variables that already exist, which means an empty
// `export E2B_API_KEY=` in a shell profile would permanently shadow the value in
// .env. Keep the parsed file values so an empty variable can fall back to them.
const fromDotenv = dotenv.config().parsed ?? {}

const KEEP_SANDBOX = process.env.KEEP_SANDBOX === '1'

function requireKey(name: string): string {
  // a non-empty value in the environment wins; empty counts as unset
  const value = ((process.env[name] ?? '').trim() || (fromDotenv[name] ?? '').trim())
  if (!value) {
    console.error(
      `error: ${name} is not set.\n` +
        `  get one: https://e2b.dev/dashboard\n` +
        `  then:    export ${name}=...\n` +
        `  or put ${name}=... in a .env file next to index.ts (gitignored)`
    )
    process.exit(2)
  }
  return value
}

requireKey('E2B_API_KEY')

let sbx: Sandbox | undefined
try {
  // Creates a persistent sandbox session: it keeps running until you kill it or
  // it times out. We kill it in `finally` so a quickstart does not leave a
  // billable sandbox behind -- set KEEP_SANDBOX=1 to keep it alive instead.
  sbx = await Sandbox.create()

  const execution = await sbx.runCode('print("hello world")') // Execute Python inside the sandbox
  console.log(execution.logs)

  // A failed cell is not an exception: it comes back on the execution object.
  if (execution.error) {
    console.error(`\nthe cell raised ${execution.error.name}: ${execution.error.value}`)
    process.exitCode = 1
  }

  const files = await sbx.files.list('/')
  console.log(files)
} catch (err) {
  const message = err instanceof Error ? `${err.name}: ${err.message}` : String(err)
  const networkish = /connect|network|tls|handshake|fetch failed|ENOTFOUND|ETIMEDOUT/i.test(message)
  console.error(`error: could not create or use the sandbox.\n  ${message}`)
  if (networkish) {
    console.error(
      '  That looks like a network problem rather than a bad key: this script\n' +
        '  must reach api.e2b.app. A locked-down sandbox or an offline machine\n' +
        '  fails here even with a valid E2B_API_KEY.',
    )
  }
  process.exitCode = 3
} finally {
  if (sbx && !KEEP_SANDBOX) {
    await sbx.kill()
    console.log('\nsandbox killed (set KEEP_SANDBOX=1 to leave it running)')
  }
}
