# Known issues

Open problems that are understood but deliberately left for after the demo. Each one says what
breaks, when it matters, and the intended fix.

## 1. Signed-in users can see and delete each other's jobs

**Severity: high once more than one person can sign in.** Fine for a single-operator demo.

The frontend's API routes check only that *someone* is signed in
(`guard()` in [frontend/src/lib/server/http.ts](frontend/src/lib/server/http.ts)). Every job route
then calls the backend with the one shared `MLREDACT_API_TOKEN`, and a job record has no owner, so
neither side can tell whose job it is.

With two entries in `MLREDACT_UI_USERS`, or a Google domain in
`MLREDACT_UI_GOOGLE_ALLOWED_DOMAINS`, any signed-in user can:

- list every job on `/jobs` and in the recent-jobs panel;
- open any job and get fresh signed links to its redacted PDF, manifest and sealed manifest
  (`GET /api/jobs/<id>`, `/api/jobs/<id>/files/<name>`);
- cancel or delete any job (`POST /api/jobs/<id>/cancel`, `DELETE /api/jobs/<id>`).

**Until it is fixed:** configure a single user (or a single trusted team that may share all jobs).

**Intended fix:**

1. Backend: add an `owner` to `CreateJob` and `JobRecord` (an opaque id, such as a hash of the
   session email, not the email itself, so job records stay free of personal data). Have the
   frontend send it as a header on every server-to-server call, and filter `GET /v1/jobs` by it.
   Return `404` (not `403`) for another owner's job on get, cancel, delete and link signing, so job
   ids cannot be probed.
2. Frontend: derive the owner id from the next-auth session in `guard()` and pass it through
   `backend()` ([frontend/src/lib/server/backend.ts](frontend/src/lib/server/backend.ts)) on every
   job route.
3. Decide whether there is an admin role that sees all jobs (the System page's queue view
   probably wants it).
4. Tests: a two-user API test covering list, get, files, cancel and delete; and an e2e case in
   `frontend/e2e/`.

## Residual limits of the fixes on `fix/pr1-review-findings`

Not bugs today, but worth knowing:

- **Windows tree kill after the CLI exits.** `kill_tree` now reaches page workers that outlive the
  CLI on POSIX, by signalling the process group. On Windows, `taskkill /T` walks the tree from
  the CLI's pid, so it cannot find workers once the CLI has exited. Production runs on Linux; a
  Windows fix would need a Job Object.
- **Job list cost.** `GET /v1/jobs` no longer re-parses unchanged records, but it still lists the
  jobs directory and stats each `job.json` on every poll. If history grows into the thousands, add
  a retention policy or a paged `GET /v1/jobs?limit=`.
- **Records from before the size check.** A job created before `expectedBytes` was recorded skips
  the declared-size check on upload. It only affects uploads in flight during the deploy.
