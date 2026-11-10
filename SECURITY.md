# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for vulnerabilities.

Use GitHub's private vulnerability reporting on this repository
(*Security → Advisories → Report a vulnerability*). If that is unavailable,
contact the maintainer directly and include:

* affected version / commit,
* a minimal reproduction (redacted input is fine),
* impact assessment — especially anything that could make GuardLens execute
  log content, escape the guarded SQL sandbox, or leak credentials.

Expect an acknowledgement within a few days. Please give a reasonable window
before public disclosure.

## Scope

GuardLens is a local analysis tool. The security-relevant surfaces are:

| Surface | Guarantee today |
|---|---|
| Audit log content | parsed as data only; commands are tokenised with `shlex`, never executed |
| SQL from questions | `SELECT`/`WITH` only, executed on a copied read-only DuckDB file, dry-run + row cap |
| Credentials | never hard-coded, never logged; read from environment only |
| Network | no outbound calls except the LLM endpoint you configure |
| Serving | Qdrant and Streamlit bind to localhost; no auth is provided — do not expose them |

## Out of scope

* Running GuardLens against untrusted input on a host you do not control, or
  exposing the dashboard / API / Qdrant to the public internet.
* Prompt-injection attempts that only produce a wrong *answer*: the citation
  and abstention checks are the mitigation, and answers are advisory.
* The bundled synthetic sample.
