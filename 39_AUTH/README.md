# Module 39: Authentication & Authorization

Identity and access control for AI backends — the layer that keeps your
models, data, and users safe.

## What You Will Learn

- Authentication vs authorization
- Passwords: hashing (bcrypt/argon2), salting, timing attacks
- Sessions vs tokens
- JWT: structure, signing, expiry, refresh flows
- OAuth2 concepts and scopes
- API keys and rate limiting
- Role-based access control (RBAC)
- Securing ML endpoints: who may call your model?
- Multi-user RAG/agent systems: per-user data isolation
- Common auth failures and how they get exploited

## Module Files

| File | Topic |
|------|-------|
| auth_complete.txt | Full auth course |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

## Prerequisites

- 38_REST_APIS; 35_FASTAPI or 37_DJANGO to practice

## Exit Criteria

- [ ] You can implement password hashing correctly
- [ ] You can issue and verify JWTs
- [ ] You can design per-user authorization for an AI feature
