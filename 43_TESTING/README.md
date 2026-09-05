# Module 43: Testing

Writing tests that catch broken data, broken models, and broken APIs —
before your users do.

## What You Will Learn

- Unit vs integration vs end-to-end tests
- pytest: fixtures, parametrize, assertions, plugins
- Testing data: schemas, distributions, invariants
- Testing preprocessing (the usual silent-bug zone)
- Testing models: shapes, determinism, basic sanity metrics
- Testing APIs: FastAPI TestClient, Django test client
- Mocking external calls (LLM/API) for fast tests
- Test-driven fixes: write the failing test first
- CI integration: run tests on every push

## Module Files

| File | Topic |
|------|-------|
| testing_complete.txt | Full testing course |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

## Prerequisites

- 07_DATA_PREPROCESSING, 35_FASTAPI (things to test)

## Exit Criteria

- [ ] Your preprocessing + model + API all have passing tests
- [ ] You can mock an external LLM call in a test
- [ ] Tests run in CI on every push
