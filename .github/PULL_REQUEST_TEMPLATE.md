## What and why

<!-- One paragraph: the change, the motivation, the trade-off you accepted. -->

## Type

- [ ] feature
- [ ] fix
- [ ] performance
- [ ] docs
- [ ] test / refactor

## Design invariants

- [ ] Numbers still come from guarded SQL only (no model arithmetic)
- [ ] Citations are still verified against the retrieved set
- [ ] Abstention still happens when evidence is missing
- [ ] SQL remains SELECT/WITH on a read-only copy with a row cap
- [ ] No new outbound network call, no secret, no real log line or personal path

## Verification

<!-- Commands you ran and what they printed. Retrieval changes need ablation numbers. -->

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Notes for reviewers

<!-- Anything delicate, alternatives rejected, follow-up work. -->
