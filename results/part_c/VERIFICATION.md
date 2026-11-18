# Part C verification

Verified on 2026-11-18:
- `python -m unittest discover -s tests -v`: 13 tests passed.
- Synthetic decoder smoke test: predicted route [10, 11, 12], including connecting arc 11.
- API contract tested with FastAPI TestClient and a synthetic scorer.
- Changing reference routes or removing them entirely leaves predictions identical.
- Syntax compilation passed for src/api, src/hmm, and scripts/run_decode.py.

The full test suite ran successfully outside the Codex sandbox; native imports and
TestClient stalled in sandboxed attempts. This is not a two-laptop network test.

Test environment: Python 3.12, FastAPI 0.115.12, Starlette 0.46.2, HTTPX 0.28.1,
scikit-learn 1.9.1, SciPy 1.18.1, joblib 1.6.0. These are test environment versions,
not a claim about B's model serialization environment. Match B's versions for deployment.

Not yet verified: saved-model inference, all real trips, decoder latency at full scale,
UI rendering of multiple segments, or two-laptop deployment. No trained model bundle
was present in the cloned repository. Synthetic fixtures are not experimental evidence
of map-matching accuracy on the assignment dataset.
