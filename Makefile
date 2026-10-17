# Entry points for both halves.  Each half also has its own tooling: backend/Makefile (uv),
# frontend/package.json (npm).
.DEFAULT_GOAL := help

.PHONY: help backend frontend test backend-test frontend-test

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

backend:  ## run the backend API on http://127.0.0.1:8100 (see backend/README.md)
	$(MAKE) -C backend serve

frontend: frontend/node_modules  ## run the frontend on http://127.0.0.1:3100 (see frontend/README.md)
	cd frontend && npm run dev

frontend/node_modules: frontend/package-lock.json
	cd frontend && npm ci
	@touch frontend/node_modules

backend-test:  ## backend fast suite (engine and API; no models)
	$(MAKE) -C backend test

frontend-test: frontend/node_modules  ## frontend type-check and Vitest
	cd frontend && npm run typecheck && npm test

test: backend-test frontend-test  ## both
