# SatQuery AI — Frontend

React + TypeScript + Vite web app. See the repository root `README.md` for the full
overview and `.kiro/steering/frontend-ux.md` for UX rules.

## Scripts
```bash
npm install
npm run dev         # dev server (http://localhost:5173)
npm run build       # type-check + production build
npm run preview     # preview the production build
npm run test        # vitest (run once)
npm run lint        # eslint
```

## Structure
```
src/
├── main.tsx           entry (Router + QueryClient providers)
├── App.tsx            routes
├── lib/               api client (single source of backend calls) + utils
├── hooks/             data hooks (useHealth)
├── components/
│   ├── ui/            shadcn-style primitives (button, card, badge)
│   ├── layout/        Header
│   ├── panels/        Input / Visualization / Assistant / History
│   └── SystemStatus   accessible health indicator (icon + text)
└── pages/             Workspace, NotFound
tests/                 Vitest + React Testing Library
```

Config: set `VITE_API_BASE_URL` (see `.env.example`); defaults to
`http://localhost:8000/api/v1`.
