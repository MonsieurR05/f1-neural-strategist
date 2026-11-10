# Web app

Next.js 16 (App Router, Cache Components) + TypeScript + Tailwind 4. A dark "pit wall"
interface on top of the strategy API (`services/api`).

| Screen | What it does |
|---|---|
| Race list (`/`) | Bahrain races up front, every season below; wet races and races with unrepresentative qualifying are shown but not simulatable |
| Race workspace (`/race/[raceId]`) | **Replay**: 2D circuit map with every car placed by its gap to the leader, play/scrub controls, SC/VSC status. **Gap chart**: gap to the leader by lap, focus and comparison drivers highlighted, neutralisations shaded, hover for values, click to jump. **Strategy builder**: drag stops, click stints to change tyres, then simulate as raced vs yours (300 races each, same random draws). **Optimiser**: ranked strategies with odds, time vs as raced, cost breakdown and explanation; load any into the builder. **Classification**: the table view |

Simulated data is always labelled and drawn dashed; historical data is solid. Car positions
on the map are interpolated from lap timing.

## Run

Start the API first (see `services/api/README.md`), then:

```bash
cp .env.example .env.local   # optional: only if the API isn't on localhost:8000
npm install
npm run dev
```

Open http://localhost:3000.

```bash
npm run lint
npx tsc --noEmit
npm run build
```

## Deployment (Vercel)

Live at https://f1-neural-strategist.vercel.app. The Vercel project (`f1-neural-strategist` in
the "rahul's projects" team) is not connected to GitHub, so pushing does not deploy it: deploy
with the CLI. It has `API_URL` and `NEXT_PUBLIC_API_URL` set to the Render API for production.
The API only accepts browser requests from the origins in `F1S_CORS_ORIGINS` (`render.yaml`), so
a new address must be added there. To set it up or point it at a different API, then deploy (from `apps/web`):

```powershell
& .\scripts\vercel-setup.ps1 -ApiUrl https://f1-neural-strategist-api.onrender.com
vercel deploy --prod --yes
```

## Browser simulator

The Strategy Builder simulates in the browser: the Rust simulator (`crates/strategist-sim`)
compiled to WebAssembly runs in `public/sim/worker.js` and needs no API call. If it can't load,
`src/lib/simulate.ts` falls back to the API. The files in `public/sim` are committed, because
Vercel's build has no Rust; regenerate them after retraining or a simulator change (from the
repository root):

```bash
python -m strategist.sim.export --web apps/web/public/sim
node crates/strategist-sim/scripts/build_web.mjs
```

## Structure

- `src/lib/api.ts`: typed API client (mirrors `services/api/app/schemas.py`)
- `src/lib/simulate.ts`: simulate in the browser, or via the API
- `src/lib/race.ts`: tyre colours, strategy editing, replay interpolation
- `src/components/workspace/`: track map, gap chart, strategy timeline, builder, optimiser
- Charts are hand-written SVG with `d3-scale`/`d3-shape`. Colours are CSS roles in
  `globals.css`; the driver highlights were validated for colour-blind separation on the dark
  surface, and tyres always carry their S/M/H letter as well as their colour.
