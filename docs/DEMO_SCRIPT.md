# Demo script

About 90 seconds, screen recording with voiceover. Every step below was run on the live app
on 2026-11-01. Simulated numbers can change after a redeploy, so read them off the screen.

## Before recording

- Redeploy the frontend first. The live site still says "300 races each" and "usually under a
  minute"; the repo has already dropped the first of these.
- Open https://f1-neural-strategist.vercel.app about a minute early. The API sleeps when idle
  and takes about a minute to wake.
- Use a desktop window and start on `/race/2025_04` (Bahrain 2025), with PIA as focus and RUS as
  compare.
- Plan to cut the waits. Simulate takes about 25 s and the optimiser about 2 minutes on the free
  tier. The API allows 30 simulations and 4 optimiser runs per hour, and Simulate uses two
  simulations per click, so rehearse sparingly.

## Script

| Time | On screen | Voiceover |
|---|---|---|
| 0:00 | Race page, circuit map | "This is an F1 strategy simulator. It's built on a neural network trained on five seasons of real lap times, and it asks what would have happened if a team had chosen differently." |
| 0:10 | Set the speed to 4 lap/s, press ▶, let the cars move | "Here's the 2025 Bahrain Grand Prix, replayed from real timing data." |
| 0:20 | Scroll to **Gap to the leader**, hover the SC band | "Each driver's gap to the leader, with the safety car shaded." |
| 0:28 | **Strategy builder**: click the last stint so it reads H, then **Simulate** (cut the wait) | "Now I can change Piastri's strategy. Here I've put him on hards for the last stint. It simulates the race both ways using the same random draws, so the difference comes from the strategy." |
| 0:42 | Table **PIA: as raced vs yours** | "No meaningful difference. The app says so rather than inventing a winner, and anything simulated is labelled." |
| 0:52 | **Optimiser** tab, **Find the best strategy** (cut the wait) | "The optimiser tries every legal strategy, 316 here, then re-runs the best ones." |
| 1:02 | Ranked list; point at the `as raced` row and a tyre wear / compound pace line | "It ranks the strategy he really used close to the top, within a tenth of a place of the best, and it breaks down the trade-off in tyre wear and compound pace." |
| 1:14 | Return to the replay, or show the case study's failed-experiment chart | "On the sealed 2025 season the network beat every baseline. Then I tried updating it mid-race for a live pit-wall mode. Lap forecasts improved by about a quarter, but race predictions didn't clearly improve, so it failed the test I'd set in advance and I didn't build it." |
| 1:28 | Live URL on screen | "It's live, and the full write-up is linked below." |

## If something goes wrong

- A page says it can't load: the API is still waking. Wait a minute and reload.
- Simulate or the optimiser reports a limit: the hourly quota is used up. Record that step later.
- A value differs from this script: use what's on screen. Say "the best" only if the `as raced`
  row is still close to the top.
