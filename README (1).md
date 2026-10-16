# Kept

*An AI agent that keeps proof of work hours, so low-income Americans don't lose their health coverage. Powered by NVIDIA Nemotron on Nebius Token Factory.*

**Try it online:** https://kept-agent.onrender.com (open Keisha's demo). It runs on a free server, so the first load can take up to a minute if nobody has visited recently.

**People don't lose Medicaid because they don't work. They lose it because they can't prove it.**

Medicaid is the public health insurance for people with low incomes in the United States. From January 1, 2027, adults aged 19 to 64 covered by Medicaid expansion must show, every month, 80 hours of work, volunteering, job training or school, or at least $580 of pay. The state checks when they apply and at every renewal. If its own data can't confirm the hours, it sends a letter and they have 30 days to send proof.

The people the state's data can't see are the ones with irregular work: delivery apps, cash shifts, volunteering, several small jobs. Kept is built for them. It keeps the proof day by day, warns them while a month can still be fixed, and has everything ready when the state asks.

## Why

- **About 18.5 million people** will be subject to the rule each year (CBO, cited by KFF).
- **Nearly 1 in 5** eligible adults in expansion states are at risk because their hours are too few or too irregular (Boston University, *JAMA Health Forum*, 2026).
- **Arkansas, 2018:** about 18,000 people lost coverage in seven months, yet a Harvard study (*NEJM*, 2019) found about 97% of those aged 30 to 49 were already working or should have been exempt.
- **Louisiana, 2025:** 13,000 people were texted a link to verify their income; fewer than 7% completed it (KFF Health News).

## What it does

- **Log work in the way that's easiest right now:** a photo (pay stub, app screenshot, a handwritten note from the boss, a signed volunteer sheet), a **screen recording** of an app's earnings history, a **voice note** in English or Spanish, or one typed line.
- **Every proof is checked against her log.** A photo given to the wrong day, or $46 typed when the app says $64 because she forgot her tips: Kept points it out and asks.
- **The agent works every day without being asked.** It warns while a month can still be fixed, proposes a plan for the rest of the month, counts down to the renewal, closes each month on the 1st and prepares the volunteer page to sign, and starts a 30-day countdown when a state letter arrives.
- **When the state asks, the folder is ready:** a proof packet PDF with every hour and every document.
- **The human makes the final click.** Kept never signs or sends anything.

## How Kept uses NVIDIA Nemotron on Token Factory

### Nemotron is the agent's brain

Every day, without being asked, Kept's rules check her situation (letters, exemptions, the month, the state check, missing proof, quiet days) and write down what they found. Then **Nemotron 3 Super** takes over as the agent (`kept/agent/brain.py`). Using **tool calling**, it reads what it needs: `get_rule_findings`, `get_situation`, `get_missing_proof`, `get_week`. It then calls `decide`: which findings deserve a message today and which can wait, how to say each one about her real week, and at most one idea of its own (for example, asking the pantry coordinator to sign October's page on Saturday, when she'll be there anyway).

Its decision goes back through the rules before anything reaches her: urgent letters are always sent, even if the model held them; every number and date must come from what the tools returned; a month that is at risk can't be called "on track"; nothing private; buttons only to real screens; her weekly message limit still applies. Anything that fails falls back to the rules' own wording, and the model's reasoning and every check are shown on the **What Kept checked** screen. The brain thinks once a day (about 4 seconds); saving work afterwards stays instant.

### The right model for each job

Every AI call goes to **Nebius Token Factory** (`https://api.tokenfactory.nebius.com/v1`, OpenAI-compatible chat completions). Kept picks the NVIDIA model that fits each job (`kept/capture/llm.py`):

| Job | NVIDIA model | Why this one |
|---|---|---|
| Read photos, letters, and the frames of an app-history video | `nvidia/Nemotron-3.5-Super-VL` | Reads crumpled stubs, screenshots and handwriting like a person; up to 8 frames per call |
| Understand a typed line or a voice note, in English or Spanish | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | About 1 second and a fraction of a cent, for the most frequent action |
| The agent's daily decision (tool calling), and the month plan | `nvidia/nemotron-3-super-120b-a12b` | Better reasoning over her situation, dates, fixed days and totals |
| Explain a state letter in plain English and Spanish | `nvidia/Nemotron-3-Ultra-550b-a55b` | The hardest reading, where clarity matters most |

All calls run with reasoning traces off (`chat_template_kwargs: {"enable_thinking": false}`) and JSON answers, so they are fast and cheap: a photo takes about 2 seconds, a 16-frame video about 10 seconds.

### AI reads, rules decide

- **Two readings, one value.** Super VL transcribes every line of a photo and gives the values it sees. Kept's rules re-read the transcription on their own (`kept/capture/extract.py`). Agreement means "Read clearly"; disagreement means Kept asks her.
- **Never invented.** A number from the model is kept only if it is written in what it read. A shift from a video must match the row the model quoted (`3.5` must appear as `3.5` or `3h 30m`). A letter explanation may not contain a date that isn't in the letter.
- **AI proposes, rules verify.** Nemotron Super proposes a month plan; the rules (`kept/agent/planner.py`) drop impossible lines, keep classes and volunteering on their real days and add them if the model forgot, require a free day in any 7, top up to 80 hours or trim to "just enough", and then Nemotron Nano writes the message for the checked plan, using only its numbers.
- **Rules decide what counts.** Hours, pay, the months the state checks, deadlines and exemptions come from a versioned rules file per state (`kept/rules/states/MT.json`). A model doesn't decide someone's health coverage.
- **Works when the AI doesn't.** No key, a timeout, or a spending limit: Kept falls back to its rules and asks her to type what it couldn't read.

### Spending limits

Kept stops calling the AI before any limit is passed: $1 a day and $35 in total by default (`KEPT_AI_DAILY_USD`, `KEPT_AI_TOTAL_USD`), and 60 AI calls per visitor per day on the public demo. Usage is tracked from the token counts Token Factory returns. The live numbers are on the app's **How Kept uses NVIDIA** page (More, then How Kept uses NVIDIA, or `/ai`).

### How Token Factory accelerated our workflow

- **One API, four NVIDIA models.** Token Factory speaks the OpenAI chat-completions format, so one small client (`kept/capture/llm.py`, about 350 lines, no SDK) serves Nano, Super, Ultra and Super VL. Trying a different model for a job meant changing one model name, which is how we chose Super over Nano for planning and the agent's brain.
- **No GPUs to set up.** We went from creating a key to Nemotron reading our first pay stub in minutes, with nothing to download, host or scale.
- **Fast enough to build with.** With reasoning off, answers come in 1 to 2 seconds (Nano, Ultra) or about 2 seconds per photo (Super VL), so we could test every change against the live models while building, not only at the end.
- **Cheap enough to test everything.** Building and testing the whole project, every feature run live many times, cost under 5 cents of credits.
- **Agent features ready to use.** Tool calling and JSON answers worked without extra setup, which is what made the agent's brain possible in the time we had.

### Nebius tools we used, and why only Token Factory

We used **Nebius Token Factory** only, for every model call. We did not use Nebius Serverless Endpoints, Serverless Jobs or AI Cloud: Kept doesn't host or fine-tune its own model, all four models it needs are already served on Token Factory, and its daily agent is a short job that runs inside the web app. The app itself is a small Python web server, which we host on a free container service (Render) so the demo stays free to run until judging ends. Moving it to Nebius Serverless would be a one-step change of host, using the same `Dockerfile`.

### What we wanted, and the constraint we designed around

We wanted Nemotron to **listen** to voice notes and **watch** app videos directly. NVIDIA's model for that, Nemotron Nano Omni, isn't offered on Token Factory yet. So the phone's built-in speech-to-text writes the voice note down and Nemotron Nano understands it; and Kept keeps only the frames of a screen recording where the screen changed (ffmpeg plus a small image-difference check) and Super VL reads them, 8 per call.

## Feedback on Token Factory and NVIDIA models

- **What worked well:** the OpenAI-compatible API meant one small client for four models; tool calling worked out of the box on Nemotron 3 Super and Nano; Nemotron Super VL read every one of our sample documents correctly, including handwriting; turning reasoning off took a photo read from no answer at all within 1,500 tokens to a complete answer in about 2 seconds; `response_format: json_object` worked on the text models; prices are low enough that a full run through the demo costs about one cent.
- **What we ran into:** a vision request accepts at most 8 images, which we only learned from the error message; with reasoning on, Super VL used the whole `max_tokens` budget on its reasoning and returned no answer, so the enable-thinking switch deserves a place in the quick-start; a JSON answer once stopped one bracket short, so the client repairs that; and an Omni (audio and video) model on Token Factory would make voice notes and screen recordings fully native.

## The demo

Open the app and choose **Open Keisha's demo**. Keisha is fictional: a Montana gig worker with two delivery apps, cash shifts at a corner store, Saturday volunteering and a Tuesday GED class. It's October 20 and her month is at risk. **More, Demo controls** has sample documents to try: a Dashly screenshot, a handwritten note from her manager, a screen recording of her Dashly history, a pay stub, a signed volunteer sheet, and two state letters (a PDF and a photo). All are fictional and marked SAMPLE.

## Run it yourself

1. Install **Python 3.11 or newer** (on Windows, tick "Add Python to PATH").
2. Get a key from Nebius Token Factory (https://tokenfactory.nebius.com, then API keys). Make a copy of `env.example`, name the copy `.env`, and paste the key after `NEBIUS_API_KEY=`. Never commit the `.env` file.
3. Double-click **`start.bat`** (Windows) or run `sh start.sh` (Mac, Linux). The first time, it installs what it needs.
4. Open **http://localhost:8000**. On your phone (same Wi-Fi), open the second link it prints; the camera button works in the phone's browser.

Without a key, Kept still runs with its rules: you type the values from photos. A local Ollama model can handle the text jobs offline (`KEPT_MODEL`, default `qwen2.5:3b`).

**Deploy:** the `Dockerfile` runs on any container host (we use Render's free plan). Set `NEBIUS_API_KEY` in the host's environment settings.

## Tests

`python -m unittest discover tests` runs 102 tests, all offline: the rules (thresholds, income path, seasonal average, pay periods across months, duplicates, months checked, deadlines, exemptions), the NVIDIA layer with Nemotron's answers stood in (model routing, spending limits, JSON repair, the vision model and the rules agreeing or disagreeing, video rows checked against their quotes, Spanish voice notes, letters with invented dates refused, the month plan fixed by the rules, the agent's brain: its tool loop, and its decisions checked and overruled by the rules), the agent's whole demo month, and every screen.

## Project layout

```
kept/capture/llm.py       NVIDIA Nemotron on Token Factory: model routing, limits, JSON answers
kept/capture/vision.py    photos and app-history videos, read by Nemotron Super VL
kept/capture/extract.py   the rules that re-read every photo and compare
kept/capture/textlog.py   typed lines and voice notes (English, Spanish)
kept/capture/letters.py   state letters: deadline by rules, explanation by Nemotron Ultra
kept/agent/watchdog.py    the daily agent: rule checks, weekly limit, decision log
kept/agent/brain.py       the agent's brain: Nemotron decides with tools, the rules check it
kept/agent/planner.py     the month plan: Nemotron proposes, rules verify
kept/rules/               the rules engine and one versioned rules file per state
kept/services.py          saving entries, documents, videos, proof, sign-offs
kept/pdfs.py              reporting log, volunteer page, proof packet
kept/web/                 the app (Flask) and its screens
tests/                    102 tests
tools/                    scripts that drew the fictional samples
```

## Sources

- CMS interim final rule, June 2026: https://www.hklaw.com/en/insights/publications/2026/06/cms-issues-interim-final-rule-implementing-medicaid-community
- KFF, work requirement provisions: https://www.kff.org/60c1aad
- KFF, state policy decisions (months checked): https://www.kff.org/medicaid/an-early-look-at-policy-decisions-as-states-get-ready-to-implement-work-requirements/
- CBPP, verification and the 30-day notice: https://www.cbpp.org/research/health/a-guide-to-reducing-coverage-losses-through-effective-implementation-of-medicaids
- Montana documentation list: https://dphhs.mt.gov/medicaidchanges/VerificationMatrix
- Arkansas (NEJM study): https://kffhealthnews.org/medicaid/study-arkansas-medicaid-work-requirements-hit-those-already-employed/
- Louisiana pilot: https://kffhealthnews.org/news/article/medicaid-eligibility-tool-pilot-test-work-requirements-ai-louisiana-arizona-georgia/
- 1 in 5 at risk: https://www.bu.edu/sph/news/articles/2026/nearly-1-in-5-medicaid-eligible-adults-in-expansion-states-are-at-risk-of-losing-health-coverage-due-to-insufficient-or-inconsistent-work-hours/

Kept isn't affiliated with any government agency and isn't legal advice. Keisha and all sample documents are fictional. MIT licence.
