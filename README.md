# AI Data Augmentor

Takes a list of company names and augments each row with website, headquarters
location, and phone number — using Claude with the web_search tool, not a
static dataset or a blind LLM guess.

## Why this design

The core risk with this kind of task is hallucination: an LLM asked "what's
Company X's phone number" will happily produce a plausible-looking number
that does not exist. This script closes that gap two ways:

1. **Grounding.** The model is given the `web_search` tool and instructed to
   search before answering, not answer from parametric memory.
2. **A null is a correct answer.** The system prompt explicitly tells the
   model that returning `null` for a field it cannot verify is success, not
   failure. Many smaller brands in this dataset (e.g. Kuhl, Oboz, Sonder)
   do not publish a public phone number — a good agent says so instead of
   inventing one.

Every returned field is expected to carry a `source_url` so a human reviewer
can spot-check it in seconds, rather than re-researching from scratch.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your real key
export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d= -f2)
```

## Run

```bash
python src/augment.py --input data/starter-companies.csv --output data/augmented-companies.csv
```

The script is resumable — if it stops partway through (rate limit, network
blip), re-running it skips companies already written to the output file.

## Files

| Path | Purpose |
|---|---|
| `src/augment.py` | The agent |
| `data/starter-companies.csv` | Input: 50 apparel/outdoor company names |
| `data/augmented-companies.csv` | Output: augmented with website, HQ, phone |
| `requirements.txt` | Python dependencies |
| `.env.example` | Credential template — never commit a real key |

