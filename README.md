# JobLens

[![CI](https://github.com/jonahernan99/joblens/actions/workflows/ci.yml/badge.svg)](https://github.com/jonahernan99/joblens/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Explainable resume-to-job matching with NLP skill extraction and gap analysis.**

Most applicant tracking systems (ATS) score your resume against a job posting
and never tell you why. JobLens does the same comparison in the open: it
extracts skills from both documents, separates a posting's **required** skills
from its **nice-to-haves**, and tells you exactly which gaps are costing you.

```text
$ joblens match examples/resume_ml_engineer.md examples/job_ml_engineer_toronto.md

Match score       56.1 / 100  [#################-------------]
Skill coverage   71%
Text similarity  12% (tfidf)

Good match - tailor your resume to close the gaps below.

+ Required skills you have   Docker, Embeddings, FastAPI, Git, Hugging Face, LLMs, Machine Learning, ...
- Required skills missing    Airflow, AWS, Code Review, Data Pipelines, Statistics
+ Preferred skills you have  MLflow, MLOps
- Preferred skills missing   Kubernetes, LangChain, LlamaIndex, Terraform
* Other skills on resume     CI/CD, FAISS, Fine-tuning, GitHub Actions, JavaScript, Linux, ...
~ Credited via related      LLMs (via RAG), MLOps (via MLflow), REST APIs (via FastAPI), Vector Databases (via FAISS)
```

## Features

- **Skill extraction** over a curated taxonomy of ~100 tech skills and their
  aliases (`k8s` -> Kubernetes, `sklearn` -> scikit-learn). Symbol-aware
  matching handles `C++`, `C#`, `CI/CD` and `Node.js` without false positives
  like `git` inside `GitHub`.
- **Skill inference** through an implication graph: listing FAISS satisfies a
  "vector databases" requirement, and PyTorch implies Deep Learning and Machine
  Learning. Every inferred credit is reported with its source skill.
- **Requirement parsing** that recognizes section headings (`Requirements`,
  `Nice to Have`, `Bonus`) and inline phrases such as
  *"experience with Kubernetes is an asset"*, common in Canadian postings.
- **Explainable scoring** that blends weighted skill coverage (a missing
  requirement costs twice as much as a missing nice-to-have) with text
  similarity.
- **Semantic matching** (optional) with sentence-transformer embeddings, which
  relate paraphrases that share no vocabulary.
- **Batch ranking** of many postings against one resume, with a summary of the
  required skills you are missing most often across all of them.
- **Multiple outputs**: human-readable text, Markdown reports, JSON for
  scripting and CSV for spreadsheets. In `match`, `--min-score` sets a non-zero
  exit code for automation.
- Reads `.txt`, `.md` and `.pdf` files.

## Quick start

```bash
git clone https://github.com/jonahernan99/joblens.git
cd joblens
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[pdf]"

joblens match path/to/resume.pdf path/to/job.txt
joblens match resume.pdf job.txt --format markdown > report.md
joblens skills resume.pdf

# Rank a folder of saved postings and see which skills to learn first
joblens rank resume.pdf saved_jobs/ --top 10
joblens rank resume.pdf saved_jobs/ --format csv > ranking.csv

# Optional semantic similarity (installs PyTorch; the model downloads on first use)
pip install -e ".[embeddings]"
joblens match resume.pdf job.txt --similarity embeddings
```

Or use it as a library:

```python
from joblens import get_backend, match

result = match(resume_text, job_text, similarity=get_backend("embeddings"))
print(result.score, result.missing_required)
```

## Ranking many postings

```text
$ joblens rank examples/resume_ml_engineer.md examples/job_ml_engineer_toronto.md examples/jobs

Ranked 4 job postings

  #  Score  Skills  Missing  Job
  1   57.5     74%        2  examples/jobs/applied_ai_engineer_fintech.md
  2   56.1     71%        5  examples/job_ml_engineer_toronto.md
  3   37.0     46%        6  examples/jobs/backend_engineer_toronto.md
  4   29.7     38%        4  examples/jobs/data_analyst_toronto.md

Text similarity backend: tfidf

Most common missing required skills (all 4 postings)
  AWS               2/4  [#####-----]
  Code Review       2/4  [#####-----]
  Kubernetes        2/4  [#####-----]
  Statistics        2/4  [#####-----]
  ...
```

`--top` and `--min-score` shorten the list, while the skill gaps always describe
every posting analyzed. Unreadable or empty files, such as scanned PDFs with no
text layer, are skipped with a warning so one bad file does not stop the batch.

## How scoring works

1. **Extract** canonical skills from the resume and the job description.
   Resume skills are expanded with the broader skills they imply
   (disable with `--no-inference`).
2. **Split** the job's skills into *required* and *preferred* using section
   headings and inline cues.
3. **Coverage** = matched weight / total weight, with required skills weighted
   `1.0` and preferred skills `0.5`.
4. **Similarity** captures overlap beyond the skill taxonomy, using one of the
   backends below.
5. **Score** = `100 x (0.75 x coverage + 0.25 x similarity)`. If no known skills
   appear in the posting, the score falls back to text similarity alone.

Weights are configurable through `joblens.MatchWeights`.

### Similarity backends

| Backend | How it works | Trade-off |
| --- | --- | --- |
| `tfidf` (default) | Cosine similarity of TF-IDF vectors (unigrams + bigrams) over both documents | Fast, no heavy dependencies, only sees shared words |
| `embeddings` | Sentence-transformer embeddings (`all-MiniLM-L6-v2` by default, `--model` to change) | Understands paraphrases, requires PyTorch |

The embeddings backend splits both documents into line-level chunks to stay
under the model's token limit. Each job chunk is compared with its closest
resume chunk and the best scores are averaged, which answers *"how much of this
job does the resume cover?"*. Unrelated extra resume content does not lower the
result.

Measured on two sentences against the job line *"Build LLM-powered assistants
that retrieve answers from internal knowledge bases"*:

| Resume line | `tfidf` | `embeddings` |
| --- | --- | --- |
| Developed GPT-based chatbots answering customer questions from company documents | 0.00 | 0.38 |
| Prepared sourdough bread and managed a busy bakery kitchen every morning | 0.00 | 0.10 |

The two backends use different scales, so only compare scores produced by the
same backend. The report states which one was used. On the example files the
overall score is 56.1 with `tfidf` and 63.7 with `embeddings`.

## Project structure

```text
src/joblens/
  skills.py       # taxonomy loading + symbol-aware skill extraction
  sections.py     # required vs. preferred requirement parsing
  similarity.py   # pluggable TF-IDF and embedding backends
  matcher.py      # scoring and the MatchResult model
  ranker.py       # batch ranking and skill gap summary
  report.py       # text / markdown / json / csv renderers
  loaders.py      # .txt / .md / .pdf loading
  cli.py          # command-line interface
  data/skills.json        # skill taxonomy and aliases
  data/implications.json  # skill -> broader skills it implies
tests/            # pytest suite
examples/         # fictional resume and job postings
```

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest --cov=joblens

# Real-model test (needs the embeddings extra)
pip install -e ".[embeddings]"
pytest -m integration
```

## Roadmap

- [x] Skill hierarchy, so specific skills imply general ones (FAISS -> Vector Databases)
- [x] Semantic similarity with sentence-transformer embeddings (optional backend)
- [ ] `.docx` resume support
- [x] Batch mode: rank many job postings against one resume
- [ ] Understand alternatives in requirements ("Power BI or Tableau" currently counts as two required skills)
- [ ] REST API with FastAPI + Docker image
- [ ] Web UI demo
- [ ] Optional LLM-powered suggestions for rewriting resume bullets
- [ ] Labeled evaluation set with precision/recall metrics for skill extraction
- [ ] Publish to PyPI

## License

[MIT](LICENSE)
