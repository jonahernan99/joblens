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

Match score       45.1 / 100  [##############----------------]
Skill coverage   56%
Text similarity  12%

Partial match - consider it a stretch role.

+ Required skills you have   Docker, Embeddings, FastAPI, Git, Hugging Face, Machine Learning, NLP, ...
- Required skills missing    Airflow, AWS, Code Review, Data Pipelines, LLMs, REST APIs, Statistics, ...
+ Preferred skills you have  MLflow
- Preferred skills missing   Kubernetes, LangChain, LlamaIndex, MLOps, Terraform
* Other skills on resume     CI/CD, FAISS, Fine-tuning, GitHub Actions, JavaScript, Linux, ...
```

## Features

- **Skill extraction** over a curated taxonomy of ~100 tech skills and their
  aliases (`k8s` -> Kubernetes, `sklearn` -> scikit-learn). Symbol-aware
  matching handles `C++`, `C#`, `CI/CD` and `Node.js` without false positives
  like `git` inside `GitHub`.
- **Requirement parsing** that recognizes section headings (`Requirements`,
  `Nice to Have`, `Bonus`) and inline phrases such as
  *"experience with Kubernetes is an asset"*, common in Canadian postings.
- **Explainable scoring** that blends weighted skill coverage (a missing
  requirement costs twice as much as a missing nice-to-have) with TF-IDF text
  similarity.
- **Multiple outputs**: human-readable text, Markdown reports and JSON for
  scripting. `--min-score` sets a non-zero exit code for automation.
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
```

Or use it as a library:

```python
from joblens import match

result = match(resume_text, job_text)
print(result.score, result.missing_required)
```

## How scoring works

1. **Extract** canonical skills from the resume and the job description.
2. **Split** the job's skills into *required* and *preferred* using section
   headings and inline cues.
3. **Coverage** = matched weight / total weight, with required skills weighted
   `1.0` and preferred skills `0.5`.
4. **Similarity** = cosine similarity of TF-IDF vectors (unigrams + bigrams) of
   both documents, capturing overlap beyond the skill taxonomy.
5. **Score** = `100 x (0.75 x coverage + 0.25 x similarity)`. If no known skills
   appear in the posting, the score falls back to text similarity alone.

Weights are configurable through `joblens.MatchWeights`.

## Project structure

```text
src/joblens/
  skills.py       # taxonomy loading + symbol-aware skill extraction
  sections.py     # required vs. preferred requirement parsing
  similarity.py   # TF-IDF text similarity
  matcher.py      # scoring and the MatchResult model
  report.py       # text / markdown / json renderers
  loaders.py      # .txt / .md / .pdf loading
  cli.py          # command-line interface
  data/skills.json
tests/            # pytest suite
examples/         # fictional resume and job posting
```

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest --cov=joblens
```

## Roadmap

- [ ] Skill hierarchy, so specific skills imply general ones (FAISS -> Vector Databases)
- [ ] Semantic similarity with sentence-transformer embeddings (optional backend)
- [ ] `.docx` resume support
- [ ] Batch mode: rank many job postings against one resume
- [ ] REST API with FastAPI + Docker image
- [ ] Web UI demo
- [ ] Optional LLM-powered suggestions for rewriting resume bullets
- [ ] Labeled evaluation set with precision/recall metrics for skill extraction
- [ ] Publish to PyPI

## License

[MIT](LICENSE)
