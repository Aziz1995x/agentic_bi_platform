# Multimodal Corpus Generator

Generates the synthetic visual knowledge base for the Agentic BI Platform's
multimodal RAG pipeline.

## What it produces

```
documents/multimodal/
├── charts/          # 10 PNG/JPG business charts
├── tables/          # 5 PNG table images
├── pdfs/            # 3 mixed-content PDF reports
└── metadata/        # JSON metadata for each artefact
```

## Run

```bash
# From project root — generates all 18 artefacts
python scripts/generate_corpus/generate_charts.py
python scripts/generate_corpus/generate_tables.py
python scripts/generate_corpus/generate_pdfs.py
```

## Design decisions

- All numbers in `data_layer.py` — single source of truth so charts,
  tables, and PDFs tell a **consistent story**
- Charts use both PNG and JPG to test the ingestion pipeline's
  format handling
- PDFs embed chart and table images — genuinely mixed content,
  not just text PDFs
- Metadata JSON files carry `key_insight` fields used by the
  caption-then-embed pipeline as supplementary context

## Data disclaimer

All figures are synthetic, modelled on the **Olist Brazilian E-Commerce
dataset** (CC BY-NC-SA). Raw Olist CSVs are gitignored and not
redistributed.
