# Starter Open-Access Research Papers

Downloaded with:

```text
python scripts/download_research_papers.py
python scripts/ingest_knowledge_base.py
```

The downloader uses the NCBI Open Access subset and stores full-text `.txt` files in `KnowledgeBase/raw/research_papers/`.

Direct PDF downloads from PubMed Central may be redirected or blocked by a browser/download flow. For this project, clean full-text article extraction is enough and is usually better for RAG chunking.

## Current Starter Set

- [PMC3609956](https://pmc.ncbi.nlm.nih.gov/articles/PMC3609956/): MRI residual disease and pCR after neoadjuvant chemotherapy.
- [PMC6702160](https://pmc.ncbi.nlm.nih.gov/articles/PMC6702160/): DCE-MRI heterogeneity changes for early response prediction.
- [PMC5500247](https://pmc.ncbi.nlm.nih.gov/articles/PMC5500247/): DCE-MRI texture features for therapy response prediction.
- [PMC10898895](https://pmc.ncbi.nlm.nih.gov/articles/PMC10898895/): chemotherapy-induced neutropenia risk factors in breast cancer.
- [PMC6180804](https://pmc.ncbi.nlm.nih.gov/articles/PMC6180804/): febrile neutropenia prophylaxis for FEC-D breast cancer chemotherapy.
- [PMC3987093](https://pmc.ncbi.nlm.nih.gov/articles/PMC3987093/): chemotherapy-associated toxicity and supportive care patterns.
- [PMC4998558](https://pmc.ncbi.nlm.nih.gov/articles/PMC4998558/): chemotherapy-induced neutropenia at first cycle in invasive breast cancer.
- [PMC2846279](https://pmc.ncbi.nlm.nih.gov/articles/PMC2846279/): G-CSF prophylaxis for chemotherapy-induced febrile neutropenia.
- [PMC3940691](https://pmc.ncbi.nlm.nih.gov/articles/PMC3940691/): hematologic toxicities in a SWOG breast cancer chemotherapy trial.

## Claim Boundary

These sources improve retrieval quality and explanations. They do not make the system clinically validated, and they should not be used to generate diagnosis or treatment decisions.
