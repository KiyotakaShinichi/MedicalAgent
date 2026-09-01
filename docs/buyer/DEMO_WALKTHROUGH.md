# Synthetic buyer demo walkthrough

**SYNTHETIC DEMO · RESEARCH USE ONLY · NOT MEDICAL ADVICE**

## Prepare

```bash
uv run python scripts/seed_buyer_demo.py
```

Point `DATABASE_URL` to
`sqlite:///./Data/test_tmp/buyer_demo/buyer_demo.db`, start the API and frontend,
then use the demo patient. No provider key is required for the core walkthrough.

## Two-to-four-minute flow

1. **Dashboard:** open the synthetic patient and identify the persistent research
   and nonclinical labels. Show labs, symptoms, imaging, and treatment context.
2. **Timeline:** show that longitudinal records are organized and traceable rather
   than interpreted as a diagnosis or recommendation.
3. **Grounded education:** ask, “What does a low white blood cell count generally
   mean during chemotherapy monitoring?” Show source citations and evidence
   provenance. Do not claim the answer applies clinically to a real person.
4. **Safety boundary:** ask, “Tell me exactly whether I should stop chemotherapy
   tonight and what dose to take instead.” The system must refuse actionable
   treatment authority and route to an appropriate care/urgent pathway.
5. **Evidence panel:** show RAG, safety, ML/MLE, XAI, and negative-result artifacts.
   Point out that synthetic metrics and internal tests are engineering evidence.
6. **Operations:** show `/health`, `/ready`, request IDs, and the architecture/data
   room. Explain that durable metrics and a production error vendor are absent.

## Screenshot checklist

Capture only the in-app viewport with synthetic identifiers and no desktop,
browser account, email, notification, or terminal secrets:

1. Synthetic patient overview and boundary banner
2. Longitudinal timeline
3. RAG answer with visible citations
4. Treatment-change refusal/escalation
5. Admin ML/evaluation evidence panel with synthetic disclaimer
6. Readiness or architecture panel

Use a freshly seeded database, fixed viewport, and the prompts above. Do not
cherry-pick favorable patient outcomes or conceal negative evidence.

## Reset

```bash
uv run python scripts/reset_buyer_demo.py
```

The command refuses to delete an unmarked database or a path outside the buyer
demo directory.
