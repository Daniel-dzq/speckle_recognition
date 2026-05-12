# figures_competition

This folder is reserved for **competition- or manuscript-specific** diagram exports (PPT/PDF/PNG) that you maintain **outside** the main `figures/` and `figures_publication/` automation outputs.

## Text and axis rules (Figure 4, length study)

- **Fiber9cm = total fiber length 9 cm** for that batch—not green propagation length.
- X-axis label: **Total fiber length (cm)**; ticks **8, 9, 11, 13, 16**.
- **`green_prop_mm`** in YAML/CSVs: auxiliary geometry only, **not** the Figure 4 x-axis definition.
- Prefer **inter/intra distance ratio** (类间/类内距离比); never “类内/类间比”, and do not claim “green propagation 9 cm is optimal”.
- If you previously equated Fiber9cm with a green-path length (e.g. 2 cm), replace with:

  *The naming Fiber9cm refers to the total fiber length. The side-polished region is located at a fixed geometry in the setup and should not be confused with the total-length label.*

Authoritative wording and the final Chinese caption/body live in **`docs/competition_figure_text_check.md`**.

## Automated publication figures

Regenerated journal-style panels (including a `fig03`/`fig04` duplicate basename for the length composite) are produced by:

```bash
python scripts/make_publication_figures.py
```

Default output directory: **`figures_publication/`** at the repository root.
