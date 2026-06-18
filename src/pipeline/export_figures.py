"""Export all thesis figures as single-column PDFs into thesis/thesis_draft_2/figures/."""

from __future__ import annotations

from pathlib import Path

from src.analysis.findings import (
    NotebookPaths,
    configure_matplotlib,
    export_figures_as_pdf,
    load_eda_summary,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIRECTORY = ROOT / "thesis/thesis_draft_2/figures"


def main() -> None:
    configure_matplotlib()
    paths = NotebookPaths.from_root(ROOT)
    eda_summary = load_eda_summary(paths.eda_directory)

    written = export_figures_as_pdf(
        output_directory=DEFAULT_OUTPUT_DIRECTORY,
        eda_summary=eda_summary,
        paths=paths,
    )
    for path in written:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
