from pathlib import Path

from cobalt_alpha.graph import build_graph


def main() -> None:
    output_path = Path("graph.png")
    graph_view = build_graph().get_graph()

    try:
        graph_view.draw_mermaid_png(output_file_path=str(output_path))
        print(f"Graph image saved to {output_path}")
    except Exception as exc:
        fallback_path = output_path.with_suffix(".mmd")
        fallback_path.write_text(graph_view.draw_mermaid(), encoding="utf-8")
        print(f"Failed to render PNG: {exc}")
        print(f"Mermaid graph saved to {fallback_path}")


if __name__ == "__main__":
    main()
