"""Construit le test externe UDHR français–sérère pour le probe.

Les paragraphes sont alignés par numéro d'article puis par position dans
l'article. Le préambule est volontairement exclu : ses segmentations française
et sérère ne sont pas identiques.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


BASE_URL = "https://raw.githubusercontent.com/eric-muller/udhr/main/data/udhr"
SOURCES = {
    "fra": f"{BASE_URL}/udhr_fra.xml",
    "srr": f"{BASE_URL}/udhr_srr.xml",
}
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "udhr_fr_srr_probe_test.json"


def download_xml(url: str) -> ET.Element:
    request = urllib.request.Request(url, headers={"User-Agent": "fr-srr-nmt-reproducibility/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return ET.fromstring(response.read())


def article_paragraphs(root: ET.Element) -> dict[str, list[str]]:
    namespace = {"u": root.tag.partition("}")[0].removeprefix("{")}
    result: dict[str, list[str]] = {}
    for article in root.findall("u:article", namespace):
        number = article.attrib["number"]
        paragraphs = [
            " ".join("".join(paragraph.itertext()).split())
            for paragraph in article.findall(".//u:para", namespace)
        ]
        result[number] = paragraphs
    return result


def build_rows(french_root: ET.Element, serer_root: ET.Element) -> list[dict[str, str]]:
    french = article_paragraphs(french_root)
    serer = article_paragraphs(serer_root)
    if french.keys() != serer.keys():
        raise ValueError("Les numéros d'article français et sérère ne correspondent pas")

    rows: list[dict[str, str]] = []
    for article in french:
        if len(french[article]) != len(serer[article]):
            raise ValueError(
                f"Article {article}: {len(french[article])} paragraphes français "
                f"contre {len(serer[article])} paragraphes sérère"
            )
        rows.extend(
            {"francais": source, "serere": target}
            for source, target in zip(french[article], serer[article])
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_rows(download_xml(SOURCES["fra"]), download_xml(SOURCES["srr"]))
    if len(rows) != 50:
        raise ValueError(f"50 paires attendues, {len(rows)} obtenues")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} paires écrites dans {args.output}")


if __name__ == "__main__":
    main()
