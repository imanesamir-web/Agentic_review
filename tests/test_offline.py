from src.tools.dedup import deduplicate
from src.tools.openalex import rebuild_abstract
from src.tools.pubmed import parse_pubmed_xml
from src.llm import extract_json


def test_dedup():
    recs = [{"id": "a", "doi": "10.1/x", "title": "Deep Learning for BUS"},
            {"id": "b", "doi": "10.1/X", "title": "other"},
            {"id": "c", "doi": "", "title": "deep learning for BUS!"},
            {"id": "d", "doi": "", "title": "Different paper"}]
    u, r = deduplicate(recs)
    assert [x["id"] for x in u] == ["a", "d"] and len(r) == 2


def test_abstract():
    assert rebuild_abstract({"Hello": [0], "world": [1]}) == "Hello world"


def test_json():
    assert extract_json('text ```json {"a": 1} ```') == {"a": 1}


def test_pubmed_xml():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
    <Journal><Title>J</Title></Journal><ArticleTitle>T</ArticleTitle>
    <Abstract><AbstractText>Abs</AbstractText></Abstract></Article></MedlineCitation>
    <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/y</ArticleId></ArticleIdList></PubmedData>
    </PubmedArticle></PubmedArticleSet>"""
    r = parse_pubmed_xml(xml)
    assert r[0]["doi"] == "10.1/y" and r[0]["abstract"] == "Abs"
