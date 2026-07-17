from src.utils.skill_taxonomy import normalize, _alias_map


def test_known_alias_returns_taxonomy_id():
    assert normalize("postgres") == "postgresql"
    assert normalize("Postgres") == "postgresql"
    assert normalize("postgre") == "postgresql"


def test_taxonomy_key_itself_maps_to_itself():
    assert normalize("python") == "python"
    assert normalize("docker") == "docker"


def test_alias_case_insensitive():
    assert normalize("K8S") == "kubernetes"
    assert normalize("k8s") == "kubernetes"
    assert normalize("GOLANG") == "go"


def test_unknown_skill_returns_slugified_fallback():
    assert normalize("Microsoft Excel") == "microsoft-excel"
    assert normalize("Some Unknown Tool") == "some-unknown-tool"


def test_taxonomy_id_with_aliases():
    # node.js and variants all map to nodejs
    assert normalize("node.js") == "nodejs"
    assert normalize("Node JS") == "nodejs"
    assert normalize("nodejs") == "nodejs"


def test_different_skills_map_to_different_ids():
    assert normalize("react") != normalize("vue")
    assert normalize("python") != normalize("javascript")


def test_slugify_strips_leading_trailing_dashes():
    # Leading/trailing special chars get stripped
    result = normalize("  Python  ")
    assert not result.startswith("-")
    assert not result.endswith("-")


def test_alias_map_has_no_duplicate_keys_colliding_incorrectly():
    mapping = _alias_map()
    # js maps to javascript (not overwritten by something else)
    assert mapping.get("js") == "javascript"
    # ts maps to typescript
    assert mapping.get("ts") == "typescript"
