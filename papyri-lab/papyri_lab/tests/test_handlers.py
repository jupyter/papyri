import json


async def test_get_example(jp_fetch):
    # When
    response = await jp_fetch("papyri-lab", "get-example")

    # Then
    assert response.code == 200
    payload = json.loads(response.body)
    assert "papyri" in payload["data"]
    assert len(payload["data"]) >= 400, "we have at least 400 papyri items"
