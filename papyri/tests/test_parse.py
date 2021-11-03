from papyri.ts import parse


def test_parse_space():

    [section] = parse(
        "Element-wise maximum of two arrays, propagating any NaNs.".encode()
    )
    assert (
        section.children[0].children[0].value
        == "Element-wise maximum of two arrays, propagating any NaNs."
    )


def test_parse_reference():

    [section] = parse("This is a `reference <to this>`_".encode())
    [paragraph] = section.children
    [text, reference] = paragraph.children
    assert reference.text == "reference <to this>"
    assert text.value == "This is a "
