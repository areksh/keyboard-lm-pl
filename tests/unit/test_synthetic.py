from pl_keyboard import synthetic


def test_build_prompt_asks_for_n_polish_sentences_about_the_topic():
    prompt = synthetic.build_prompt("technologia i komputery", 7)
    assert "technologia i komputery" in prompt
    assert "7" in prompt
    # It must forbid numbering/bullets so parse_response gets clean lines.
    assert "numeracji" in prompt


def test_parse_response_strips_numbering_bullets_and_quotes():
    raw = (
        "1. Kupiłem nowy laptop wczoraj.\n"
        "2) Zainstalowałem najnowszy system.\n"
        "- Bateria trzyma cały dzień.\n"
        '"Polecam ten model każdemu."\n'
    )
    assert synthetic.parse_response(raw) == [
        "Kupiłem nowy laptop wczoraj.",
        "Zainstalowałem najnowszy system.",
        "Bateria trzyma cały dzień.",
        "Polecam ten model każdemu.",
    ]


def test_parse_response_drops_blank_and_too_short_lines():
    raw = "Ok.\n   \nTo jest pełne zdanie tutaj.\nDwa słowa\n"
    # min_words default is 3: "Ok." and "Dwa słowa" are dropped.
    assert synthetic.parse_response(raw) == ["To jest pełne zdanie tutaj."]


def test_generate_calls_client_once_per_topic_and_flattens_parsed_lines():
    prompts: list[str] = []

    def fake_client(prompt: str) -> str:
        prompts.append(prompt)
        return "Pierwsze zdanie jest tutaj.\nDrugie zdanie też tutaj."

    lines = list(synthetic.generate(fake_client, ["sport i rekreacja", "gotowanie"], per_topic=2))

    assert len(prompts) == 2
    assert "sport i rekreacja" in prompts[0] and "gotowanie" in prompts[1]
    assert lines == [
        "Pierwsze zdanie jest tutaj.",
        "Drugie zdanie też tutaj.",
        "Pierwsze zdanie jest tutaj.",
        "Drugie zdanie też tutaj.",
    ]
