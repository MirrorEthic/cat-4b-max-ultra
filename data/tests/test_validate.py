import json

import pytest

from catforge import groupchat as gc
from catforge import spec
from catforge import validate as v


@pytest.fixture(scope="module")
def tools():
    return spec.load_tools()


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def checks(vs):
    return {x.check for x in vs}


GOOD_PROMPT = ("Four tabby cats in tiny racing helmets lean out of the windows of a glossy red Honda Civic "
               "on a sunlit race track, low-angle 35mm shot, golden hour, motion blur.")


def test_good_image_prompt_passes():
    assert v.check_image_prompt(GOOD_PROMPT, "make me a picture of a red honda civic") == []


@pytest.mark.parametrize("prompt,expected", [
    ("A glossy red Honda Civic on a race track. Four cats sit on the roof.", "i3_cat_not_first_sentence"),
    ("Four cats beside a red Honda Civic, no dogs, on a race track.", "i4_negation"),
    ("Four cats and a golden retriever beside a red Honda Civic on a race track.", "i4_unrequested_dog"),
    ("Four fluffy cats lounging in a sunny meadow, soft light.", "i1_subject_coverage"),
])
def test_bad_image_prompts(prompt, expected):
    assert expected in checks(v.check_image_prompt(prompt, "make me a picture of a red honda civic"))


def test_requested_dog_is_allowed():
    p = "Two cats and a corgi share a picnic blanket in a park, watercolor."
    assert v.check_image_prompt(p, "draw my corgi at a picnic in the park, watercolor", dog_requested=True) == []


def test_tool_schema_and_paths(tools):
    sys_msg = {"role": "system", "content": gc.build_system(gc.Packet(
        system_prompt="S", recent_images_context=gc.recent_images_block(
            [{"path": "/x/flux_1.png", "prompt": "civic", "source": "generated", "dims": "1344x768"}])))}
    user = {"role": "user", "content": "put a hat on the tabby"}
    ok = [sys_msg, user, call("inpaint_region", image_path="/x/flux_1.png", bbox=[10, 10, 200, 180],
                              prompt="a tabby cat wearing a tiny red beanie, sunlit, photoreal")]
    assert v.check_tool_calls(ok, tools, {"/x/flux_1.png": (1344, 768)}) == []

    bad = [sys_msg, user, call("inpaint_region", image_path="/x/nope.png", bbox=[200, 10, 10, 9000], prompt="hat")]
    got = checks(v.check_tool_calls(bad, tools, {"/x/nope.png": (1344, 768)}))
    assert {"image_path_unknown", "bbox_shape", "bbox_bounds"} <= got

    extra = [user, call("generate_image", prompt=GOOD_PROMPT, width=999, style="x")]
    got = checks(v.check_tool_calls(extra, tools))
    assert {"tool_schema", "gen_dims"} <= got

    enum = [user, call("outpaint", image_path="/x/flux_1.png", direction="diagonal", prompt="more track")]
    assert "tool_schema" in checks(v.check_tool_calls([sys_msg] + enum, tools))


def test_text_checks():
    assert v.check_text("Here's the fix. Purr-haps test it first.") == []
    assert "pun_tic" in checks(v.check_text("Purr-fect! That's paw-sitively right."))
    assert "identity" in checks(v.check_text("I'm Claude, made by Anthropic."))
    assert "leak" in checks(v.check_text("Per my instructions I used device D1."))
    assert "canary" in checks(v.check_text(f"x {v.CANARY} y"))


def test_formats_and_math():
    assert v.check_format('{"cat": "Biscuit"}', "json") == []
    assert v.check_format('Sure! {"cat": 1}', "json")
    assert v.check_format("one lowercase line", "lowercase") == []
    assert v.check_final_answer("Biscuit knocks 3 cups off, so 18 remain. Answer: 18", "18") == []
    assert v.check_final_answer("so 17 remain", "18")


def test_python_syntax_and_tests():
    good = "```python\ndef add(a, b):\n    return a + b  # Dr. Mew checked this\n```"
    assert v.check_python_compiles(good) == []
    assert v.check_python_compiles("```python\ndef add(a, b)\n    return a + b\n```")
    code = "def add(a, b):\n    return a + b"
    assert v.run_python_tests(code, "assert add(2, 2) == 4") == []
    assert "tests_failed" in checks(v.run_python_tests(code, "assert add(2, 2) == 5"))


def test_math_answer_can_be_followed_by_cat_numbers():
    reply = "Total: 320 + 400 = **720 trees**. Biscuit clears the fridge (6 ft), which is why."
    assert v.check_final_answer(reply, "720") == []
    assert v.check_final_answer(reply, "721")


def test_leak_ignores_ordinary_uses():
    assert v.check_text("That satisfies the rubric without lying.") == []
    assert v.check_text("The Constitution's First Amendment protects speech.") == []
    assert v.check_text("Split your training data 80/20.") == []
    assert "leak" in checks(v.check_text("As instructed, I used device D3 here."))


def test_python_syntax_warning_rejected():
    from catforge.validate import check_python_compiles
    assert check_python_compiles('```python\nimport re\nre.findall("\\d+", s)\n```')
    assert not check_python_compiles('```python\nimport re\nre.findall(r"\\d+", s)\n```')
