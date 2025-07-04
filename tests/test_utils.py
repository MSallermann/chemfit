from scme_fitting.utils import (
    flatten_dict,
    unflatten_dict,
    iterate_nested_dict,
    set_nested_value,
    get_nested_value,
)


def test_flatten_dict():
    inp = {"a": {"b": 1.0, "c": 2.0, "d": {"e": "test"}}, "f": [1, 2]}
    out_expected = {"a.b": 1.0, "a.c": 2.0, "a.d.e": "test", "f": [1, 2]}

    out = flatten_dict(inp)
    inp2 = unflatten_dict(out)

    assert out == out_expected
    assert inp == inp2

    for keys, value in iterate_nested_dict(inp):
        assert value == get_nested_value(inp, keys)
        set_nested_value(inp, keys, 1.0)

    for keys, value in iterate_nested_dict(inp):
        assert value == 1.0
