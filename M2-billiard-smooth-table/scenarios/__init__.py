import config

from . import (
    catch_up,
    grazing,
    head_on_counter,
    head_on_rest,
    oblique_rest,
    steel_target,
    unequal_radius,
)

MODULES = (
    head_on_rest,
    oblique_rest,
    grazing,
    head_on_counter,
    catch_up,
    unequal_radius,
    steel_target,
)

REGISTRY = {module.__name__.split(".")[-1]: module for module in MODULES}


def get(name):
    if name == "working":
        return config
    if name not in REGISTRY:
        available = ", ".join(["working"] + list(REGISTRY))
        raise KeyError(f"Сценарий {name!r} не найден. Доступны: {available}")
    return REGISTRY[name]


def names():
    return list(REGISTRY)


def check_uniform():
    def public_names(module):
        return {
            name for name in vars(module)
            if not name.startswith("_") and name.isupper()
        }

    expected = public_names(config)
    report = {}
    for name in names():
        found = public_names(get(name))
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if missing or extra:
            report[name] = {"нет в файле": missing, "лишнее в файле": extra}
    return report
