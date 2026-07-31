import importlib.util
from pathlib import Path


def _load_script_module():
    path = Path("scripts/prepare_conversations_smoke_data.py")
    spec = importlib.util.spec_from_file_location("prepare_conversations_smoke_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_preparation_copies_all_ten_rounds_for_each_selected_combination(tmp_path: Path):
    module = _load_script_module()
    source = tmp_path / "source"
    output = tmp_path / "smoke" / "conversations"
    source.mkdir()
    for combination in ("00000", "00001"):
        for round_number in range(1, 11):
            (source / f"tongue-{combination}-r{round_number:02d}-without.json").write_text(
                "{}", encoding="utf-8"
            )

    selected = module.prepare_smoke_data(source, output, combinations=1, seed=20260731)

    assert len(selected) == 1
    copied = sorted(path.name for path in output.glob("*.json"))
    assert len(copied) == 10
    assert {name.split("-r")[0] for name in copied} == {f"tongue-{selected[0]}"}
    assert {name.split("-r")[1][:2] for name in copied} == {f"{number:02d}" for number in range(1, 11)}
