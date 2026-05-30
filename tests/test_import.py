import babappa


def test_version_exists() -> None:
    assert hasattr(babappa, "__version__")
    assert babappa.__version__
