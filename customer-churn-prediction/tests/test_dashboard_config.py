from src.utils import load_yaml


def test_config_has_display_settings():
    cfg = load_yaml('config.yaml')
    assert 'id_column' in cfg, "config.yaml must define 'id_column'"
    assert 'display_columns' in cfg, "config.yaml must define 'display_columns'"
    assert isinstance(cfg['display_columns'], list), "'display_columns' should be a list"