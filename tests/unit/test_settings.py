import pytest
from pydantic import ValidationError
from agentic_bi.config.settings import Settings, get_settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("ENVIRONMENT", "dev")

    settings = Settings()

    assert isinstance(settings.postgres_port, int)
    assert settings.postgres_port == 5432
    assert settings.database_url == "postgresql+asyncpg://testuser:testpass@localhost:5432/testdb"


def test_settings_rejects_invalid_environment(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(ValidationError):
        Settings()