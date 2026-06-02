"""Minimal config manager stub."""


class ConfigManager:
    def __getattr__(self, name):
        return None


config_manager = ConfigManager()
token_tracker = None
