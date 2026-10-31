"""Runtime settings for the exporter service."""

SETTINGS = {
    "app_name": "exporter",
    "version": "1.0.0",
    "debug": False,
    "port": 8080,
    "workers": 2,
    "timeout_secs": 30,
    "retries": 3,
    "log_level": "INFO",
}


def get(key, default=None):
    return SETTINGS.get(key, default)
