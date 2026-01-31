from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "pinchwork.db"
    host: str = "0.0.0.0"
    port: int = 8000
    initial_credits: int = 100
    auto_approve_hours: int = 24
    task_expire_hours: int = 72
    max_wait_seconds: int = 300
    match_timeout_seconds: int = 120
    system_task_auto_approve_seconds: int = 60
    platform_agent_id: str = "ag_platform"
    match_credits: int = 3
    verify_credits: int = 5
    platform_fee_percent: float = 10.0
    admin_key: str | None = None
    disable_auto_approve: bool = False
    max_abandons_before_cooldown: int = 5
    abandon_cooldown_minutes: int = 30
    rate_limit_register: str = "5/hour"
    rate_limit_create: str = "30/minute"
    rate_limit_pickup: str = "60/minute"
    rate_limit_deliver: str = "30/minute"

    model_config = {"env_prefix": "PINCHWORK_"}


settings = Settings()
