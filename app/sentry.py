from app.settings import SENTRY_DSN
import sentry_sdk


if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=1.0,
    )