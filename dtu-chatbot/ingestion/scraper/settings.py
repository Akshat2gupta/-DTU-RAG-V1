"""
Scrapy settings for DTU crawler.
"""
import os

BOT_NAME = "dtu_crawler"

SPIDER_MODULES = ["ingestion.scraper.spiders"]
NEWSPIDER_MODULE = "ingestion.scraper.spiders"

# ---------------------------------------------------------------------------
# Politeness
# ---------------------------------------------------------------------------
ROBOTSTXT_OBEY = False
DOWNLOAD_DELAY = 2.0

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
RANDOMIZE_DOWNLOAD_DELAY = True

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 10.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.5
AUTOTHROTTLE_DEBUG = False

CONCURRENT_REQUESTS = 4
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [429, 500, 502, 503, 504]

# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------
JOBDIR = os.environ.get("SCRAPY_JOBDIR", None)

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
ITEM_PIPELINES = {
    "ingestion.scraper.manifest_pipeline.ManifestPipeline": 100,
}

# ---------------------------------------------------------------------------
# Feed / output
# ---------------------------------------------------------------------------
FEEDS = {}  # All output handled by ManifestPipeline

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
LOG_LEVEL = "INFO"
