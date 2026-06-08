"""
Scrapy item definitions for DTU crawler.
"""
import scrapy


class CrawlRecord(scrapy.Item):
    url             = scrapy.Field()
    category        = scrapy.Field()
    doc_type        = scrapy.Field()
    http_status     = scrapy.Field()
    content_type    = scrapy.Field()
    last_modified   = scrapy.Field()
    etag            = scrapy.Field()
    content_length  = scrapy.Field()
    crawl_timestamp = scrapy.Field()
    link_date_label = scrapy.Field()   # date text scraped from anchor context
    html_body       = scrapy.Field()   # None for PDFs; populated for HTML pages
    html_body_path  = scrapy.Field()   # relative path to saved .html file
