"""
爬虫百宝箱 - 一个功能强大的网络爬虫工具包

主要功能:
- 异步网络请求
- 智能重试机制
- 代理管理
- 验证码处理
- 评论爬取
- 视频下载
- 文档爬取
- 图片采集

使用示例:
    from spider_toolkit import SpiderToolkit
    
    toolkit = SpiderToolkit()
    comments = await toolkit.crawl_comments("https://example.com")
"""

from .core import SpiderToolkit
from .config import CrawlerConfig, config
from .network import AsyncCrawler, AsyncRequestThrottler, ProxyManager
from .parsers import CommentParser, VideoParser, DocumentParser
from .captcha import CaptchaSolver
from .database import CrawlHistory, history
from .utils import *

__version__ = "3.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # 核心类
    'SpiderToolkit',
    
    # 配置
    'CrawlerConfig',
    'config',
    
    # 网络
    'AsyncCrawler',
    'AsyncRequestThrottler', 
    'ProxyManager',
    
    # 解析器
    'CommentParser',
    'VideoParser',
    'DocumentParser',
    
    # 验证码
    'CaptchaSolver',
    
    # 数据库
    'CrawlHistory',
    'history',
    
    # 工具函数
    'setup_logging',
    'sanitize_filename',
    'extract_domain',
    'is_valid_url',
    'save_json_data',
    'load_json_data',
]