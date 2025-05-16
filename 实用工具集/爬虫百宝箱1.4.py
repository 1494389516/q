import requests
import re
import os
import time
import random
from bs4 import BeautifulSoup
import shutil
from urllib.parse import urlparse, unquote, urljoin
import xlwt
import traceback
from datetime import datetime
import threading
import queue
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from pathlib import Path
import logging
import csv
import asyncio
import aiohttp
import aiofiles
from typing import Optional, Dict, List, Any, Union
import yaml
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

DEFAULT_TIMEOUT = 30
COMMENT_TIMEOUT = 10
CONFIG_FILE = "crawler_config.yaml"
g_robots_parsers_lock = asyncio.Lock()
g_robots_parsers: Dict[str, "RobotsTxtParser"] = {}

class SmartRetry:
    """智能重试机制"""
    def __init__(self, max_retries=3, backoff_factor=0.3, 
                 status_forcelist=(500, 502, 503, 504, 429)):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.status_forcelist = status_forcelist
        self.history = {}
        
    def should_retry(self, method, status_code):
        """检查是否应该重试"""
        if method.upper() not in ('GET', 'HEAD', 'OPTIONS'):
            return False
        return status_code in self.status_forcelist
    
    def get_delay(self, attempt, status_code=None):
        """计算重试延迟"""
        base_delay = self.backoff_factor * (2 ** (attempt - 1))
        jitter = random.uniform(0, base_delay * 0.1)
        return base_delay + jitter
    
    def record(self, url, status_code):
        """记录请求历史"""
        self.history[url] = {
            'last_status': status_code,
            'last_time': time.time(),
            'retry_count': self.history.get(url, {}).get('retry_count', 0) + 1
        }
    
    def can_retry(self, url):
        """检查是否可以重试"""
        history = self.history.get(url, {})
        return history.get('retry_count', 0) < self.max_retries

class RobotsTxtParser:
    """解析和遵守robots.txt规则"""
    def __init__(self, domain):
        self.domain = domain
        self.rules: Dict[str, Dict[str, List[str]]] = {}
        self.crawl_delay = 0
        self.last_checked = 0
        self.lock = threading.Lock()
        
    def parse(self, content: str):
        """解析robots.txt内容"""
        lines = content.split('\n')
        current_agent = '*'
        
        for line in lines:
            line = line.strip().lower()
            if not line or line.startswith('#'):
                continue
                
            if line.startswith('user-agent:'):
                current_agent = line.split(':', 1)[1].strip()
                if current_agent not in self.rules:
                    self.rules[current_agent] = {'allow': [], 'disallow': []}
            elif line.startswith('disallow:'):
                path = line.split(':', 1)[1].strip()
                if path and current_agent in self.rules:
                    self.rules[current_agent]['disallow'].append(path)
            elif line.startswith('allow:'):
                path = line.split(':', 1)[1].strip()
                if path and current_agent in self.rules:
                    self.rules[current_agent]['allow'].append(path)
            elif line.startswith('crawl-delay:'):
                try:
                    self.crawl_delay = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass

    def can_fetch(self, user_agent, path):
        """检查是否允许爬取指定路径"""
        now = time.time()
        if now - self.last_checked < self.crawl_delay:
            return False
            
        with self.lock:
            rules = self.rules.get(user_agent, self.rules.get('*', {'allow': [], 'disallow': []}))
            
            for allow_path in rules['allow']:
                if path.startswith(allow_path):
                    self.last_checked = now
                    return True
                    
            for disallow_path in rules['disallow']:
                if path.startswith(disallow_path):
                    return False
                    
            self.last_checked = now
            return True

class AsyncCrawler:
    """异步爬虫核心类"""
    def __init__(self, max_connections=100, timeout=30, retries=3):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            force_close=True,
            enable_cleanup_closed=True
        )
        self.retries = retries
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            trust_env=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def fetch(self, url: str, method: str = 'GET', headers: Optional[Dict[str, str]] = None, 
                    proxy: Optional[str] = None, params: Optional[Dict[str, Any]] = None, 
                    data: Any = None, json: Any = None) -> Optional[aiohttp.ClientResponse]:
        """异步获取URL内容，返回整个响应对象"""
        headers = headers or {}
        for attempt in range(self.retries):
            try:
                if not self.session or self.session.closed:
                    logger.error("Attempted to use a closed or uninitialized aiohttp session.")
                    return None

                async with self.session.request(
                    method,
                    url,
                    headers=headers,
                    proxy=proxy,
                    params=params,
                    data=data,
                    json=json,
                    allow_redirects=True,
                    timeout=self.timeout
                ) as response:
                    if response.status in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    response.raise_for_status()
                    return response
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"异步请求尝试 {attempt + 1}/{self.retries} 失败: {url}, 错误: {str(e)}")
                if attempt == self.retries - 1:
                    return None
                await asyncio.sleep(0.5 * (2 ** attempt))
        return None

class AsyncProxyManager:
    """代理管理器，处理代理IP的获取和轮换 (异步版本)"""
    def __init__(self, async_crawler_instance: AsyncCrawler, proxy_list: Optional[List[str]] = None, 
                 proxy_api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.proxies = proxy_list or []
        self.proxy_api_url = proxy_api_url
        self.api_key = api_key
        self.current_index = 0
        self.lock = asyncio.Lock()
        self.async_crawler = async_crawler_instance
        
    async def get_proxy(self) -> Optional[str]:
        """获取当前代理 (异步)"""
        async with self.lock:
            if not self.proxies and self.proxy_api_url:
                await self._refresh_proxies_async()
            if not self.proxies:
                return None
            proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)
            return proxy
            
    async def _refresh_proxies_async(self):
        """从API获取新代理 (异步)"""
        if not self.proxy_api_url:
            return
        try:
            params_dict: Dict[str, str] = {}
            if self.api_key:
                params_dict['api_key'] = self.api_key
                
            response_obj = await self.async_crawler.fetch(
                self.proxy_api_url, 
                method='GET',
                params=params_dict
            )

            if response_obj and response_obj.status == 200:
                new_proxies = await response_obj.json()
                if isinstance(new_proxies, list):
                    self.proxies = new_proxies
                    self.current_index = 0
                    logger.info(f"成功刷新代理列表，共获取 {len(self.proxies)} 个代理")
                else:
                    logger.error(f"代理API返回非列表格式: {type(new_proxies)}")
            elif response_obj:
                logger.error(f"代理API返回错误: {response_obj.status}")
            else:
                logger.error(f"代理API请求失败 (无响应对象)")

        except Exception as e:
            logger.error(f"刷新代理失败: {e}")

def load_config(config_path=CONFIG_FILE):
    """加载配置文件"""
    default_config = {
        'max_threads': 5,
        'max_depth': 2,
        'max_urls_per_page': 20,
        'domain_restrict': True,
        'filter_keywords': [],
        'filter_regex': '',
        'save_metadata': True,
        'auto_categorize': True,
        'history_enabled': True,
        'resume_enabled': True,
        'avoid_duplicates': True,
        'request_throttling': {
            'min_interval': 2.0,
            'max_interval': 5.0,
            'burst_requests': 3,
            'burst_period': 60
        },
        'proxy_settings': {
            'enabled': False,
            'proxy_list': [],
            'proxy_api_url': '',
            'api_key': ''
        },
        'user_agents': [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
        ]
    }

    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
                def update_config(default, user):
                    for key, value in user.items():
                        if key in default and isinstance(value, dict) and isinstance(default[key], dict):
                            update_config(default[key], value)
                        else:
                            default[key] = value
                
                update_config(default_config, user_config)
                logger.info(f"从 {config_path} 加载配置成功")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
    
    return default_config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crawler.log', 'a', 'utf-8')
    ]
)
logger = logging.getLogger('爬虫百宝箱')

class CrawlerConfig:
    """爬虫配置类，存储爬虫的各种配置参数"""
    def __init__(self):
        self.max_threads = 5  
        self.max_depth = 2   
        self.max_urls_per_page = 20  
        self.domain_restrict = True  
        self.filter_keywords = []    
        self.filter_regex = ""       
        self.save_metadata = True  
        self.auto_categorize = True  
        self.history_enabled = True 
        self.resume_enabled = True   
        self.avoid_duplicates = True 


config = CrawlerConfig()
class CrawlHistory:
    """爬取历史管理类，记录已爬取的URL和内容"""
    def __init__(self, db_path="crawler_history.db"):
        self.db_path = db_path
        self.enabled = config.history_enabled
        if self.enabled:
            self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建URL历史表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS url_history (
                url TEXT PRIMARY KEY,
                timestamp TEXT,
                status TEXT,
                content_hash TEXT,
                depth INTEGER,
                parent_url TEXT
            )
            ''')
            
            # 创建内容历史表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_history (
                content_hash TEXT PRIMARY KEY,
                content_type TEXT,
                file_path TEXT,
                metadata TEXT,
                timestamp TEXT
            )
            ''')
            
            # 创建任务队列表，用于断点恢复
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_queue (
                url TEXT PRIMARY KEY,
                depth INTEGER,
                parent_url TEXT,
                timestamp TEXT,
                status TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"历史数据库初始化成功: {self.db_path}")
        except Exception as e:
            logger.error(f"历史数据库初始化失败: {e}")
            self.enabled = False
    
    def add_url(self, url, status="completed", content_hash=None, depth=0, parent_url=None):
        """添加URL到历史记录"""
        if not self.enabled:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT OR REPLACE INTO url_history VALUES (?, ?, ?, ?, ?, ?)",
                (url, timestamp, status, content_hash, depth, parent_url)
            )
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"添加URL到历史记录失败: {e}")
    
    def add_content(self, content_hash, content_type, file_path, metadata=None):
        """添加内容到历史记录"""
        if not self.enabled:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            if metadata and isinstance(metadata, dict):
                metadata = json.dumps(metadata)
            
            cursor.execute(
                "INSERT OR REPLACE INTO content_history VALUES (?, ?, ?, ?, ?)",
                (content_hash, content_type, file_path, metadata, timestamp)
            )
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"添加内容到历史记录失败: {e}")
    
    def url_exists(self, url):
        """检查URL是否已经爬取过"""
        if not self.enabled or not config.avoid_duplicates:
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT url FROM url_history WHERE url = ?", (url,))
            result = cursor.fetchone()
            
            conn.close()
            return result is not None
        except Exception as e:
            logger.error(f"检查URL历史失败: {e}")
            return False
    
    def add_task(self, url, depth, parent_url=None):
        """添加任务到队列，用于断点恢复"""
        if not self.enabled or not config.resume_enabled:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT OR REPLACE INTO task_queue VALUES (?, ?, ?, ?, ?)",
                (url, depth, parent_url, timestamp, "pending")
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"添加任务到队列失败: {e}")
    def mark_task_completed(self, url):
        """标记任务为已完成"""
        if not self.enabled or not config.resume_enabled:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE task_queue SET status = ? WHERE url = ?",
                ("completed", url)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"标记任务完成失败: {e}")
    
    def get_pending_tasks(self):
        """获取待处理的任务，用于断点恢复"""
        if not self.enabled or not config.resume_enabled:
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT url, depth, parent_url FROM task_queue WHERE status = ?", ("pending",))
            tasks = cursor.fetchall()
            
            conn.close()
            return tasks
        except Exception as e:
            logger.error(f"获取待处理任务失败: {e}")
            return []
class ContentFilter:
    """内容过滤器，基于关键词和正则表达式过滤内容"""
    def __init__(self):
        self.keywords = config.filter_keywords
        self.regex_pattern = config.filter_regex
        self.compiled_regex = None
        if self.regex_pattern:
            try:
                self.compiled_regex = re.compile(self.regex_pattern, re.IGNORECASE)
            except Exception as e:
                logger.error(f"正则表达式编译失败: {e}")
    
    def should_keep(self, text):
        """判断内容是否应该保留"""
        if not self.keywords and not self.compiled_regex:
            return True
        
        if self.keywords:
            for keyword in self.keywords:
                if keyword.lower() in text.lower():
                    return True
        
        if self.compiled_regex:
            if self.compiled_regex.search(text):
                return True
        
        return not (self.keywords or self.compiled_regex)
    
    def categorize_content(self, url, content_type, text=None):
        """根据内容类型和URL自动分类"""
        if not config.auto_categorize:
            return "downloads"
        
        if content_type:
            if 'image' in content_type:
                return "images"
            elif 'video' in content_type:
                return "videos"
            elif 'audio' in content_type:
                return "audios"
            elif 'pdf' in content_type or 'document' in content_type:
                return "documents"
            elif 'text/html' in content_type:
                return "html_pages"
        
        path = urlparse(url).path
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return "images"
        elif ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm']:
            return "videos"
        elif ext in ['.mp3', '.wav', '.ogg', '.flac', '.aac']:
            return "audios"
        elif ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
            return "documents"
        elif ext in ['.html', '.htm']:
            return "html_pages"
        
        return "others"
    
    def extract_metadata(self, url, content_type, html_content=None):
        """提取内容的元数据"""
        if not config.save_metadata:
            return None
        
        metadata = {
            "url": url,
            "content_type": content_type,
            "timestamp": datetime.now().isoformat()
        }
        
        if html_content and ('text/html' in content_type or content_type == ''):
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                
                title_tag = soup.find('title')
                if title_tag:
                    metadata["title"] = title_tag.get_text(strip=True)
                
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc:
                    metadata["description"] = meta_desc.get('content', '')
                
                meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
                if meta_keywords:
                    metadata["keywords"] = meta_keywords.get('content', '')
            except Exception as e:
                logger.error(f"提取元数据失败: {e}")
        
        return metadata

# 全局历史记录实例
history = CrawlHistory()
# 全局内容过滤器实例
content_filter = ContentFilter()

class ThreadPoolManager:
    """线程池管理器，管理多线程爬取"""
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or config.max_threads
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.tasks = {}
        self.task_results = {}
        self.lock = threading.Lock()
    
    def submit_task(self, task_id, func, *args, **kwargs):
        """提交任务到线程池"""
        future = self.executor.submit(func, *args, **kwargs)
        with self.lock:
            self.tasks[task_id] = future
        return future
    
    def get_result(self, task_id, timeout=None):
        """获取任务结果"""
        with self.lock:
            if task_id not in self.tasks:
                return None
            future = self.tasks[task_id]
        
        try:
            result = future.result(timeout=timeout)
            with self.lock:
                self.task_results[task_id] = result
            return result
        except Exception as e:
            logger.error(f"任务 {task_id} 执行失败: {e}")
            return None
    
    def wait_for_all(self, timeout=None):
        """等待所有任务完成"""
        with self.lock:
            futures = list(self.tasks.values())
        
        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"任务执行失败: {e}")
    
    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)

# 全局线程池实例
thread_pool = ThreadPoolManager()

class AsyncRequestThrottler:
    def __init__(self, min_interval=2.0, max_interval=5.0, burst_requests=3, burst_period=60):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.burst_requests = burst_requests
        self.burst_period = burst_period
        self.last_request_time = 0
        self.request_history: List[float] = []
        self.lock = asyncio.Lock()

    async def wait_async(self):
        async with self.lock:
            current_time = time.time()
            self.request_history = [t for t in self.request_history if current_time - t < self.burst_period]

            if len(self.request_history) >= self.burst_requests:
                oldest_request_in_burst_window = self.request_history[-(self.burst_requests -1)] if len(self.request_history) >= self.burst_requests else self.request_history[0]

                wait_time_for_burst = (oldest_request_in_burst_window + self.burst_period) - current_time

                if wait_time_for_burst > 0:
                    logger.info(f"达到突发请求限制，等待 {wait_time_for_burst:.2f} 秒...")
                    await asyncio.sleep(wait_time_for_burst)
                    current_time = time.time()

            elapsed_since_last = current_time - self.last_request_time
            wait_time_interval = 0
            if elapsed_since_last < self.min_interval:
                wait_time_interval = self.min_interval - elapsed_since_last

            if self.max_interval > self.min_interval:
                 wait_time_interval += random.uniform(0, self.max_interval - self.min_interval)

            if wait_time_interval > 0:
                logger.info(f"请求限速，等待 {wait_time_interval:.2f} 秒...")
                await asyncio.sleep(wait_time_interval)

            self.last_request_time = time.time()
            self.request_history.append(self.last_request_time)

class LinkExtractor:
    """链接提取器，从网页中提取链接并支持递归爬取"""
    def __init__(self):
        self.visited_urls = set()
        self.url_queue = queue.Queue()
        self.domain_restrict = config.domain_restrict
    
    def extract_links(self, html_content, base_url):
        """从HTML内容中提取链接"""
        links = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                absolute_url = urljoin(base_url, href)
                links.append(absolute_url)
        except Exception as e:
            logger.error(f"提取链接失败: {e}")
        
        return links
    
    def filter_links(self, links, base_url):
        """过滤链接，只保留符合条件的链接"""
        filtered_links = []
        base_domain = urlparse(base_url).netloc
        
        for link in links:
            if not link.startswith(('http://', 'https://')):
                continue
            
            if self.domain_restrict:
                link_domain = urlparse(link).netloc
                if link_domain != base_domain:
                    continue
            
            if link in self.visited_urls:
                continue
            
            filtered_links.append(link)
        return filtered_links[:config.max_urls_per_page]
    
    def add_to_queue(self, url, depth=0, parent_url=None):
        """将URL添加到队列中"""
        if url not in self.visited_urls:
            self.url_queue.put((url, depth, parent_url))
            self.visited_urls.add(url)
            history.add_task(url, depth, parent_url)
    
    def crawl_recursive(self, start_url, max_depth=None, throttler=None, session=None, headers_dict=None):
        """递归爬取链接"""
        if max_depth is None:
            max_depth = config.max_depth
        
        if not throttler:
            throttler = AsyncRequestThrottler()
        
        if not session:
            session = requests.Session()
            if headers_dict:
                session.headers.update(headers_dict)
        self.add_to_queue(start_url)
        pending_tasks = history.get_pending_tasks()
        for url, depth, parent_url in pending_tasks:
            if url not in self.visited_urls:
                self.add_to_queue(url, depth, parent_url)
        results = []
        while not self.url_queue.empty():
            url, depth, parent_url = self.url_queue.get()
            
            if history.url_exists(url):
                history.mark_task_completed(url)
                continue
            
            if depth > max_depth:
                continue
            
            logger.info(f"爬取链接 (深度 {depth}/{max_depth}): {url}")
            
            try:
                throttler.wait_async()
                
                response = session.get(url, timeout=DEFAULT_TIMEOUT)
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '')
                
                metadata = content_filter.extract_metadata(url, content_type, response.text)
                
                content_hash = hashlib.md5(response.content).hexdigest()
                
                history.add_url(url, "completed", content_hash, depth, parent_url)
                history.mark_task_completed(url)
                
                if 'text/html' in content_type and content_filter.should_keep(response.text):
                    result = {
                        'url': url,
                        'content': response.text,
                        'content_type': content_type,
                        'depth': depth,
                        'metadata': metadata
                    }
                    results.append(result)
                    
                    if depth < max_depth:
                        links = self.extract_links(response.text, url)
                        filtered_links = self.filter_links(links, url)
                        
                        for link in filtered_links:
                            self.add_to_queue(link, depth + 1, url)
                
                if config.auto_categorize:
                    category = content_filter.categorize_content(url, content_type, response.text)
                    save_dir = os.path.join("downloads", category)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    
                    filename = os.path.basename(urlparse(url).path)
                    if not filename:
                        filename = f"{content_hash[:10]}"
                        if 'text/html' in content_type:
                            filename += ".html"
                    
                    file_path = os.path.join(save_dir, filename)
                    
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    if metadata:
                        meta_path = f"{file_path}.meta.json"
                        with open(meta_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                    history.add_content(content_hash, content_type, file_path, metadata)
            
            except Exception as e:
                logger.error(f"爬取链接失败 {url}: {e}")
                history.add_url(url, f"failed: {str(e)}", None, depth, parent_url)
        
        return results

async def make_request_async(
    url: str,
    async_crawler: AsyncCrawler,
    throttler: Optional[AsyncRequestThrottler] = None,
    headers: Optional[Dict[str, str]] = None,
    proxy_manager: Optional[AsyncProxyManager] = None,
    method: str = 'GET',
    params: Optional[Dict[str, Any]] = None,
    data: Any = None,
    json_payload: Any = None,
) -> Optional[aiohttp.ClientResponse]:
    """
    通用的异步请求函数，支持代理、节流和robots.txt检查。
    Retry is handled by the AsyncCrawler instance.
    """
    effective_headers = headers.copy() if headers else {}
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path or "/"
    if 'User-Agent' not in effective_headers and config.user_agents: # Assuming config is loaded globally
        effective_headers['User-Agent'] = random.choice(config.user_agents)
    elif 'User-Agent' not in effective_headers:
        effective_headers['User-Agent'] = "PythonAsyncCrawler/1.0"
    robots_parser: Optional[RobotsTxtParser] = None
    async with g_robots_parsers_lock:
        if domain in g_robots_parsers:
            robots_parser = g_robots_parsers[domain]
        else:
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
            logger.info(f"Fetching robots.txt for {domain} from {robots_url}")
            try:
                robots_headers = {'User-Agent': effective_headers['User-Agent']}
                robots_response = await async_crawler.fetch(robots_url, headers=robots_headers, method='GET')
                if robots_response and robots_response.status == 200:
                    content = await robots_response.text()
                    new_parser = RobotsTxtParser(domain)
                    new_parser.parse(content)
                    g_robots_parsers[domain] = new_parser
                    robots_parser = new_parser
                    logger.info(f"Successfully fetched and parsed robots.txt for {domain}")
                elif robots_response:
                    logger.warning(f"Failed to fetch robots.txt for {domain}, status: {robots_response.status}. Assuming allowed.")
                    g_robots_parsers[domain] = RobotsTxtParser(domain) 
                    robots_parser = g_robots_parsers[domain]
                else: 
                    logger.warning(f"Request for robots.txt for {domain} failed. Assuming allowed.")
                    g_robots_parsers[domain] = RobotsTxtParser(domain)
                    robots_parser = g_robots_parsers[domain]

            except Exception as e_robots:
                logger.error(f"Error fetching/parsing robots.txt for {domain}: {e_robots}. Assuming allowed.")
                if domain not in g_robots_parsers: 
                    g_robots_parsers[domain] = RobotsTxtParser(domain) 
                robots_parser = g_robots_parsers[domain]

    if robots_parser and not robots_parser.can_fetch(effective_headers.get('User-Agent', '*'), path):
        logger.warning(f"根据robots.txt规则，不允许爬取: {url}")
        return None

    if throttler:
        await throttler.wait_async()
    
    proxy_url: Optional[str] = None
    if proxy_manager:
        proxy_url = await proxy_manager.get_proxy()
        if proxy_url:
            logger.debug(f"使用代理: {proxy_url} for url {url}")
        else:
            logger.debug(f"No proxy available from manager for {url}")

    response = await async_crawler.fetch(
        url,
        method=method,
        headers=effective_headers,
        proxy=proxy_url,
        params=params,
        data=data,
        json=json_payload
    )
    
    return response

def douban_spider(url, headers_dict, throttler):
    try:
        throttler.wait()
        response = requests.get(url, headers=headers_dict, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup
    except requests.RequestException as e:
        print(f"请求失败: {e}")
        return None

def parse_bilibili_video(bvid, referer, headers, throttler):
    try:
        api_url = f"https://api.bilibili.com/x/player/playurl?bvid={bvid}&qn=116&fnval=16"
        api_headers = {
            **headers,
            'Referer': referer,
            'Origin': 'https://www.bilibili.com',
            'Accept': 'application/json, text/plain, */*'
        }
        throttler.wait()
        response = requests.get(api_url, headers=api_headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if data['code'] != 0:
            print(f"B站API错误: {data.get('message')}")
            return None
        return download_bilibili_m3u8(data, referer, headers, throttler)
    except requests.RequestException as e:
        print(f"请求失败: {e}")
    except Exception as e:
        print(f"B站解析失败: {str(e)}")
    return None

def download_bilibili_m3u8(data, referer_param, headers_param, throttler):
    download_dir = "downloaded_videos"
    os.makedirs(download_dir, exist_ok=True)
    video_info = data['data']['dash']['video'][0]
    audio_info = data['data']['dash']['audio'][0]
    print(f"发现视频流: {video_info['id']} | 分辨率: {video_info['width']}x{video_info['height']}")
    video_url = video_info['baseUrl']
    audio_url = audio_info['baseUrl']
    temp_dir = f"temp_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)

    def download_stream(url, name, stream_headers, stream_referer, stream_throttler_inner):
        try:
            print(f"开始下载{name}流...")
            current_headers = stream_headers.copy()
            current_headers['Referer'] = stream_referer
            stream_throttler_inner.wait()
            response = requests.get(url, headers=current_headers, stream=True, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            with open(f"{temp_dir}/{name}.mp4", 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"{name}流下载完成")
        except Exception as e:
            print(f"{name}流下载失败:", str(e))
            return False
        return True

    if download_stream(video_url, 'video', headers_param, referer_param, throttler) and \
       download_stream(audio_url, 'audio', headers_param, referer_param, throttler):
        output_file = f"{download_dir}/bilibili_{int(time.time())}.mp4"
        try:
            print("开始合并音视频...")
            shutil.move(f"{temp_dir}/video.mp4", output_file)
            shutil.rmtree(temp_dir)
            print(f"视频已保存至: {output_file}")
        except Exception as e:
            print("合并失败:", str(e))
    else:
        print("一个或多个B站媒体流下载失败，无法合并。")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


async def crawl_images_revised_async(page_url, html_content_for_images, headers_dict, throttler, async_crawler, recursive=True, max_depth=None, proxy_manager=None, retry=None):
    """
    爬取图片链接，支持多线程下载和递归爬取
    参数:
        page_url: 目标页面URL
        html_content_for_images: 页面HTML内容
        headers_dict: 请求头
        recursive: 是否递归爬取 (默认True)
        max_depth: 最大递归深度 (默认None)
        proxy_manager: 代理管理器实例 (可选)
        retry: 重试机制实例 (可选)
    """
    if max_depth is None:
        max_depth = config.max_depth
        
    if retry is None:
        retry = SmartRetry()
        
    # 创建图片下载目录
    download_dir = os.path.join("downloads", "images")
    os.makedirs(download_dir, exist_ok=True)
    
    # 初始化统计
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0
    }

    # 初始化链接提取器
    link_extractor = LinkExtractor()
    
    def process_url(url):
        """处理URL，确保使用HTTPS"""
        if url.startswith("http://"):
            return "https://" + url[len("http://"):]
        elif not url.startswith("https://") and "://" not in url:
            return "https://" + url
        return url

    async def download_image(img_url, referer_url, task_id):
        """下载单个图片的函数"""
        try:
            # 检查是否已下载过
            url_hash = hashlib.md5(img_url.encode()).hexdigest()
            if history.url_exists(img_url):
                logger.info(f"图片已下载过: {img_url}")
                stats['skipped'] += 1
                return

            # 准备请求头
            local_headers = headers_dict.copy()
            local_headers['Referer'] = referer_url

            # 使用新的请求函数
            response = await make_request_async(
                img_url,
                headers=local_headers,
                proxy_manager=proxy_manager,
                retry=retry
            )
            
            if not response:
                logger.error(f"请求图片失败: {img_url}")
                stats['failed'] += 1
                return

            # 检查内容类型
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"非图片内容类型: {content_type}, URL: {img_url}")
                return

            # 生成更规范的文件名
            parsed_url = urlparse(img_url)
            base_name = os.path.splitext(os.path.basename(parsed_url.path))[0]
            if not base_name or len(base_name) < 3:
                base_name = url_hash[:8]
                
            ext = content_type.split('/')[-1].lower()
            if ext == 'jpeg':
                ext = 'jpg'
                
            filename = f"{base_name[:50]}_{int(time.time())}_{task_id}.{ext}"
            file_path = os.path.join(download_dir, filename)

            # 保存图片
            with open(file_path, 'wb') as f:
                f.write(response.content)

            # 提取和保存元数据
            metadata = {
                'url': img_url,
                'referer': referer_url,
                'content_type': content_type,
                'size': len(response.content),
                'download_time': datetime.now().isoformat(),
                'dimensions': None  # 可以添加图片尺寸信息
            }

            # 保存元数据
            meta_path = f"{file_path}.meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # 添加到历史记录
            history.add_url(img_url, "completed", url_hash, 0, referer_url)
            history.add_content(url_hash, content_type, file_path, metadata)

            logger.info(f"成功下载图片: {img_url} -> {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"下载图片失败 {img_url}: {e}")
            history.add_url(img_url, f"failed: {str(e)}", None, 0, referer_url)
            return None

    async def process_page(url, html_content, depth=0):  # Make process_page async
        """处理单个页面，提取图片和链接"""
        processed_urls = set()
        base_url = process_url(url)
        
        # 提取图片URL
        image_pattern = r'<img.*?src=["\'](.*?)["\'].*?>'
        img_urls_found = re.findall(image_pattern, html_content, re.IGNORECASE)
        
        # 处理每个图片URL
        for img_url in img_urls_found:
            try:
                
                if img_url.startswith("data:"):
                    continue

                
                if img_url.startswith("//"):
                    final_url = f"https:{img_url}"
                elif img_url.startswith("/"):
                    final_url = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}{img_url}"
                else:
                    final_url = urljoin(base_url, img_url)
                final_url = process_url(final_url)

                if final_url not in processed_urls and final_url.startswith("https://"):
                    processed_urls.add(final_url)
                    task_id = hashlib.md5(final_url.encode()).hexdigest()[:6]
                    thread_pool.submit_task(task_id, download_image, final_url, base_url, task_id)

            except Exception as e:
                logger.error(f"处理图片URL失败: {e}")

       
        if recursive and depth < max_depth:
            links = link_extractor.extract_links(html_content, base_url)
            filtered_links = link_extractor.filter_links(links, base_url)
            for link in filtered_links:
                try:
                    if link not in link_extractor.visited_urls:
                        link_extractor.visited_urls.add(link)
                      
                        await throttler.wait_async()
                        response_obj = await make_request_async(
                            link,
                            async_crawler=async_crawler, 
                            headers=headers_dict,
                            throttler=throttler,
                            proxy_manager=proxy_manager
                        )
                        if response_obj:
                            link_html_content = await response_obj.text()
                            await process_page(link, link_html_content, depth + 1) # Await recursive call
                        else:
                            logger.error(f"Failed to fetch content for link: {link}")
                except Exception as e:
                    logger.error(f"处理链接失败 {link}: {e}")
    thread_pool.wait_for_all()
    
    logger.info("图片爬取任务完成")

async def crawl_comments_async(target_url: str, headers_dict: Dict, throttler: AsyncRequestThrottler, async_crawler: AsyncCrawler, proxy_manager: Optional[AsyncProxyManager]):
    print(f"正在尝试从 {target_url} 爬取评论...")
    try:
        await throttler.wait_async()
        
        response = await make_request_async(
            target_url,
            async_crawler=async_crawler,
            headers=headers_dict,
            throttler=throttler, # make_request_async handles its own throttling if passed
            proxy_manager=proxy_manager,
            method='GET'
        )

        if not response:
            print(f"请求页面内容失败 (无响应对象): {target_url}")
            return

        # Try to determine encoding, similar to how requests does, or use a common default
        # For aiohttp, response.charset is often None, rely on content_type or default to utf-8
        content_type_header = response.headers.get('Content-Type', '').lower()
        charset = response.charset
        if not charset:
            if 'charset=' in content_type_header:
                charset = content_type_header.split('charset=')[-1].split(';')[0].strip()
            else:
                charset = 'utf-8' # Default
        
        try:
            html_content = await response.text(encoding=charset, errors='replace')
        except UnicodeDecodeError:
            logger.warning(f"UnicodeDecodeError with detected charset {charset} for {target_url}, trying with 'utf-8' errors replace.")
            html_content = await response.text(encoding='utf-8', errors='replace')


        soup = BeautifulSoup(html_content, 'html.parser')
        comment_selectors = [
            '.comment-text', '.comment-body', '.comment-content', '.comment',
            'div[class*="comment"]', 'article[class*="comment"]',
            '.reply-content', '.post-text', '.comment-message',
            '#comments', '.comments-area', 'section.comments'
        ]
        found_comments = []
        for selector in comment_selectors:
            elements = soup.select(selector)
            for element in elements:
                is_nested_comment_container = False
                for inner_selector in comment_selectors:
                    if selector != inner_selector and element.select_one(inner_selector):
                        is_nested_comment_container = True
                        break
                if not is_nested_comment_container:
                    comment_text = element.get_text(separator=' ', strip=True)
                    if comment_text and len(comment_text) > 10:
                        found_comments.append(comment_text)
        if found_comments:
            print(f"找到 {len(found_comments)} 条可能的评论内容:")
            for i, comment in enumerate(set(found_comments)):
                print(f"--- 评论 {i+1} ---")
                print(comment)
                print("--------------------")
        else:
            print("未能自动提取到明确的评论内容。")
            print("提示: 评论可能是动态加载的 (需要JavaScript执行)，或者使用了网站特定的HTML结构。")
            print("VIP内容或需要登录的评论也可能无法直接获取。")
    except aiohttp.ClientResponseError as e_http:
        print(f"HTTP错误 {e_http.status}: {e_http.message} - URL: {target_url}")
    except Exception as e:
        print(f"处理评论时发生错误: {e} - URL: {target_url}")
        traceback.print_exc()


def crawl_videos(target_url, headers_dict, html_content, throttler):
    print(f"正在尝试从 {target_url} 爬取视频到本地...")
    soup = BeautifulSoup(html_content, 'html.parser')
    found_video_sources = set()
    processed_page_url_for_video = target_url
    if processed_page_url_for_video.startswith("http://"):
        processed_page_url_for_video = "https://" + processed_page_url_for_video[len("http://"):]
    elif not processed_page_url_for_video.startswith("https://") and "://" not in processed_page_url_for_video:
        processed_page_url_for_video = "https://" + processed_page_url_for_video
    base_domain_url_video = ""
    match_domain_video = re.match(r'(https?://[^/]+)', processed_page_url_for_video)
    if match_domain_video:
        base_domain_url_video = match_domain_video.group(1)
        if base_domain_url_video.startswith("http://"):
            base_domain_url_video = "https://" + base_domain_url_video[len("http://"):]
    else:
        domain_part = processed_page_url_for_video.split('/')[0]
        if "://" not in domain_part:
            base_domain_url_video = "https://" + domain_part
        else:
            print(f"警告: 无法从 {processed_page_url_for_video} 解析域名。根相对路径的视频可能无法正确处理。")
    current_path_base_video = processed_page_url_for_video
    if not current_path_base_video.endswith('/'):
        last_slash_idx = current_path_base_video.rfind('/')
        if last_slash_idx > current_path_base_video.find("://") + 2:
            current_path_base_video = current_path_base_video[:last_slash_idx + 1]
        else:
            current_path_base_video += '/'
    if current_path_base_video.startswith("http://"):
        current_path_base_video = "https://" + current_path_base_video[len("http://"):]
    elif not current_path_base_video.startswith("https://") and "://" not in current_path_base_video.split('/')[0]:
        current_path_base_video = "https://" + current_path_base_video

    def resolve_video_url(video_url_param):
        final_url_res = ""
        if video_url_param.startswith("data:"):
            return None
        elif video_url_param.startswith("//"):
            final_url_res = f"https:{video_url_param}"
        elif video_url_param.startswith("/"):
            if base_domain_url_video:
                final_url_res = f"{base_domain_url_video}{video_url_param}"
            else:
                final_url_res = video_url_param
        elif video_url_param.startswith("http://"):
            final_url_res = "https://" + video_url_param[len("http://"):]
        elif video_url_param.startswith("https://"):
            final_url_res = video_url_param
        else:
            if current_path_base_video:
                final_url_res = current_path_base_video + video_url_param
                if final_url_res.startswith("http://"):
                     final_url_res = "https://" + final_url_res[len("http://"):]
            else:
                final_url_res = video_url_param
        if final_url_res:
            if final_url_res.startswith("http://"):
                final_url_res = "https://" + final_url_res[len("http://"):]
            elif not final_url_res.startswith("https://") and not final_url_res.startswith("data:") and "://" not in final_url_res.split('/')[0]:
                 final_url_res = "https://" + final_url_res
            return final_url_res
        return None

    for video_tag in soup.find_all('video'):
        src = video_tag.get('src')
        if src:
            resolved = resolve_video_url(src)
            if resolved: found_video_sources.add(resolved)
        poster = video_tag.get('poster')
        if poster:
            resolved = resolve_video_url(poster)
            if resolved: found_video_sources.add(resolved)
        for source_tag in video_tag.find_all('source'):
            src = source_tag.get('src')
            if src:
                resolved = resolve_video_url(src)
                if resolved: found_video_sources.add(resolved)
    for iframe_tag in soup.find_all(['iframe', 'embed']):
        src = iframe_tag.get('src')
        if src:
            resolved = resolve_video_url(src)
            if resolved: found_video_sources.add(resolved)

    html_text_content = str(html_content)
    
    generic_video_pattern = r"['\"`]([^'\"`]*?\.(?:mp4|webm|ogg|mov|avi|flv|wmv|mkv|3gp))['\"`]"
    js_video_pattern = r"video[Uu]rl\s*[:=]\s*['\"`]([^'\"`]+?)['\"`]"
    generic_src_pattern = r"src\s*[:=]\s*['\"`]([^'\"`]+?)['\"`]"

    for pattern in [generic_video_pattern, js_video_pattern, generic_src_pattern]:
        for match in re.finditer(pattern, html_text_content, re.IGNORECASE):
            url_match = match.group(1)
            if url_match and not url_match.startswith("data:"):
                resolved = resolve_video_url(url_match)
                if resolved:
                    if any(ext in resolved.lower() for ext in ['.js', '.css', '.html', '.php', '.json', '.xml', '.txt', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp']):
                        continue
                    if any(ext in resolved.lower() for ext in ['.mp4', '.webm', '.ogg', '.mov', '.avi', '.flv', '.wmv', '.mkv', '.3gp', '.m3u8']) or \
                       ('video' in resolved.lower() or 'stream' in resolved.lower() or 'embed' in resolved.lower() or 'player' in resolved.lower() or 'media' in resolved.lower() or 'playurl' in resolved.lower()) and \
                       not any(non_video_ext in resolved.lower() for non_video_ext in ['.js', '.css', '.php', '.html']):
                        found_video_sources.add(resolved)

    print("正在尝试检测可能的m3u8播放列表...")
    m3u8_pattern = r"['\"`]([^'\"`]*?\.m3u8[^'\"`]*?)['\"`]"
    for match in re.finditer(m3u8_pattern, html_text_content, re.IGNORECASE):
        m3u8_url = match.group(1)
        resolved_m3u8 = resolve_video_url(m3u8_url)
        if resolved_m3u8:
            print(f"发现可能的m3u8播放列表 (已处理): {resolved_m3u8}")
            found_video_sources.add(resolved_m3u8)
            
    all_urls = list(found_video_sources)
    if not all_urls:
        print("未找到直接可见的视频链接或m3u8播放列表。")
        return
    print(f"\n找到 {len(all_urls)} 个可能的视频相关链接")
    download_dir = "downloaded_videos"
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    for i, video_url_item in enumerate(all_urls):
        processed_url = video_url_item
        print(f"\n正在处理 #{i+1}: {processed_url}")
        if ".m3u8" in processed_url.lower():
            print("检测到m3u8播放列表，这通常是流媒体内容")
            print("提示: 流媒体内容通常需要特殊工具如ffmpeg进行下载")
            print("该链接可能是视频的加密播放列表，可能需要额外的解密步骤")
            continue
        if "iframe" in processed_url.lower() or "embed" in processed_url.lower():
            print("这是嵌入式播放器链接，可能需要进一步分析")
            continue
        try:
            file_name_prefix = re.sub(r'[^a-zA-Z0-9_]', '_', urlparse(processed_url).path.split('/')[-1] or f"video_{i+1}")
            file_name = f"{download_dir}/{file_name_prefix[:50]}_{int(time.time())}_{random.randint(1000, 9999)}"
            
            extension = ""
            parsed_url_path = urlparse(processed_url).path
            if '.' in parsed_url_path.split('/')[-1]:
                potential_ext = '.' + parsed_url_path.split('/')[-1].split('.')[-1].lower()
                valid_video_exts = ['.mp4', '.webm', '.ogg', '.mov', '.avi', '.flv', '.wmv', '.mkv', '.3gp']
                if len(potential_ext) <= 5 and potential_ext in valid_video_exts:
                    extension = potential_ext
                else:
                    extension = '.mp4' 
            else:
                extension = '.mp4'
            file_name += extension
            
            print(f"正在尝试下载到: {file_name}")
            download_headers = headers_dict.copy()
            download_headers['Referer'] = target_url
            retry_count = 3
            for attempt in range(retry_count):
                try:
                    throttler.wait()
                    response = requests.get(processed_url, headers=download_headers, stream=True, timeout=DEFAULT_TIMEOUT)
                    response.raise_for_status()
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'text/html' in content_type or 'application/javascript' in content_type or 'text/javascript' in content_type or 'text/css' in content_type:
                        print(f"警告: 返回的是{content_type}内容而非视频文件。跳过此链接。")
                        break 
                    if not ('video' in content_type or 'stream' in content_type or 'application/octet-stream' in content_type or 'binary/octet-stream' in content_type):
                        print(f"警告: 内容类型 {content_type} 可能不是视频。仍尝试下载。")
                    
                    total_size = int(response.headers.get('Content-Length', 0))
                    if 0 < total_size < 10000:
                        print(f"警告: 文件大小只有 {total_size} 字节，可能不是完整的视频文件。")

                    with open(file_name, 'wb') as file:
                        downloaded_bytes = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                                downloaded_bytes += len(chunk)
                        print(f"下载了 {downloaded_bytes / (1024*1024) :.2f} MB.")

                    if os.path.getsize(file_name) < 10000 and total_size == 0 :
                         print(f"警告: 实际下载文件大小只有 {os.path.getsize(file_name)} 字节，可能不是有效的视频。")

                    print(f"下载完成: {file_name}")
                    break
                except requests.exceptions.RequestException as e_req:
                    print(f"尝试 {attempt+1}/{retry_count} 失败: {e_req}")
                    if attempt < retry_count - 1:
                        time.sleep(2 + random.uniform(0,1))
                    else:
                        print("下载失败，已达到最大重试次数。")
                except Exception as e_inner:
                    print(f"下载过程中发生内部错误 (尝试 {attempt+1}/{retry_count}): {e_inner}")
                    if attempt < retry_count - 1:
                        time.sleep(2 + random.uniform(0,1))
                    else:
                        print("下载失败，已达到最大重试次数。")
        except Exception as e:
            print(f"处理视频URL {processed_url} 时出错: {e}")
    print(f"\n视频爬取任务完成。下载的文件保存在 {os.path.abspath(download_dir)} 目录中")
    print("注意: 某些下载内容可能不完整或需要进一步处理才能播放")

async def crawl_document_async(target_url: str, headers_dict: Dict, throttler: AsyncRequestThrottler, async_crawler: AsyncCrawler, proxy_manager: Optional[AsyncProxyManager]):
    print(f"\n=== 开始文档爬取: {target_url} ===")
    download_dir = "downloaded_documents"
    os.makedirs(download_dir, exist_ok=True)
    CONTENT_TYPE_MAP = {
        'application/pdf': '.pdf', 'text/plain': '.txt', 'text/html': '.html',
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.ms-excel': '.xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'application/vnd.ms-powerpoint': '.ppt',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
        'application/zip': '.zip', 'application/x-zip-compressed': '.zip',
        'application/epub+zip': '.epub', 'application/rtf': '.rtf', 'text/csv': '.csv',
        'application/json': '.json', 'application/xml': '.xml', 'text/xml': '.xml'
    }
    retry_count = 3 
    for attempt in range(1, retry_count + 1):
        try:
            print(f"\n尝试下载 ({attempt}/{retry_count})...")
            await throttler.wait_async() # Use the passed async throttler

            download_headers = headers_dict.copy()
            response = await make_request_async(
                target_url,
                async_crawler=async_crawler,
                headers=download_headers,
                throttler=throttler,
                proxy_manager=proxy_manager,
                method='GET'
            )

            if not response:
                print(f"请求文档失败 (无响应对象) 尝试 {attempt}/{retry_count}: {target_url}")
                if attempt == retry_count: 
                    print("已达到最大重试次数，下载失败。")
                    return
                await asyncio.sleep(2 + random.uniform(0,1)) # Simple delay before retrying loop
                continue

            content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
            content_disposition = response.headers.get('Content-Disposition')
            file_name_from_disposition = None
            if content_disposition:
                fn_match = re.search(r'filename\\*?=(?:UTF-8\\\'\\\')?([^;\\n]+)', content_disposition, flags=re.IGNORECASE)
                if fn_match:
                    file_name_from_disposition = unquote(fn_match.group(1).strip('\\"'))

            base_file_name = f"document_{int(time.time())}"
            file_ext = ""

            if file_name_from_disposition:
                base_file_name_from_disp, file_ext_from_disp = os.path.splitext(file_name_from_disposition)
                base_file_name = re.sub(r'[^\\w\\-. ]', '_', base_file_name_from_disp)
                if file_ext_from_disp and len(file_ext_from_disp) < 10 : file_ext = file_ext_from_disp

            if not file_ext:
                if content_type in CONTENT_TYPE_MAP:
                    file_ext = CONTENT_TYPE_MAP[content_type]
                else:
                    path_obj = urlparse(target_url).path
                    file_ext_from_url = os.path.splitext(path_obj)[1]
                    file_ext = file_ext_from_url[:10] if file_ext_from_url else '.dat'

            final_file_name = f"{base_file_name}{file_ext}"
            full_path = os.path.join(download_dir, final_file_name)
            content_length = int(response.headers.get('Content-Length', 0))
            print(f"内容类型: {content_type}")
            print(f"文件名: {final_file_name}")
            print(f"文件大小: {content_length/1024:.1f} KB" if content_length else "文件大小: 未知")
            print(f"保存路径: {os.path.abspath(full_path)}")

            downloaded_size = 0
            start_time = time.time()
            with open(full_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        # Progress reporting can be kept simple or enhanced for async if needed
                        # For simplicity, a basic report after loop or by time interval
                # Simplified progress reporting for this async version
                print(f"\r下载完成: {downloaded_size/1024:.1f} KB", end='')
            print()
            final_size = os.path.getsize(full_path)
            if content_length > 0 and final_size != content_length:
                print(f"\n警告: 文件大小不匹配 (预期: {content_length} 字节, 实际: {final_size} 字节)")
            if final_size == 0 and (content_length == 0 or content_length == -1) : # content_length can be -1 if chunked
                 print("警告: 下载的文件大小为0字节，可能下载失败或文件为空。")
            elif final_size < 1024 and (content_length <=0 or content_length == -1):
                print("警告: 文件尺寸过小，可能是错误页面或不完整的文档。")
            print(f"\n文档下载成功: {final_file_name}")
            return 
        except aiohttp.ClientResponseError as e_http:
            print(f"HTTP错误 ({e_http.status}): {e_http.message} - URL: {target_url}")
            if attempt == retry_count: print("已达到最大重试次数")
        except aiohttp.ClientConnectionError as e_conn:
            print(f"连接错误: {e_conn} - URL: {target_url}")
            if attempt == retry_count: print("已达到最大重试次数")
        except asyncio.TimeoutError:
            print(f"请求超时 - URL: {target_url}")
            if attempt == retry_count: print("无法在超时时间内完成下载")
        except Exception as e:
            print(f"未知错误: {str(e)}\n{traceback.format_exc()} - URL: {target_url}")
            return # Exit on unknown error
        
        if attempt < retry_count:
            print("等待2秒后重试...")
            await asyncio.sleep(2 + random.uniform(0,1))

def make_request(url, headers=None, proxy_manager=None, retry=None, session=None, method='GET', timeout=30):
    """通用的请求函数，支持代理、重试和robots.txt检查"""
    headers = headers or {}
    session = session or requests.Session()
    retry = retry or SmartRetry()
    
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    
    if domain not in globals().get('robots_parsers', {}):
        robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
        try:
            response = session.get(robots_url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                robots_parser = RobotsTxtParser(domain)
                robots_parser.parse(response.text)
                globals().setdefault('robots_parsers', {})[domain] = robots_parser
        except Exception:
            pass
    
    robots_parser = globals().get('robots_parsers', {}).get(domain)
    if robots_parser and not robots_parser.can_fetch(headers.get('User-Agent', ''), path):
        logger.warning(f"根据robots.txt规则，不允许爬取: {url}")
        return None
    
    proxy = None
    if proxy_manager:
        proxy = proxy_manager.get_proxy()
        if proxy:
            logger.debug(f"使用代理: {proxy}")
    
    for attempt in range(retry.max_retries):
        try:
            if proxy:
                proxies = {'http': proxy, 'https': proxy}
            else:
                proxies = None
                
            response = session.request(
                method,
                url,
                headers=headers,
                proxies=proxies,
                timeout=timeout,
                allow_redirects=True
            )           
            if retry.should_retry(method, response.status_code):
                delay = retry.get_delay(attempt, response.status_code)
                logger.warning(f"请求失败，状态码: {response.status_code}, 等待 {delay:.1f}秒后重试...")
                time.sleep(delay)
                continue    
            response.raise_for_status()
            return response
            
        except Exception as e:
            logger.error(f"请求失败 (尝试 {attempt+1}/{retry.max_retries}): {str(e)}")
            if attempt < retry.max_retries - 1:
                delay = retry.get_delay(attempt)
                time.sleep(delay)
            else:
                raise
    return None
def load_counter():
    try:
        if os.path.exists('.request_counter'):
            with open('.request_counter', 'r') as f:
                return int(f.read().strip())
    except:
        pass
    return 0

def save_counter(count):
    try:
        with open('.request_counter', 'w') as f:
            f.write(str(count))
    except Exception as e:
        print(f"无法保存计数器: {e}")

DOUBAN_COLUMN_HEADERS = ['名称', '图片', '排名', '评分', '信息', '简介']

def parse_douban_page_data(soup):
    page_movie_data = []
    movie_list_ul = soup.find(class_='grid_view')
    if not movie_list_ul:
        print("未能找到电影列表容器 (class 'grid_view')，页面结构可能已更改或当前页面无内容。")
        return page_movie_data
        
    movie_list = movie_list_ul.find_all('li')
    if not movie_list:
        print("在 'grid_view' 内未能找到电影条目 (li elements)，页面结构可能已更改或当前页面无内容。")
        return page_movie_data
        
    for item in movie_list:
        item_name_str = "N/A"
        item_name_tags = item.find_all(class_='title')
        if item_name_tags:
            item_name_str = item_name_tags[0].get_text(strip=True) 
            if len(item_name_tags) > 1:
                 other_title_span = item_name_tags[1].find('span')
                 if other_title_span:
                    other_title = other_title_span.get_text(strip=True).replace('/', '').strip()
                    if other_title and other_title not in item_name_str: item_name_str += f" / {other_title}"
                 elif item_name_tags[1].get_text(strip=True) and item_name_tags[1].get_text(strip=True) not in item_name_str:
                    item_name_str += f" / {item_name_tags[1].get_text(strip=True).replace('/','').strip()}"

        item_img_str = item.find('a').find('img').get('src') if item.find('a') and item.find('a').find('img') else "N/A"
        item_index_str = item.find('em', class_='').get_text(strip=True) if item.find('em', class_='') else "N/A"
        item_score_str = item.find(class_='rating_num').get_text(strip=True) if item.find(class_='rating_num') else "N/A"
        
        p_tag = item.find('p', class_='')
        item_author_info_str = ' '.join(p_tag.get_text(strip=True).split()) if p_tag else ""
        
        item_intr_tag = item.find(class_='inq')
        item_intr_str = item_intr_tag.get_text(strip=True) if item_intr_tag else 'NOT AVAILABLE'
        
        print(f"  解析电影：{item_index_str} | {item_name_str} | {item_score_str}")
        
        movie_item_data = {
            DOUBAN_COLUMN_HEADERS[0]: item_name_str,
            DOUBAN_COLUMN_HEADERS[1]: item_img_str,
            DOUBAN_COLUMN_HEADERS[2]: item_index_str,
            DOUBAN_COLUMN_HEADERS[3]: item_score_str,
            DOUBAN_COLUMN_HEADERS[4]: item_author_info_str,
            DOUBAN_COLUMN_HEADERS[5]: item_intr_str
        }
        page_movie_data.append(movie_item_data)
    return page_movie_data

def _get_douban_save_path(base_url_to_use, extension, sub_dir="downloaded_douban"):
    os.makedirs(sub_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parsed_url = urlparse(base_url_to_use)
    url_path_part = parsed_url.path.replace('/', '_').strip('_') if parsed_url.path else "list"
    sanitized_netloc = re.sub(r'[^a-zA-Z0-9_.-]', '', parsed_url.netloc) if parsed_url.netloc else "douban"
    
    base_filename_part = f"{sanitized_netloc}_{url_path_part}"
    base_filename_part = re.sub(r'_+', '_', base_filename_part).strip('_')
    
    filename = f'豆瓣电影_{base_filename_part[:50]}_{timestamp}.{extension}'
    return os.path.join(sub_dir, filename)

def export_to_excel(all_movie_data, base_url_to_use):
    if not all_movie_data:
        print("没有数据可以导出到Excel。")
        return
    save_path = _get_douban_save_path(base_url_to_use, "xlsx")
    try:
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet('豆瓣电影', cell_overwrite_ok=True)
        
        for col, header_text in enumerate(DOUBAN_COLUMN_HEADERS):
            sheet.write(0, col, header_text)
        
        for row, movie_data in enumerate(all_movie_data):
            for col, header in enumerate(DOUBAN_COLUMN_HEADERS):
                sheet.write(row + 1, col, movie_data.get(header, ""))
        
        book.save(save_path)
        print(f"\n豆瓣电影数据已成功导出到Excel: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"导出到Excel失败: {e}")
        traceback.print_exc()

def export_to_json(all_movie_data, base_url_to_use):
    if not all_movie_data:
        print("没有数据可以导出到JSON。")
        return
    save_path = _get_douban_save_path(base_url_to_use, "json")
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_movie_data, f, ensure_ascii=False, indent=4)
        print(f"\n豆瓣电影数据已成功导出到JSON: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"导出到JSON失败: {e}")
        traceback.print_exc()

def export_to_csv(all_movie_data, base_url_to_use):
    if not all_movie_data:
        print("没有数据可以导出到CSV。")
        return
    save_path = _get_douban_save_path(base_url_to_use, "csv")
    try:
        with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=DOUBAN_COLUMN_HEADERS)
            writer.writeheader()
            for movie_data in all_movie_data:
                writer.writerow(movie_data)
        print(f"\n豆瓣电影数据已成功导出到CSV: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"导出到CSV失败: {e}")
        traceback.print_exc()

def export_to_database(all_movie_data, base_url_to_use):
    if not all_movie_data:
        print("没有数据可以导出到数据库。")
        return
    
    db_save_dir = "downloaded_douban"
    os.makedirs(db_save_dir, exist_ok=True)
    db_name = "douban_movies_library.db" 
    db_path = os.path.join(db_save_dir, db_name)
    
    parsed_url = urlparse(base_url_to_use)
    table_name_part = parsed_url.netloc.replace('.', '_') + (parsed_url.path.replace('/', '_') if parsed_url.path else "_list")
    table_name = "douban_" + re.sub(r'[^a-zA-Z0-9_]', '', table_name_part)[:50]
    table_name = table_name.strip('_')

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cols_for_create = ", ".join([f'"{col_name}" TEXT' for col_name in DOUBAN_COLUMN_HEADERS])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols_for_create})")
        
        cols_for_insert = ", ".join([f'"{col_name}"' for col_name in DOUBAN_COLUMN_HEADERS])
        placeholders = ", ".join(["?"] * len(DOUBAN_COLUMN_HEADERS))
        
        for movie_data in all_movie_data:
            values = [movie_data.get(header, "") for header in DOUBAN_COLUMN_HEADERS]
            cursor.execute(f"INSERT INTO {table_name} ({cols_for_insert}) VALUES ({placeholders})", values)
        
        conn.commit()
        print(f"\n豆瓣电影数据已成功导出到SQLite数据库: {os.path.abspath(db_path)} (表名: {table_name})")
    except Exception as e:
        print(f"导出到SQLite数据库失败: {e}")
        traceback.print_exc()
    finally:
        if conn:
            conn.close()

async def crawl_douban_top250_movies_async(headers_dict, throttler, async_crawler, proxy_manager=None):
    """异步版本的豆瓣电影爬取功能"""
    print("\n--- 开始爬取豆瓣电影 (异步版) ---")
    all_collected_movie_data = []
    
    try:
        # 获取用户输入URL
        url_input_prompt = "\n请输入要爬取的豆瓣URL (例如: https://movie.douban.com/top250 或 https://movie.douban.com/chart, 留空则默认为Top250): "
        user_provided_url = input(url_input_prompt).strip()
        
        base_url_to_use = ""
        is_top250_like = False

        if not user_provided_url:
            base_url_to_use = 'https://movie.douban.com/top250'
            is_top250_like = True
            print(f"使用默认URL: {base_url_to_use}")
        else:
            if not user_provided_url.startswith("http://") and not user_provided_url.startswith("https://"):
                base_url_to_use = "https://" + user_provided_url
            else:
                base_url_to_use = user_provided_url
            print(f"使用用户提供的URL: {base_url_to_use}")
            if "movie.douban.com/top250" in base_url_to_use or "movie.douban.com/chart" in base_url_to_use:
                 is_top250_like = True

        # 异步请求豆瓣页面
        async def request_douban_page_async(url, page_desc):
            print(f"正在请求 {page_desc}: {url}")
            try:
                response = await make_request_async(
                    url,
                    async_crawler=async_crawler,
                    headers={'User-Agent': headers_dict.get('User-Agent', 'Mozilla/5.0'), 
                           'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'},
                    throttler=throttler,
                    proxy_manager=proxy_manager
                )
                
                if not response:
                    print(f"请求豆瓣页面失败: {url}")
                    return None
                    
                content = await response.text()
                return content
            except Exception as e_req:
                print(f"请求豆瓣页面失败: {url}, 错误: {e_req}")
                return None
        
        num_pages_to_crawl = 10 if is_top250_like and "top250" in base_url_to_use else 1
        for i in range(num_pages_to_crawl):
            page_num_for_display = i + 1
            current_page_url = ""
            page_description = f"第 {page_num_for_display} 页"

            if is_top250_like and "top250" in base_url_to_use:
                url_parts = urlparse(base_url_to_use)
                clean_base_url = f"{url_parts.scheme}://{url_parts.netloc}{url_parts.path.split('?')[0]}"
                current_page_url = f'{clean_base_url}?start={i * 25}&filter='
            else:
                if i > 0 : break 
                current_page_url = base_url_to_use
                page_description = "指定页面"

            html_content = await request_douban_page_async(current_page_url, page_description)
            
            if html_content:
                soup = BeautifulSoup(html_content, 'lxml') 
                page_data = parse_douban_page_data(soup)
                all_collected_movie_data.extend(page_data)
            else:
                print(f"跳过 {page_description}，因为未能获取内容。")
            
            if i < num_pages_to_crawl - 1 and num_pages_to_crawl > 1:
                sleep_duration = random.uniform(throttler.min_interval, throttler.max_interval + 1.0) 
                print(f"处理完一页，额外暂停 {sleep_duration:.2f} 秒...")
                await asyncio.sleep(sleep_duration)
        if not all_collected_movie_data:
            print("未能收集到任何电影数据。")
        else:
            print(f"\n成功收集到 {len(all_collected_movie_data)} 条电影数据。")
            while True:
                export_choice = input("请选择导出格式 (1: Excel, 2: JSON, 3: CSV, 4: SQLite数据库, N: 不导出): ").strip().lower()
                if export_choice == '1':
                    await export_to_excel_async(all_collected_movie_data, base_url_to_use)
                    break
                elif export_choice == '2':
                    await export_to_json_async(all_collected_movie_data, base_url_to_use)
                    break
                elif export_choice == '3':
                    await export_to_csv_async(all_collected_movie_data, base_url_to_use)
                    break
                elif export_choice == '4':
                    await export_to_database_async(all_collected_movie_data, base_url_to_use)
                    break
                elif export_choice == 'n':
                    print("用户选择不导出数据。")
                    break
                else:
                    print("无效的选项，请重新输入。")

    except Exception as e:
        print(f"爬取豆瓣电影过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        print("--- 豆瓣电影爬取任务结束 ---")

async def export_to_excel_async(movie_data, base_url):
    """异步导出到Excel"""
    if not movie_data:
        print("没有数据可以导出到Excel。")
        return      
    save_path = _get_douban_save_path(base_url, "xlsx")
    try:
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet('豆瓣电影', cell_overwrite_ok=True)
        
        for col, header_text in enumerate(DOUBAN_COLUMN_HEADERS):
            sheet.write(0, col, header_text)
        
        for row, data in enumerate(movie_data):
            for col, header in enumerate(DOUBAN_COLUMN_HEADERS):
                sheet.write(row + 1, col, data.get(header, ""))
        
        temp_path = f"{save_path}.tmp"
        book.save(temp_path)
        os.replace(temp_path, save_path)
        
        print(f"\n豆瓣电影数据已成功导出到Excel: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"导出到Excel失败: {e}")
        traceback.print_exc()

async def export_to_json_async(movie_data, base_url):
    """异步导出到JSON"""
    if not movie_data:
        print("没有数据可以导出到JSON。")
        return
        
    save_path = _get_douban_save_path(base_url, "json")
    try:
        temp_path = f"{save_path}.tmp"
        async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(movie_data, ensure_ascii=False, indent=4))
        os.replace(temp_path, save_path)
        
        print(f"\n豆瓣电影数据已成功导出到JSON: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"导出到JSON失败: {e}")
        traceback.print_exc()

async def export_to_csv_async(movie_data, base_url):
    """异步导出到CSV"""
    if not movie_data:
        print("没有数据可以导出到CSV。")
        return
        
    save_path = _get_douban_save_path(base_url, "csv")
    try:
        # 使用临时文件确保原子性写入
        temp_path = f"{save_path}.tmp"
        async with aiofiles.open(temp_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=DOUBAN_COLUMN_HEADERS)
            await writer.writeheader()
            for data in movie_data:
                await writer.writerow(data)
        os.replace(temp_path, save_path)
        
        print(f"\n豆瓣电影数据已成功导出到CSV: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"导出到CSV失败: {e}")
        traceback.print_exc()

async def export_to_database_async(movie_data, base_url):
    """异步导出到SQLite数据库"""
    if not movie_data:
        print("没有数据可以导出到数据库。")
        return  
    db_save_dir = "downloaded_douban"
    os.makedirs(db_save_dir, exist_ok=True)
    db_name = "douban_movies_library.db" 
    db_path = os.path.join(db_save_dir, db_name)
    
    parsed_url = urlparse(base_url)
    table_name_part = parsed_url.netloc.replace('.', '_') + (parsed_url.path.replace('/', '_') if parsed_url.path else "_list")
    table_name = "douban_" + re.sub(r'[^a-zA-Z0-9_]', '', table_name_part)[:50]
    table_name = table_name.strip('_')

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, 
            lambda: _sync_export_to_database(movie_data, db_path, table_name)
        )
        print(f"\n豆瓣电影数据已成功导出到SQLite数据库: {os.path.abspath(db_path)} (表名: {table_name})")
    except Exception as e:
        print(f"导出到SQLite数据库失败: {e}")
        traceback.print_exc()

def _sync_export_to_database(movie_data, db_path, table_name):
    """同步的数据库导出函数，供线程池使用"""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cols_for_create = ", ".join([f'"{col_name}" TEXT' for col_name in DOUBAN_COLUMN_HEADERS])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols_for_create})")
        
        cols_for_insert = ", ".join([f'"{col_name}"' for col_name in DOUBAN_COLUMN_HEADERS])
        placeholders = ", ".join(["?"] * len(DOUBAN_COLUMN_HEADERS))
        
        for data in movie_data:
            values = [data.get(header, "") for header in DOUBAN_COLUMN_HEADERS]
            cursor.execute(f"INSERT INTO {table_name} ({cols_for_insert}) VALUES ({placeholders})", values)
        
        conn.commit()
    finally:
        conn.close()

async def async_main():
    config_data = load_config()
    request_counter = load_counter()
    
    custom_headers = {
        'User-Agent': random.choice(config_data['user_agents'])
    }

    if request_counter >= 20:
        print("\n已达到使用次数上限（或更多），需要更新请求头以继续。")
        user_agent_input = input(f"请输入有效的User-Agent (当前随机选择: {custom_headers['User-Agent']}): ").strip()
        if user_agent_input:
            custom_headers['User-Agent'] = user_agent_input
        save_counter(0)
        request_counter = 0
        print("User-Agent已更新，计数器已重置。")

    # Initialize async components
    default_async_throttler = AsyncRequestThrottler(
        min_interval=config_data['request_throttling']['min_interval'],
        max_interval=config_data['request_throttling']['max_interval'],
        burst_requests=config_data['request_throttling']['burst_requests'],
        burst_period=config_data['request_throttling']['burst_period']
    )
    async_proxy_manager: Optional[AsyncProxyManager] = None

    async with AsyncCrawler(max_connections=10, timeout=DEFAULT_TIMEOUT) as async_crawler_instance:
        if config_data['proxy_settings']['enabled']:
            async_proxy_manager = AsyncProxyManager(
                async_crawler_instance=async_crawler_instance, # Pass the crawler instance
                proxy_list=config_data['proxy_settings']['proxy_list'],
                proxy_api_url=config_data['proxy_settings']['proxy_api_url'],
                api_key=config_data['proxy_settings']['api_key']
            )
        print("\n请选择操作:")
        print("1. 爬取图片链接")
        print("2. 爬取评论")
        print("3. 爬取视频到本地")
        print("4. 爬取豆瓣电影信息")
        print("5. 爬取文档内容")
        choice = input("请输入选项 (1-5): ").strip()
        tasks_increment_counter = True

        if choice == '4':
            try:
                await crawl_douban_top250_movies_async(
                    custom_headers,
                    default_async_throttler,
                    async_crawler_instance,
                    async_proxy_manager
                )
            except Exception as e_douban:
                print(f"豆瓣电影爬取失败: {e_douban}")
                traceback.print_exc()


        elif choice in ['1', '2', '3', '5']:
            target_url_input = input("\n请输入目标网址: ").strip()
            if not target_url_input:
                print("错误: 未输入目标网址。程序将退出。")
                tasks_increment_counter = False
            else:
                processed_target_url = target_url_input
                if not processed_target_url.startswith("http://") and not processed_target_url.startswith("https://"):
                    if "://" in processed_target_url:
                        print(f"错误: 不支持的URL协议。请输入以 http:// 或 https:// 开头的网址，或纯域名。")
                        tasks_increment_counter = False
                    else:
                        processed_target_url = "https://" + processed_target_url
                elif processed_target_url.startswith("http://"):
                    processed_target_url = "https://" + processed_target_url[len("http://"):]

                if tasks_increment_counter:
                    print(f"目标URL (将使用HTTPS协议): {processed_target_url}")

                    if choice == '3' and ('bilibili.com' in processed_target_url or 'b23.tv' in processed_target_url):
                        print("\n检测到B站链接，尝试专门解析视频...")
                        bvid = None
                        bvid_match = re.search(r'(?:bvid=|video/)(BV[a-zA-Z0-9]+)', processed_target_url, re.IGNORECASE)
                        if bvid_match:
                            bvid = bvid_match.group(1)
                        if bvid:
                            print("B站视频解析功能尚未完全迁移到异步，可能无法正常工作或报错。")                         
                        else:
                            print("无法从URL中提取B站视频ID (BVID)。请确保URL包含如 'bvid=BV...' 或 '/video/BV...' 的部分。")
                    elif choice in ['1', '2', '3']: 
                        html_content_main = None
                        try:
                            print(f"\n正在请求目标页面: {processed_target_url} ...") 
                            response = await make_request_async(
                                processed_target_url,
                                async_crawler=async_crawler_instance,
                                headers=custom_headers,
                                throttler=default_async_throttler,
                                proxy_manager=async_proxy_manager
                            )

                            if response:
                                content_type_header = response.headers.get('Content-Type', '').lower()
                                charset = response.charset
                                if not charset:
                                    if 'charset=' in content_type_header:
                                        charset = content_type_header.split('charset=')[-1].split(';')[0].strip()
                                    else:
                                        charset = 'utf-8' # Default
                                try:
                                    html_content_main = await response.text(encoding=charset, errors='replace')
                                except UnicodeDecodeError:
                                    logger.warning(f"UnicodeDecodeError with detected charset {charset} for {processed_target_url}, trying 'utf-8'")
                                    html_content_main = await response.text(encoding='utf-8', errors='replace')
                                print(f"成功获取目标页面内容。")
                            else:
                                print(f"未能获取目标页面 {processed_target_url} (无响应对象)")

                        except aiohttp.ClientResponseError as http_err:
                             print(f"HTTP错误: {http_err.status} - {http_err.message} - URL: {processed_target_url}")
                        except Exception as e_fetch:
                            print(f"请求目标页面失败: {e_fetch} - URL: {processed_target_url}")
                            traceback.print_exc()
                        
                        if html_content_main:
                            if choice == '1':
                                print("\n--- 开始爬取图片链接 ---")
                                print("图片爬取功能尚未完全迁移到异步，可能无法正常工作或报错。")
                            elif choice == '2':
                                print("\n--- 开始爬取评论 ---")
                                await crawl_comments_async(processed_target_url, custom_headers, default_async_throttler, async_crawler_instance, async_proxy_manager)
                            elif choice == '3':
                                print("\n--- 开始爬取视频到本地 (通用方法) ---")
                                print("通用视频爬取功能尚未完全迁移到异步，可能无法正常工作或报错。")                   
                        elif tasks_increment_counter:
                             print("未能获取页面内容，无法继续执行所选操作。")

                    elif choice == '5':
                        print("\n--- 开始爬取文档内容 ---")
                        await crawl_document_async(processed_target_url, custom_headers, default_async_throttler, async_crawler_instance, async_proxy_manager)
        else:
            print("无效的选项。程序将退出。")
            tasks_increment_counter = False

        if tasks_increment_counter and choice in ['1','2','3','4','5']:
            request_counter += 1
            save_counter(request_counter)
            print("\n本次任务使用已记录。")

    print("--- 主任务结束 ---")

if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    except Exception as e_global:
        print(f"发生未捕获的全局错误: {e_global}")
        traceback.print_exc()
    