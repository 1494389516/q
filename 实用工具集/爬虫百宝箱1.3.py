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

# 配置常量
DEFAULT_TIMEOUT = 30
COMMENT_TIMEOUT = 10

# 设置日志
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
        self.max_threads = 5  # 最大线程数
        self.max_depth = 2    # 最大爬取深度
        self.max_urls_per_page = 20  # 每页最大爬取URL数
        self.domain_restrict = True  # 是否限制在同一域名下爬取
        self.filter_keywords = []    # 过滤关键词列表
        self.filter_regex = ""       # 过滤正则表达式
        self.save_metadata = True    # 是否保存元数据
        self.auto_categorize = True  # 是否自动分类
        self.history_enabled = True  # 是否启用历史记录
        self.resume_enabled = True   # 是否启用断点恢复
        self.avoid_duplicates = True # 是否避免重复爬取

# 全局配置实例
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
        # 如果没有设置过滤条件，保留所有内容
        if not self.keywords and not self.compiled_regex:
            return True
        
        # 关键词过滤
        if self.keywords:
            for keyword in self.keywords:
                if keyword.lower() in text.lower():
                    return True
        
        # 正则表达式过滤
        if self.compiled_regex:
            if self.compiled_regex.search(text):
                return True
        
        # 如果设置了过滤条件但都不匹配，则不保留
        return not (self.keywords or self.compiled_regex)
    
    def categorize_content(self, url, content_type, text=None):
        """根据内容类型和URL自动分类"""
        if not config.auto_categorize:
            return "downloads"
        
        # 根据内容类型分类
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
        
        # 根据URL扩展名分类
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
        
        # 如果是HTML内容，尝试提取标题、描述等
        if html_content and ('text/html' in content_type or content_type == ''):
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # 提取标题
                title_tag = soup.find('title')
                if title_tag:
                    metadata["title"] = title_tag.get_text(strip=True)
                
                # 提取描述
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc:
                    metadata["description"] = meta_desc.get('content', '')
                
                # 提取关键词
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

class RequestThrottler:
    def __init__(self, min_interval=2.0, max_interval=5.0, burst_requests=3, burst_period=60):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.burst_requests = burst_requests
        self.burst_period = burst_period
        self.last_request_time = 0
        self.request_history = []
        self.lock = threading.Lock()  # 添加线程锁，确保线程安全

    def wait(self):
        with self.lock:  # 使用线程锁保护共享资源
            current_time = time.time()
            self.request_history = [t for t in self.request_history if current_time - t < self.burst_period]
            if len(self.request_history) >= self.burst_requests:
                oldest_request = self.request_history[0]
                wait_time = self.burst_period - (current_time - oldest_request)
                if wait_time > 0:
                    print(f"达到突发请求限制，等待 {wait_time:.2f} 秒...")
                    time.sleep(wait_time)
                    current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                wait_time += random.uniform(0, self.max_interval - self.min_interval)
                print(f"请求限速，等待 {wait_time:.2f} 秒...")
                time.sleep(wait_time)
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
                # 将相对URL转换为绝对URL
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
            # 跳过非HTTP链接
            if not link.startswith(('http://', 'https://')):
                continue
            
            # 如果启用了域名限制，只保留同一域名下的链接
            if self.domain_restrict:
                link_domain = urlparse(link).netloc
                if link_domain != base_domain:
                    continue
            
            # 跳过已访问的链接
            if link in self.visited_urls:
                continue
            
            filtered_links.append(link)
        
        # 限制每页提取的链接数量
        return filtered_links[:config.max_urls_per_page]
    
    def add_to_queue(self, url, depth=0, parent_url=None):
        """将URL添加到队列中"""
        if url not in self.visited_urls:
            self.url_queue.put((url, depth, parent_url))
            self.visited_urls.add(url)
            # 添加到任务队列数据库，用于断点恢复
            history.add_task(url, depth, parent_url)
    
    def crawl_recursive(self, start_url, max_depth=None, throttler=None, session=None, headers_dict=None):
        """递归爬取链接"""
        if max_depth is None:
            max_depth = config.max_depth
        
        if not throttler:
            throttler = RequestThrottler()
        
        if not session:
            session = requests.Session()
            if headers_dict:
                session.headers.update(headers_dict)
        
        # 添加起始URL到队列
        self.add_to_queue(start_url)
        
        # 恢复未完成的任务
        pending_tasks = history.get_pending_tasks()
        for url, depth, parent_url in pending_tasks:
            if url not in self.visited_urls:
                self.add_to_queue(url, depth, parent_url)
        
        results = []
        
        while not self.url_queue.empty():
            url, depth, parent_url = self.url_queue.get()
            
            # 检查是否已经爬取过
            if history.url_exists(url):
                history.mark_task_completed(url)
                continue
            
            # 检查深度是否超过限制
            if depth > max_depth:
                continue
            
            logger.info(f"爬取链接 (深度 {depth}/{max_depth}): {url}")
            
            try:
                # 限制请求频率
                throttler.wait()
                
                # 发送请求
                response = session.get(url, timeout=DEFAULT_TIMEOUT)
                response.raise_for_status()
                
                # 处理响应
                content_type = response.headers.get('Content-Type', '')
                
                # 提取元数据
                metadata = content_filter.extract_metadata(url, content_type, response.text)
                
                # 计算内容哈希值
                content_hash = hashlib.md5(response.content).hexdigest()
                
                # 添加到历史记录
                history.add_url(url, "completed", content_hash, depth, parent_url)
                history.mark_task_completed(url)
                
                # 过滤内容
                if 'text/html' in content_type and content_filter.should_keep(response.text):
                    # 保存结果
                    result = {
                        'url': url,
                        'content': response.text,
                        'content_type': content_type,
                        'depth': depth,
                        'metadata': metadata
                    }
                    results.append(result)
                    
                    # 如果深度未达到最大值，继续提取链接
                    if depth < max_depth:
                        links = self.extract_links(response.text, url)
                        filtered_links = self.filter_links(links, url)
                        
                        # 将新链接添加到队列
                        for link in filtered_links:
                            self.add_to_queue(link, depth + 1, url)
                
                # 保存内容到文件
                if config.auto_categorize:
                    category = content_filter.categorize_content(url, content_type, response.text)
                    save_dir = os.path.join("downloads", category)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    
                    # 生成文件名
                    filename = os.path.basename(urlparse(url).path)
                    if not filename:
                        filename = f"{content_hash[:10]}"
                        if 'text/html' in content_type:
                            filename += ".html"
                    
                    file_path = os.path.join(save_dir, filename)
                    
                    # 保存文件
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    # 保存元数据
                    if metadata:
                        meta_path = f"{file_path}.meta.json"
                        with open(meta_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                    # 添加内容到历史记录
                    history.add_content(content_hash, content_type, file_path, metadata)
            
            except Exception as e:
                logger.error(f"爬取链接失败 {url}: {e}")
                history.add_url(url, f"failed: {str(e)}", None, depth, parent_url)
        
        return results

default_throttler = RequestThrottler()

def douban_spider(url, headers_dict, throttler=default_throttler):
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

def parse_bilibili_video(bvid, referer, headers, throttler=default_throttler):
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

def download_bilibili_m3u8(data, referer_param, headers_param, throttler=default_throttler):
    download_dir = "downloaded_videos"
    os.makedirs(download_dir, exist_ok=True)
    video_info = data['data']['dash']['video'][0]
    audio_info = data['data']['dash']['audio'][0]
    print(f"发现视频流: {video_info['id']} | 分辨率: {video_info['width']}x{video_info['height']}")
    video_url = video_info['baseUrl']
    audio_url = audio_info['baseUrl']
    temp_dir = f"temp_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)

    def download_stream(url, name, stream_headers, stream_referer, stream_throttler):
        try:
            print(f"开始下载{name}流...")
            current_headers = stream_headers.copy()
            current_headers['Referer'] = stream_referer
            stream_throttler.wait()
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


def crawl_images_revised(page_url, html_content_for_images, headers_dict, recursive=True, max_depth=None):
    """
    爬取图片链接，支持多线程下载和递归爬取
    """
    if max_depth is None:
        max_depth = config.max_depth

    # 创建图片下载目录
    download_dir = os.path.join("downloads", "images")
    os.makedirs(download_dir, exist_ok=True)

    # 初始化链接提取器
    link_extractor = LinkExtractor()
    
    def process_url(url):
        """处理URL，确保使用HTTPS"""
        if url.startswith("http://"):
            return "https://" + url[len("http://"):]
        elif not url.startswith("https://") and "://" not in url:
            return "https://" + url
        return url

    def download_image(img_url, referer_url, task_id):
        """下载单个图片的函数"""
        try:
            # 检查是否已下载过
            url_hash = hashlib.md5(img_url.encode()).hexdigest()
            if history.url_exists(img_url):
                logger.info(f"图片已下载过: {img_url}")
                return

            # 准备请求头
            local_headers = headers_dict.copy()
            local_headers['Referer'] = referer_url

            # 发送请求
            default_throttler.wait()
            response = requests.get(img_url, headers=local_headers, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()

            # 检查内容类型
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"非图片内容类型: {content_type}, URL: {img_url}")
                return

            # 生成文件名
            ext = content_type.split('/')[-1].lower()
            if ext == 'jpeg':
                ext = 'jpg'
            filename = f"{url_hash[:10]}_{int(time.time())}_{task_id}.{ext}"
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
                'download_time': datetime.now().isoformat()
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

    def process_page(url, html_content, depth=0):
        """处理单个页面，提取图片和链接"""
        processed_urls = set()
        base_url = process_url(url)
        
        # 提取图片URL
        image_pattern = r'<img.*?src=["\'](.*?)["\'].*?>'
        img_urls_found = re.findall(image_pattern, html_content, re.IGNORECASE)
        
        # 处理每个图片URL
        for img_url in img_urls_found:
            try:
                # 跳过数据URI
                if img_url.startswith("data:"):
                    continue

                # 处理相对URL
                if img_url.startswith("//"):
                    final_url = f"https:{img_url}"
                elif img_url.startswith("/"):
                    final_url = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}{img_url}"
                else:
                    final_url = urljoin(base_url, img_url)

                # 确保使用HTTPS
                final_url = process_url(final_url)

                # 如果是新的URL，添加到处理队列
                if final_url not in processed_urls and final_url.startswith("https://"):
                    processed_urls.add(final_url)
                    # 提交下载任务到线程池
                    task_id = hashlib.md5(final_url.encode()).hexdigest()[:6]
                    thread_pool.submit_task(task_id, download_image, final_url, base_url, task_id)

            except Exception as e:
                logger.error(f"处理图片URL失败: {e}")

        # 如果启用了递归爬取，且未达到最大深度
        if recursive and depth < max_depth:
            # 提取页面中的链接
            links = link_extractor.extract_links(html_content, base_url)
            filtered_links = link_extractor.filter_links(links, base_url)

            # 递归处理每个链接
            for link in filtered_links:
                try:
                    if link not in link_extractor.visited_urls:
                        link_extractor.visited_urls.add(link)
                        # 获取链接页面内容
                        default_throttler.wait()
                        response = requests.get(link, headers=headers_dict, timeout=DEFAULT_TIMEOUT)
                        response.raise_for_status()
                        # 递归处理新页面
                        process_page(link, response.text, depth + 1)
                except Exception as e:
                    logger.error(f"处理链接失败 {link}: {e}")

    # 开始处理初始页面
    process_page(page_url, html_content_for_images)
    
    # 等待所有下载任务完成
    thread_pool.wait_for_all()
    
    logger.info("图片爬取任务完成")

def crawl_comments(target_url, headers_dict, throttler=default_throttler):
    print(f"正在尝试从 {target_url} 爬取评论...")
    try:
        throttler.wait()
        response = requests.get(target_url, headers=headers_dict, timeout=COMMENT_TIMEOUT)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        html_content = response.text
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
    except requests.exceptions.RequestException as e:
        print(f"请求页面内容失败: {e}")
    except Exception as e:
        print(f"处理评论时发生错误: {e}")

def crawl_videos(target_url, headers_dict, html_content, throttler=default_throttler):
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

def crawl_document(target_url, headers_dict, throttler=default_throttler):
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
            throttler.wait()
            response = requests.get(target_url, headers=headers_dict, timeout=DEFAULT_TIMEOUT, stream=True, allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
            content_disposition = response.headers.get('Content-Disposition')
            file_name_from_disposition = None
            if content_disposition:
                fn_match = re.search(r'filename\*?=(?:UTF-8\'\')?([^;\n]+)', content_disposition, flags=re.IGNORECASE)
                if fn_match:
                    file_name_from_disposition = unquote(fn_match.group(1).strip('"'))

            base_file_name = f"document_{int(time.time())}"
            file_ext = ""

            if file_name_from_disposition:
                base_file_name_from_disp, file_ext_from_disp = os.path.splitext(file_name_from_disposition)
                base_file_name = re.sub(r'[^\w\-. ]', '_', base_file_name_from_disp)
                if file_ext_from_disp and len(file_ext_from_disp) < 10 : file_ext = file_ext_from_disp      
            
            if not file_ext:
                if content_type in CONTENT_TYPE_MAP:
                    file_ext = CONTENT_TYPE_MAP[content_type]
                else:
                    path = urlparse(target_url).path
                    file_ext_from_url = os.path.splitext(path)[1]
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
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if time.time() - start_time > 1 or downloaded_size == content_length :
                            elapsed_time = time.time() - start_time if time.time() - start_time > 0 else 1
                            speed = downloaded_size / elapsed_time / 1024 if elapsed_time > 0 else 0
                            progress = f"{downloaded_size/1024:.1f} KB"
                            if content_length > 0:
                                percent = (downloaded_size / content_length) * 100
                                progress = f"{downloaded_size/1024:.1f}/{content_length/1024:.1f} KB ({percent:.1f}%)"
                            print(f"\r下载进度: {progress} ({speed:.1f} KB/s)", end='')
            print()
            final_size = os.path.getsize(full_path)
            if content_length > 0 and final_size != content_length:
                print(f"\n警告: 文件大小不匹配 (预期: {content_length} 字节, 实际: {final_size} 字节)")
            if final_size == 0 and (content_length == 0 or content_length == -1) :
                 print("警告: 下载的文件大小为0字节，可能下载失败或文件为空。")
            elif final_size < 1024 and (content_length <=0 or content_length == -1):
                print("警告: 文件尺寸过小，可能是错误页面或不完整的文档。")
            print(f"\n文档下载成功: {final_file_name}")
            return
        except requests.exceptions.HTTPError as e:
            print(f"HTTP错误 ({e.response.status_code}): {str(e)}")
            if attempt == retry_count: print("已达到最大重试次数")
        except requests.exceptions.Timeout:
            print("请求超时")
            if attempt == retry_count: print("无法在超时时间内完成下载")
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {str(e)}")
            return
        except Exception as e:
            print(f"未知错误: {str(e)}\n{traceback.format_exc()}")
            return
        if attempt < retry_count:
            print("等待2秒后重试...")
            time.sleep(2 + random.uniform(0,1))

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
        with open(save_path, 'w', newline='', encoding='utf-8-sig') as f: # utf-8-sig for Excel compatibility
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
    
    # Sanitize table name from base_url_to_use
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

def crawl_douban_top250_movies(headers_dict, throttler=default_throttler):
    print("\n--- 开始爬取豆瓣电影 ---")
    all_collected_movie_data = []
    try:
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

        def request_douban_page(url, page_desc, req_headers):
            print(f"正在请求 {page_desc}: {url}")
            try:
                throttler.wait()
                response = requests.get(url, headers=req_headers, timeout=DEFAULT_TIMEOUT)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e_req:
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

            douban_req_headers = {'User-Agent': headers_dict.get('User-Agent', 'Mozilla/5.0'), 'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'}
            html_content = request_douban_page(current_page_url, page_description, douban_req_headers)
            
            if html_content:
                soup = BeautifulSoup(html_content, 'lxml') 
                page_data = parse_douban_page_data(soup)
                all_collected_movie_data.extend(page_data)
            else:
                print(f"跳过 {page_description}，因为未能获取内容。")
            
            if i < num_pages_to_crawl - 1 and num_pages_to_crawl > 1:
                 sleep_duration = random.uniform(throttler.min_interval, throttler.max_interval + 1.0) 
                 print(f"处理完一页，额外暂停 {sleep_duration:.2f} 秒...")
                 time.sleep(sleep_duration)
        
        if not all_collected_movie_data:
            print("未能收集到任何电影数据。")
        else:
            print(f"\n成功收集到 {len(all_collected_movie_data)} 条电影数据。")
            while True:
                export_choice = input("请选择导出格式 (1: Excel, 2: JSON, 3: CSV, 4: SQLite数据库, N: 不导出): ").strip().lower()
                if export_choice == '1':
                    export_to_excel(all_collected_movie_data, base_url_to_use)
                    break
                elif export_choice == '2':
                    export_to_json(all_collected_movie_data, base_url_to_use)
                    break
                elif export_choice == '3':
                    export_to_csv(all_collected_movie_data, base_url_to_use)
                    break
                elif export_choice == '4':
                    export_to_database(all_collected_movie_data, base_url_to_use)
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

def main():
    request_counter = load_counter()
    custom_headers = {}

    if request_counter >= 20: # Adjusted counter limit
        print("\n已达到使用次数上限（或更多），需要更新请求头以继续。")
        user_agent_input = input("请重新输入有效的User-Agent (例如: Mozilla/5.0 ...): ").strip()
        if not user_agent_input:
            print("错误: User-Agent是必填项。程序将退出。")
            return
        custom_headers['User-Agent'] = user_agent_input
        save_counter(0) 
        request_counter = 0 
        print("User-Agent已更新，计数器已重置。")
    else:
        user_agent_input = input("请输入User-Agent (例如: Mozilla/5.0 ...): ").strip()
        if not user_agent_input:
            print("错误: User-Agent是必填项。程序将退出。")
            return
        custom_headers['User-Agent'] = user_agent_input

    print("\n请选择操作:")
    print("1. 爬取图片链接")
    print("2. 爬取评论")
    print("3. 爬取视频到本地")
    print("4. 爬取豆瓣电影信息")
    print("5. 爬取文档内容")
    choice = input("请输入选项 (1-5): ").strip()
    
    tasks_increment_counter = True

    if choice == '4':
        crawl_douban_top250_movies(custom_headers, default_throttler)
    elif choice in ['1', '2', '3', '5']:
        target_url_input = input("\n请输入目标网址: ").strip()
        if not target_url_input:
            print("错误: 未输入目标网址。程序将退出。")
            tasks_increment_counter = False # Don't increment if no URL
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
            
            if tasks_increment_counter: # Only proceed if URL is valid so far
                print(f"目标URL (将使用HTTPS协议): {processed_target_url}")

                if choice == '3' and ('bilibili.com' in processed_target_url or 'b23.tv' in processed_target_url):
                    print("\n检测到B站链接，尝试专门解析视频...")
                    bvid = None
                    bvid_match = re.search(r'(?:bvid=|video/)(BV[a-zA-Z0-9]+)', processed_target_url, re.IGNORECASE)
                    if bvid_match:
                        bvid = bvid_match.group(1)
                    
                    if bvid:
                        parse_bilibili_video(bvid, processed_target_url, custom_headers, default_throttler)
                    else:
                        print("无法从URL中提取B站视频ID (BVID)。请确保URL包含如 'bvid=BV...' 或 '/video/BV...' 的部分。")
                
                elif choice in ['1', '2', '3']:
                    html_content_main = None
                    try:
                        print(f"\n正在请求目标页面: {processed_target_url} ...")
                        default_throttler.wait()
                        response = requests.get(processed_target_url, headers=custom_headers, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
                        response.raise_for_status()
                        response.encoding = response.apparent_encoding 
                        html_content_main = response.text
                        print(f"成功获取目标页面内容。")
                    except requests.exceptions.Timeout:
                        print(f"请求超时: 无法在{DEFAULT_TIMEOUT}秒内连接到 {processed_target_url}")
                    except requests.exceptions.HTTPError as http_err:
                        print(f"HTTP错误: {http_err} - {response.status_code if hasattr(response, 'status_code') else 'N/A'}")
                    except requests.exceptions.RequestException as e:
                        print(f"请求目标页面失败: {e}")
                    
                    if html_content_main:
                        if choice == '1':
                            print("\n--- 开始爬取图片链接 ---")
                            crawl_images_revised(processed_target_url, html_content_main, custom_headers)
                        elif choice == '2':
                            print("\n--- 开始爬取评论 ---")
                            crawl_comments(processed_target_url, custom_headers, default_throttler)
                        elif choice == '3': 
                            print("\n--- 开始爬取视频到本地 (通用方法) ---")
                            crawl_videos(processed_target_url, custom_headers, html_content_main, default_throttler)
                    elif tasks_increment_counter: # Only if we intended to run a task
                         print("未能获取页面内容，无法继续执行所选操作。")

                elif choice == '5':
                    print("\n--- 开始爬取文档内容 ---")
                    crawl_document(processed_target_url, custom_headers, default_throttler)
    else:
        print("无效的选项。程序将退出。")
        tasks_increment_counter = False # Don't increment for invalid choice

    if tasks_increment_counter and choice in ['1','2','3','4','5']:
        request_counter += 1
        save_counter(request_counter)
        print("\n本次任务使用已记录。")
    
    print("--- 主任务结束 ---")

if __name__ == "__main__":
    main()








    