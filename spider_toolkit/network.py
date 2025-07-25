"""
网络请求模块
"""

import asyncio
import aiohttp
import time
import random
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse
import logging

from .config import config
from .utils import random_delay


class AsyncRequestThrottler:
    """异步请求节流器，控制请求频率"""
    
    def __init__(self, min_delay: float = 1.0, max_delay: float = 3.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0
        self.lock = asyncio.Lock()
    
    async def wait_async(self):
        """异步等待，确保请求间隔"""
        async with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            
            delay = random.uniform(self.min_delay, self.max_delay)
            if elapsed < delay:
                wait_time = delay - elapsed
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()


class SmartRetry:
    """智能重试机制"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 0.3, 
                 status_forcelist: tuple = (500, 502, 503, 504, 429)):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.status_forcelist = status_forcelist
        self.history = {}
        
    def should_retry(self, method: str, status_code: int) -> bool:
        """检查是否应该重试"""
        if method.upper() not in ('GET', 'HEAD', 'OPTIONS'):
            return False
        return status_code in self.status_forcelist
    
    def get_delay(self, attempt: int, status_code: Optional[int] = None) -> float:
        """计算重试延迟"""
        base_delay = self.backoff_factor * (2 ** (attempt - 1))
        jitter = random.uniform(0, base_delay * 0.1)
        return base_delay + jitter
    
    def record(self, url: str, status_code: int):
        """记录请求历史"""
        self.history[url] = {
            'last_status': status_code,
            'last_time': time.time(),
            'retry_count': self.history.get(url, {}).get('retry_count', 0) + 1
        }
    
    def can_retry(self, url: str) -> bool:
        """检查是否可以重试"""
        history = self.history.get(url, {})
        return history.get('retry_count', 0) < self.max_retries


class ResponseWrapper:
    """响应包装器，提供统一的响应接口"""
    
    def __init__(self, status: int, headers: dict, url: str, content: bytes):
        self.status = status
        self.headers = headers
        self.url = url
        self._content = content
        self._text = None
        self._json = None
    
    async def read(self) -> bytes:
        """返回响应内容的字节"""
        return self._content
    
    async def text(self, encoding: str = None, errors: str = 'strict') -> str:
        """返回响应内容的文本形式"""
        if self._text is None:
            if encoding is None:
                # 尝试从Content-Type头部获取编码
                content_type = self.headers.get('Content-Type', '').lower()
                if 'charset=' in content_type:
                    encoding = content_type.split('charset=')[-1].split(';')[0].strip()
                else:
                    encoding = 'utf-8'
            
            try:
                self._text = self._content.decode(encoding, errors=errors)
            except UnicodeDecodeError:
                # 如果指定编码失败，回退到utf-8
                self._text = self._content.decode('utf-8', errors='replace')
        
        return self._text
    
    async def json(self, encoding: str = None) -> dict:
        """返回响应内容的JSON解析结果"""
        if self._json is None:
            import json
            text = await self.text(encoding=encoding)
            self._json = json.loads(text)
        return self._json


class AsyncCrawler:
    """异步爬虫核心类"""
    
    def __init__(self, max_connections: int = 10, timeout: int = 30):
        self.max_connections = max_connections
        self.timeout = timeout
        self.session = None
        self.retry_handler = SmartRetry()
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': config.user_agent}
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def request(self, 
                     method: str,
                     url: str,
                     headers: Optional[Dict[str, str]] = None,
                     data: Optional[Any] = None,
                     json_data: Optional[Dict] = None,
                     params: Optional[Dict] = None,
                     proxy: Optional[str] = None,
                     **kwargs) -> Optional[ResponseWrapper]:
        """发送HTTP请求"""
        
        if not self.session:
            raise RuntimeError("AsyncCrawler must be used as async context manager")
        
        # 合并请求头
        request_headers = {}
        if headers:
            request_headers.update(headers)
        
        attempt = 0
        while attempt <= self.retry_handler.max_retries:
            try:
                self.logger.debug(f"请求 {method} {url} (尝试 {attempt + 1})")
                
                async with self.session.request(
                    method=method,
                    url=url,
                    headers=request_headers,
                    data=data,
                    json=json_data,
                    params=params,
                    proxy=proxy,
                    **kwargs
                ) as response:
                    content = await response.read()
                    
                    # 记录请求历史
                    self.retry_handler.record(url, response.status)
                    
                    # 检查是否需要重试
                    if (response.status in self.retry_handler.status_forcelist and 
                        attempt < self.retry_handler.max_retries):
                        
                        delay = self.retry_handler.get_delay(attempt + 1, response.status)
                        self.logger.warning(f"请求失败 {response.status}，{delay:.2f}秒后重试")
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    
                    return ResponseWrapper(
                        status=response.status,
                        headers=dict(response.headers),
                        url=str(response.url),
                        content=content
                    )
            
            except asyncio.TimeoutError:
                self.logger.error(f"请求超时: {url}")
                if attempt < self.retry_handler.max_retries:
                    delay = self.retry_handler.get_delay(attempt + 1)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                break
            
            except Exception as e:
                self.logger.error(f"请求异常: {url}, 错误: {e}")
                if attempt < self.retry_handler.max_retries:
                    delay = self.retry_handler.get_delay(attempt + 1)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                break
        
        return None
    
    async def get(self, url: str, **kwargs) -> Optional[ResponseWrapper]:
        """发送GET请求"""
        return await self.request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> Optional[ResponseWrapper]:
        """发送POST请求"""
        return await self.request('POST', url, **kwargs)
    
    async def head(self, url: str, **kwargs) -> Optional[ResponseWrapper]:
        """发送HEAD请求"""
        return await self.request('HEAD', url, **kwargs)


class ProxyManager:
    """代理管理器"""
    
    def __init__(self, proxy_list: Optional[List[str]] = None):
        self.proxy_list = proxy_list or []
        self.current_index = 0
        self.failed_proxies = set()
        self.last_rotation_time = time.time()
        self.rotation_interval = config.get('proxy.rotation_interval', 300)
    
    def get_proxy(self) -> Optional[str]:
        """获取可用的代理"""
        if not self.proxy_list:
            return None
        
        # 检查是否需要轮换代理
        current_time = time.time()
        if current_time - self.last_rotation_time > self.rotation_interval:
            self.rotate_proxy()
            self.last_rotation_time = current_time
        
        # 获取当前代理
        available_proxies = [p for p in self.proxy_list if p not in self.failed_proxies]
        if not available_proxies:
            # 重置失败代理列表
            self.failed_proxies.clear()
            available_proxies = self.proxy_list
        
        if available_proxies:
            return available_proxies[self.current_index % len(available_proxies)]
        
        return None
    
    def rotate_proxy(self):
        """轮换代理"""
        if self.proxy_list:
            self.current_index = (self.current_index + 1) % len(self.proxy_list)
    
    def mark_proxy_failed(self, proxy: str):
        """标记代理为失败"""
        self.failed_proxies.add(proxy)
    
    def add_proxy(self, proxy: str):
        """添加代理"""
        if proxy not in self.proxy_list:
            self.proxy_list.append(proxy)
    
    def remove_proxy(self, proxy: str):
        """移除代理"""
        if proxy in self.proxy_list:
            self.proxy_list.remove(proxy)
        if proxy in self.failed_proxies:
            self.failed_proxies.remove(proxy)