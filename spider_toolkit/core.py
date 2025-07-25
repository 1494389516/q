"""
爬虫工具包核心模块
"""

import asyncio
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse, urljoin
import logging

from .config import config
from .network import AsyncCrawler, AsyncRequestThrottler, ProxyManager
from .utils import (
    setup_logging, sanitize_filename, extract_domain, 
    save_json_data, create_timestamp_filename
)
from .parsers import CommentParser, VideoParser, DocumentParser
from .captcha import CaptchaSolver


class SpiderToolkit:
    """爬虫工具包主类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化爬虫工具包
        
        Args:
            config_file: 配置文件路径
        """
        # 加载配置
        if config_file:
            config.config_file = config_file
            config.config_data = config._load_config()
        
        # 设置日志
        self.logger = setup_logging(
            level=config.get('logging.level', 'INFO'),
            log_file=config.get('logging.file', 'crawler.log')
        )
        
        # 初始化组件
        self.throttler = AsyncRequestThrottler(
            min_delay=config.get('request.delay_range', [1, 3])[0],
            max_delay=config.get('request.delay_range', [1, 3])[1]
        )
        
        self.proxy_manager = None
        if config.proxy_enabled:
            self.proxy_manager = ProxyManager(config.get('proxy.proxy_list', []))
        
        self.captcha_solver = None
        if config.captcha_enabled:
            self.captcha_solver = CaptchaSolver()
        
        # 解析器
        self.comment_parser = CommentParser()
        self.video_parser = VideoParser()
        self.document_parser = DocumentParser()
    
    async def crawl_comments(self, 
                           url: str, 
                           headers: Optional[Dict[str, str]] = None,
                           save_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """爬取评论数据
        
        Args:
            url: 目标URL
            headers: 自定义请求头
            save_file: 保存文件路径
            
        Returns:
            评论数据列表
        """
        self.logger.info(f"开始爬取评论: {url}")
        
        # 准备请求头
        request_headers = {'User-Agent': config.user_agent}
        if headers:
            request_headers.update(headers)
        
        async with AsyncCrawler(timeout=config.request_timeout) as crawler:
            # 获取页面内容
            await self.throttler.wait_async()
            response = await crawler.get(
                url, 
                headers=request_headers,
                proxy=self.proxy_manager.get_proxy() if self.proxy_manager else None
            )
            
            if not response or response.status != 200:
                self.logger.error(f"请求失败: {url}")
                return []
            
            html_content = await response.text()
            
            # 解析评论
            comments = await self.comment_parser.parse(url, html_content, crawler, request_headers, self.throttler)
            
            # 保存数据
            if comments and save_file:
                self._save_comments(comments, url, save_file)
            
            self.logger.info(f"成功爬取 {len(comments)} 条评论")
            return comments
    
    async def crawl_videos(self, 
                          url: str,
                          headers: Optional[Dict[str, str]] = None,
                          download_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """爬取视频数据
        
        Args:
            url: 目标URL
            headers: 自定义请求头
            download_dir: 下载目录
            
        Returns:
            视频信息列表
        """
        self.logger.info(f"开始爬取视频: {url}")
        
        # 准备请求头
        request_headers = {'User-Agent': config.user_agent}
        if headers:
            request_headers.update(headers)
        
        # 设置下载目录
        if not download_dir:
            download_dir = os.path.join(config.download_dir, 'videos')
        
        async with AsyncCrawler(timeout=config.request_timeout) as crawler:
            # 获取页面内容
            await self.throttler.wait_async()
            response = await crawler.get(
                url,
                headers=request_headers,
                proxy=self.proxy_manager.get_proxy() if self.proxy_manager else None
            )
            
            if not response or response.status != 200:
                self.logger.error(f"请求失败: {url}")
                return []
            
            html_content = await response.text()
            
            # 解析视频
            videos = await self.video_parser.parse(
                url, html_content, crawler, request_headers, 
                self.throttler, download_dir
            )
            
            self.logger.info(f"成功处理 {len(videos)} 个视频")
            return videos
    
    async def crawl_documents(self,
                            url: str,
                            headers: Optional[Dict[str, str]] = None,
                            download_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """爬取文档数据
        
        Args:
            url: 目标URL
            headers: 自定义请求头
            download_dir: 下载目录
            
        Returns:
            文档信息列表
        """
        self.logger.info(f"开始爬取文档: {url}")
        
        # 准备请求头
        request_headers = {'User-Agent': config.user_agent}
        if headers:
            request_headers.update(headers)
        
        # 设置下载目录
        if not download_dir:
            download_dir = os.path.join(config.download_dir, 'documents')
        
        async with AsyncCrawler(timeout=config.request_timeout) as crawler:
            # 获取页面内容
            await self.throttler.wait_async()
            response = await crawler.get(
                url,
                headers=request_headers,
                proxy=self.proxy_manager.get_proxy() if self.proxy_manager else None
            )
            
            if not response or response.status != 200:
                self.logger.error(f"请求失败: {url}")
                return []
            
            html_content = await response.text()
            
            # 解析文档
            documents = await self.document_parser.parse(
                url, html_content, crawler, request_headers,
                self.throttler, download_dir
            )
            
            self.logger.info(f"成功处理 {len(documents)} 个文档")
            return documents
    
    async def crawl_images(self,
                          url: str,
                          headers: Optional[Dict[str, str]] = None,
                          download_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """爬取图片
        
        Args:
            url: 目标URL
            headers: 自定义请求头
            download_dir: 下载目录
            
        Returns:
            图片信息列表
        """
        self.logger.info(f"开始爬取图片: {url}")
        
        # 准备请求头
        request_headers = {'User-Agent': config.user_agent}
        if headers:
            request_headers.update(headers)
        
        # 设置下载目录
        if not download_dir:
            download_dir = os.path.join(config.download_dir, 'images')
        
        os.makedirs(download_dir, exist_ok=True)
        
        async with AsyncCrawler(timeout=config.request_timeout) as crawler:
            # 获取页面内容
            await self.throttler.wait_async()
            response = await crawler.get(
                url,
                headers=request_headers,
                proxy=self.proxy_manager.get_proxy() if self.proxy_manager else None
            )
            
            if not response or response.status != 200:
                self.logger.error(f"请求失败: {url}")
                return []
            
            html_content = await response.text()
            
            # 提取图片URL
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            images = []
            img_tags = soup.find_all('img')
            
            for i, img in enumerate(img_tags):
                img_url = img.get('src') or img.get('data-src') or img.get('data-original')
                if not img_url:
                    continue
                
                # 转换为绝对URL
                if not img_url.startswith(('http://', 'https://')):
                    img_url = urljoin(url, img_url)
                
                # 下载图片
                await self.throttler.wait_async()
                img_response = await crawler.get(
                    img_url,
                    headers=request_headers,
                    proxy=self.proxy_manager.get_proxy() if self.proxy_manager else None
                )
                
                if img_response and img_response.status == 200:
                    # 生成文件名
                    filename = f"image_{i+1:03d}.jpg"
                    if '.' in img_url.split('/')[-1]:
                        ext = img_url.split('/')[-1].split('.')[-1]
                        filename = f"image_{i+1:03d}.{ext}"
                    
                    filepath = os.path.join(download_dir, sanitize_filename(filename))
                    
                    # 保存图片
                    content = await img_response.read()
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    
                    images.append({
                        'url': img_url,
                        'filepath': filepath,
                        'size': len(content),
                        'alt': img.get('alt', ''),
                        'title': img.get('title', '')
                    })
                    
                    self.logger.debug(f"下载图片: {filename}")
            
            self.logger.info(f"成功下载 {len(images)} 张图片")
            return images
    
    def _save_comments(self, comments: List[Dict], source_url: str, filepath: str):
        """保存评论数据"""
        data = {
            'source_url': source_url,
            'crawl_time': datetime.now().isoformat(),
            'comment_count': len(comments),
            'comments': comments
        }
        
        save_json_data(data, filepath)
        self.logger.info(f"评论数据已保存到: {filepath}")
    
    async def batch_crawl(self, 
                         urls: List[str],
                         crawl_type: str = 'comments',
                         headers: Optional[Dict[str, str]] = None,
                         max_concurrent: int = 5) -> Dict[str, Any]:
        """批量爬取
        
        Args:
            urls: URL列表
            crawl_type: 爬取类型 ('comments', 'videos', 'documents', 'images')
            headers: 自定义请求头
            max_concurrent: 最大并发数
            
        Returns:
            批量爬取结果
        """
        self.logger.info(f"开始批量爬取 {len(urls)} 个URL，类型: {crawl_type}")
        
        # 选择爬取方法
        crawl_methods = {
            'comments': self.crawl_comments,
            'videos': self.crawl_videos,
            'documents': self.crawl_documents,
            'images': self.crawl_images
        }
        
        if crawl_type not in crawl_methods:
            raise ValueError(f"不支持的爬取类型: {crawl_type}")
        
        crawl_method = crawl_methods[crawl_type]
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def crawl_single(url):
            async with semaphore:
                try:
                    return await crawl_method(url, headers)
                except Exception as e:
                    self.logger.error(f"爬取失败 {url}: {e}")
                    return []
        
        # 并发执行
        tasks = [crawl_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        success_count = 0
        total_items = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"URL {urls[i]} 爬取异常: {result}")
            elif isinstance(result, list):
                success_count += 1
                total_items += len(result)
        
        summary = {
            'total_urls': len(urls),
            'success_count': success_count,
            'failed_count': len(urls) - success_count,
            'total_items': total_items,
            'results': dict(zip(urls, results))
        }
        
        self.logger.info(f"批量爬取完成: 成功 {success_count}/{len(urls)}, 总计 {total_items} 项")
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """获取爬虫统计信息"""
        return {
            'config': {
                'timeout': config.request_timeout,
                'max_retries': config.max_retries,
                'proxy_enabled': config.proxy_enabled,
                'captcha_enabled': config.captcha_enabled
            },
            'proxy_manager': {
                'proxy_count': len(self.proxy_manager.proxy_list) if self.proxy_manager else 0,
                'failed_proxies': len(self.proxy_manager.failed_proxies) if self.proxy_manager else 0
            } if self.proxy_manager else None
        }