"""
解析器模块 - 处理不同类型的内容解析
"""

import re
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import logging

from .utils import sanitize_filename, extract_domain, clean_text


class BaseParser:
    """基础解析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_json_from_html(self, html_content: str) -> List[Dict]:
        """从HTML中提取JSON数据"""
        json_data_list = []
        
        # 查找script标签中的JSON数据
        json_patterns = [
            r'<script[^>]*>\s*window\.__INITIAL_STATE__\s*=\s*({.+?})\s*</script>',
            r'<script[^>]*>\s*window\.__NUXT__\s*=\s*({.+?})\s*</script>',
            r'<script[^>]*>\s*var\s+\w+\s*=\s*({.+?})\s*</script>',
            r'<script[^>]*type=["\']application/json["\'][^>]*>(.+?)</script>',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    json_obj = json.loads(match)
                    json_data_list.append(json_obj)
                except json.JSONDecodeError:
                    continue
        
        return json_data_list


class CommentParser(BaseParser):
    """评论解析器"""
    
    async def parse(self, url: str, html_content: str, crawler, headers: Dict, throttler) -> List[Dict[str, Any]]:
        """解析评论数据"""
        domain = extract_domain(url)
        
        # 根据不同平台选择解析策略
        if 'xiaohongshu.com' in domain:
            return await self._parse_xiaohongshu_comments(url, html_content, crawler, headers, throttler)
        elif 'bilibili.com' in domain:
            return await self._parse_bilibili_comments(url, html_content, crawler, headers, throttler)
        elif 'douyin.com' in domain:
            return await self._parse_douyin_comments(url, html_content, crawler, headers, throttler)
        else:
            return await self._parse_generic_comments(url, html_content, crawler, headers, throttler)
    
    async def _parse_xiaohongshu_comments(self, url: str, html_content: str, crawler, headers: Dict, throttler) -> List[Dict]:
        """解析小红书评论"""
        comments = []
        
        # 提取笔记ID
        note_id_match = re.search(r'item/(\w+)', url)
        if not note_id_match:
            return comments
        
        note_id = note_id_match.group(1)
        
        # 构建评论API URL
        api_url = f"https://edith.xiaohongshu.com/api/sns/web/v1/comment/page?note_id={note_id}&cursor=&top_comment_id="
        
        try:
            await throttler.wait_async()
            response = await crawler.get(api_url, headers=headers)
            
            if response and response.status == 200:
                json_data = await response.json()
                
                if 'data' in json_data and 'comments' in json_data['data']:
                    for comment in json_data['data']['comments']:
                        user_info = comment.get('user_info', {})
                        comments.append({
                            'platform': 'xiaohongshu',
                            'user': user_info.get('nickname', '未知用户'),
                            'content': comment.get('content', ''),
                            'time': comment.get('create_time', ''),
                            'likes': comment.get('like_count', 0),
                            'id': comment.get('id', '')
                        })
        except Exception as e:
            self.logger.error(f"解析小红书评论失败: {e}")
        
        return comments
    
    async def _parse_bilibili_comments(self, url: str, html_content: str, crawler, headers: Dict, throttler) -> List[Dict]:
        """解析B站评论"""
        comments = []
        
        # 提取视频ID
        bv_match = re.search(r'BV(\w+)', url)
        if not bv_match:
            return comments
        
        bvid = 'BV' + bv_match.group(1)
        
        # 获取视频信息以获取aid和cid
        info_url = f'https://api.bilibili.com/x/web-interface/view?bvid={bvid}'
        
        try:
            await throttler.wait_async()
            response = await crawler.get(info_url, headers=headers)
            
            if response and response.status == 200:
                video_info = await response.json()
                
                if video_info.get('code') == 0:
                    aid = video_info['data']['aid']
                    
                    # 获取评论
                    comment_url = f'https://api.bilibili.com/x/v2/reply?type=1&oid={aid}&sort=2'
                    
                    await throttler.wait_async()
                    comment_response = await crawler.get(comment_url, headers=headers)
                    
                    if comment_response and comment_response.status == 200:
                        comment_data = await comment_response.json()
                        
                        if comment_data.get('code') == 0 and 'replies' in comment_data.get('data', {}):
                            for reply in comment_data['data']['replies']:
                                comments.append({
                                    'platform': 'bilibili',
                                    'user': reply['member']['uname'],
                                    'content': reply['content']['message'],
                                    'time': datetime.fromtimestamp(reply['ctime']).strftime('%Y-%m-%d %H:%M:%S'),
                                    'likes': reply['like'],
                                    'id': reply['rpid']
                                })
        except Exception as e:
            self.logger.error(f"解析B站评论失败: {e}")
        
        return comments
    
    async def _parse_douyin_comments(self, url: str, html_content: str, crawler, headers: Dict, throttler) -> List[Dict]:
        """解析抖音评论"""
        comments = []
        
        # 抖音评论需要更复杂的逆向，这里提供基础框架
        try:
            # 从HTML中提取可能的JSON数据
            json_data_list = self.extract_json_from_html(html_content)
            
            for json_data in json_data_list:
                # 递归查找评论数据
                found_comments = self._find_comments_in_json(json_data)
                comments.extend(found_comments)
        
        except Exception as e:
            self.logger.error(f"解析抖音评论失败: {e}")
        
        return comments
    
    async def _parse_generic_comments(self, url: str, html_content: str, crawler, headers: Dict, throttler) -> List[Dict]:
        """通用评论解析"""
        comments = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 常见的评论容器选择器
            comment_selectors = [
                '.comment', '.comments', '.comment-item', '.comment-list',
                '[class*="comment"]', '[id*="comment"]',
                '.reply', '.replies', '.review', '.reviews'
            ]
            
            for selector in comment_selectors:
                comment_elements = soup.select(selector)
                
                if comment_elements and len(comment_elements) > 2:  # 至少找到几个评论元素
                    for i, element in enumerate(comment_elements[:50]):  # 限制数量
                        # 提取用户名
                        user_selectors = ['.user', '.username', '.author', '.name', '[class*="user"]']
                        user = "未知用户"
                        for user_sel in user_selectors:
                            user_elem = element.select_one(user_sel)
                            if user_elem:
                                user = clean_text(user_elem.get_text())
                                break
                        
                        # 提取评论内容
                        content_selectors = ['.content', '.text', '.message', '.body', 'p']
                        content = ""
                        for content_sel in content_selectors:
                            content_elem = element.select_one(content_sel)
                            if content_elem:
                                content = clean_text(content_elem.get_text())
                                if len(content) > 10:  # 确保内容不是太短
                                    break
                        
                        # 提取时间
                        time_selectors = ['.time', '.date', '[class*="time"]', '[class*="date"]']
                        time_str = ""
                        for time_sel in time_selectors:
                            time_elem = element.select_one(time_sel)
                            if time_elem:
                                time_str = clean_text(time_elem.get_text())
                                break
                        
                        if content and len(content) > 5:  # 确保有有效内容
                            comments.append({
                                'platform': 'generic',
                                'user': user,
                                'content': content,
                                'time': time_str,
                                'likes': 0,
                                'id': f"comment_{i}"
                            })
                    
                    if comments:  # 如果找到了评论就停止尝试其他选择器
                        break
        
        except Exception as e:
            self.logger.error(f"通用评论解析失败: {e}")
        
        return comments
    
    def _find_comments_in_json(self, data: Any, depth: int = 0) -> List[Dict]:
        """在JSON数据中递归查找评论"""
        if depth > 5:  # 限制递归深度
            return []
        
        comments = []
        comment_keys = ['comments', 'comment_list', 'commentList', 'replies', 'reply_list']
        
        if isinstance(data, dict):
            # 检查是否是评论对象
            if any(key in data for key in ['content', 'text', 'message']) and any(key in data for key in ['user', 'author', 'nickname']):
                comment = {
                    'platform': 'json_extracted',
                    'user': str(data.get('user') or data.get('author') or data.get('nickname', '未知用户')),
                    'content': str(data.get('content') or data.get('text') or data.get('message', '')),
                    'time': str(data.get('time') or data.get('create_time') or data.get('date', '')),
                    'likes': data.get('like_count') or data.get('likes') or 0,
                    'id': str(data.get('id') or data.get('comment_id', ''))
                }
                comments.append(comment)
            
            # 递归查找
            for key, value in data.items():
                if key in comment_keys and isinstance(value, list):
                    for item in value:
                        comments.extend(self._find_comments_in_json(item, depth + 1))
                elif isinstance(value, (dict, list)):
                    comments.extend(self._find_comments_in_json(value, depth + 1))
        
        elif isinstance(data, list):
            for item in data:
                comments.extend(self._find_comments_in_json(item, depth + 1))
        
        return comments


class VideoParser(BaseParser):
    """视频解析器"""
    
    async def parse(self, url: str, html_content: str, crawler, headers: Dict, throttler, download_dir: str) -> List[Dict[str, Any]]:
        """解析视频数据"""
        videos = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 查找视频标签
            video_tags = soup.find_all(['video', 'source'])
            
            for video_tag in video_tags:
                video_url = video_tag.get('src') or video_tag.get('data-src')
                if not video_url:
                    continue
                
                # 转换为绝对URL
                if not video_url.startswith(('http://', 'https://')):
                    video_url = urljoin(url, video_url)
                
                # 下载视频
                video_info = await self._download_video(video_url, crawler, headers, throttler, download_dir)
                if video_info:
                    videos.append(video_info)
            
            # 从JSON数据中查找视频URL
            json_data_list = self.extract_json_from_html(html_content)
            for json_data in json_data_list:
                video_urls = self._find_video_urls_in_json(json_data)
                for video_url in video_urls:
                    video_info = await self._download_video(video_url, crawler, headers, throttler, download_dir)
                    if video_info:
                        videos.append(video_info)
        
        except Exception as e:
            self.logger.error(f"解析视频失败: {e}")
        
        return videos
    
    async def _download_video(self, video_url: str, crawler, headers: Dict, throttler, download_dir: str) -> Optional[Dict]:
        """下载单个视频"""
        try:
            await throttler.wait_async()
            response = await crawler.get(video_url, headers=headers)
            
            if not response or response.status != 200:
                return None
            
            # 生成文件名
            filename = os.path.basename(urlparse(video_url).path)
            if not filename or '.' not in filename:
                filename = f"video_{int(datetime.now().timestamp())}.mp4"
            
            filepath = os.path.join(download_dir, sanitize_filename(filename))
            os.makedirs(download_dir, exist_ok=True)
            
            # 保存视频
            content = await response.read()
            with open(filepath, 'wb') as f:
                f.write(content)
            
            return {
                'url': video_url,
                'filepath': filepath,
                'size': len(content),
                'filename': filename
            }
        
        except Exception as e:
            self.logger.error(f"下载视频失败 {video_url}: {e}")
            return None
    
    def _find_video_urls_in_json(self, data: Any, depth: int = 0) -> List[str]:
        """在JSON数据中查找视频URL"""
        if depth > 5:
            return []
        
        video_urls = []
        video_keys = ['video_url', 'videoUrl', 'src', 'url', 'play_url', 'playUrl']
        
        if isinstance(data, dict):
            for key, value in data.items():
                if key in video_keys and isinstance(value, str) and ('mp4' in value or 'video' in value):
                    video_urls.append(value)
                elif isinstance(value, (dict, list)):
                    video_urls.extend(self._find_video_urls_in_json(value, depth + 1))
        
        elif isinstance(data, list):
            for item in data:
                video_urls.extend(self._find_video_urls_in_json(item, depth + 1))
        
        return video_urls


class DocumentParser(BaseParser):
    """文档解析器"""
    
    async def parse(self, url: str, html_content: str, crawler, headers: Dict, throttler, download_dir: str) -> List[Dict[str, Any]]:
        """解析文档数据"""
        documents = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 查找文档链接
            doc_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt']
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # 检查是否是文档链接
                if any(ext in href.lower() for ext in doc_extensions):
                    # 转换为绝对URL
                    if not href.startswith(('http://', 'https://')):
                        href = urljoin(url, href)
                    
                    # 下载文档
                    doc_info = await self._download_document(href, crawler, headers, throttler, download_dir)
                    if doc_info:
                        doc_info['title'] = clean_text(link.get_text())
                        documents.append(doc_info)
        
        except Exception as e:
            self.logger.error(f"解析文档失败: {e}")
        
        return documents
    
    async def _download_document(self, doc_url: str, crawler, headers: Dict, throttler, download_dir: str) -> Optional[Dict]:
        """下载单个文档"""
        try:
            await throttler.wait_async()
            response = await crawler.get(doc_url, headers=headers)
            
            if not response or response.status != 200:
                return None
            
            # 生成文件名
            filename = os.path.basename(urlparse(doc_url).path)
            if not filename:
                filename = f"document_{int(datetime.now().timestamp())}.pdf"
            
            filepath = os.path.join(download_dir, sanitize_filename(filename))
            os.makedirs(download_dir, exist_ok=True)
            
            # 保存文档
            content = await response.read()
            with open(filepath, 'wb') as f:
                f.write(content)
            
            return {
                'url': doc_url,
                'filepath': filepath,
                'size': len(content),
                'filename': filename
            }
        
        except Exception as e:
            self.logger.error(f"下载文档失败 {doc_url}: {e}")
            return None