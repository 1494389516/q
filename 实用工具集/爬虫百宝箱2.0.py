import requests
import re
import os
import time
import random
from bs4 import BeautifulSoup
from twocaptcha import TwoCaptcha
import shutil
from urllib.parse import urlparse, unquote, urljoin
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
import xlwt
import aiohttp
import aiofiles
from typing import Optional, Dict, List, Any, Union, TYPE_CHECKING
import asyncio
import random
import time
from datetime import datetime
import aiohttp
from urllib.parse import urlparse, urljoin
import logging
import json
import os
import re
import hashlib
import yaml
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import base64
import io
from PIL import Image
import pytesseract
import cssutils
import jsbeautifier
from PIL import Image, ImageFilter

class ResponseCopy:
    """存储响应的关键信息并提供类似aiohttp.ClientResponse的接口"""
    def __init__(self, status: int, headers: dict, url: str, content: bytes):
        self.status = status
        self.headers = headers
        self.url = url
        self._content = content
        self._text = None
        self.charset = None

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

    async def json(self, encoding: str = None, loads=None, content_type: str = 'application/json') -> dict:
        """返回响应内容的JSON解析结果"""
        text = await self.text(encoding=encoding)
        if loads is None:
            import json
            loads = json.loads
        return loads(text)



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

class CaptchaSolver:
    def __init__(self):
        self.tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd=self.tesseract_path
    def solve_image_captcha(self,image_data):
        try:
            if isinstance(image_data,bytes):
                img=Image.open(io.BytesIO(image_data))
            elif isinstance(image_data,str):
                if image_data.startswith('data:image'):
                    b64_data=image_data.split(',')[1]
                    img=Image.open(io.BytesIO(base64.b64decode(b64_data)))
                else:
                    img=Image.open(image_data)
            else:
                img=image_data
            img=img.convert('L')
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img=img.point(lambda x:0 if x<128 else 255,'1')
            result=pytesseract.image_to_string(img,config='--psm 7')
            return result.strip()
        except Exception as e:
            logger.error(f"验证码识别失败: {e}")
            return None
    def solve_slider_captcha(self,bg_image,slider_image):
        try:
            bg=Image.open(io.BytesIO(bg_image)) if isinstance(bg_image,bytes) else Image.open(bg_image)
            slider=Image.open(io.BytesIO(slider_image)) if isinstance(slider_image,bytes) else Image.open(slider_image)
            bg_data=bg.load()
            slider_data=slider.load()
            for x in range(bg.width):
                for y in range(bg.height):
                    if x < slider.width and y < slider.height:
                        if abs(sum(bg_data[x,y])-sum(slider_data[x,y])) > 60:
                            return x
            return None
        except Exception as e:
            logger.error(f"滑块验证码处理失败: {e}")
            return None

    def solve_with_external_service(self, service_name, api_key, page_url=None, site_key=None, image_data_bytes=None, image_base64=None, method='image'):
        """Solves CAPTCHA using a third-party service.

        Args:
            service_name (str): Name of the service (e.g., '2captcha').
            api_key (str): API key for the service.
            page_url (str, optional): URL of the page where CAPTCHA appears (for reCAPTCHA).
            site_key (str, optional): Site key for reCAPTCHA/hCaptcha.
            image_data_bytes (bytes, optional): Byte data of an image CAPTCHA.
            image_base64 (str, optional): Base64 encoded string of an image CAPTCHA.
            method (str): Type of CAPTCHA to solve ('image', 'recaptcha', etc.).

        Returns:
            str: The solved CAPTCHA text, or None if solving failed.
        """
        logger.info(f"Attempting to solve CAPTCHA using external service: {service_name} for method {method}")

        if service_name.lower() == '2captcha':
            try:
                import tempfile 
                import os       
            except ImportError:
                logger.error("The 'twocaptcha' library is not installed. Please install it: pip install twocaptcha-python")
                return None

            solver = TwoCaptcha(api_key)
            try:
                if method == 'image':
                    captcha_solution = None
                    if image_base64:
                        logger.info("Sending base64 image to 2Captcha...")
                        
                        response = solver.normal(image_base64)
                        captcha_solution = response['code']
                    elif image_data_bytes:
                        logger.info("Sending image bytes to 2Captcha (via temporary file)...")
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                            tmpfile.write(image_data_bytes)
                            temp_image_path = tmpfile.name
                        try:
                            response = solver.normal(temp_image_path)
                            captcha_solution = response['code']
                        finally:
                            os.remove(temp_image_path) 
                    else:
                        logger.error("2Captcha 'image' method called without image_data_bytes or image_base64.")
                        return None
                    logger.info(f"2Captcha solved image CAPTCHA: {captcha_solution}")
                    return captcha_solution

                elif method == 'recaptcha' and site_key and page_url:
                    logger.info(f"Sending reCAPTCHA (sitekey: {site_key}, page: {page_url}) to 2Captcha...")
                    response = solver.recaptcha(sitekey=site_key, url=page_url)
                    captcha_solution = response['code']
                    logger.info(f"2Captcha solved reCAPTCHA: {captcha_solution}")
                    return captcha_solution
                else:
                    logger.warning(f"Unsupported method '{method}' or missing data for 2Captcha.")
                    return None
            except Exception as e_2captcha:
                logger.error(f"Error using 2Captcha service: {e_2captcha}")
               
                return None

       

        else:
            logger.error(f"Unsupported or unknown CAPTCHA solving service configured: {service_name}")
            return None

class CSSDecryptor:
    def __init__(self):
        self.known_patterns={'rot13':lambda x:x.encode('rot13'),'base64':base64.b64decode}
    def decrypt_css(self,css_content):
        try:
            sheet=cssutils.parseString(css_content)
            for rule in sheet:
                if rule.type==rule.STYLE_RULE:
                    for prop in rule.style:
                        if any(pattern in prop.value for pattern in self.known_patterns):
                            for pattern,decoder in self.known_patterns.items():
                                if pattern in prop.value:
                                    decoded=decoder(prop.value.replace(pattern+'(','').rstrip(')'))
                                    prop.value=decoded
            return sheet.cssText.decode()
        except Exception as e:
            logger.error(f"CSS解密失败: {e}")
            return css_content
    def beautify_css(self,css_content):
        try:
            return jsbeautifier.beautify_css(css_content)
        except:
            return css_content

def save_comments_to_file(comments_data, source_url, source_type="general"):
    """保存评论数据到文件"""
    if not comments_data:
        return
        
    # 创建保存目录
    save_dir = "downloaded_comments"
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    domain = urlparse(source_url).netloc
    sanitized_domain = re.sub(r'[^a-zA-Z0-9_.-]', '', domain)
    filename = f"{source_type}_comments_{sanitized_domain}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    
    # 添加元数据
    data_to_save = {
        "source_url": source_url,
        "source_type": source_type,
        "crawl_time": datetime.now().isoformat(),
        "comment_count": len(comments_data),
        "comments": comments_data
    }
    
    # 保存为JSON文件
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"\n评论数据已保存到: {os.path.abspath(filepath)}")
    except Exception as e:
        print(f"保存评论数据失败: {e}")

def extract_api_endpoints(html_content, base_url):
    """从HTML内容中提取可能的API端点"""
    api_endpoints = set()
    
    # 提取可能的API URL
    api_patterns = [
        r'["\']((?:/api/|/v\d+/|/rest/|/graphql)[^"\']+)["\']',
        r'["\'](https?://[^"\']+(?:/api/|/v\d+/|/rest/|/graphql)[^"\']+)["\']',
        r'["\'](/[^"\']*?comment[^"\']*?)["\']',
        r'["\'](https?://[^"\']*?comment[^"\']*?)["\']'
    ]
    
    for pattern in api_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        for match in matches:
            if match.startswith('http'):
                api_endpoints.add(match)
            else:
                # 相对路径转绝对路径
                parsed_url = urlparse(base_url)
                base = f"{parsed_url.scheme}://{parsed_url.netloc}"
                if not match.startswith('/'):
                    match = '/' + match
                api_endpoints.add(base + match)
    
    return list(api_endpoints)

async def try_api_endpoint(api_url, headers_dict, throttler, async_crawler, proxy_manager):
    """尝试请求API端点并解析评论数据"""
    try:
        print(f"尝试请求API: {api_url}")
        await throttler.wait_async()
        
        api_headers = headers_dict.copy()
        api_headers.update({
            'Accept': 'application/json, text/plain, */*',
            'X-Requested-With': 'XMLHttpRequest'
        })
        
        response = await make_request_async(
            api_url,
            async_crawler=async_crawler,
            headers=api_headers,
            throttler=throttler,
            proxy_manager=proxy_manager,
            method='GET'
        )
        
        if not response:
            return False
            
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/json' in content_type:
            json_data = await response.json()
            
            # 检查是否包含评论数据
            if await parse_json_comments(json_data, api_url):
                return True
        else:
            text_content = await response.text()
            # 尝试从响应中提取JSON
            json_match = re.search(r'({[\s\S]*})', text_content)
            if json_match:
                try:
                    json_data = json.loads(json_match.group(1))
                    if await parse_json_comments(json_data, api_url):
                        return True
                except:
                    pass
    except Exception as e:
        print(f"请求API端点失败: {e}")
    
    return False

async def parse_json_comments(json_data, source_url):
    """解析JSON格式的评论数据"""
    # 常见的评论数据字段名
    comment_keys = ['comments', 'comment_list', 'commentList', 'data', 'list', 'items']
    user_keys = ['user', 'author', 'userInfo', 'user_info']
    content_keys = ['content', 'text', 'comment', 'body', 'message']
    time_keys = ['time', 'date', 'createTime', 'create_time', 'createdAt', 'created_at', 'publishTime']
    
    # 递归查找评论数据
    def find_comments(data, depth=0):
        if depth > 3:  # 限制递归深度
            return None
            
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # 检查列表中的对象是否可能是评论
            if any(key in data[0] for key in content_keys + user_keys):
                return data
                
        if isinstance(data, dict):
            # 直接检查常见的评论列表字段
            for key in comment_keys:
                if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                    return data[key]
            
            # 递归检查所有字典值
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    result = find_comments(value, depth + 1)
                    if result:
                        return result
        
        return None
    
    comments_list = find_comments(json_data)
    if not comments_list:
        return False
        
    print(f"\n在JSON数据中找到 {len(comments_list)} 条可能的评论:")
    
    comments_data = []
    for i, comment in enumerate(comments_list[:20]):  # 只处理前20条
        try:
            # 提取用户信息
            user_info = None
            for key in user_keys:
                if key in comment and comment[key]:
                    user_info = comment[key]
                    break
                    
            nickname = "未知用户"
            if user_info:
                if isinstance(user_info, dict):
                    for name_key in ['nickname', 'name', 'userName', 'user_name', 'username']:
                        if name_key in user_info:
                            nickname = user_info[name_key]
                            break
                else:
                    nickname = str(user_info)
            
            # 提取评论内容
            content = None
            for key in content_keys:
                if key in comment and comment[key]:
                    content = comment[key]
                    break
            
            if not content:
                continue
                
            # 提取时间
            time_str = ""
            for key in time_keys:
                if key in comment and comment[key]:
                    time_val = comment[key]
                    if isinstance(time_val, int) and time_val > 1000000000:
                        # 可能是时间戳
                        time_str = datetime.fromtimestamp(time_val / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        time_str = str(time_val)
                    break
            
            comment_data = {
                "user": nickname,
                "content": content,
                "time": time_str
            }
            
            comments_data.append(comment_data)
            
            if i < 10:  # 只显示前10条
                print(f"--- 评论 {i+1} ---")
                print(f"用户: {nickname}")
                print(f"内容: {content}")
                if time_str:
                    print(f"时间: {time_str}")
                print("--------------------")
        except Exception as e:
            print(f"解析评论项失败: {e}")
    
    # 保存评论数据
    if comments_data:
        save_comments_to_file(comments_data, source_url, "json_api")
        return True
    
    return False

async def crawl_xiaohongshu_comments(
    target_url: str, 
    headers_dict: Dict, 
    throttler: 'AsyncRequestThrottler', 
    async_crawler: 'AsyncCrawler', 
    proxy_manager: Optional['AsyncProxyManager']
):
    """专门针对小红书评论的爬取函数"""
    print("\n=== 小红书评论爬取 ===")
    
    # 提取笔记ID
    note_id = None
    note_id_patterns = [
        r'item/(\w+)', 
        r'discovery/item/(\w+)',
        r'note/(\w+)',
        r'xhslink\.com/\w+',  # 短链接需要跟踪重定向
    ]
    
    for pattern in note_id_patterns:
        match = re.search(pattern, target_url)
        if match and pattern != r'xhslink\.com/\w+':
            note_id = match.group(1)
            break
    
    # 处理短链接
    if not note_id and "xhslink.com" in target_url:
        print("检测到小红书短链接，尝试获取重定向后的URL...")
        try:
            # 设置不跟随重定向，以便获取重定向URL
            redirect_headers = headers_dict.copy()
            redirect_response = await make_request_async(
                target_url,
                async_crawler=async_crawler,
                headers=redirect_headers,
                throttler=throttler,
                proxy_manager=proxy_manager,
                method='GET'
            )
            
            if redirect_response and redirect_response.status in (301, 302):
                redirect_url = redirect_response.headers.get('Location')
                if redirect_url:
                    print(f"获取到重定向URL: {redirect_url}")
                    for pattern in note_id_patterns:
                        match = re.search(pattern, redirect_url)
                        if match and pattern != r'xhslink\.com/\w+':
                            note_id = match.group(1)
                            break
        except Exception as e:
            print(f"处理重定向失败: {e}")
    
    if not note_id:
        print("无法从URL中提取小红书笔记ID，尝试从页面内容中提取...")
        
        try:
            response = await make_request_async(
                target_url,
                async_crawler=async_crawler,
                headers=headers_dict,
                throttler=throttler,
                proxy_manager=proxy_manager,
                method='GET'
            )
            
            if not response:
                print("请求页面失败，无法提取笔记ID")
                return
                
            html_content = await response.text(encoding='utf-8', errors='replace')
            
            # 尝试从HTML中提取笔记ID
            id_match = re.search(r'"noteId":\s*"(\w+)"', html_content)
            if id_match:
                note_id = id_match.group(1)
            else:
                id_match = re.search(r'data-note-id="(\w+)"', html_content)
                if id_match:
                    note_id = id_match.group(1)
        except Exception as e:
            print(f"从页面提取笔记ID失败: {e}")
    
    if not note_id:
        print("无法获取小红书笔记ID，无法爬取评论")
        return
        
    print(f"成功获取小红书笔记ID: {note_id}")
    
    # 构建评论API请求
    api_url = f"https://www.xiaohongshu.com/fe_api/burdock/weixin/v2/note/{note_id}/comment?pageSize=30&sort=time"
    
    # 设置必要的请求头
    api_headers = headers_dict.copy()
    api_headers.update({
        'Referer': f'https://www.xiaohongshu.com/discovery/item/{note_id}',
        'Accept': 'application/json, text/plain, */*',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://www.xiaohongshu.com'
    })
    
    print(f"请求评论API: {api_url}")
    
    try:
        await throttler.wait_async()
        
        response = await make_request_async(
            api_url,
            async_crawler=async_crawler,
            headers=api_headers,
            throttler=throttler,
            proxy_manager=proxy_manager,
            method='GET'
        )
        
        if not response:
            print("请求评论API失败")
            
            # 尝试备用API
            backup_api_url = f"https://www.xiaohongshu.com/api/sns/web/v1/comment/page?note_id={note_id}&cursor=&top_comment_id="
            print(f"尝试备用API: {backup_api_url}")
            
            backup_response = await make_request_async(
                backup_api_url,
                async_crawler=async_crawler,
                headers=api_headers,
                throttler=throttler,
                proxy_manager=proxy_manager,
                method='GET'
            )
            
            if not backup_response:
                print("备用API也请求失败，尝试模拟移动端请求...")
                
                # 模拟移动端请求
                mobile_headers = {
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/8.0.16(0x18001042) NetType/WIFI Language/zh_CN',
                    'Accept': 'application/json',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                    'Referer': f'https://www.xiaohongshu.com/discovery/item/{note_id}',
                    'Origin': 'https://www.xiaohongshu.com'
                }
                
                mobile_api_url = f"https://edith.xiaohongshu.com/api/sns/web/v1/comment/page?note_id={note_id}&cursor=&top_comment_id="
                
                mobile_response = await make_request_async(
                    mobile_api_url,
                    async_crawler=async_crawler,
                    headers=mobile_headers,
                    throttler=throttler,
                    proxy_manager=proxy_manager,
                    method='GET'
                )
                
                if not mobile_response:
                    print("所有API请求均失败，无法获取评论数据")
                    return
                    
                response = mobile_response
            else:
                response = backup_response
        
        # 解析评论数据
        try:
            json_data = await response.json()
            
            # 检查API响应格式
            if 'data' in json_data and ('comments' in json_data['data'] or 'comment_list' in json_data['data'] or 'comments_list' in json_data['data']):
                comments_list = json_data['data'].get('comments') or json_data['data'].get('comment_list') or json_data['data'].get('comments_list') or []
                
                if not comments_list:
                    print("API返回的评论列表为空")
                    return
                    
                print(f"\n成功获取 {len(comments_list)} 条评论:")
                
                comments_data = []
                for i, comment in enumerate(comments_list):
                    try:
                        user_info = comment.get('user_info') or comment.get('user') or {}
                        nickname = user_info.get('nickname') or user_info.get('user_name') or "未知用户"
                        content = comment.get('content') or comment.get('comment_text') or "无内容"
                        time_str = comment.get('create_time') or comment.get('time') or ""
                        
                        if isinstance(time_str, int):
                            # 转换时间戳为可读时间
                            time_str = datetime.fromtimestamp(time_str / 1000).strftime('%Y-%m-%d %H:%M:%S')
                            
                        likes = comment.get('like_count') or comment.get('likes') or 0
                        
                        comment_data = {
                            "user": nickname,
                            "content": content,
                            "time": time_str,
                            "likes": likes
                        }
                        
                        comments_data.append(comment_data)
                        
                        if i < 10:  # 只显示前10条
                            print(f"--- 评论 {i+1} ---")
                            print(f"用户: {nickname}")
                            print(f"内容: {content}")
                            print(f"时间: {time_str}")
                            print(f"点赞: {likes}")
                            print("--------------------")
                    except Exception as e:
                        print(f"解析评论项失败: {e}")
                
                # 保存评论数据
                if comments_data:
                    save_comments_to_file(comments_data, target_url, "xiaohongshu")
            else:
                print("API响应格式不符合预期:")
                print(json.dumps(json_data, ensure_ascii=False, indent=2)[:500] + "...")  # 只显示前500个字符
        except Exception as e:
            print(f"解析评论JSON数据失败: {e}")
            content = await response.text()
            print(f"API响应内容: {content[:500]}...")  # 只显示前500个字符
    except Exception as e:
        print(f"请求小红书评论数据失败: {e}")
        traceback.print_exc()

async def try_fetch_json_comments(
    target_url: str, 
    headers_dict: Dict, 
    throttler: 'AsyncRequestThrottler', 
    async_crawler: 'AsyncCrawler', 
    proxy_manager: Optional['AsyncProxyManager']
):
    """尝试获取JSON格式的评论数据"""
    # 常见的评论API路径模式
    api_patterns = [
        "/api/comments",
        "/api/comment/list",
        "/api/v1/comments",
        "/comments/api",
        "/api/sns/web/v1/comment/page",
        "/fe_api/burdock/weixin/v2/note/{id}/comment"
    ]
    
    # 从URL中提取可能的ID
    id_match = re.search(r'/(\w+)(?:/|$)', target_url)
    possible_id = id_match.group(1) if id_match else ""
    
    parsed_url = urlparse(target_url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    for pattern in api_patterns:
        api_url = base_url + pattern.replace("{id}", possible_id)
        
        try:
            print(f"尝试请求可能的评论API: {api_url}")
            await throttler.wait_async()
            
            api_headers = headers_dict.copy()
            api_headers['Accept'] = 'application/json'
            
            response = await make_request_async(
                api_url,
                async_crawler=async_crawler,
                headers=api_headers,
                throttler=throttler,
                proxy_manager=proxy_manager,
                method='GET'
            )
            
            if not response:
                continue
                
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/json' in content_type:
                json_data = await response.json()
                
                # 检查是否包含评论数据
                if await parse_json_comments(json_data, target_url):
                    return True
        except Exception:
            continue
    return False
# 定义基础Spider类
class Spider:
    """基础爬虫类，用于继承"""
    name = 'base_spider'
    allowed_domains = []
    
    def __init__(self, *args, **kwargs):
        self.start_urls = []
    
    def start_requests(self):
        """开始请求"""
        pass
    
    def parse(self, response):
        """解析响应"""
        pass

# 请求类，模拟requests库的Request
class Request:
    """请求类，用于创建请求"""
    def __init__(self, url, headers=None, meta=None):
        self.url = url
        self.headers = headers or {}
        self.meta = meta or {}

class XiaohongshuCommentSpider(Spider):
    name = 'xiaohongshu_comment'
    allowed_domains = ['xiaohongshu.com']

    def __init__(self, start_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.config = load_config()
        self.history = CrawlHistory()

    def start_requests(self):
        for url in self.start_urls:
            yield Request(url, headers={'User-Agent': random.choice(self.config['user_agents'])},
                          meta={'proxy': self.config['proxy_settings']['proxy_list'][0] if self.config['proxy_settings']['enabled'] else None})

    def parse(self, response):
        # 解析小红书评论页面
        soup = BeautifulSoup(response.text, 'html.parser')
        comments = soup.select('div.comment-item')
        for comment in comments:
            comment_text = comment.select_one('.comment-text').get_text(strip=True)
            comment_author = comment.select_one('.author-name').get_text(strip=True)
            comment_data = {
                'url': response.url,
                'author': comment_author,
                'text': comment_text,
                'timestamp': datetime.now().isoformat()
            }
            # 记录到历史数据库
            content_hash = hashlib.md5(json.dumps(comment_data).encode()).hexdigest()
            self.history.add_content(content_hash, 'comment', None, json.dumps(comment_data))
            yield comment_data

class AsyncCrawler:
    """
    异步爬虫核心类，处理网络请求、验证码和CSS解密
    
    Attributes:
        timeout (aiohttp.ClientTimeout): 请求超时设置
        connector (aiohttp.TCPConnector): TCP连接管理器
        retries (int): 请求重试次数
        session (Optional[aiohttp.ClientSession]): 异步HTTP会话
        proxy_manager (Optional[AsyncProxyManager]): 代理管理器
        captcha_solver (CaptchaSolver): 验证码解决器
        css_decryptor (CSSDecryptor): CSS解密器
    """
    def __init__(self, max_connections: int = 100, timeout: int = 30, retries: int = 3) -> None:
        self.timeout: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=timeout)
        self.connector: aiohttp.TCPConnector = aiohttp.TCPConnector(
            limit=max_connections,
            force_close=True,
            enable_cleanup_closed=True
        )
        self.retries: int = retries
        self.session: Optional[aiohttp.ClientSession] = None
        self.proxy_manager: Optional[AsyncProxyManager] = None
        self.captcha_solver: CaptchaSolver = CaptchaSolver()
        self.css_decryptor: CSSDecryptor = CSSDecryptor()

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
                    data: Any = None, json: Any = None) -> Optional[ResponseCopy]:
        """
        异步获取URL内容，返回响应对象
        
        Args:
            url (str): 要请求的URL
            method (str): HTTP方法，默认为'GET'
            headers (Optional[Dict[str, str]]): 请求头
            proxy (Optional[str]): 代理URL
            params (Optional[Dict[str, Any]]): URL参数
            data (Any): 请求体数据
            json (Any): JSON格式的请求体数据
            
        Returns:
            Optional[ResponseCopy]: 响应对象的副本，如果请求失败则返回None
            
        Raises:
            aiohttp.ClientError: 当HTTP请求失败时
            asyncio.TimeoutError: 当请求超时时
        """
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
                    
                    # 创建一个响应副本，包含已经读取的内容
                    response_copy = ResponseCopy(
                        status=response.status,
                        headers=response.headers.copy(),
                        url=response.url,
                        content=await response.read()
                    )
                    return response_copy
                    
                    # 检测是否存在验证码
                    content_type = response.headers.get('Content-Type', '')
                    if response.status == 403 or 'captcha' in url.lower() or 'verify' in url.lower():
                        logger.info(f"检测到可能的验证码页面: {url}")
                        # 获取响应内容
                        content = await response.read()
                        if 'image' in content_type:
                            captcha_result = self.captcha_solver.solve_image_captcha(content)
                            if captcha_result:
                                logger.info(f"成功识别图片验证码: {captcha_result}")
                                # 使用识别结果重新提交请求
                                if data is None:
                                    data = {}
                                if isinstance(data, dict):
                                    data['captcha'] = captcha_result
                                    continue 
                            else: # Local image solving failed
                                logger.warning(f"本地识别图片验证码失败: {url}")
                                current_config = load_config()
                                service_config = current_config.get('captcha_solving_services', {})
                                if service_config.get('enabled'):
                                    api_key = service_config.get('api_key')
                                    service_name = service_config.get('service_name')
                                    if api_key and service_name:
                                        # 'content' here is the image data bytes
                                        external_captcha_result = self.captcha_solver.solve_with_external_service(
                                            service_name, api_key, image_data_bytes=content, method='image', page_url=url
                                        )
                                        if external_captcha_result:
                                            logger.info(f"外部服务 {service_name} 解决验证码: {external_captcha_result}")
                                            if data is None: data = {}
                                            if isinstance(data, dict):
                                                data['captcha'] = external_captcha_result
                                                continue # 重试请求
                                        else:
                                            logger.error(f"外部服务 {service_name} 未能解决验证码 for {url}")
                                    else:
                                        logger.warning("外部验证码服务已启用，但API密钥或服务名称在配置中缺失。")
                        if 'text/html' in content_type:
                            html_content = content.decode('utf-8', errors='ignore')
                            if 'captcha' in html_content.lower():
                                soup = BeautifulSoup(html_content, 'html.parser')
                                captcha_img = soup.find('img', {'id': lambda x: x and 'captcha' in x.lower()})
                                if captcha_img and captcha_img.get('src'):
                                    img_url = captcha_img['src']
                                    if img_url.startswith('data:image'):
                                        captcha_result = self.captcha_solver.solve_image_captcha(img_url)
                                    else:
                                        if not img_url.startswith(('http://', 'https://')):
                                            base_url = '/'.join(url.split('/')[:3])
                                            img_url = urljoin(base_url, img_url)
                                        async with self.session.get(img_url, headers=headers) as img_response:
                                            img_content = await img_response.read()
                                            captcha_result = self.captcha_solver.solve_image_captcha(img_content)
                                    
                                    if captcha_result:
                                        logger.info(f"成功识别HTML中的验证码: {captcha_result}")
                                        
                                        if data is None:
                                            data = {}
                                        if isinstance(data, dict):
                                            data['captcha'] = captcha_result
                                            continue  
                                    else: 
                                        logger.warning(f"本地识别HTML中的验证码失败: {url}")
                                        current_config = load_config()
                                        service_config = current_config.get('captcha_solving_services', {})
                                        if service_config.get('enabled'):
                                            api_key = service_config.get('api_key')
                                            service_name = service_config.get('service_name')
                                            if api_key and service_name:
                                                
                                                external_captcha_result = self.captcha_solver.solve_with_external_service(
                                                    service_name, api_key, image_data_bytes=img_content, method='image', page_url=url
                                                )
                                                if external_captcha_result:
                                                    logger.info(f"外部服务 {service_name} 解决验证码: {external_captcha_result}")
                                                    if data is None: data = {}
                                                    if isinstance(data, dict):
                                                        data['captcha'] = external_captcha_result
                                                        continue # 
                                                else:
                                                    logger.error(f"外部服务 {service_name} 未能解决验证码 for {url}")
                                            else:
                                                logger.warning("外部验证码服务已启用，但API密钥或服务名称在配置中缺失。")
                    
                    response.raise_for_status()
                    return response
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"异步请求尝试 {attempt + 1}/{self.retries} 失败: {url}, 错误: {str(e)}")
                if attempt == self.retries - 1:
                    return None
                await asyncio.sleep(0.5 * (2 ** attempt))
        return None
        
    async def decrypt_css(self, css_content: str) -> str:
        """解密CSS内容"""
        return self.css_decryptor.decrypt_css(css_content)
    
    async def beautify_css(self, css_content: str) -> str:
        """美化CSS内容"""
        return self.css_decryptor.beautify_css(css_content)

class AsyncProxyManager:
    """
    代理管理器，处理代理IP的获取和轮换 (异步版本)
    
    负责管理代理IP池，包括从列表或API获取代理、验证代理可用性和轮换使用代理。
    支持异步操作，适用于高并发爬虫场景。
    
    Attributes:
        proxies (List[str]): 代理列表
        proxy_api_url (Optional[str]): 代理API的URL
        api_key (Optional[str]): 代理API的密钥
        current_index (int): 当前使用的代理索引
        lock (asyncio.Lock): 异步锁，用于线程安全操作
        async_crawler (AsyncCrawler): 异步爬虫实例，用于请求代理API
    """
    def __init__(self, async_crawler_instance: AsyncCrawler, proxy_list: Optional[List[str]] = None, 
                 proxy_api_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.proxies: List[str] = proxy_list or []
        self.proxy_api_url: Optional[str] = proxy_api_url
        self.api_key: Optional[str] = api_key
        self.current_index: int = 0
        self.lock: asyncio.Lock = asyncio.Lock()
        self.async_crawler: AsyncCrawler = async_crawler_instance
        
    async def get_proxy(self) -> Optional[str]:
        """
        获取当前代理 (异步)
        
        如果代理列表为空且设置了代理API，会尝试刷新代理列表。
        使用轮询方式返回代理列表中的代理。
        
        Returns:
            Optional[str]: 代理URL字符串，如果没有可用代理则返回None
        """
        async with self.lock:
            if not self.proxies and self.proxy_api_url:
                await self._refresh_proxies_async()
            if not self.proxies:
                return None
            proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)
            return proxy
            
    async def _refresh_proxies_async(self) -> None:
        """
        从API获取新代理 (异步)
        
        通过配置的API URL获取新的代理列表，并更新内部代理池。
        如果API请求失败或返回格式不正确，会记录错误但不会抛出异常。
        
        Returns:
            None
        """
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
        ],
        'captcha_solving_services': {
            'enabled': False,
            'service_name': '', 
            'api_key': '',
            
            'recaptcha_site_key_selectors': ['div.g-recaptcha', 'div.h-captcha'], 
            'recaptcha_site_key_attribute': 'data-sitekey'
        }
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
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_history (
                content_hash TEXT PRIMARY KEY,
                content_type TEXT,
                file_path TEXT,
                metadata TEXT,
                timestamp TEXT
            )
            ''')
        
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
    """异步请求限流器，控制请求频率和突发请求"""
    def __init__(self, min_interval: float = 2.0, max_interval: float = 5.0, 
                 burst_requests: int = 3, burst_period: float = 60.0) -> None:
        self.min_interval: float = min_interval
        self.max_interval: float = max_interval
        self.burst_requests: int = burst_requests
        self.burst_period: float = burst_period
        self.last_request_time: float = 0.0
        self.request_history: List[float] = []
        self.lock: asyncio.Lock = asyncio.Lock()

    async def wait_async(self) -> None:
        """
        异步等待函数，实现请求频率和突发请求的控制
        
        实现两种限流机制：
        1. 突发请求控制：在burst_period时间窗口内限制最大请求数为burst_requests
        2. 间隔控制：确保请求之间的最小间隔，并可选添加随机延迟
        
        Returns:
            None
        """
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
    自动处理验证码和CSS内容。
    """
    effective_headers = headers.copy() if headers else {}
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path or "/"
    if 'User-Agent' not in effective_headers and config.user_agents:
        effective_headers['User-Agent'] = random.choice(config.user_agents)
    elif 'User-Agent' not in effective_headers:
        effective_headers['User-Agent'] = "PythonAsyncCrawler/1.0"
    
    # 检查robots.txt
    robots_parser: Optional[RobotsTxtParser] = None
    async with g_robots_parsers_lock:
        if domain in g_robots_parsers:
            robots_parser = g_robots_parsers[domain]
        else:
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
            logger.info(f"Fetching robots.txt for {domain} from {robots_url}")
            try:
                robots_headers = {'User-Agent': effective_headers['User-Agent']}
                try:
                    robots_response = await async_crawler.fetch(robots_url, headers=robots_headers, method='GET')
                    if robots_response and robots_response.status == 200:
                        try:
                            content = await robots_response.text()
                            new_parser = RobotsTxtParser(domain)
                            new_parser.parse(content)
                            g_robots_parsers[domain] = new_parser
                            robots_parser = new_parser
                            logger.info(f"Successfully fetched and parsed robots.txt for {domain}")
                        except Exception as e_text:
                            logger.warning(f"Error reading robots.txt content for {domain}: {e_text}. Assuming allowed.")
                            g_robots_parsers[domain] = RobotsTxtParser(domain)
                            robots_parser = g_robots_parsers[domain]
                    else:
                        logger.warning(f"Failed to fetch robots.txt for {domain}. Assuming allowed.")
                        g_robots_parsers[domain] = RobotsTxtParser(domain)
                        robots_parser = g_robots_parsers[domain]
                except Exception as e_fetch:
                    logger.warning(f"Error fetching robots.txt for {domain}: {e_fetch}. Assuming allowed.")
                    g_robots_parsers[domain] = RobotsTxtParser(domain)
                    robots_parser = g_robots_parsers[domain]
            except Exception as e_robots:
                logger.error(f"Error fetching/parsing robots.txt for {domain}: {e_robots}. Assuming allowed.")
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

    # 发起请求并处理验证码
    response = await async_crawler.fetch(
        url,
        method=method,
        headers=effective_headers,
        proxy=proxy_url,
        params=params,
        data=data,
        json=json_payload
    )

    if response:
        content_type = response.headers.get('Content-Type', '').lower()
        
        # 自动处理验证码
        if response.status == 403 or 'captcha' in url.lower():
            logger.info(f"检测到可能需要验证码，尝试自动处理: {url}")
            try:
                captcha_response = await bypass_captcha(
                    url,
                    effective_headers,
                    throttler,
                    async_crawler,
                    proxy_manager
                )
                if captcha_response:
                    response = captcha_response
            except Exception as e:
                logger.error(f"验证码处理失败: {e}")

        # 自动处理CSS内容
        if 'text/css' in content_type or url.lower().endswith('.css'):
            logger.info(f"检测到CSS内容，尝试解密: {url}")
            try:
                css_content = await response.text()
                decrypted_css = await async_crawler.decrypt_css(css_content)
                beautified_css = await async_crawler.beautify_css(decrypted_css)
                
                # 创建一个新的响应对象，包含解密后的CSS
                new_response = aiohttp.ClientResponse(
                    response.method,
                    response.url,
                    writer=None,
                    continue100=None,
                    timer=None,
                    request_info=response.request_info,
                    traces=None,
                    loop=response._loop,
                    session=response._session
                )
                new_response.status = response.status
                new_response.headers = response.headers.copy()
                new_response._body = beautified_css.encode('utf-8')
                response = new_response
                
            except Exception as e:
                logger.error(f"CSS解密失败: {e}")
    
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

async def parse_bilibili_video_async(bvid: str, referer: str, headers: Dict, throttler: AsyncRequestThrottler, async_crawler: AsyncCrawler, proxy_manager: Optional[AsyncProxyManager] = None) -> Optional[str]:
    """异步解析B站视频"""
    try:
        api_url = f"https://api.bilibili.com/x/player/playurl?bvid={bvid}&qn=116&fnval=16"
        api_headers = {
            **headers,
            'Referer': referer,
            'Origin': 'https://www.bilibili.com',
            'Accept': 'application/json, text/plain, */*'
        }

        await throttler.wait_async()
        response = await make_request_async(
            api_url,
            async_crawler=async_crawler,
            headers=api_headers,
            throttler=throttler,
            proxy_manager=proxy_manager,
            method='GET'
        )

        if not response:
            print(f"B站API请求失败: {api_url}")
            return None

        data = await response.json()
        if data['code'] != 0:
            print(f"B站API错误: {data.get('message')}")
            return None

        return await download_bilibili_m3u8_async(data, referer, headers, throttler, async_crawler)

    except aiohttp.ClientError as e:
        print(f"B站API请求失败: {e}")
    except Exception as e:
        print(f"B站解析失败: {e}")
        traceback.print_exc()
    return None

async def download_bilibili_m3u8_async(data: Dict, referer: str, headers: Dict, throttler: AsyncRequestThrottler, async_crawler: AsyncCrawler) -> Optional[str]:
    """异步下载B站视频"""
    download_dir = "downloaded_videos"
    os.makedirs(download_dir, exist_ok=True)
    temp_dir = f"temp_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        video_info = data['data']['dash']['video'][0]
        audio_info = data['data']['dash']['audio'][0]
        print(f"发现视频流: {video_info['id']} | 分辨率: {video_info['width']}x{video_info['height']}")

        video_url = video_info['baseUrl']
        audio_url = audio_info['baseUrl']

        # 设置请求头
        stream_headers = headers.copy()
        stream_headers['Referer'] = referer

        # 异步下载视频和音频流
        async def download_stream(url: str, file_path: str, stream_type: str) -> bool:
            """异步下载单个媒体流"""
            try:
                print(f"开始下载{stream_type}流...")
                await throttler.wait_async()

                async with async_crawler.session.get(
                    url,
                    headers=stream_headers,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5分钟超时
                ) as response:
                    if response.status != 200:
                        print(f"{stream_type}流下载失败 ({response.status}): {url}")
                        return False

                    async with aiofiles.open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            if chunk:
                                await f.write(chunk)

                    print(f"{stream_type}流下载完成")
                    return True

            except Exception as e:
                print(f"{stream_type}流下载失败: {e}")
                return False

        # 并发下载视频和音频
        video_path = os.path.join(temp_dir, 'video.mp4')
        audio_path = os.path.join(temp_dir, 'audio.mp4')

        video_task = download_stream(video_url, video_path, '视频')
        audio_task = download_stream(audio_url, audio_path, '音频')

        video_result, audio_result = await asyncio.gather(video_task, audio_task)

        if video_result and audio_result:
            output_file = os.path.join(download_dir, f"bilibili_{int(time.time())}.mp4")

            # 使用FFmpeg合并视频和音频（如果可用）
            try:
                print("开始合并音视频...")
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    output_file
                ]

                # 异步执行FFmpeg命令
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    print(f"视频已成功合并并保存至: {output_file}")
                    return output_file
                else:
                    print(f"FFmpeg合并失败: {stderr.decode()}")
                    # 如果FFmpeg失败，只保存视频流
                    import shutil
                    shutil.copy(video_path, output_file)
                    print(f"仅保存视频流至: {output_file}")
                    return output_file

            except (ImportError, FileNotFoundError):
                # 如果没有FFmpeg，只保存视频流
                import shutil
                shutil.copy(video_path, output_file)
                print(f"未找到FFmpeg，仅保存视频流至: {output_file}")
                return output_file
        else:
            print("一个或多个B站媒体流下载失败，无法合并。")

    except Exception as e:
        print(f"下载B站视频时发生错误: {e}")
        traceback.print_exc()
    finally:
        # 清理临时文件
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return None


async def crawl_images_revised_async(page_url, html_content_for_images, headers_dict, throttler, async_crawler, recursive=True, max_depth=None, proxy_manager=None, retry=None):
    """
    异步爬取图片链接，支持并发下载和递归爬取
    参数:
        page_url: 目标页面URL
        html_content_for_images: 页面HTML内容
        headers_dict: 请求头
        throttler: 异步请求限流器
        async_crawler: 异步爬虫实例
        recursive: 是否递归爬取 (默认True)
        max_depth: 最大递归深度 (默认None)
        proxy_manager: 代理管理器实例 (可选)
        retry: 重试机制实例 (可选)
    """
    if max_depth is None:
        max_depth = config.max_depth
        
    if retry is None:
        retry = SmartRetry()

    download_dir = os.path.join("downloads", "images")
    os.makedirs(download_dir, exist_ok=True)
    
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0
    }

    link_extractor = LinkExtractor()
    processed_urls = set()

    def process_url(url):
        """处理URL，确保使用HTTPS"""
        if url.startswith("http://"):
            return "https://" + url[len("http://"):]
        elif not url.startswith("https://") and "://" not in url:
            return "https://" + url
        return url

    async def download_image(img_url, referer_url):
        """异步下载单个图片"""
        try:
            url_hash = hashlib.md5(img_url.encode()).hexdigest()
            if history.url_exists(img_url):
                logger.info(f"图片已下载过: {img_url}")
                stats['skipped'] += 1
                return None

            local_headers = headers_dict.copy()
            local_headers['Referer'] = referer_url

            response = await make_request_async(
                img_url,
                async_crawler=async_crawler,
                headers=local_headers,
                throttler=throttler,
                proxy_manager=proxy_manager,
                method='GET'
            )
            
            if not response:
                logger.error(f"请求图片失败: {img_url}")
                stats['failed'] += 1
                return None

            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"非图片内容类型: {content_type}, URL: {img_url}")
                return None

            parsed_url = urlparse(img_url)
            base_name = os.path.splitext(os.path.basename(parsed_url.path))[0]
            if not base_name or len(base_name) < 3:
                base_name = url_hash[:8]
                
            ext = content_type.split('/')[-1].lower()
            if ext == 'jpeg':
                ext = 'jpg'
                
            filename = f"{base_name[:50]}_{int(time.time())}_{url_hash[:6]}.{ext}"
            file_path = os.path.join(download_dir, filename)

            # 异步保存图片
            content = await response.read()
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)

            # 异步保存元数据
            metadata = {
                'url': img_url,
                'referer': referer_url,
                'content_type': content_type,
                'size': len(content),
                'download_time': datetime.now().isoformat()
            }
            
            meta_path = f"{file_path}.meta.json"
            async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))

            history.add_url(img_url, "completed", url_hash, 0, referer_url)
            history.add_content(url_hash, content_type, file_path, metadata)
            
            logger.info(f"成功下载图片: {img_url} -> {file_path}")
            stats['success'] += 1
            return file_path

        except Exception as e:
            logger.error(f"下载图片失败 {img_url}: {e}")
            history.add_url(img_url, f"failed: {str(e)}", None, 0, referer_url)
            stats['failed'] += 1
            return None

    async def process_page(url, html_content, depth=0):
        """异步处理单个页面，提取图片和链接"""
        base_url = process_url(url)
        
        # 使用BeautifulSoup提取图片URL
        soup = BeautifulSoup(html_content, 'html.parser')
        img_tasks = []
        
        # 处理<img>标签
        for img_tag in soup.find_all('img'):
            img_url = img_tag.get('src')
            if not img_url or img_url.startswith('data:'):
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
                img_tasks.append(download_image(final_url, base_url))
                stats['total'] += 1

        # 处理CSS背景图片
        for tag in soup.find_all(['div', 'span', 'a'], style=True):
            style = tag.get('style', '')
            if 'background-image' in style:
                url_match = re.search(r'url\([\'"]?(.*?)[\'"]?\)', style)
                if url_match:
                    img_url = url_match.group(1)
                    if not img_url.startswith('data:'):
                        final_url = process_url(urljoin(base_url, img_url))
                        if final_url not in processed_urls and final_url.startswith("https://"):
                            processed_urls.add(final_url)
                            img_tasks.append(download_image(final_url, base_url))
                            stats['total'] += 1

        # 并发下载图片
        if img_tasks:
            await asyncio.gather(*img_tasks)

        # 递归处理链接
        if recursive and depth < max_depth:
            links = link_extractor.extract_links(html_content, base_url)
            filtered_links = link_extractor.filter_links(links, base_url)
            link_tasks = []
            
            for link in filtered_links:
                if link not in link_extractor.visited_urls:
                    link_extractor.visited_urls.add(link)
                    link_tasks.append(async_crawler.fetch(
                        link,
                        headers=headers_dict,
                        proxy=proxy_manager.get_proxy() if proxy_manager else None
                    ))

            if link_tasks:
                responses = await asyncio.gather(*link_tasks, return_exceptions=True)
                for link, response in zip(filtered_links, responses):
                    if isinstance(response, Exception):
                        logger.error(f"获取链接失败 {link}: {response}")
                        continue
                    if response and response.status == 200:
                        try:
                            link_html_content = await response.text()
                            await process_page(link, link_html_content, depth + 1)
                        except Exception as e:
                            logger.error(f"处理链接内容失败 {link}: {e}")

    # 开始处理主页面
    try:
        await process_page(page_url, html_content_for_images, 0)
    except Exception as e:
        logger.error(f"处理主页面失败: {e}")
        traceback.print_exc()

    logger.info(f"图片爬取任务完成 - 总计: {stats['total']}, 成功: {stats['success']}, "
                f"失败: {stats['failed']}, 跳过: {stats['skipped']}")
    return stats

async def crawl_bilibili_comments(target_url: str, headers_dict: Dict, throttler: 'AsyncRequestThrottler', async_crawler: 'AsyncCrawler', proxy_manager: Optional['AsyncProxyManager']):
    """专门针对B站评论的爬取函数"""
    print("\n=== B站评论爬取 ===")
    
    # 提取视频ID
    bvid = None
    aid = None
    
    # 尝试从URL中提取BV号或AV号
    bvid_match = re.search(r'(?:bvid=|video/)(BV[a-zA-Z0-9]+)', target_url, re.IGNORECASE)
    aid_match = re.search(r'(?:aid=|av)(\d+)', target_url, re.IGNORECASE)
    
    if bvid_match:
        bvid = bvid_match.group(1)
        print(f"提取到B站视频BV号: {bvid}")
    elif aid_match:
        aid = aid_match.group(1)
        print(f"提取到B站视频AV号: {aid}")
    
    # 处理短链接
    if not bvid and not aid and "b23.tv" in target_url:
        print("检测到B站短链接，尝试获取重定向后的URL...")
        try:
            # 设置不跟随重定向，以便获取重定向URL
            redirect_headers = headers_dict.copy()
            response = await make_request_async(
                target_url,
                async_crawler=async_crawler,
                headers=redirect_headers,
                throttler=throttler,
                proxy_manager=proxy_manager,
                method='GET'
            )
            
            if response and response.status in (301, 302):
                redirect_url = response.headers.get('Location')
                if redirect_url:
                    print(f"获取到重定向URL: {redirect_url}")
                    bvid_match = re.search(r'(?:bvid=|video/)(BV[a-zA-Z0-9]+)', redirect_url, re.IGNORECASE)
                    aid_match = re.search(r'(?:aid=|av)(\d+)', redirect_url, re.IGNORECASE)
                    
                    if bvid_match:
                        bvid = bvid_match.group(1)
                        print(f"从重定向URL提取到BV号: {bvid}")
                    elif aid_match:
                        aid = aid_match.group(1)
                        print(f"从重定向URL提取到AV号: {aid}")
        except Exception as e:
            print(f"处理重定向失败: {e}")
    
    # 如果没有从URL中提取到视频ID，尝试从页面内容中提取
    if not bvid and not aid:
        print("无法从URL中提取B站视频ID，尝试从页面内容中提取...")
        
        try:
            response = await make_request_async(
                target_url,
                async_crawler=async_crawler,
                headers=headers_dict,
                throttler=throttler,
                proxy_manager=proxy_manager,
                method='GET'
            )
            
            if not response:
                print("请求页面失败，无法提取视频ID")
                return
                
            html_content = await response.text(encoding='utf-8', errors='replace')
            
            # 尝试从HTML中提取视频ID
            bvid_match = re.search(r'"bvid":\s*"(BV[a-zA-Z0-9]+)"', html_content)
            if bvid_match:
                bvid = bvid_match.group(1)
                print(f"从页面内容提取到BV号: {bvid}")
            else:
                aid_match = re.search(r'"aid":\s*(\d+)', html_content)
                if aid_match:
                    aid = aid_match.group(1)
                    print(f"从页面内容提取到AV号: {aid}")
        except Exception as e:
            print(f"从页面提取视频ID失败: {e}")
    
    if not bvid and not aid:
        print("无法获取B站视频ID，无法爬取评论")
        return
    
    # 如果有BV号但没有AV号，需要先获取AV号
    if bvid and not aid:
        try:
            print(f"正在通过BV号获取AV号...")
            api_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
            
            api_headers = headers_dict.copy()
            api_headers.update({
                'Referer': 'https://www.bilibili.com/',
                'Origin': 'https://www.bilibili.com'
            })
            
            await throttler.wait_async()
            response = await make_request_async(
                api_url,
                async_crawler=async_crawler,
                headers=api_headers,
                throttler=throttler,
                proxy_manager=proxy_manager,
                method='GET'
            )
            
            if not response:
                print("获取AV号失败")
                return
                
            data = await response.json()
            if data.get('code') == 0 and 'data' in data:
                aid = str(data['data']['aid'])
                print(f"成功获取AV号: {aid}")
            else:
                print(f"获取AV号失败: {data.get('message', '未知错误')}")
                return
        except Exception as e:
            print(f"获取AV号失败: {e}")
            return
    
    # 构建评论API请求
    api_url = f"https://api.bilibili.com/x/v2/reply?type=1&oid={aid}&pn=1&ps=30&sort=2"
    
    # 设置必要的请求头
    api_headers = headers_dict.copy()
    api_headers.update({
        'Referer': f'https://www.bilibili.com/video/{bvid if bvid else "av" + aid}',
        'Accept': 'application/json, text/plain, */*',
        'Origin': 'https://www.bilibili.com'
    })
    
    print(f"请求评论API: {api_url}")
    
    try:
        await throttler.wait_async()
        
        response = await make_request_async(
            api_url,
            async_crawler=async_crawler,
            headers=api_headers,
            throttler=throttler,
            proxy_manager=proxy_manager,
            method='GET'
        )
        
        if not response:
            print("请求评论API失败")
            return
        
        # 解析评论数据
        try:
            json_data = await response.json()
            
            # 检查API响应格式
            if json_data.get('code') == 0 and 'data' in json_data:
                replies = json_data['data'].get('replies', [])
                
                if not replies:
                    print("API返回的评论列表为空")
                    return
                    
                print(f"\n成功获取 {len(replies)} 条评论:")
                
                comments_data = []
                for i, reply in enumerate(replies):
                    try:
                        member = reply.get('member', {})
                        content = reply.get('content', {})
                        
                        username = member.get('uname', "未知用户")
                        message = content.get('message', "无内容")
                        ctime = reply.get('ctime', 0)
                        
                        # 转换时间戳为可读时间
                        time_str = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S') if ctime else ""
                            
                        likes = reply.get('like', 0)
                        
                        comment_data = {
                            "user": username,
                            "content": message,
                            "time": time_str,
                            "likes": likes
                        }
                        
                        comments_data.append(comment_data)
                        
                        if i < 10:  # 只显示前10条
                            print(f"--- 评论 {i+1} ---")
                            print(f"用户: {username}")
                            print(f"内容: {message}")
                            print(f"时间: {time_str}")
                            print(f"点赞: {likes}")
                            print("--------------------")
                    except Exception as e:
                        print(f"解析评论项失败: {e}")
                
                # 保存评论数据
                if comments_data:
                    save_comments_to_file(comments_data, target_url, "bilibili")
                    
                # 检查是否有热门评论
                top_replies = json_data['data'].get('top_replies', [])
                if top_replies:
                    print(f"\n另外发现 {len(top_replies)} 条热门评论:")
                    
                    hot_comments_data = []
                    for i, reply in enumerate(top_replies):
                        try:
                            member = reply.get('member', {})
                            content = reply.get('content', {})
                            
                            username = member.get('uname', "未知用户")
                            message = content.get('message', "无内容")
                            ctime = reply.get('ctime', 0)
                            
                            # 转换时间戳为可读时间
                            time_str = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S') if ctime else ""
                                
                            likes = reply.get('like', 0)
                            
                            comment_data = {
                                "user": username,
                                "content": message,
                                "time": time_str,
                                "likes": likes,
                                "is_hot": True
                            }
                            
                            hot_comments_data.append(comment_data)
                            
                            print(f"--- 热门评论 {i+1} ---")
                            print(f"用户: {username}")
                            print(f"内容: {message}")
                            print(f"时间: {time_str}")
                            print(f"点赞: {likes}")
                            print("--------------------")
                        except Exception as e:
                            print(f"解析热门评论项失败: {e}")
                    
                    # 将热门评论也添加到保存的数据中
                    if hot_comments_data:
                        save_comments_to_file(hot_comments_data, target_url, "bilibili_hot")
            else:
                print(f"API响应错误: {json_data.get('message', '未知错误')}")
        except Exception as e:
            print(f"解析评论JSON数据失败: {e}")
            content = await response.text()
            print(f"API响应内容: {content[:500]}...")  # 只显示前500个字符
    except Exception as e:
        print(f"请求B站评论数据失败: {e}")
        traceback.print_exc()

async def crawl_comments_async(target_url: str, headers_dict: Dict, throttler: AsyncRequestThrottler, async_crawler: AsyncCrawler, proxy_manager: Optional[AsyncProxyManager]):
    """爬取评论的主函数"""
    print(f"正在尝试从 {target_url} 爬取评论...")
    
    # 检测是否为B站链接
    if "bilibili.com" in target_url or "b23.tv" in target_url:
        print("检测到B站链接，使用专用爬取方法...")
        await crawl_bilibili_comments(target_url, headers_dict, throttler, async_crawler, proxy_manager)
        return
    
    # 检测是否为小红书链接
    if "xiaohongshu.com" in target_url or "xhslink.com" in target_url:
        print("检测到小红书链接，使用专用爬取方法...")
        await crawl_xiaohongshu_comments(target_url, headers_dict, throttler, async_crawler, proxy_manager)
        return
    
    # 尝试检测并爬取JSON格式的评论数据
    json_comments = await try_fetch_json_comments(target_url, headers_dict, throttler, async_crawler, proxy_manager)
    if json_comments:
        return
    
    # 如果JSON爬取失败，回退到HTML解析方法
    try:
        await throttler.wait_async()
        
        response = await make_request_async(
            target_url,
            async_crawler=async_crawler,
            headers=headers_dict,
            throttler=throttler,
            proxy_manager=proxy_manager,
            method='GET'
        )

        if not response:
            print(f"请求页面内容失败 (无响应对象): {target_url}")
            return
        
        content_type_header = response.headers.get('Content-Type', '').lower()
        
        # 检查是否为JSON响应
        if 'application/json' in content_type_header:
            try:
                json_data = await response.json()
                await parse_json_comments(json_data, target_url)
                return
            except Exception as e_json:
                print(f"解析JSON数据失败: {e_json}")
        
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

        # 尝试从HTML中提取可能的API端点
        api_endpoints = extract_api_endpoints(html_content, target_url)
        if api_endpoints:
            print(f"发现可能的API端点: {len(api_endpoints)} 个")
            for i, endpoint in enumerate(api_endpoints[:3]):  # 只显示前3个
                print(f"  - {endpoint}")
            
            # 尝试请求第一个API端点
            if api_endpoints:
                print("\n尝试请求发现的API端点...")
                for endpoint in api_endpoints[:2]:  # 只尝试前2个
                    await try_api_endpoint(endpoint, headers_dict, throttler, async_crawler, proxy_manager)

        # 常规HTML解析
        soup = BeautifulSoup(html_content, 'html.parser')
        comment_selectors = [
            '.comment-text', '.comment-body', '.comment-content', '.comment',
            'div[class*="comment"]', 'article[class*="comment"]',
            '.reply-content', '.post-text', '.comment-message',
            '#comments', '.comments-area', 'section.comments',
            # 增加更多选择器以提高匹配率
            '.review-item', '.user-comment', '.comment-list li',
            '[data-testid*="comment"]', '[class*="CommentItem"]',
            '[class*="commentItem"]', '.comment-wrapper'
        ]
        
        # 尝试查找评论中的用户名和时间
        comment_data = []
        for element in soup.select('[class*="comment"], [class*="review"], .comment, .review-item'):
            try:
                user_elem = element.select_one('.user-name, .author, [class*="user"], [class*="author"], [class*="name"]')
                time_elem = element.select_one('.time, .date, [class*="time"], [class*="date"]')
                content_elem = element.select_one('.content, .text, [class*="content"], [class*="text"]')
                
                user = user_elem.get_text(strip=True) if user_elem else "未知用户"
                time = time_elem.get_text(strip=True) if time_elem else "未知时间"
                content = content_elem.get_text(strip=True) if content_elem else element.get_text(strip=True)
                
                if content and len(content) > 10:
                    comment_data.append({
                        "user": user,
                        "time": time,
                        "content": content
                    })
            except Exception:
                continue
        
        if comment_data:
            print(f"\n找到 {len(comment_data)} 条带用户信息的评论:")
            for i, comment in enumerate(comment_data[:10]):  # 只显示前10条
                print(f"--- 评论 {i+1} ---")
                print(f"用户: {comment['user']}")
                print(f"时间: {comment['time']}")
                print(f"内容: {comment['content']}")
                print("--------------------")
            
            # 保存评论数据
            save_comments_to_file(comment_data, target_url)
            return
        
        # 如果没有找到带用户信息的评论，尝试简单的评论提取
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
            print(f"\n找到 {len(found_comments)} 条可能的评论内容:")
            for i, comment in enumerate(set(found_comments)[:10]):  # 只显示前10条
                print(f"--- 评论 {i+1} ---")
                print(comment)
                print("--------------------")
            
            # 保存简单评论数据
            simple_comment_data = [{"content": c} for c in set(found_comments)]
            save_comments_to_file(simple_comment_data, target_url)
        else:
            print("未能自动提取到明确的评论内容。")
            print("提示: 评论可能是动态加载的 (需要JavaScript执行)，或者使用了网站特定的HTML结构。")
            print("VIP内容或需要登录的评论也可能无法直接获取。")
            
            # 尝试查找可能的评论加载脚本
            scripts = soup.find_all('script')
            for script in scripts:
                script_text = script.string if script.string else ""
                if script_text and ("comment" in script_text.lower() or "评论" in script_text):
                    print("\n发现可能包含评论数据的脚本:")
                    print(f"脚本长度: {len(script_text)} 字符")
                    # 尝试从脚本中提取JSON数据
                    json_matches = re.findall(r'({[\s\S]*?})', script_text)
                    for json_str in json_matches[:3]:  # 只检查前3个匹配项
                        try:
                            json_data = json.loads(json_str)
                            if isinstance(json_data, dict) and any(k for k in json_data.keys() if "comment" in k.lower()):
                                print("从脚本中提取到可能的评论数据:")
                                print(json.dumps(json_data, ensure_ascii=False, indent=2)[:500] + "...")  # 只显示前500个字符
                                break
                        except:
                            continue
                    break
    except aiohttp.ClientResponseError as e_http:
        print(f"HTTP错误 {e_http.status}: {e_http.message} - URL: {target_url}")
    except Exception as e:
        print(f"处理评论时发生错误: {e} - URL: {target_url}")
        traceback.print_exc()


async def crawl_videos_async(target_url, headers_dict, html_content, throttler, async_crawler, proxy_manager=None):
    """异步爬取视频到本地"""
    print(f"正在尝试从 {target_url} 爬取视频到本地...")
    soup = BeautifulSoup(html_content, 'html.parser')
    found_video_sources = set()
    
    # 处理URL，确保使用HTTPS
    processed_page_url_for_video = target_url
    if processed_page_url_for_video.startswith("http://"):
        processed_page_url_for_video = "https://" + processed_page_url_for_video[len("http://"):]
    elif not processed_page_url_for_video.startswith("https://") and "://" not in processed_page_url_for_video:
        processed_page_url_for_video = "https://" + processed_page_url_for_video
    
    # 获取基础域名
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
    
    # 获取当前路径基础
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
        """解析视频URL为绝对URL"""
        if video_url_param.startswith("data:"):
            return None
        elif video_url_param.startswith("//"):
            return f"https:{video_url_param}"
        elif video_url_param.startswith("/"):
            if base_domain_url_video:
                return f"{base_domain_url_video}{video_url_param}"
            else:
                return video_url_param
        elif video_url_param.startswith("http://"):
            return "https://" + video_url_param[len("http://"):]
        elif video_url_param.startswith("https://"):
            return video_url_param
        else:
            if current_path_base_video:
                final_url = current_path_base_video + video_url_param
                if final_url.startswith("http://"):
                    return "https://" + final_url[len("http://"):]
                return final_url
            else:
                return video_url_param

    # 从HTML中提取视频源
    # 1. 处理video标签
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
    
    # 2. 处理iframe和embed标签
    for iframe_tag in soup.find_all(['iframe', 'embed']):
        src = iframe_tag.get('src')
        if src:
            resolved = resolve_video_url(src)
            if resolved: found_video_sources.add(resolved)

    # 3. 从HTML文本中查找视频链接
    html_text_content = str(html_content)
    
    # 通用视频模式
    generic_video_pattern = r"['\"`]([^'\"`]*?\.(?:mp4|webm|ogg|mov|avi|flv|wmv|mkv|3gp))['\"`]"
    js_video_pattern = r"video[Uu]rl\s*[:=]\s*['\"`]([^'\"`]+?)['\"`]"
    generic_src_pattern = r"src\s*[:=]\s*['\"`]([^'\"`]+?)['\"`]"

    for pattern in [generic_video_pattern, js_video_pattern, generic_src_pattern]:
        for match in re.finditer(pattern, html_text_content, re.IGNORECASE):
            url_match = match.group(1)
            if url_match and not url_match.startswith("data:"):
                resolved = resolve_video_url(url_match)
                if resolved:
                    # 过滤掉非视频文件
                    if any(ext in resolved.lower() for ext in ['.js', '.css', '.html', '.php', '.json', '.xml', '.txt', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp']):
                        continue
                    
                    # 添加可能的视频链接
                    if any(ext in resolved.lower() for ext in ['.mp4', '.webm', '.ogg', '.mov', '.avi', '.flv', '.wmv', '.mkv', '.3gp', '.m3u8']) or \
                       ('video' in resolved.lower() or 'stream' in resolved.lower() or 'embed' in resolved.lower() or 'player' in resolved.lower() or 'media' in resolved.lower() or 'playurl' in resolved.lower()) and \
                       not any(non_video_ext in resolved.lower() for non_video_ext in ['.js', '.css', '.php', '.html']):
                        found_video_sources.add(resolved)

    # 4. 查找m3u8播放列表
    print("正在尝试检测可能的m3u8播放列表...")
    m3u8_pattern = r"['\"`]([^'\"`]*?\.m3u8[^'\"`]*?)['\"`]"
    for match in re.finditer(m3u8_pattern, html_text_content, re.IGNORECASE):
        m3u8_url = match.group(1)
        resolved_m3u8 = resolve_video_url(m3u8_url)
        if resolved_m3u8:
            print(f"发现可能的m3u8播放列表: {resolved_m3u8}")
            found_video_sources.add(resolved_m3u8)
    
    # 处理找到的视频链接
    all_urls = list(found_video_sources)
    if not all_urls:
        print("未找到直接可见的视频链接或m3u8播放列表。")
        return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
    
    print(f"\n找到 {len(all_urls)} 个可能的视频相关链接")
    download_dir = "downloaded_videos"
    os.makedirs(download_dir, exist_ok=True)
    
    stats = {'total': len(all_urls), 'success': 0, 'failed': 0, 'skipped': 0}
    
    # 异步下载视频
    async def download_video(video_url, index):
        """异步下载单个视频"""
        processed_url = video_url
        print(f"\n正在处理 #{index+1}: {processed_url}")
        
        # 跳过m3u8文件，因为需要特殊处理
        if ".m3u8" in processed_url.lower():
            print("检测到m3u8播放列表，这通常是流媒体内容")
            print("提示: 流媒体内容通常需要特殊工具如ffmpeg进行下载")
            stats['skipped'] += 1
            return None
        
        # 跳过嵌入式播放器链接
        if "iframe" in processed_url.lower() or "embed" in processed_url.lower():
            print("这是嵌入式播放器链接，可能需要进一步分析")
            stats['skipped'] += 1
            return None
        
        try:
            # 生成文件名
            file_name_prefix = re.sub(r'[^a-zA-Z0-9_]', '_', urlparse(processed_url).path.split('/')[-1] or f"video_{index+1}")
            file_name = f"{file_name_prefix[:50]}_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # 确定文件扩展名
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
            file_path = os.path.join(download_dir, file_name)
            
            print(f"正在尝试下载到: {file_path}")
            
            # 设置请求头
            download_headers = headers_dict.copy()
            download_headers['Referer'] = target_url
            
            # 异步下载视频
            for attempt in range(3):  # 最多重试3次
                try:
                    await throttler.wait_async()
                    
                    response = await make_request_async(
                        processed_url,
                        async_crawler=async_crawler,
                        headers=download_headers,
                        throttler=throttler,
                        proxy_manager=proxy_manager,
                        method='GET'
                    )
                    
                    if not response:
                        print(f"请求视频失败 (尝试 {attempt+1}/3): {processed_url}")
                        if attempt == 2:  # 最后一次尝试
                            stats['failed'] += 1
                            return None
                        await asyncio.sleep(2 + random.uniform(0, 1))
                        continue
                    
                    # 检查内容类型
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'text/html' in content_type or 'application/javascript' in content_type or 'text/javascript' in content_type or 'text/css' in content_type:
                        print(f"警告: 返回的是{content_type}内容而非视频文件。跳过此链接。")
                        stats['failed'] += 1
                        return None
                    
                    if not ('video' in content_type or 'stream' in content_type or 'application/octet-stream' in content_type or 'binary/octet-stream' in content_type):
                        print(f"警告: 内容类型 {content_type} 可能不是视频。仍尝试下载。")
                    
                    # 检查文件大小
                    total_size = int(response.headers.get('Content-Length', 0))
                    if 0 < total_size < 10000:
                        print(f"警告: 文件大小只有 {total_size} 字节，可能不是完整的视频文件。")
                    
                    # 异步下载文件
                    downloaded_bytes = 0
                    async with aiofiles.open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            if chunk:
                                await f.write(chunk)
                                downloaded_bytes += len(chunk)
                    
                    print(f"下载了 {downloaded_bytes / (1024*1024):.2f} MB.")
                    
                    # 检查下载的文件大小
                    file_size = os.path.getsize(file_path)
                    if file_size < 10000 and total_size == 0:
                        print(f"警告: 实际下载文件大小只有 {file_size} 字节，可能不是有效的视频。")
                    
                    print(f"下载完成: {file_path}")
                    stats['success'] += 1
                    return file_path
                    
                except Exception as e:
                    print(f"下载过程中发生错误 (尝试 {attempt+1}/3): {e}")
                    if attempt < 2:  # 不是最后一次尝试
                        await asyncio.sleep(2 + random.uniform(0, 1))
                    else:
                        stats['failed'] += 1
                        return None
        
        except Exception as e:
            print(f"处理视频URL {processed_url} 时出错: {e}")
            stats['failed'] += 1
            return None
    
    # 并发下载所有视频
    download_tasks = []
    for i, video_url in enumerate(all_urls):
        download_tasks.append(download_video(video_url, i))
    
    # 等待所有下载任务完成
    results = await asyncio.gather(*download_tasks, return_exceptions=True)
    
    # 处理异常
    for result in results:
        if isinstance(result, Exception):
            print(f"下载过程中发生未捕获的异常: {result}")
    
    print(f"\n视频爬取任务完成。")
    print(f"总计: {stats['total']} 个视频链接")
    print(f"成功下载: {stats['success']} 个")
    print(f"下载失败: {stats['failed']} 个")
    print(f"已跳过: {stats['skipped']} 个")
    print(f"下载的文件保存在 {os.path.abspath(download_dir)} 目录中")
    print("注意: 某些下载内容可能不完整或需要进一步处理才能播放")
    
    return stats

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

async def decrypt_css_content(url: str, headers_dict: Dict, throttler: AsyncRequestThrottler, async_crawler: AsyncCrawler, proxy_manager: Optional[AsyncProxyManager] = None):
    """获取并解密CSS内容"""
    print(f"\n=== 开始CSS解密: {url} ===")
    try:
        await throttler.wait_async()
        
        response = await make_request_async(
            url,
            async_crawler=async_crawler,
            headers=headers_dict,
            throttler=throttler,
            proxy_manager=proxy_manager,
            method='GET'
        )
        
        if not response:
            print(f"请求CSS内容失败 (无响应对象): {url}")
            return
            
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/css' not in content_type and '.css' not in url:
            print(f"警告: 内容类型 {content_type} 可能不是CSS。")
            
        css_content = await response.text()
        print(f"成功获取CSS内容，长度: {len(css_content)} 字符")
        print("正在解密CSS内容...")
        decrypted_css = await async_crawler.decrypt_css(css_content)
        
        print("正在美化CSS内容...")
        beautified_css = await async_crawler.beautify_css(decrypted_css)
        
        # 保存结果
        download_dir = "downloaded_css"
        os.makedirs(download_dir, exist_ok=True)
        
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path)
        if not file_name or not file_name.endswith('.css'):
            file_name = f"decrypted_css_{int(time.time())}.css"
            
        file_path = os.path.join(download_dir, file_name)
        
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(beautified_css)
            
        print(f"CSS解密完成，已保存到: {os.path.abspath(file_path)}")
        
        # 显示解密前后的差异
        print("\n解密前后的差异:")
        if css_content == beautified_css:
            print("未检测到加密内容，CSS未发生变化。")
        else:
            print(f"原始CSS长度: {len(css_content)} 字符")
            print(f"解密后CSS长度: {len(beautified_css)} 字符")
            print("解密可能已成功处理了混淆或加密的内容。")
            
    except Exception as e:
        print(f"CSS解密过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        print("=== CSS解密任务结束 ===")

async def bypass_captcha(url: str, headers_dict: Dict, throttler: AsyncRequestThrottler, async_crawler: AsyncCrawler, proxy_manager: Optional[AsyncProxyManager] = None):
    """绕过验证码并访问页面内容"""
    print(f"\n=== 开始绕过验证码: {url} ===")
    try:
        await throttler.wait_async()
        
        print("正在尝试访问页面并检测验证码...")
        response = await make_request_async(
            url,
            async_crawler=async_crawler,
            headers=headers_dict,
            throttler=throttler,
            proxy_manager=proxy_manager,
            method='GET'
        )
        
        if not response:
            print(f"请求页面失败 (无响应对象): {url}")
            return
            
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            print(f"警告: 内容类型 {content_type} 不是HTML，可能无法正确处理验证码。")
            
        html_content = await response.text()
        
        # 检查是否存在验证码
        if 'captcha' not in html_content.lower() and response.status != 403:
            print("未检测到验证码，页面可能已经可以正常访问。")
            return
            
        print("检测到可能存在验证码，正在尝试识别和绕过...")
        
        # 解析HTML查找验证码图片
        soup = BeautifulSoup(html_content, 'html.parser')
        captcha_img = soup.find('img', {'id': lambda x: x and 'captcha' in x.lower()})
        
        if not captcha_img or not captcha_img.get('src'):
            print("未找到验证码图片，无法自动识别。")
            return
            
        img_url = captcha_img['src']
        captcha_result = None
        img_content_bytes_for_service = None # To store fetched image bytes for external service
        img_base64_for_service = None # To store base64 data for external service
        
        if img_url.startswith('data:image'):
            print("检测到内联验证码图片，正在识别...")
            img_base64_for_service = img_url.split(',', 1)[1] if ',' in img_url else img_url # Get only base64 part
            captcha_result = async_crawler.captcha_solver.solve_image_captcha(img_url)
        else:
            # 获取完整URL
            if not img_url.startswith(('http://', 'https://')):
                base_url = '/'.join(url.split('/')[:3])
                img_url = urljoin(base_url, img_url)
                
            print(f"正在获取验证码图片: {img_url}")
            # Use async_crawler's session to fetch the image
            async with async_crawler.session.get(img_url, headers=headers_dict) as img_response:
                img_content_bytes_for_service = await img_response.read()
                captcha_result = async_crawler.captcha_solver.solve_image_captcha(img_content_bytes_for_service)
                
        if not captcha_result:
            print("本地验证码识别失败，尝试外部服务...")
            current_config = load_config()
            service_config = current_config.get('captcha_solving_services', {})
            if service_config.get('enabled'):
                api_key = service_config.get('api_key')
                service_name = service_config.get('service_name')
                if api_key and service_name:
                    captcha_result = async_crawler.captcha_solver.solve_with_external_service(
                        service_name, 
                        api_key, 
                        image_data_bytes=img_content_bytes_for_service, 
                        image_base64=img_base64_for_service, 
                        method='image',
                        page_url=url
                    )
                    if captcha_result:
                        print(f"外部服务 {service_name} 成功解决验证码: {captcha_result}")
                    else:
                        print(f"外部服务 {service_name} 未能解决验证码。")
                else:
                    print("外部验证码服务已启用，但API密钥或服务名称在配置中缺失。")
            else:
                print("外部验证码服务未在配置中启用。")

        if not captcha_result:
            print("验证码识别失败 (本地和外部服务均失败)，无法自动绕过。")
            return # Changed from return None to just return, as function returns response or None
            
        print(f"成功识别验证码: {captcha_result}")
        
        # 查找表单并提交验证码
        form = soup.find('form')
        if not form:
            print("未找到表单，无法自动提交验证码。")
            return # Changed from return None
            
        # 提取表单字段
        form_data = {}
        for input_field in form.find_all('input'):
            name = input_field.get('name')
            value = input_field.get('value', '')
            if name:
                form_data[name] = value
                
        # 添加验证码
        captcha_field_found = False
        possible_captcha_names = [
            'captcha', 'validateCode', 'verifyCode', 'verification', 'code', 'vcode', 
            'captcha_code', 'security_code', 'pin', 'secucode', 'authcode'
        ]
        
       
        for field_name_key in list(form_data.keys()): 
            if field_name_key.lower() in possible_captcha_names or 'captcha' in field_name_key.lower():
                form_data[field_name_key] = captcha_result 
                captcha_field_found = True
                print(f"找到并填充验证码字段: {field_name_key}")
                break
        
        if not captcha_field_found:
            
            for possible_name in possible_captcha_names:
                input_tag = soup.find('input', {'name': possible_name})
                if input_tag:
                    form_data[possible_name] = captcha_result
                    captcha_field_found = True
                    print(f"找到并填充验证码字段 (HTML): {possible_name}")
                    break
                    
        if not captcha_field_found:
            
            default_field_name = 'captcha'
            form_data[default_field_name] = captcha_result
            print(f"未找到特定验证码输入字段，使用默认字段名 '{default_field_name}'")
            
        # 获取表单提交URL
        form_action = form.get('action', '')
        if not form_action:
            form_action = url
        elif not form_action.startswith(('http://', 'https://')):
            base_url = '/'.join(url.split('/')[:3])
            form_action = urljoin(base_url, form_action)
            
        # 提交表单
        print(f"正在提交验证码到: {form_action}")
        form_method = form.get('method', 'post').lower()
        
        submit_response = await make_request_async(
            form_action,
            async_crawler=async_crawler,
            headers=headers_dict,
            throttler=throttler,
            proxy_manager=proxy_manager,
            method=form_method.upper(),
            data=form_data
        )
        
        if not submit_response:
            print("提交验证码失败 (无响应对象)")
            return
            
        if submit_response.status == 200:
            print("验证码提交成功，已绕过验证码保护！")
            
            # 保存结果
            result_content = await submit_response.text()
            download_dir = "bypassed_pages"
            os.makedirs(download_dir, exist_ok=True)
            
            file_name = f"bypassed_page_{int(time.time())}.html"
            file_path = os.path.join(download_dir, file_name)
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(result_content)
                
            print(f"已保存绕过验证码后的页面内容到: {os.path.abspath(file_path)}")
        else:
            print(f"验证码提交后收到非200状态码: {submit_response.status}")
            
    except Exception as e:
        print(f"绕过验证码过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        print("=== 验证码绕过任务结束 ===")

async def async_main():
    try:
        config_data = load_config()
    except Exception as e:
        print(f"配置加载出错: {e}")
        config_data = {
                'user_agents': [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
                ],
                'request_throttling': {
                    'min_interval': 2.0,
                    'max_interval': 5.0,
                    'burst_requests': 3,
                    'burst_period': 60
                },
                'proxy_settings': {
                    'enabled': False,
                    'proxy_list': []
                },
                'captcha_solving_services': {
                    'enabled': False,
                    'service_name': '', 
                    'api_key': '',
                  
                    'recaptcha_site_key_selectors': ['div.g-recaptcha', 'div.h-captcha'], 
                    'recaptcha_site_key_attribute': 'data-sitekey'
                }
            }

    request_counter = load_counter()
    
    custom_headers = {
        'User-Agent': random.choice(config_data['user_agents']),
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'
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
            async_proxy_manager = None
            try:
                if config_data['proxy_settings']['enabled']:
                    if not config_data['proxy_settings']['proxy_list'] and not config_data['proxy_settings']['proxy_api_url']:
                        print("警告: 代理已启用但未配置代理列表或API URL")
                    else:
                        async_proxy_manager = AsyncProxyManager(
                            async_crawler_instance=async_crawler_instance,
                            proxy_list=config_data['proxy_settings']['proxy_list'],
                            proxy_api_url=config_data['proxy_settings']['proxy_api_url'],
                            api_key=config_data['proxy_settings']['api_key']
                        )
                        print("代理管理器初始化成功")
            except Exception as proxy_error:
                print(f"代理管理器初始化失败: {proxy_error}")
                async_proxy_manager = None
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
                html_content_main = None # Initialize html_content_main
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
                                print(f"提取到B站视频ID: {bvid}")
                                result = await parse_bilibili_video_async(
                                    bvid,
                                    processed_target_url,
                                    custom_headers,
                                    default_async_throttler,
                                    async_crawler_instance,
                                    async_proxy_manager
                                )
                                if result:
                                    print(f"\nB站视频下载成功: {result}")
                                else:
                                    print("\nB站视频下载失败")
                            else:
                                print("无法从URL中提取B站视频ID (BVID)。请确保URL包含如 'bvid=BV...' 或 '/video/BV...' 的部分。")
                                print("尝试使用通用视频爬取方法...")
                                if html_content_main:
                                    await crawl_videos_async(
                                        processed_target_url,
                                        custom_headers,
                                        html_content_main,
                                        default_async_throttler,
                                        async_crawler_instance,
                                        async_proxy_manager
                                    )
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
                                            charset = 'utf-8' 
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
                                    await crawl_images_revised_async(
                                        processed_target_url,
                                        html_content_main,
                                        custom_headers,
                                        default_async_throttler,
                                        async_crawler_instance,
                                        recursive=True,
                                        max_depth=config.max_depth,
                                        proxy_manager=async_proxy_manager
                                    )
                                elif choice == '2':
                                    print("\n--- 开始爬取评论 ---")
                                    await crawl_comments_async(processed_target_url, custom_headers, default_async_throttler, async_crawler_instance, async_proxy_manager)
                                elif choice == '3':
                                    print("\n--- 开始爬取视频到本地 (通用方法) ---")
                                    await crawl_videos_async(
                                        processed_target_url,
                                        custom_headers,
                                        html_content_main,
                                        default_async_throttler,
                                        async_crawler_instance,
                                        async_proxy_manager
                                    )
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

    