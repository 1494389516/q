"""
工具函数模块
"""

import re
import os
import json
import hashlib
import time
import random
from datetime import datetime
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Any, Optional
import logging


def setup_logging(level: str = 'INFO', log_file: str = 'crawler.log') -> logging.Logger:
    """设置日志配置"""
    logger = logging.getLogger('spider_toolkit')
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除非法字符"""
    # 移除或替换非法字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除控制字符
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    # 限制长度
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200-len(ext)] + ext
    
    return filename.strip()


def extract_domain(url: str) -> str:
    """从URL中提取域名"""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return ""


def is_valid_url(url: str) -> bool:
    """检查URL是否有效"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def generate_file_hash(content: bytes) -> str:
    """生成文件内容的MD5哈希"""
    return hashlib.md5(content).hexdigest()


def save_json_data(data: Any, filepath: str, ensure_ascii: bool = False) -> bool:
    """保存数据为JSON文件"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=2)
        return True
    except Exception as e:
        logging.error(f"保存JSON文件失败: {e}")
        return False


def load_json_data(filepath: str) -> Optional[Any]:
    """从JSON文件加载数据"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载JSON文件失败: {e}")
        return None


def extract_urls_from_text(text: str, base_url: str = "") -> List[str]:
    """从文本中提取URL"""
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[^\s<>"\']+\.[a-z]{2,}[^\s<>"\']*'
    urls = re.findall(url_pattern, text, re.IGNORECASE)
    
    result = []
    for url in urls:
        if not url.startswith(('http://', 'https://')):
            if url.startswith('www.'):
                url = 'http://' + url
            elif base_url:
                url = urljoin(base_url, url)
        
        if is_valid_url(url):
            result.append(url)
    
    return list(set(result))


def extract_emails_from_text(text: str) -> List[str]:
    """从文本中提取邮箱地址"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return list(set(re.findall(email_pattern, text)))


def extract_phone_numbers_from_text(text: str) -> List[str]:
    """从文本中提取电话号码"""
    phone_patterns = [
        r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
        r'\b\d{3}\.\d{3}\.\d{4}\b',  # 123.456.7890
        r'\b\d{10}\b',  # 1234567890
        r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (123) 456-7890
        r'\b1[3-9]\d{9}\b',  # 中国手机号
    ]
    
    phones = []
    for pattern in phone_patterns:
        phones.extend(re.findall(pattern, text))
    
    return list(set(phones))


def random_delay(min_seconds: float = 1.0, max_seconds: float = 3.0):
    """随机延迟"""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def get_file_extension(url: str) -> str:
    """从URL获取文件扩展名"""
    parsed = urlparse(url)
    path = parsed.path
    return os.path.splitext(path)[1].lower()


def is_image_url(url: str) -> bool:
    """检查URL是否指向图片"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
    return get_file_extension(url) in image_extensions


def is_video_url(url: str) -> bool:
    """检查URL是否指向视频"""
    video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v'}
    return get_file_extension(url) in video_extensions


def create_timestamp_filename(prefix: str = "", extension: str = "") -> str:
    """创建带时间戳的文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        filename = f"{prefix}_{timestamp}"
    else:
        filename = timestamp
    
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    return filename + extension


def parse_content_type(content_type: str) -> tuple:
    """解析Content-Type头"""
    if not content_type:
        return "", ""
    
    parts = content_type.split(';')
    mime_type = parts[0].strip()
    
    charset = ""
    for part in parts[1:]:
        if 'charset=' in part:
            charset = part.split('charset=')[1].strip()
            break
    
    return mime_type, charset


def detect_encoding(content: bytes) -> str:
    """检测文本编码"""
    try:
        import chardet
        result = chardet.detect(content)
        return result.get('encoding', 'utf-8') or 'utf-8'
    except ImportError:
        # 简单的编码检测
        encodings = ['utf-8', 'gbk', 'gb2312', 'big5', 'latin1']
        for encoding in encodings:
            try:
                content.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        return 'utf-8'


def clean_text(text: str) -> str:
    """清理文本，移除多余的空白字符"""
    if not text:
        return ""
    
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 移除首尾空白
    text = text.strip()
    
    return text


def extract_numbers_from_text(text: str) -> List[float]:
    """从文本中提取数字"""
    number_pattern = r'-?\d+\.?\d*'
    numbers = re.findall(number_pattern, text)
    return [float(num) for num in numbers if num]


def calculate_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度（简单版本）"""
    if not text1 or not text2:
        return 0.0
    
    # 转换为小写并分词
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # 计算交集和并集
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


# 导出所有工具函数
__all__ = [
    'setup_logging',
    'sanitize_filename',
    'extract_domain',
    'is_valid_url',
    'generate_file_hash',
    'save_json_data',
    'load_json_data',
    'extract_urls_from_text',
    'extract_emails_from_text',
    'extract_phone_numbers_from_text',
    'random_delay',
    'format_file_size',
    'get_file_extension',
    'is_image_url',
    'is_video_url',
    'create_timestamp_filename',
    'parse_content_type',
    'detect_encoding',
    'clean_text',
    'extract_numbers_from_text',
    'calculate_similarity'
]