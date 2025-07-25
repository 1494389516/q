"""
配置管理模块
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class CrawlerConfig:
    """爬虫配置类，管理所有配置参数"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "crawler_config.yaml"
        self.config_data = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                return self._get_default_config()
        else:
            # 创建默认配置文件
            default_config = self._get_default_config()
            self.save_config(default_config)
            return default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'request': {
                'timeout': 30,
                'max_retries': 3,
                'delay_range': [1, 3],
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            },
            'proxy': {
                'enabled': False,
                'proxy_list': [],
                'proxy_api_url': '',
                'api_key': '',
                'rotation_interval': 300
            },
            'captcha': {
                'enabled': False,
                'service': '2captcha',
                'api_key': '',
                'tesseract_path': r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            },
            'download': {
                'base_dir': './downloads',
                'max_file_size': 100 * 1024 * 1024,  # 100MB
                'allowed_extensions': ['.jpg', '.png', '.gif', '.mp4', '.pdf', '.doc', '.docx']
            },
            'database': {
                'enabled': True,
                'db_path': 'crawler_history.db'
            },
            'logging': {
                'level': 'INFO',
                'file': 'crawler.log',
                'max_size': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5
            },
            'splash': {
                'enabled': False,
                'url': 'http://localhost:8050',
                'wait_time': 3,
                'timeout': 30
            }
        }
    
    def save_config(self, config_data: Optional[Dict[str, Any]] = None):
        """保存配置到文件"""
        data = config_data or self.config_data
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键"""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值，支持点号分隔的嵌套键"""
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()
    
    @property
    def request_timeout(self) -> int:
        return self.get('request.timeout', 30)
    
    @property
    def max_retries(self) -> int:
        return self.get('request.max_retries', 3)
    
    @property
    def user_agent(self) -> str:
        return self.get('request.user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    @property
    def proxy_enabled(self) -> bool:
        return self.get('proxy.enabled', False)
    
    @property
    def captcha_enabled(self) -> bool:
        return self.get('captcha.enabled', False)
    
    @property
    def download_dir(self) -> str:
        return self.get('download.base_dir', './downloads')
    
    @property
    def database_enabled(self) -> bool:
        return self.get('database.enabled', True)
    
    @property
    def splash_enabled(self) -> bool:
        return self.get('splash.enabled', False)


# 全局配置实例
config = CrawlerConfig()