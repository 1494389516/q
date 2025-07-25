# 🕷️ 爬虫百宝箱 (Spider Toolkit)

一个功能强大、易于使用的Python网络爬虫工具包，专为高效数据采集而设计。

## ✨ 核心特性

### 🚀 异步高性能
- 基于aiohttp的异步网络请求
- 智能重试机制和请求节流
- 支持高并发批量处理
- 内置连接池管理

### 🛡️ 反爬虫对策
- 智能代理轮换管理
- 随机请求延迟
- 验证码自动识别
- 自定义User-Agent

### 📊 多平台支持
- **小红书**: 评论、视频、图片采集
- **B站**: 评论、视频信息获取
- **抖音**: 评论数据提取
- **通用网站**: 智能内容识别

### �  数据管理
- SQLite数据库存储
- JSON/CSV多格式导出
- 重复数据检测
- 增量更新支持

## 📦 安装

```bash
# 克隆项目
git clone <repository-url>
cd spider_toolkit

# 安装依赖
pip install -r requirements.txt
```

## 🚀 快速开始

### 基础使用

```python
import asyncio
from spider_toolkit import SpiderToolkit

async def main():
    toolkit = SpiderToolkit()
    
    # 爬取评论
    comments = await toolkit.crawl_comments("https://www.bilibili.com/video/BV1xx411c7mu")
    print(f"获取到 {len(comments)} 条评论")
    
    # 下载视频
    videos = await toolkit.crawl_videos("https://example.com/video-page")
    
    # 采集图片
    images = await toolkit.crawl_images("https://example.com/gallery")

asyncio.run(main())
```

### 命令行使用

```bash
# 爬取B站视频评论
python -m spider_toolkit.cli comments https://www.bilibili.com/video/BV1xx411c7mu -o comments.json

# 批量下载图片
python -m spider_toolkit.cli images https://example.com/gallery --download-dir ./images

# 批量处理
python -m spider_toolkit.cli batch urls.txt --type comments --concurrent 5
```

## 📖 详细功能

### 1. 评论爬取

```python
# B站评论
comments = await toolkit.crawl_comments("https://www.bilibili.com/video/BV1xx411c7mu")

# 小红书评论
comments = await toolkit.crawl_comments("https://www.xiaohongshu.com/explore/xxx")

# 自定义请求头
headers = {"Cookie": "your_cookie_here"}
comments = await toolkit.crawl_comments(url, headers=headers)
```

### 2. 视频下载

```python
videos = await toolkit.crawl_videos(
    url="https://example.com/video-page",
    download_dir="./downloads/videos"
)

for video in videos:
    print(f"已下载: {video['filename']} ({video['size']} bytes)")
```

### 3. 图片采集

```python
images = await toolkit.crawl_images(
    url="https://example.com/gallery",
    download_dir="./downloads/images"
)
```

### 4. 批量处理

```python
urls = [
    "https://www.bilibili.com/video/BV1xx411c7mu",
    "https://www.bilibili.com/video/BV1yy411c7mu"
]

results = await toolkit.batch_crawl(
    urls=urls,
    crawl_type='comments',
    max_concurrent=5
)

print(f"成功: {results['success_count']}/{results['total_urls']}")
```

## ⚙️ 配置管理

### 配置文件 (crawler_config.yaml)

```yaml
request:
  timeout: 30
  max_retries: 3
  delay_range: [1, 3]

proxy:
  enabled: true
  proxy_list:
    - "http://proxy1:8080"
    - "http://proxy2:8080"

captcha:
  enabled: true
  service: "2captcha"
  api_key: "your_api_key"

download:
  base_dir: "./downloads"
  max_file_size: 104857600

database:
  enabled: true
  db_path: "crawler_history.db"
```

### 代码中配置

```python
from spider_toolkit import config

# 设置代理
config.set('proxy.enabled', True)
config.set('proxy.proxy_list', ['http://proxy:8080'])

# 设置请求延迟
config.set('request.delay_range', [2, 5])
```

## 🔍 验证码处理

```python
from spider_toolkit import CaptchaSolver

solver = CaptchaSolver()

# 识别图片验证码
result = solver.solve_image_captcha(image_data)

# 处理滑块验证码
distance = solver.solve_slider_captcha(bg_image, slider_image)

# 使用2captcha服务
result = solver.solve_with_2captcha(
    api_key="your_api_key",
    image_data=image_bytes
)
```

## 📊 数据存储

```python
from spider_toolkit import history

# 检查是否最近爬取过
if not history.is_recently_crawled(url, 'comments', hours=24):
    comments = await toolkit.crawl_comments(url)

# 获取统计信息
stats = history.get_statistics()
print(f"总计爬取: {stats['total_records']} 条记录")

# 导出数据
history.export_data('export.json', crawl_type='comments')
```

## 🛠️ 高级用法

### 自定义解析器

```python
from spider_toolkit.parsers import BaseParser

class CustomParser(BaseParser):
    async def parse(self, url, html_content, crawler, headers, throttler):
        # 自定义解析逻辑
        return parsed_data

toolkit.comment_parser = CustomParser()
```

### 代理管理

```python
from spider_toolkit import ProxyManager

proxy_manager = ProxyManager([
    'http://proxy1:8080',
    'http://proxy2:8080'
])

proxy = proxy_manager.get_proxy()
proxy_manager.mark_proxy_failed(proxy)
```

## 📋 命令行工具

```bash
# 基本命令
spider-toolkit comments <URL> [选项]
spider-toolkit videos <URL> --download-dir ./videos
spider-toolkit images <URL> --download-dir ./images
spider-toolkit batch urls.txt --type comments --concurrent 5

# 配置管理
spider-toolkit config get request.timeout
spider-toolkit config set proxy.enabled true
spider-toolkit config list

# 高级选项
spider-toolkit comments <URL> --proxy http://proxy:8080 --delay 2 5 --verbose
```

## 📁 项目结构

```
spider_toolkit/
├── __init__.py          # 包初始化
├── core.py             # 核心爬虫类
├── network.py          # 网络请求模块
├── parsers.py          # 内容解析器
├── database.py         # 数据库管理
├── captcha.py          # 验证码处理
├── config.py           # 配置管理
├── utils.py            # 工具函数
├── cli.py              # 命令行接口
└── requirements.txt    # 依赖列表
```

## 📊 支持平台

| 平台 | 评论 | 视频 | 图片 | 特殊功能 |
|------|------|------|------|----------|
| 哔哩哔哩 | ✅ | ✅ | ✅ | 弹幕、用户信息 |
| 小红书 | ✅ | ✅ | ✅ | 用户资料、标签 |
| 抖音 | ✅ | ⚠️ | ✅ | 音乐信息 |
| 通用网站 | ✅ | ✅ | ✅ | 自定义解析 |

## ⚠️ 注意事项

### 使用建议
- 遵守网站robots.txt协议
- 合理设置请求频率
- 尊重网站使用条款
- 不采集敏感信息

### 常见问题

**Q: 如何处理反爬虫？**
A: 启用代理轮换、调整请求频率、使用验证码识别

**Q: 如何提高效率？**
A: 使用批量处理、增加并发数、启用数据库缓存

**Q: 遇到动态内容怎么办？**
A: 可结合Selenium或Playwright进行浏览器自动化

## 🧪 测试

```bash
# 运行测试
python -m pytest spider_toolkit/tests/ -v

# 运行示例
python example_usage.py
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

### 开发规范
- 遵循PEP 8代码风格
- 添加类型注解
- 编写单元测试
- 更新文档

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🙏 致谢

感谢以下开源项目：
- [aiohttp](https://github.com/aio-libs/aiohttp) - 异步HTTP客户端
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) - HTML解析
- [Tesseract](https://github.com/tesseract-ocr/tesseract) - OCR引擎

---

⭐ **如果这个项目对你有帮助，请给个Star支持一下！**

**免责声明**: 本工具仅供学习和研究使用，使用者需要遵守相关法律法规和网站使用条款。