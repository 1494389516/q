"""
命令行接口模块
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

from .core import SpiderToolkit
from .config import config
from .utils import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="爬虫百宝箱 - 功能强大的网络爬虫工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  spider-toolkit comments https://www.bilibili.com/video/BV1xx411c7mu
  spider-toolkit videos https://example.com/video-page --download-dir ./videos
  spider-toolkit images https://example.com/gallery -o images.json
  spider-toolkit batch urls.txt --type comments --concurrent 5
        """
    )
    
    # 全局选项
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    parser.add_argument('-q', '--quiet', action='store_true', help='静默模式')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--timeout', type=int, default=30, help='请求超时时间（秒）')
    parser.add_argument('--retries', type=int, default=3, help='最大重试次数')
    parser.add_argument('--delay', type=float, nargs=2, default=[1, 3], 
                       metavar=('MIN', 'MAX'), help='请求延迟范围（秒）')
    
    # 代理选项
    parser.add_argument('--proxy', type=str, action='append', help='代理服务器（可多次使用）')
    parser.add_argument('--proxy-file', type=str, help='代理列表文件')
    
    # 输出选项
    parser.add_argument('-o', '--output', type=str, help='输出文件路径')
    parser.add_argument('--format', choices=['json', 'csv'], default='json', help='输出格式')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 评论爬取命令
    comments_parser = subparsers.add_parser('comments', help='爬取评论')
    comments_parser.add_argument('url', help='目标URL')
    comments_parser.add_argument('--headers', type=str, help='自定义请求头（JSON格式）')
    comments_parser.add_argument('--save-html', action='store_true', help='保存原始HTML')
    
    # 视频下载命令
    videos_parser = subparsers.add_parser('videos', help='下载视频')
    videos_parser.add_argument('url', help='目标URL')
    videos_parser.add_argument('--download-dir', type=str, default='./downloads/videos', 
                              help='下载目录')
    videos_parser.add_argument('--max-size', type=int, help='最大文件大小（MB）')
    
    # 图片下载命令
    images_parser = subparsers.add_parser('images', help='下载图片')
    images_parser.add_argument('url', help='目标URL')
    images_parser.add_argument('--download-dir', type=str, default='./downloads/images',
                              help='下载目录')
    images_parser.add_argument('--min-size', type=int, default=1024, help='最小文件大小（字节）')
    
    # 文档下载命令
    docs_parser = subparsers.add_parser('documents', help='下载文档')
    docs_parser.add_argument('url', help='目标URL')
    docs_parser.add_argument('--download-dir', type=str, default='./downloads/documents',
                            help='下载目录')
    docs_parser.add_argument('--types', type=str, nargs='+', 
                            default=['pdf', 'doc', 'docx', 'xls', 'xlsx'],
                            help='文档类型')
    
    # 批量处理命令
    batch_parser = subparsers.add_parser('batch', help='批量处理')
    batch_parser.add_argument('file', help='包含URL列表的文件')
    batch_parser.add_argument('--type', choices=['comments', 'videos', 'images', 'documents'],
                             default='comments', help='处理类型')
    batch_parser.add_argument('--concurrent', type=int, default=5, help='并发数')
    batch_parser.add_argument('--download-dir', type=str, help='下载目录')
    
    # 配置命令
    config_parser = subparsers.add_parser('config', help='配置管理')
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    config_get = config_subparsers.add_parser('get', help='获取配置值')
    config_get.add_argument('key', help='配置键')
    
    config_set = config_subparsers.add_parser('set', help='设置配置值')
    config_set.add_argument('key', help='配置键')
    config_set.add_argument('value', help='配置值')
    
    config_list = config_subparsers.add_parser('list', help='列出所有配置')
    
    return parser


async def handle_comments(args) -> int:
    """处理评论爬取命令"""
    toolkit = SpiderToolkit()
    
    # 解析自定义请求头
    headers = None
    if args.headers:
        try:
            headers = json.loads(args.headers)
        except json.JSONDecodeError:
            print("错误：无效的请求头JSON格式", file=sys.stderr)
            return 1
    
    try:
        comments = await toolkit.crawl_comments(args.url, headers=headers)
        
        if not comments:
            print("未找到评论数据")
            return 0
        
        print(f"成功爬取 {len(comments)} 条评论")
        
        # 输出结果
        if args.output:
            output_data(comments, args.output, args.format)
        else:
            # 显示前几条评论
            for i, comment in enumerate(comments[:5], 1):
                print(f"\n--- 评论 {i} ---")
                print(f"用户: {comment.get('user', '未知')}")
                print(f"内容: {comment.get('content', '')[:100]}...")
                print(f"时间: {comment.get('time', '')}")
        
        return 0
    
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1


async def handle_videos(args) -> int:
    """处理视频下载命令"""
    toolkit = SpiderToolkit()
    
    try:
        videos = await toolkit.crawl_videos(args.url, download_dir=args.download_dir)
        
        if not videos:
            print("未找到视频")
            return 0
        
        print(f"成功下载 {len(videos)} 个视频")
        
        for video in videos:
            size_mb = video.get('size', 0) / (1024 * 1024)
            print(f"- {video.get('filename', 'unknown')} ({size_mb:.1f}MB)")
        
        return 0
    
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1


async def handle_images(args) -> int:
    """处理图片下载命令"""
    toolkit = SpiderToolkit()
    
    try:
        images = await toolkit.crawl_images(args.url, download_dir=args.download_dir)
        
        if not images:
            print("未找到图片")
            return 0
        
        # 过滤小图片
        filtered_images = [img for img in images if img.get('size', 0) >= args.min_size]
        
        print(f"成功下载 {len(filtered_images)} 张图片")
        
        if args.output:
            output_data(filtered_images, args.output, args.format)
        
        return 0
    
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1


async def handle_documents(args) -> int:
    """处理文档下载命令"""
    toolkit = SpiderToolkit()
    
    try:
        documents = await toolkit.crawl_documents(args.url, download_dir=args.download_dir)
        
        if not documents:
            print("未找到文档")
            return 0
        
        print(f"成功下载 {len(documents)} 个文档")
        
        for doc in documents:
            size_mb = doc.get('size', 0) / (1024 * 1024)
            print(f"- {doc.get('filename', 'unknown')} ({size_mb:.1f}MB)")
        
        return 0
    
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1


async def handle_batch(args) -> int:
    """处理批量命令"""
    # 读取URL列表
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误：文件 {args.file} 不存在", file=sys.stderr)
        return 1
    
    if not urls:
        print("错误：URL列表为空", file=sys.stderr)
        return 1
    
    toolkit = SpiderToolkit()
    
    try:
        results = await toolkit.batch_crawl(
            urls, 
            crawl_type=args.type,
            max_concurrent=args.concurrent
        )
        
        print(f"批量处理完成:")
        print(f"- 总计URL: {results['total_urls']}")
        print(f"- 成功: {results['success_count']}")
        print(f"- 失败: {results['failed_count']}")
        print(f"- 总计项目: {results['total_items']}")
        
        if args.output:
            output_data(results, args.output, args.format)
        
        return 0
    
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1


def handle_config(args) -> int:
    """处理配置命令"""
    if args.config_action == 'get':
        value = config.get(args.key)
        if value is not None:
            print(f"{args.key} = {value}")
        else:
            print(f"配置项 {args.key} 不存在")
        return 0
    
    elif args.config_action == 'set':
        # 尝试解析值的类型
        value = args.value
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)
        
        config.set(args.key, value)
        print(f"已设置 {args.key} = {value}")
        return 0
    
    elif args.config_action == 'list':
        print("当前配置:")
        print(json.dumps(config.config_data, indent=2, ensure_ascii=False))
        return 0
    
    else:
        print("错误：未知的配置操作", file=sys.stderr)
        return 1


def output_data(data, filepath: str, format_type: str):
    """输出数据到文件"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif format_type == 'csv':
        import csv
        if isinstance(data, list) and data:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                if isinstance(data[0], dict):
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                else:
                    writer = csv.writer(f)
                    for item in data:
                        writer.writerow([item])
    
    print(f"结果已保存到: {filepath}")


def setup_config(args):
    """设置配置"""
    if args.config:
        config.config_file = args.config
        config.config_data = config._load_config()
    
    # 应用命令行参数
    if args.timeout:
        config.set('request.timeout', args.timeout)
    if args.retries:
        config.set('request.max_retries', args.retries)
    if args.delay:
        config.set('request.delay_range', args.delay)
    
    # 设置代理
    proxy_list = []
    if args.proxy:
        proxy_list.extend(args.proxy)
    if args.proxy_file:
        try:
            with open(args.proxy_file, 'r') as f:
                proxy_list.extend([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            print(f"警告：代理文件 {args.proxy_file} 不存在")
    
    if proxy_list:
        config.set('proxy.enabled', True)
        config.set('proxy.proxy_list', proxy_list)


async def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        log_level = 'DEBUG'
    elif args.quiet:
        log_level = 'ERROR'
    else:
        log_level = 'INFO'
    
    setup_logging(log_level)
    
    # 设置配置
    setup_config(args)
    
    # 处理命令
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == 'comments':
            return await handle_comments(args)
        elif args.command == 'videos':
            return await handle_videos(args)
        elif args.command == 'images':
            return await handle_images(args)
        elif args.command == 'documents':
            return await handle_documents(args)
        elif args.command == 'batch':
            return await handle_batch(args)
        elif args.command == 'config':
            return handle_config(args)
        else:
            print(f"错误：未知命令 {args.command}", file=sys.stderr)
            return 1
    
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        return 130
    except Exception as e:
        print(f"未预期的错误：{e}", file=sys.stderr)
        return 1


def cli_main():
    """CLI入口点"""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    cli_main()