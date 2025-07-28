#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描工具2.3 - 启动脚本
集成了JSFinder功能的综合扫描工具
"""

import sys
import os

def main():
    print("=" * 50)
    print("    扫描工具2.3 - JSFinder集成版")
    print("=" * 50)
    print()
    print("选择启动模式:")
    print("1. 命令行模式 (完整功能)")
    print("2. 图形界面模式 (推荐)")
    print("3. 直接JSFinder扫描")
    print("0. 退出")
    print()
    
    while True:
        choice = input("请选择模式 (0-3): ").strip()
        
        if choice == "0":
            print("程序已退出")
            sys.exit(0)
        elif choice == "1":
            print("启动命令行模式...")
            import importlib.util
            spec = importlib.util.spec_from_file_location("scan_tool", "扫描工具2.3.py")
            scan_tool = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scan_tool)
            scan_tool.main()
            break
        elif choice == "2":
            print("启动图形界面模式...")
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("scan_tool", "扫描工具2.3.py")
                scan_tool = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(scan_tool)
                scan_tool.start_gui()
            except ImportError as e:
                print(f"启动图形界面失败: {e}")
                print("请确保已安装tkinter库")
            except Exception as e:
                print(f"启动失败: {e}")
            break
        elif choice == "3":
            print("直接JSFinder扫描模式...")
            url = input("请输入目标URL: ").strip()
            if url:
                cookie = input("请输入Cookie (可选): ").strip() or None
                import importlib.util
                spec = importlib.util.spec_from_file_location("scan_tool", "扫描工具2.3.py")
                scan_tool = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(scan_tool)
                results = scan_tool.js_finder_scan(url, cookie, False)
                if results:
                    print(f"\n找到 {len(results['urls'])} 个URL:")
                    for found_url in results['urls']:
                        print(f"[+] {found_url}")
                    
                    print(f"\n找到 {len(results['subdomains'])} 个子域名:")
                    for subdomain in results['subdomains']:
                        print(f"[+] {subdomain}")
                else:
                    print("扫描未返回结果或发生错误")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()