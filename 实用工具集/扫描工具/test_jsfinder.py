#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSFinder功能测试脚本
"""

import sys
import importlib.util

def test_jsfinder():
    """测试JSFinder功能"""
    try:
        # 导入扫描工具模块
        spec = importlib.util.spec_from_file_location("scan_tool", "扫描工具2.5.py")
        scan_tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scan_tool)
        
        # 测试多种输入格式
        test_cases = [
            "baidu.com",  # 只有域名
            "www.baidu.com",  # 带www的域名
            "https://www.baidu.com",  # 完整URL
        ]
        
        for test_target in test_cases:
            print(f"\n测试目标: {test_target}")
            print("=" * 50)
            
            # 执行JSFinder扫描
            results = scan_tool.js_finder_scan(test_target, None, False)
            
            if results:
                print(f"✅ 扫描成功!")
                print(f"找到 {len(results['urls'])} 个URL:")
                for i, url in enumerate(results['urls'][:5], 1):  # 只显示前5个
                    print(f"  {i}. {url}")
                if len(results['urls']) > 5:
                    print(f"  ... 还有 {len(results['urls']) - 5} 个URL")
                
                print(f"\n找到 {len(results['subdomains'])} 个子域名:")
                for i, subdomain in enumerate(results['subdomains'][:10], 1):  # 只显示前10个
                    print(f"  {i}. {subdomain}")
                if len(results['subdomains']) > 10:
                    print(f"  ... 还有 {len(results['subdomains']) - 10} 个子域名")
            else:
                print("❌ 扫描失败或无结果")
            
            print("\n" + "-" * 50)
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_jsfinder()