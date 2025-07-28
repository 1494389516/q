#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描工具2.5 - 图形界面版本
集成JSFinder功能的综合扫描工具
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import requests
from bs4 import BeautifulSoup
import urllib.parse
import re
import nmap
import whois
import sublist3r
import dns.resolver
import ssl
from cryptography import x509
from cryptography.hazmat.backends import default_backend
import datetime
import subprocess
import time
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning

# 禁用SSL警告
disable_warnings(InsecureRequestWarning)

class ScanToolGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("扫描工具2.5 - 图形界面版")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # 设置图标（如果有的话）
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        self.setup_ui()
        self.current_function = None
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧功能选择框架
        left_frame = ttk.Frame(main_frame, width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # 标题
        title_label = ttk.Label(left_frame, text="扫描工具2.5", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # 功能选择标签
        ttk.Label(left_frame, text="选择扫描功能:", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        # 功能按钮
        self.create_function_buttons(left_frame)
        
        # 创建右侧输入和结果框架
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 输入框架
        input_frame = ttk.LabelFrame(right_frame, text="输入参数", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # URL输入
        ttk.Label(input_frame, text="目标URL/IP/域名:").pack(anchor=tk.W)
        self.url_entry = ttk.Entry(input_frame, width=80, font=("Consolas", 10))
        self.url_entry.pack(fill=tk.X, pady=(5, 10))
        
        # Cookie输入
        ttk.Label(input_frame, text="Cookie (可选):").pack(anchor=tk.W)
        self.cookie_entry = ttk.Entry(input_frame, width=80, font=("Consolas", 10))
        self.cookie_entry.pack(fill=tk.X, pady=(5, 10))
        
        # 其他参数输入
        ttk.Label(input_frame, text="其他参数 (可选):").pack(anchor=tk.W)
        self.params_entry = ttk.Entry(input_frame, width=80, font=("Consolas", 10))
        self.params_entry.pack(fill=tk.X, pady=(5, 10))
        
        # 按钮框架
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 执行按钮
        self.execute_btn = ttk.Button(button_frame, text="执行扫描", command=self.execute_scan)
        self.execute_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 停止按钮
        self.stop_btn = ttk.Button(button_frame, text="停止扫描", command=self.stop_scan, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 清空按钮
        clear_btn = ttk.Button(button_frame, text="清空结果", command=self.clear_results)
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 保存按钮
        save_btn = ttk.Button(button_frame, text="保存结果", command=self.save_results)
        save_btn.pack(side=tk.LEFT)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(right_frame, text="扫描结果", padding=5)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 结果文本框
        self.result_text = scrolledtext.ScrolledText(
            result_frame, 
            wrap=tk.WORD, 
            font=("Consolas", 10),
            bg="#1e1e1e",
            fg="#ffffff",
            insertbackground="#ffffff"
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 进度条
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪 - 请选择扫描功能")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))
        
        # 扫描控制
        self.scan_thread = None
        self.stop_scanning = False
    
    def create_function_buttons(self, parent):
        """创建功能按钮"""
        functions = [
            ("🔍 JSFinder扫描", "jsfinder", "#4CAF50"),
            ("🔌 端口扫描", "port_scan", "#2196F3"),
            ("📋 WHOIS查询", "whois", "#FF9800"),
            ("🌐 Shodan查询", "shodan", "#9C27B0"),
            ("🔍 Nmap扫描", "nmap", "#607D8B"),
            ("📄 获取网页", "webpage", "#795548"),
            ("🌳 子域名枚举", "subdomain", "#009688"),
            ("🔍 DNS查询", "dns", "#3F51B5"),
            ("🛡️ Web安全测试", "websec", "#F44336"),
            ("📊 信息收集", "harvest", "#E91E63"),
            ("🔒 SSL证书", "ssl", "#00BCD4"),
            ("🕷️ 网站爬取", "crawl", "#8BC34A"),
            ("⚡ 手工测试", "manual", "#FFC107"),
            ("🔧 设置代理", "proxy", "#9E9E9E")
        ]
        
        for text, func, color in functions:
            btn = ttk.Button(
                parent, 
                text=text, 
                width=20,
                command=lambda f=func: self.select_function(f)
            )
            btn.pack(pady=3, fill=tk.X)
    
    def select_function(self, function):
        """选择功能"""
        self.current_function = function
        self.status_var.set(f"已选择: {function}")
        
        # 清空输入框并设置默认值
        self.url_entry.delete(0, tk.END)
        self.cookie_entry.delete(0, tk.END)
        self.params_entry.delete(0, tk.END)
        
        # 根据功能类型调整界面
        if function == "jsfinder":
            self.url_entry.insert(0, "baidu.com")
            messagebox.showinfo("JSFinder", "JSFinder功能可以从网页中提取URL和子域名\n支持输入域名或完整URL")
        elif function == "port_scan" or function == "nmap":
            self.url_entry.insert(0, "127.0.0.1")
        elif function == "whois":
            self.url_entry.insert(0, "baidu.com")
        elif function == "ssl":
            self.url_entry.insert(0, "https://www.baidu.com")
        else:
            self.url_entry.insert(0, "www.baidu.com")
    
    def execute_scan(self):
        """执行扫描"""
        if not self.current_function:
            messagebox.showwarning("警告", "请先选择一个扫描功能")
            return
        
        target = self.url_entry.get().strip()
        if not target:
            messagebox.showwarning("警告", "请输入目标URL/IP/域名")
            return
        
        # 清空结果显示
        self.result_text.delete(1.0, tk.END)
        self.status_var.set("正在扫描...")
        self.execute_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress.start()
        self.stop_scanning = False
        
        # 在新线程中执行扫描
        self.scan_thread = threading.Thread(target=self.run_scan, args=(target,))
        self.scan_thread.daemon = True
        self.scan_thread.start()
    
    def stop_scan(self):
        """停止扫描"""
        self.stop_scanning = True
        self.status_var.set("正在停止扫描...")
    
    def clear_results(self):
        """清空结果"""
        self.result_text.delete(1.0, tk.END)
        self.status_var.set("结果已清空")
    
    def save_results(self):
        """保存结果"""
        content = self.result_text.get(1.0, tk.END)
        if not content.strip():
            messagebox.showwarning("警告", "没有结果可保存")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("成功", f"结果已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def run_scan(self, target):
        """在后台线程中运行扫描"""
        try:
            if self.current_function == "jsfinder":
                cookie = self.cookie_entry.get().strip() or None
                results = self.js_finder_scan(target, cookie)
                
                if results and not self.stop_scanning:
                    output = f"🔍 JSFinder扫描结果\n{'='*50}\n\n"
                    output += f"📊 找到 {len(results['urls'])} 个URL:\n"
                    for i, url in enumerate(results['urls'], 1):
                        if self.stop_scanning:
                            break
                        output += f"  {i:3d}. {url}\n"
                    
                    output += f"\n🌐 找到 {len(results['subdomains'])} 个子域名:\n"
                    for i, subdomain in enumerate(results['subdomains'], 1):
                        if self.stop_scanning:
                            break
                        output += f"  {i:3d}. {subdomain}\n"
                else:
                    output = "❌ JSFinder扫描未返回结果或发生错误"
            
            elif self.current_function == "port_scan":
                results = self.scan_ports(target)
                if results and not self.stop_scanning:
                    output = f"🔌 端口扫描结果\n{'='*50}\n\n"
                    for item in results:
                        if self.stop_scanning:
                            break
                        output += f"✅ {item}\n"
                else:
                    output = "❌ 端口扫描未返回结果或发生错误"
            
            elif self.current_function == "whois":
                results = self.query_whois(target)
                if results and not self.stop_scanning:
                    output = f"📋 WHOIS查询结果\n{'='*50}\n\n"
                    for key, value in results.items():
                        if self.stop_scanning:
                            break
                        output += f"📌 {key}: {value}\n"
                else:
                    output = "❌ WHOIS查询未返回结果或发生错误"
            
            elif self.current_function == "subdomain":
                results = self.subdomain_enum(target)
                if results and not self.stop_scanning:
                    output = f"🌳 子域名枚举结果\n{'='*50}\n\n"
                    for i, subdomain in enumerate(results, 1):
                        if self.stop_scanning:
                            break
                        output += f"  {i:3d}. {subdomain}\n"
                else:
                    output = "❌ 子域名枚举未返回结果或发生错误"
            
            elif self.current_function == "ssl":
                results = self.get_ssl_info(target)
                if results and not self.stop_scanning:
                    output = f"🔒 SSL证书信息\n{'='*50}\n\n"
                    for key, value in results.items():
                        if self.stop_scanning:
                            break
                        output += f"🔐 {key}: {value}\n"
                else:
                    output = "❌ SSL证书查询未返回结果或发生错误"
            
            elif self.current_function == "dns":
                results = self.get_dns_records(target)
                if results and not self.stop_scanning:
                    output = f"🔍 DNS查询结果\n{'='*50}\n\n"
                    for key, value in results.items():
                        if self.stop_scanning:
                            break
                        output += f"📍 {key}: {value}\n"
                else:
                    output = "❌ DNS查询未返回结果或发生错误"
            
            elif self.current_function == "webpage":
                results = self.get_web_page(target)
                if results and not self.stop_scanning:
                    output = f"📄 网页内容 (前2000字符)\n{'='*50}\n\n{results[:2000]}..."
                else:
                    output = "❌ 无法获取网页内容"
            
            else:
                output = f"⚠️ 功能 {self.current_function} 正在开发中..."
            
            if self.stop_scanning:
                output += "\n\n⏹️ 扫描已被用户停止"
            
            # 在主线程中更新UI
            self.root.after(0, self.update_result, output)
            
        except Exception as e:
            error_msg = f"❌ 扫描过程中发生错误: {str(e)}"
            self.root.after(0, self.update_result, error_msg)
    
    def update_result(self, output):
        """更新结果显示"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, output)
        self.status_var.set("扫描完成" if not self.stop_scanning else "扫描已停止")
        self.execute_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress.stop()
    
    def append_result(self, text):
        """追加结果到显示区域"""
        self.result_text.insert(tk.END, text + "\n")
        self.result_text.see(tk.END)
        self.root.update_idletasks()
    
    # 扫描功能实现
    def js_finder_scan(self, target, cookie=None):
        """JSFinder扫描功能"""
        try:
            # 处理输入的目标，确保是完整的URL
            if not target.startswith(('http://', 'https://')):
                target = f"https://{target}"
            
            self.append_result(f"[*] 正在扫描目标: {target}")
            
            # 获取网页源码
            html_content = self.get_webpage_content(target, cookie)
            if not html_content:
                self.append_result(f"[-] 无法访问 {target}")
                return None
            
            self.append_result(f"[+] 成功获取网页内容，长度: {len(html_content)} 字符")
            
            # 解析HTML并提取所有脚本内容
            all_scripts = self.extract_all_scripts(target, html_content, cookie)
            self.append_result(f"[+] 总共处理了 {len(all_scripts)} 个脚本源")
            
            # 从所有脚本中提取URL
            all_urls = []
            for script_info in all_scripts:
                if self.stop_scanning:
                    break
                urls = self.extract_urls_from_script(script_info['content'])
                if urls:
                    self.append_result(f"[+] 从 {script_info['source']} 中提取到 {len(urls)} 个URL")
                    for url in urls:
                        processed_url = self.normalize_url(target, url)
                        if processed_url:
                            all_urls.append(processed_url)
            
            self.append_result(f"[+] 总共提取到 {len(all_urls)} 个URL")
            
            # 过滤和分类URL
            filtered_results = self.filter_and_classify_urls(target, all_urls)
            
            self.append_result(f"[+] 过滤后保留 {len(filtered_results['urls'])} 个相关URL")
            self.append_result(f"[+] 发现 {len(filtered_results['subdomains'])} 个子域名")
            
            return filtered_results
            
        except Exception as e:
            self.append_result(f"[-] JSFinder扫描错误: {str(e)}")
            return None
    
    def get_webpage_content(self, url, cookie=None):
        """获取网页内容"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        if cookie:
            headers["Cookie"] = cookie
        
        try:
            response = requests.get(url, headers=headers, timeout=15, verify=False, allow_redirects=True)
            response.raise_for_status()
            
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding
            
            return response.text
            
        except requests.exceptions.RequestException as e:
            self.append_result(f"[-] 请求失败: {str(e)}")
            return None
    
    def extract_all_scripts(self, base_url, html_content, cookie=None):
        """提取所有脚本内容"""
        scripts = []
        
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            script_tags = soup.find_all("script")
            
            # 处理内联脚本
            inline_content = ""
            for script in script_tags:
                if not script.get("src"):
                    content = script.get_text()
                    if content.strip():
                        inline_content += content + "\n"
            
            if inline_content.strip():
                scripts.append({
                    'source': base_url + " (inline)",
                    'content': inline_content
                })
            
            # 处理外部脚本
            for script in script_tags:
                if self.stop_scanning:
                    break
                src = script.get("src")
                if src:
                    external_url = self.normalize_url(base_url, src)
                    if external_url:
                        external_content = self.get_webpage_content(external_url, cookie)
                        if external_content:
                            scripts.append({
                                'source': external_url,
                                'content': external_content
                            })
        
        except Exception as e:
            self.append_result(f"[-] 脚本提取错误: {str(e)}")
        
        return scripts
    
    def extract_urls_from_script(self, script_content):
        """从脚本内容中提取URL"""
        if not script_content:
            return []
        
        urls = []
        
        patterns = [
            r'(?:"|\'|`)(https?://[^"\'`\s<>]+)(?:"|\'|`)',
            r'(?:"|\'|`)(/[^"\'`\s<>]*\.(?:php|asp|aspx|jsp|html|htm|js|css|json|xml|txt|action))(?:"|\'|`)',
            r'(?:"|\'|`)(/api/[^"\'`\s<>]+)(?:"|\'|`)',
            r'(?:"|\'|`)(/[a-zA-Z0-9_\-/]+/[a-zA-Z0-9_\-/]*\.?[a-zA-Z0-9]*(?:\?[^"\'`\s<>]*)?)(?:"|\'|`)',
            r'(?:"|\'|`)([a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?:"|\'|`)',
        ]
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, script_content, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else (match[1] if len(match) > 1 else "")
                    
                    if match and len(match) > 1:
                        if not any(blackword in match.lower() for blackword in ['javascript:', 'data:', 'mailto:', 'tel:']):
                            urls.append(match)
            except Exception:
                continue
        
        return list(set(urls))
    
    def normalize_url(self, base_url, relative_url):
        """标准化URL"""
        try:
            if not relative_url:
                return None
            
            if relative_url.startswith(('http://', 'https://')):
                return relative_url
            
            if relative_url.startswith('//'):
                base_parsed = urllib.parse.urlparse(base_url)
                return f"{base_parsed.scheme}:{relative_url}"
            
            if relative_url.startswith('/'):
                base_parsed = urllib.parse.urlparse(base_url)
                return f"{base_parsed.scheme}://{base_parsed.netloc}{relative_url}"
            
            return urllib.parse.urljoin(base_url, relative_url)
            
        except Exception:
            return None
    
    def filter_and_classify_urls(self, base_url, urls):
        """过滤和分类URL"""
        try:
            base_parsed = urllib.parse.urlparse(base_url)
            base_domain = base_parsed.netloc.lower()
            
            domain_parts = base_domain.split('.')
            if len(domain_parts) >= 2:
                main_domain = '.'.join(domain_parts[-2:])
            else:
                main_domain = base_domain
            
            filtered_urls = []
            subdomains = set()
            
            for url in urls:
                try:
                    parsed = urllib.parse.urlparse(url)
                    url_domain = parsed.netloc.lower()
                    
                    if main_domain in url_domain or not url_domain:
                        if url not in filtered_urls:
                            filtered_urls.append(url)
                        
                        if url_domain and main_domain in url_domain:
                            subdomains.add(url_domain)
                            
                except Exception:
                    continue
            
            return {
                'urls': filtered_urls,
                'subdomains': sorted(list(subdomains))
            }
            
        except Exception as e:
            self.append_result(f"[-] URL过滤错误: {str(e)}")
            return {'urls': [], 'subdomains': []}
    
    def scan_ports(self, target):
        """端口扫描"""
        try:
            nm = nmap.PortScanner()
            self.append_result(f"[*] 正在扫描 {target} 的端口...")
            nm.scan(target, arguments='-sS')
            
            results = []
            for host in nm.all_hosts():
                if self.stop_scanning:
                    break
                for proto in nm[host].all_protocols():
                    ports = nm[host][proto].keys()
                    for port in ports:
                        if self.stop_scanning:
                            break
                        service = nm[host][proto][port]['name']
                        state = nm[host][proto][port]['state']
                        results.append(f"{host} 的 {proto.upper()}端口 {port} {state} - {service}")
            return results
        except Exception as e:
            self.append_result(f"[-] 端口扫描错误: {str(e)}")
            return None
    
    def query_whois(self, domain):
        """WHOIS查询"""
        try:
            self.append_result(f"[*] 正在查询 {domain} 的WHOIS信息...")
            w = whois.whois(domain)
            return {
                '域名': w.domain,
                '注册商': w.registrar,
                '创建日期': w.creation_date,
                '过期日期': w.expiration_date,
                '名称服务器': w.name_servers,
                '状态': w.status
            }
        except Exception as e:
            self.append_result(f"[-] WHOIS查询错误: {str(e)}")
            return None
    
    def subdomain_enum(self, domain):
        """子域名枚举"""
        try:
            self.append_result(f"[*] 正在枚举 {domain} 的子域名...")
            subdomains = sublist3r.main(domain, 40, None, ports=None, silent=True, verbose=False, enable_bruteforce=False, engines=None)
            return subdomains
        except Exception as e:
            self.append_result(f"[-] 子域名枚举错误: {str(e)}")
            return None
    
    def get_ssl_info(self, url):
        """获取SSL证书信息"""
        try:
            hostname = url.replace('https://', '').replace('http://', '').split('/')[0]
            self.append_result(f"[*] 正在获取 {hostname} 的SSL证书信息...")
            cert = ssl.get_server_certificate((hostname, 443))
            cert_obj = x509.load_pem_x509_certificate(cert.encode(), default_backend())
            
            issuer = cert_obj.issuer.rfc4514_string()
            subject = cert_obj.subject.rfc4514_string()
            valid_from = cert_obj.not_valid_before
            valid_to = cert_obj.not_valid_after
            
            return {
                '颁发机构': issuer,
                '主体': subject,
                '有效期从': valid_from.strftime('%Y-%m-%d'),
                '有效期至': valid_to.strftime('%Y-%m-%d')
            }
        except Exception as e:
            self.append_result(f"[-] SSL证书查询错误: {str(e)}")
            return None
    
    def get_dns_records(self, domain):
        """获取DNS记录"""
        try:
            self.append_result(f"[*] 正在查询 {domain} 的DNS记录...")
            records = {}
            
            # A记录
            try:
                a_records = dns.resolver.resolve(domain, 'A')
                records['A记录'] = [str(r) for r in a_records]
            except:
                records['A记录'] = ['查询失败']
            
            # MX记录
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                records['MX记录'] = [str(r.exchange) for r in mx_records]
            except:
                records['MX记录'] = ['查询失败']
            
            # NS记录
            try:
                ns_records = dns.resolver.resolve(domain, 'NS')
                records['NS记录'] = [str(r) for r in ns_records]
            except:
                records['NS记录'] = ['查询失败']
            
            return records
        except Exception as e:
            self.append_result(f"[-] DNS查询错误: {str(e)}")
            return None
    
    def get_web_page(self, url):
        """获取网页内容"""
        try:
            self.append_result(f"[*] 正在获取 {url} 的网页内容...")
            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.append_result(f"[-] 获取网页内容错误: {str(e)}")
            return None

def main():
    """主函数"""
    root = tk.Tk()
    app = ScanToolGUI(root)
    
    # 设置窗口居中
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()