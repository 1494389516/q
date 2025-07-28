#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打包脚本 - 将扫描工具打包成exe文件
"""

import os
import sys
import subprocess

def install_requirements():
    """安装必要的依赖"""
    requirements = [
        "pyinstaller",
        "requests",
        "beautifulsoup4",
        "python-nmap",
        "python-whois", 
        "sublist3r",
        "dnspython",
        "cryptography",
        "urllib3"
    ]
    
    print("正在安装依赖包...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ {req} 安装成功")
        except subprocess.CalledProcessError:
            print(f"❌ {req} 安装失败")

def create_spec_file():
    """创建PyInstaller配置文件"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['扫描工具GUI版.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.scrolledtext',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'requests',
        'bs4',
        'urllib.parse',
        'nmap',
        'whois',
        'sublist3r',
        'dns.resolver',
        'ssl',
        'cryptography',
        'cryptography.hazmat.backends.default_backend',
        'cryptography.x509'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='扫描工具2.5',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)
'''
    
    with open('scan_tool.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print("✅ 配置文件创建成功")

def build_exe():
    """构建exe文件"""
    print("开始构建exe文件...")
    
    try:
        # 使用spec文件构建
        subprocess.check_call([
            sys.executable, "-m", "PyInstaller", 
            "--clean",
            "scan_tool.spec"
        ])
        print("✅ exe文件构建成功！")
        print("📁 输出目录: dist/扫描工具2.5.exe")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 构建失败: {e}")
        
        # 尝试简单构建
        print("尝试简单构建...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "PyInstaller",
                "--onefile",
                "--windowed",
                "--name=扫描工具2.5",
                "--add-data=icon.ico;." if os.path.exists('icon.ico') else "",
                "扫描工具GUI版.py"
            ])
            print("✅ 简单构建成功！")
        except subprocess.CalledProcessError as e2:
            print(f"❌ 简单构建也失败: {e2}")

def create_icon():
    """创建简单的图标文件"""
    # 这里可以放置图标创建代码，或者提示用户放置icon.ico文件
    if not os.path.exists('icon.ico'):
        print("💡 提示: 可以在当前目录放置 icon.ico 文件作为程序图标")

def main():
    """主函数"""
    print("🚀 扫描工具2.5 打包程序")
    print("=" * 50)
    
    # 检查源文件
    if not os.path.exists('扫描工具GUI版.py'):
        print("❌ 找不到源文件: 扫描工具GUI版.py")
        return
    
    # 安装依赖
    install_requirements()
    
    # 创建图标
    create_icon()
    
    # 创建配置文件
    create_spec_file()
    
    # 构建exe
    build_exe()
    
    print("\n🎉 打包完成！")
    print("📝 使用说明:")
    print("1. exe文件位于 dist 目录中")
    print("2. 双击运行即可启动图形界面")
    print("3. 首次运行可能需要一些时间加载")

if __name__ == "__main__":
    main()