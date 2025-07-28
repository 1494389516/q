@echo off
chcp 65001
echo 扫描工具2.5 - 自动打包程序
echo ================================
echo.

echo 正在检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: 未找到Python环境
    pause
    exit /b 1
)

echo.
echo 开始打包过程...
python build_exe.py

echo.
echo 打包完成！按任意键退出...
pause