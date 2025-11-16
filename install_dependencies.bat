@echo off
REM 安装论文实验框架所需的依赖 (Windows版本)

echo ==============================================
echo   安装论文实验框架依赖
echo ==============================================
echo.

echo 正在安装依赖...
echo.

REM 更新 pip
python -m pip install --upgrade pip

REM 安装可视化相关的依赖（实验框架必需）
echo ^> 安装可视化库...
pip install seaborn==0.12.2
pip install matplotlib==3.7.5
pip install pandas==2.0.3

REM 安装深度学习相关依赖
echo ^> 安装深度学习框架...
pip install torch==2.4.1 torchvision==0.19.1
pip install torch-geometric==2.6.1

REM 安装其他必需依赖
echo ^> 安装其他工具库...
pip install tqdm==4.67.1
pip install scikit-learn==1.3.2
pip install pillow==10.4.0
pip install numpy==1.24.3

echo.
echo ==============================================
echo   依赖安装完成！
echo ==============================================
echo.
echo 现在可以运行实验了：
echo   python run_paper_experiments.py
echo.

pause

