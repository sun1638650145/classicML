# FQA: 问题解答

## 1.我需要同时安装Python后端和CC后端吗?

不需要，Python后端和CC后端的API调用完全一致。Python后端的兼容性更佳但是性能不如CC后端，如果只是简单的入门学习那么你并不需要使用CC后端.

## 2.如何在搭载Apple M1芯片的Mac上使用classicML?

目前，classicML已经提供了支持Apple M1的正式版，您可以通过pip软件包管理器直接安装.

```shell
# 安装python版
pip install classicML-python
# 或者直接安装标准版
pip install classicML
```

## 3.我如何才能联系到作者？

首先，你可以直接在[Github Issues](https://github.com/sun1638650145/classicML/issues)上进行提问；然后你也可以通过[作者邮箱](s1638650145@gmail.com)进行联系，作者看到后将会立即回复你的.

## 4.为什么我使用可视化工具，中文都是显示的小方框？

根据现有的开发经验，应该是您的系统字库中没有 classicML 的默认 Unicode 字体导致的. 目前提供了两个解决方案:

### 方案1

classicML 拥有一个控制可视化工具字体的环境变量`CLASSICML_FONT`，您可以直接修改环境变量的值选择您系统字库中存在的中文字体.

### 方案2

您可以通过安装 classicML 的默认字体，以此实现显示中文信息. 字体的下载链接在[资源](https://classicml.readthedocs.io/zh_CN/latest/resources.html)页面提供. 您需要按照以下三步操作即可(以类Unix为例)

```shell
# 步骤1(添加字库)
cp -p /path/to/ArialUnicodeMS.ttf /path/to/lib/python3.8/site-packages/matplotlib/mpl-data/fonts
cp -p /path/to/ArialUnicodeMS.ttf /usr/share/fonts
# 步骤2(注册字库信息)
vim /path/to/lib/python3.8/site-packages/matplotlib/mpl-data/matplotlibrc
# 修改第250行, 增加新的字库
font.family: ArialUnicodeMS, sans-seri
# 步骤3(刷新字库缓存)
rm -rf ~/.cache/matplotlib
```
