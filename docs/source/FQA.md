# FQA: 问题解答

## 1.我需要同时安装Python后端和CC后端吗?

不需要，Python后端和CC后端的API调用完全一致。Python后端的兼容性更佳但是性能不如CC后端，如果只是简单的使用那么你并不需要使用CC后端.

## 2.如何在搭载Apple M1芯片的Mac上使用classicML?

目前，classicML提供了一个M1预览版(只支持Python 3.8版本)，您可以通过pip软件包管理器直接安装.

```shell
# 安装python版
pip install classicML-python==0.6b2
# 或者直接安装标准版
pip install classicML==0.6b2
```

## 3.我如何才能联系到作者？

首先，你可以只直接在[Github Issues](https://github.com/sun1638650145/classicML/issues)上进行提问；然后你也可以通过[作者邮箱](s1638650145@gmail.com)进行联系，作者看到后将会立即回复你的.



