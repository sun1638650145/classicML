# FQA: 问题解答

## 1.我需要同时安装Python版和CC版吗?

不需要，Python版和CC版的API调用完全一致。Python版的兼容性更佳但是性能不如CC版，如果只是简单的使用那么你并不需要使用CC版本.

## 2.如何在搭载Apple M1芯片的Mac上使用classicML?

classicML的设计和构建依赖于Numpy，受限于现在 Numpy 没有原生支持Apple M1，Numpy支持后，classicML将在第一时间进行适配。但是，目前你可以使用如下的方式使用classicML

```shell
arch -x86_64 $SHELL  # 使用Rosetta 2转译成x86_64
arch  # 如果输出i386则表示转译成功
```

然后参照[安装页](https://classicml.readthedocs.io/zh_CN/latest/install.html)进行安装

