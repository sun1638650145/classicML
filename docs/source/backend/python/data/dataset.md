# dataset

classicML中的用来组织数据集模块.

## Dataset

数据集, 数据集提供了对输入数据的预处理和封装的功能, 使之满足cml模型输入的需要.

```python
cml.data.Dataset(dataset_type='train',
                 label_mode=None,
                 fillna=True,
                 digitization=False,
                 normalization=False,
                 standardization=False,
                 name=None)
```

### 参数

* <b>dataset_type</b>: {'train', 'validation', 'test'}，数据集的类型，如果声明为测试集，将不会生成对应的标签.
* <b>label_mode</b>: {'one-hot', 'max-margin'}，标签的编码格式.
* <b>fillna</b>: 布尔值，是否填充缺失值.
* <b>digitization</b>: 布尔值，是否使用数值化，将离散标签转化成数值.
* <b>normalization</b>: 布尔值，是否使用归一化.
* <b>standardization</b>: 布尔值，是否使用标准化.
* <b>name</b>: 字符串，数据集的名称.

### from_dataframe

```python
from_dataframe(dataframe)
```

从DataFrame中加载数据集.

#### 参数

* <b>dataframe</b>: pandas的DataFrame，原始的数据.

#### 返回

 经过预处理的特征数据和标签.

### from_csv

```python
from_csv(filepath, sep=',')
```

从CSV文件中加载数据集, 也可以从其他的结构化文本读入数据, 例如TSV等.

#### 参数

* <b>filepath</b>: 字符串，CSV文件的路径.
* <b>sep</b>: 字符串, 使用的文本分隔符.

#### 返回

 经过预处理的特征数据和标签.

### from_tensor_slices

```python
from_tensor_slices(x, y=None)
```

从张量流加载数据集.

#### 参数

* <b>filepath</b>: 字符串，CSV文件的路径.
* <b>x</b>: 一个Numpy数组，处理后的特征数据.
* <b>y</b>: 一个Numpy数组，处理后的标签.

#### 返回

 经过预处理的特征数据和标签.