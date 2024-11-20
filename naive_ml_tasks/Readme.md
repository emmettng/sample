## Structure

> epoqueTraining
>> developname folder  
>>>   project folder
>>>     

#### 模型数据集关系图

|Data sets             |Feature types |0.10               |0.11               | 0.20              |  0.21             |   0.22            | 0.23              |0.30               |
|:--:                  |:--:          |:--:               |:--:               | :--:              |:--:               |:--:               | :--:              | :--:              |
|status                |              |online             | standby           | testing           |testing            | deploying         |developing         |developing         |
|host                  |              |48 / 17            | 48 / 17 / .9      | .3                | .3                | .10               | .10               | .10               |
|v1.1                  |              |                   |                   |                   |                   |                   |                   |                   |
|                      | size         |:heavy_check_mark: |                   |                   |                   |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |
|                      | fitness      |                   |:heavy_check_mark: |                   |:heavy_check_mark: |                   |                   |                   |
|v1.8.1                |              |                   |                   |                   |                   |                   |                   |                   |
|                      | size         |                   |                   |                   |                   |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |
|                      | fitness      |                   |                   |:heavy_check_mark: |:heavy_check_mark: |                   |                   |                   |
|v1.8.2                |              |                   |                   |                   |                   |                   |                   |                   |
|                      | size         |                   |                   |                   |                   |                   |                   |                   |
|                      | fitness      |                   |                   |                   |                   |                   |                   |                   |
|Foot dataset   1/4/5  |              |                   |                   |                   |                   |                   |                   |                   |
|                      | conFeatures  |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:white_check_mark: |:white_check_mark: |:white_check_mark: |
|                      | orderedDisc  |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:white_check_mark: |:white_check_mark: |                   |
|                      | unorderedDisc|:heavy_check_mark: |:heavy_check_mark: |                   |                   |:white_check_mark: |:white_check_mark: |                   |
|Shoe dataset 1/4/5    |              |                   |                   |                   |                   |                   |                   |                   |
|                      | conFeatures  |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |                   |:white_check_mark: |                   |
|                      | orderedDisc  |:heavy_check_mark: |:heavy_check_mark: |                   |                   |                   |:white_check_mark: |                   |
|                      | unorderedDisc|:heavy_check_mark: |:heavy_check_mark: |                   |                   |                   |:white_check_mark: |                   |
|shoe dataset calibrate|              |                   |                   |                   |                   |                   |                   |                   |
|                      | conFeatures  |                   |                   |                   |                   |:white_check_mark: |                   |:white_check_mark: |
|                      | orderedDisc  |                   |                   |                   |                   |:white_check_mark: |                   |                   |
|                      | unorderedDisc|                   |                   |                   |                   |:white_check_mark: |                   |                   |


## Phase Two

|Data sets             |Feature types |0.40                |   0.41            |
|:--:                  |:--:          |:--:                |:--:               |
|status                |              |scheduling          | scheduling        |
|host                  |              |.10                 | .10               |
|v1.1                  |              |                    |                   |
|                      | size         |:heavy_check_mark:  |:heavy_check_mark: |
|                      | fitness      |:heavy_check_mark:  |:heavy_check_mark: |  
|v1.8.1                |              |                    |                   |
|                      | size         |:heavy_check_mark:  |:heavy_check_mark: |
|                      | fitness      |:heavy_check_mark:  |:heavy_check_mark: |  
|v1.8.2                |              |                    |                   |
|                      | size         |:heavy_check_mark:  |:heavy_check_mark: |
|                      | fitness      |:heavy_check_mark:  |:heavy_check_mark: |  
|Foot dataset   1/4/5  |              |                    |                   |
|                      | conFeatures  |:white_check_mark:  |:white_check_mark: |
|                      | orderedDisc  |:white_check_mark:  |:white_check_mark: |
|                      | unorderedDisc|:white_check_mark:  |:white_check_mark: |
|Shoe dataset 1/4/5    |              |                    |                   |
|                      | conFeatures  |:white_check_mark:  |                   |
|                      | orderedDisc  |:white_check_mark:  |                   |
|                      | unorderedDisc|:white_check_mark:  |                   |
|shoe dataset calibrate|              |                    |                   |
|                      | conFeatures  |                    |:white_check_mark: |
|                      | orderedDisc  |                    |:white_check_mark: |
|                      | unorderedDisc|                    |:white_check_mark: |
#### 模型开发标准流程报告：
模型名称：
  - 0. 版本号： 姓名缩写 + 数字编码
  - 1. reponse pair 反馈筛选逻辑
  - 2. 数据集匹配异常信息 （缺失，前后不一致....)
  - 3. 模型结构介绍
      - 输入输出维度
      - 归一化方法
      - 编码方案
      - 特征工程方法
      - 其他
  - 4. 定义域判断模块

例:
```
测试舒适度模型:
   0. wh+0.1
   1. reponse pair 反馈筛选逻辑
      - v1.1 数据集： 总体感觉舒适的报告，至多可以有两成报告无法接受。
   2. 数据集匹配异常信息 （缺失，前后不一致....)
      男鞋缺少sku...见...
      男鞋sku 中缺少需要维度，见...
   3. Preprocess
      data augmentation ,拓展问卷范围，手动添加未试穿的尺码， 按照已知逻辑在大尺码上标记为太松，小尺码上标记为太紧。
      找出足型数据，离散信息中不可靠的部分标记为 classicalFoot.invalidOrderedDisc
   4. 模型结构介绍
      输入输出维度
        classicalFoot.conFeatures + classicalFoot.orderedDisc + classicalShoe.contFeatures
        - classicalFoot.invalidOrderedDisc
      归一化方法
        无
      label 编码方案
        binary 编码，全局感受， 0 为可接受，1为不可接受
      特征工程方法
        无
      其他
   5. 定义域判断模块
        提供了足型，楦型 连续数据的 10%分位，90%分位信息
        提供了足型 离散数据集合
```

#### 服务部署

|Service|host|related model|type|
|:---:|:---:|:---:|:---:|:---:|
|     |10.240.117.17 | |     |     |  
|     |10.240.18.48  | |     |     |
|     |10.240.117.3  | |     |     |
|     |10.240.117.9  | |     |     |
|     |10.240.117.10 | |     |     |
