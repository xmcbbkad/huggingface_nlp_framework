## Demo for Token Classification

部署demo。

### 输入
<table>
   <tr>
        <th >参数名称</th>
        <th>数据类型</th>
        <th>重要性</th>
        <th width=60%>说明</th>
   </tr>
   <tr>
        <td>[*].raw_str</td>
        <td>str</td>
        <td>必填</td>
        <td>输入的字符串</td>
   </tr>
</table>
<br/>

输入示例：
```json
[
    {"raw_str": "SOCCER-JAPAN GET LUCKY WIN, CHINA IN SURPRISE DEFEAT."},
    {"raw_str": "Nadim Ladki"},
    {"raw_str": "AL-AIN , United Arab Emirates 1996-12-06"}
]
```

### 输出
<table>
    <tr>
        <th >参数名称</th>
        <th>数据类型</th>
        <th width=60%>说明</th>
    </tr>
    <tr>
        <td>[*].raw_str</td>
        <td>str</td>
        <td>输入的字符串</td>
    </tr>
    <tr>
        <td>[*].ne</td>
        <td>list[dict]</td>
        <td>输出的结果信息</td>
    </tr>
    <tr>
        <td>[*].ne[*].tag</td>
        <td>str</td>
        <td>tokens对应的标签</td>
    </tr>
    <tr>
        <td>[*].ne[*].offset</td>
        <td>int</td>
        <td>tokens对应在原始字符串中的起点位置</td>
    </tr>
    <tr>
        <td>[*].ne[*].length</td>
        <td>int</td>
        <td>tokens对应在原始字符串中的长度</td>
    </tr>
    <tr>
        <td>[*].ne[*].text</td>
        <td>str</td>
        <td>tokens对应在原始字符串中字符</td>
    </tr>
    <tr>
        <td>[*].ne[*].score</td>
        <td>str</td>
        <td>模型预测的置信度</td>
    </tr>
</table>
<br/>

输出示例：
```json
[
    {
        "raw_str": "SOCCER-JAPAN GET LUCKY WIN, CHINA IN SURPRISE DEFEAT.",
        "ne": [
            {
                "tag": "LOC",
                "offset": 7,
                "length": 5,
                "text": "JAPAN",
                "score": 0.99667627
            },
            {
                "tag": "LOC",
                "offset": 28,
                "length": 5,
                "text": "CHINA",
                "score": 0.99650466
            }
        ]
    },
    {
        "raw_str": "Nadim Ladki",
        "ne": [
            {
                "tag": "PER",
                "offset": 0,
                "length": 11,
                "text": "Nadim Ladki",
                "score": 0.9466292
            }
        ]
    },
    {
        "raw_str": "AL-AIN, United Arab Emirates 1996-12-06",
        "ne": [
            {
                "tag": "LOC",
                "offset": 0,
                "length": 6,
                "text": "AL-AIN",
                "score": 0.99103636
            },
            {
                "tag": "LOC",
                "offset": 9,
                "length": 20,
                "text": "United Arab Emirates",
                "score": 0.9896286
            }
        ]
    }
]

