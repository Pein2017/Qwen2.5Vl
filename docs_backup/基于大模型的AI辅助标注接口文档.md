## 背景
为提升工业领域标注效率，我们基于多模态大模型开发了一套AI辅助标注方案，并需要将其无缝集成到现有标注平台中。

## 核心设计
为支持多样化的标注任务并保持未来的可扩展性，我们设计了一个统一的、任务驱动的API接口。该方案的核心思想是：
1.  **单一入口**: 提供一个统一的API端点 `/v1.0/vision/annotate` 来处理所有视觉相关的辅助标注任务。
2.  **任务推断**: 通过请求中提供的 `prompt_text` 或 `prompt_box` 字段来自动推断任务类型。
3.  **灵活解析**: 提供两种解析方案（后端解析和前端解析），由调用方根据需求选择，兼顾了易用性与灵活性。

## 主接口

### **AI辅助标注**
`POST /v1.0/vision/annotate`

此接口为核心的辅助标注接口，根据指定的任务类型和解析方案，对输入的图片进行分析，并返回标注建议。

#### **请求参数**

| 参数名称           | 参数类型 | 是否必须 | 参数说明                                                                         |
| :----------------- | :------- | :------- | :------------------------------------------------------------------------------- |
| `appId`            | string   | 是       | 分配给调用方的应用ID。                                                           |
| `timestamp`        | string   | 是       | 请求时间戳（毫秒级），用于防止重放攻击。                                         |
| `uuid`             | string   | 是       | 当前请求的唯一ID，用于日志追踪。                                                 |
| `sign`             | string   | 是       | 请求签名，生成方法参照 **鉴权**。                                                |
| `parsing_strategy` | string   | 否       | 解析策略，支持 `backend` (默认) 和 `frontend`。决定由谁来解析模型输出。          |
| `image_url`        | string   | 是       | 待标注图片的URL。                                                                |
| `prompt_text`      | string   | 否       | **二选一**。用户的文本提示，用于文本到框的检测任务。                             |
| `prompt_box`       | array    | 否       | **二选一**。用户框选的坐标 `[x1, y1, x2, y2]`，用于框到文本的描述任务。          |
| `response_mapping` | object   | 否       | **仅在 `parsing_strategy` 为 `frontend` 时必须**。定义前端如何解析模型原始输出。 |


**关于 `prompt_text` 的交互与管理:**
对于文本到框任务，前端交互的灵活性至关重要。虽然API仅接收一个最终的 `prompt_text` 字符串，但前端应提供便利的交互方式，例如：
- **自由输入**: 一个文本框，允许用户自由输入指令，无论是宽泛的描述（"找出所有动物"）还是精确的指令（"找出红色的猫"）。
- **历史与预设**: 一个下拉列表或一组标签，方便用户快速选择常用的或之前使用过的 `prompt`。

这种方式将具体的交互逻辑保留在前端，而后端专注于接收指令并执行推理，实现了关注点分离。

系统会在内部将用户的 `prompt` 与一个预设的"系统级模板"相结合，自动附加格式要求。用户无需在 `prompt` 中手动指定输出格式。

### **解析方案**

我们提供两种方案来处理和解析大模型的输出结果。

#### **方案A: 后端解析 (Backend Parsing)**
这是**默认**且**推荐**的方案 (`parsing_strategy: "backend"`)。
后端负责完整的处理流程：调用模型、解析输出、并返回前端可直接渲染的结构化JSON。
-   **优点**: 前端集成最简单，无需关心解析逻辑。
-   **响应格式**: `data` 字段是一个包含结构化标注对象的 **数组** `[]`。

#### **方案B: 前端解析 (Frontend Parsing)**
当需要高度自定义或快速迭代解析逻辑时，可选择此方案 (`parsing_strategy: "frontend"`)。
后端将返回模型的**原始文本输出**，由前端根据 `response_mapping` 规则自行解析。
-   **优点**: 解析逻辑由前端完全控制，灵活性极高。
-   **响应格式**: `data` 字段是一个包含 `raw_model_output` 键的 **对象** `{}`。

---

### **使用示例**

#### **示例1: 文本到框任务 (Text-to-Box Detection)**
**场景**: 用户提供文本指令，模型在图中定位一个或多个对象。此任务涵盖了从精确查找（Grounding）到宽泛检测（Dense Captioning）的全部场景。

**子场景A: 精确查找 (原示例1)**
- **用户指令**: "找出图中的猫和狗"

**方案A (后端解析) 请求:**
```json
{
    "appId": "your_app_id",
    "timestamp": "1678886400000",
    "uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "parsing_strategy": "backend",
    "image_url": "http://example.com/images/cat_and_dog.jpg",
    "prompt_text": "找出图中的猫和狗",
    "sign": "generated_sign_string"
}
```
**方案A (后端解析) 响应:**
```json
{
    "code": "200",
    "message": "OK",
    "data": [
        { "box": [10, 20, 100, 120], "label": "猫", "superLabel": null, "properties": {} },
        { "box": [150, 50, 250, 200], "label": "狗", "superLabel": null, "properties": {} }
    ],
    "uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "timestamp": "1678886401000"
}
```
<br>

**方案B (前端解析) 请求:**
```json
{
    "appId": "your_app_id",
    "timestamp": "1678886400000",
    "uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "parsing_strategy": "frontend",
    "image_url": "http://example.com/images/cat_and_dog.jpg",
    "prompt_text": "找出图中的猫和狗",
    "response_mapping": {
        "item_regex": "\\{'box':\\[(.*?)\\],\\s*'ref':'(.*?)'\\}",
        "fields": { "box": "'box':\\[(.*?)\\]", "label": "'ref':'(.*?)'" }
    },
    "sign": "generated_sign_string"
}
```
**方案B (前端解析) 响应:**
```json
{
    "code": "200",
    "message": "OK",
    "data": {
        "raw_model_output": "[{'box':[10,20,100,120],'ref':'猫'} , {'box':[150,50,250,200],'ref':'狗'}]"
    },
    "uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "timestamp": "1678886401000"
}
```

---
**子场景B: 宽泛检测 (示例3)**
- **用户指令**: "找出图片中所有机柜和线缆，并描述它们的状态。"

**方案A (后端解析) 请求:**
```json
{
    "appId": "your_app_id",
    "timestamp": "1678886400000",
    "uuid": "c3d4e5f6-a7b8-9012-3456-7890abcdef2",
    "parsing_strategy": "backend",
    "image_url": "http://example.com/images/industrial_scene.jpg",
    "prompt_text": "找出图片中所有机柜和线缆，并描述它们的状态。",
    "sign": "generated_sign_string"
}
```
**方案A (后端解析) 响应:**
```json
{
    "code": "200",
    "message": "OK",
    "data": [
        { "box": [10, 50, 200, 400], "label": "cabinet_room/full", "superLabel": "cabinet_room", "properties": {} },
        { "box": [50, 410, 150, 480], "label": "cable/fiber", "superLabel": "cable", "properties": { "question": ["bend_is_true"] } }
    ],
    "uuid": "c3d4e5f6-a7b8-9012-3456-7890abcdef2",
    "timestamp": "1678886401000"
}
```
<br>

**方案B (前端解析) 请求:**
```json
{
    "appId": "your_app_id",
    "timestamp": "1678886400000",
    "uuid": "c3d4e5f6-a7b8-9012-3456-7890abcdef2",
    "parsing_strategy": "frontend",
    "image_url": "http://example.com/images/industrial_scene.jpg",
    "prompt_text": "找出图片中所有机柜和线缆，并描述它们的状态。",
    "response_mapping": {
        "item_regex": "{\\s*\"box\":\\s*\\[(.*?)\\],\\s*\"label\":\\s*\"(.*?)\",\\s*\"superLabel\":\\s*\"(.*?)\",\\s*\"properties\":\\s*(.*?)\\s*}",
        "fields": {
            "box": "\"box\":\\s*\\[(.*?)\\]",
            "label": "\"label\":\\s*\"(.*?)\"",
            "superLabel": "\"superLabel\":\\s*\"(.*?)\"",
            "properties": "\"properties\":\\s*(.*)"
        }
    },
    "sign": "generated_sign_string"
}
```
**方案B (前端解析) 响应:**
```json
{
    "code": "200",
    "message": "OK",
    "data": {
        "raw_model_output": "[{\"box\": [10, 50, 200, 400], \"label\": \"cabinet_room/full\", \"superLabel\": \"cabinet_room\", \"properties\": {}}, {\"box\": [50, 410, 150, 480], \"label\": \"cable/fiber\", \"superLabel\": \"cable\", \"properties\": {\"question\": [\"bend_is_true\"]}}]"
    },
    "uuid": "c3d4e5f6-a7b8-9012-3456-7890abcdef2",
    "timestamp": "1678886401000"
}
```

---

#### **示例2: 框到文本任务 (Box-to-Text Description)**
**场景**: 用户在图上框选一个区域，模型对该区域进行描述。

**方案A (后端解析) 请求:**
```json
{
    "appId": "your_app_id",
    "timestamp": "1678886400000",
    "uuid": "b2c3d4e5-f6a7-8901-2345-67890abcdef1",
    "parsing_strategy": "backend",
    "image_url": "http://example.com/images/cat.jpg",
    "prompt_box": [10, 20, 100, 120],
    "sign": "generated_sign_string"
}
```
**方案A (后端解析) 响应:**
```json
{
    "code": "200",
    "message": "OK",
    "data": [
        {
            "box": [10, 20, 100, 120],
            "label": "猫",
            "superLabel": "橘黄色",
            "properties": { "color": "orange", "action": "sitting" }
        }
    ],
    "uuid": "b2c3d4e5-f6a7-8901-2345-67890abcdef1",
    "timestamp": "1678886401000"
}
```
<br>

**方案B (前端解析) 请求:**
```json
{
    "appId": "your_app_id",
    "timestamp": "1678886400000",
    "uuid": "b2c3d4e5-f6a7-8901-2345-67890abcdef1",
    "parsing_strategy": "frontend",
    "image_url": "http://example.com/images/cat.jpg",
    "prompt_box": [10, 20, 100, 120],
    "response_mapping": {
        "item_regex": "\\{'Label':'(.*?)','SuperLabel':'(.*?)','properties':(.*?)\\}",
        "fields": {
            "label": "'Label':'(.*?)'",
            "superLabel": "'SuperLabel':'(.*?)'",
            "properties": "'properties':(.*?)"
        }
    },
    "sign": "generated_sign_string"
}
```
**方案B (前端解析) 响应:**
```json
{
    "code": "200",
    "message": "OK",
    "data": {
        "raw_model_output": "{'Label':'猫','SuperLabel':'橘黄色','properties':{'color': 'orange', 'action': 'sitting'}}"
    },
    "uuid": "b2c3d4e5-f6a7-8901-2345-67890abcdef1",
    "timestamp": "1678886401000"
}
```
