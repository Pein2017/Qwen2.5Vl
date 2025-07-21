Here’s a concise “data-sheet” for your cleaned **`after`** file—only the fields that still exist and what they contain.

---

## 1 . Top-level layout

```text
{
  info        : { depth, width, height }
  tagInfo     : { mode, dataId, taskId, timestamp }
  version     : "4.12.46"
  markResult  : {
      features     : [ … 29 Feature objects … ]
      groupsAttr   : []          // kept empty placeholder
      type         : "FeatureCollection"
  }
}
```

*Everything else (QA history, statistics, `qualityResult`, etc.) is gone.*

---

## 2 . Feature inventory (29 total)

| label class (`properties.content.label`) | geom type       | count  |
| ---------------------------------------- | --------------- | ------ |
| `connect_point`                          | `ExtentPolygon` | **9**  |
| `label`                                  | `Square`        | **7**  |
| `bbu`                                    | `Square`        | **2**  |
| `fiber`                                  | `LineString`    | **10** |
| `wire`                                   | `LineString`    | **1**  |

---

## 3 . Per-class attribute sets

| class              | retained attribute keys (all live under `properties.content`)                                         | typical values in this file                                                         |
| ------------------ | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **connect\_point** | `connect_point_type` <br> `connect_point_situation` <br> `connect_point_check` <br> `ex_info` (empty) | `install_screw`, `fiber_bbu` <br> `part / complete` <br> `connect_point_check_true` |
| **label**          | `label_text`                                                                                          | 5 G-equipment labels, plus two blanks `" "`                                         |
| **bbu**            | `bbu_stituation` <br> `bbu_brand` <br> `bbu_equipment` <br> `ex_info`                                 | `complete` <br> `huawei` <br> *three-way code indicating shield status*             |
| **fiber**          | `fiber_radius` <br> `fiber_block` <br> `fiber_protection` <br> `ex_info` (empty)                      | `fiber_radius_trrue` <br> `fiber_block_true` <br> `fiber_protection_true/...`       |
| **wire**           | `wire_bind` <br> `wire_block` <br> `ex_info` (empty)                                                  | `wire_bind_true`, `wire_block_false`                                                |

*(Chinese-language mirror keys live in `properties.contentZh`; they still match the English ones.)*

---

## 4 . Geometry essentials

* All **polygons** are still simple coordinate arrays—no extra area/layer metadata.
* All **lines** are basic `coordinates` + `lineType` (retained for drawing style).

---
