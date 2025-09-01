# BBU Installation Quality Inspection Dataset Overview

## Domain and Purpose
- **Industry**: Telecom (specifically Baseband Unit - BBU installation)
- **Goal**: Develop and train a vision-language model to automatically detect and evaluate BBU installation quality in telecom cabinet images. The model should:
  - Detect key components like BBU, shield, labels, screws, wires, etc.
  - Assess compliance with installation standards, safety, and organizational requirements, using both visual features and textual descriptions.

## Data Source and Format
- **Produced By**: The dataset is processed by the script `data_conversion/unified_processor.py`.
- **Data Format**: The dataset consists of flat JSONL files. Key files are:
  - `train.jsonl`, `val.jsonl`, `teacher_pool.jsonl`, `all_samples.jsonl`
  - **Sample Record Format**:
    - `images`: List of processed image file paths.
    - `objects`: List of annotations with geometry and hierarchical Chinese descriptions.
    - `width`, `height`: Final processed image dimensions.

## Object Types and Geometry Constraints
### Supported Object Types (Strict) and Geometry:
- **BBU-related components**:
  - **`bbu`** (Baseband Unit): Detects BBU equipment, its brand, and installation completeness.
  - **`bbu_shield`** (Windshield): Detects installation of the windshield for BBU.
  - **`connect_point`** (Connectors and screws): Identifies screw types and checks if connectors are correctly installed.
  - **`fiber`** (Fiber optic cable): Identifies fiber cables and checks installation integrity (e.g., protection, bend radius).
  - **`wire`** (Electrical wiring): Identifies wiring and checks tidiness and correctness of installation.
  - **`label`**: Detects textual labels on components (e.g., connector IDs, fiber types).

- **Geometrical Representation**:
  - **`bbox_2d`**: Axis-aligned bounding boxes.
  - **`quad`**: Quadrilateral for rotated bounding boxes.
  - **`line`**: Polyline for fibers and wires.

## Object Attributes and Hierarchical Descriptions
Each object is annotated with a **hierarchical Chinese description (`desc`)** that encodes:
- Object type (e.g., `bbu`, `fiber`, etc.)
- Various attributes (e.g., brand, visibility, compliance)
- Specific installation issues (e.g., improper shielding direction, unprotected fiber, etc.)

### Example Descriptions:
- `BBU设备/华为,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板`
- `挡风板/中兴,显示完整,挡风板有遮挡,安装方向正确`
- `光纤/有遮挡,有保护措施,弯曲半径合理/蛇形管`

## Recognition Task
### Primary Tasks:
1. **Object Detection & Localization**:
   - Detect and locate BBUs, shields, connectors, and labels using appropriate geometry.
2. **Attribute Inference**:
   - Predict attributes for each detected object, e.g., brand, visibility completeness, obstruction, installation direction, etc.
3. **Quality/Compliance Judgment**:
   - Aggregate results to assess the overall installation quality, e.g., check if BBU requires a shield and if it is correctly installed.

### Example Outputs:
- **Per-object**: Geometry + object type + attribute set (enough to reconstruct `desc`).
- **Per-image**: Aggregated QC metrics like:
  - `bbu_shield_required_and_present`
  - `any_incorrect_shield_direction`
  - `any_noncompliant_connect_points`
  - `any_fiber_bend_violation`
  - `wire_tidy_status`

## Key Quality Assurance Checks

### General Inspection Points:
1. **BBU Installation**:
   - Check if the BBU is correctly installed with sufficient space for the shield, and if the BBU is of the correct brand.
   - Verify that BBU installations meet the specified distance requirements (e.g., 1U spacing between BBU devices).
   
2. **Windshield Installation**:
   - Verify if a windshield is installed where necessary (e.g., ensure correct installation direction and no obstruction).
   
3. **Screws and Connectors**:
   - Check if screws are tightly fastened and correctly installed (e.g., no copper exposure, no double connections).
   - Ensure connectors and fiber optic cables are correctly installed with labels clearly visible.
   
4. **Fiber and Wiring**:
   - Ensure that the fibers have a reasonable bend radius and are adequately protected (e.g., using protective tubing like `蛇形管` or armored protection).
   - Check for tidy wire organization with no obstructions.
   
5. **Labels**:
   - Verify that all components have clear and readable labels corresponding to their function (e.g., ground cables, power cables).

### Detailed Inspection Points:
1. **BBU-related checks**:
   - **BBU Brand**: Ensure the correct brand is specified for each BBU (Huawei, Ericsson, ZTE).
   - **BBU Visibility**: Check if the BBU is fully visible or partially visible in the image.
   - **Space for Windshield**: Check if the surrounding space is sufficient for installing a windshield.
   - **Windshield Requirement**: If the BBU is of a brand that requires a windshield, verify its presence and conformity to installation guidelines.
   - **Shield Installation**:
     - **Direction**: Ensure the shield is installed in the correct direction.
     - **Obstruction**: Ensure no obstructions on the shield that might affect airflow.
     - **Brand Conformance**: Verify that the shield brand matches the BBU brand where applicable.

2. **Connector-related checks**:
   - **Screw Type**: Verify the correct type of screw for the given installation point (e.g., BBU mounting screw, grounding screw, etc.).
   - **Screw Condition**: Check if the screws are tightly fastened, exposed copper, or show signs of wear (e.g., rust).
   - **Fiber Optic Connectors**: Ensure fiber optic connectors are properly installed (e.g., BBU-end, ODF-end).
   - **Connector Visibility**: Ensure that connectors are visible and properly labeled.

3. **Cable-related checks**:
   - **Fiber Cables**:
     - **Obstruction**: Ensure no obstruction along the fiber cable path.
     - **Protection**: Verify that the fiber cables are adequately protected (e.g., with snake tubes or armor).
     - **Bend Radius**: Ensure the bend radius is within acceptable limits.
   - **Wire Cables**:
     - **Obstruction**: Ensure that there is no obstruction in the wiring path.
     - **Tidiness**: Ensure wires are neatly bundled and organized. The wiring should not be messy or loose.

4. **Label Checks**:
   - **Clarity**: Ensure labels are clear and readable.
   - **Content**: Check that labels match the context of the component (e.g., BBU grounding, fiber connectors, power distribution labels).

### Example Inspection Points from Real Samples:
1. **BBU and Shielding**:
   - Review the front of the BBU and the four screws. Check if the label content is clear, cables are organized, and fiber bend radius is reasonable.
2. **BBU Spacing**:
   - Ensure that there is at least a 1U space between adjacent BBUs, and that all BBU installations are accompanied by a shield.
3. **Huawei Equipment**:
   - Huawei BBU devices require a windshield only when space allows for installation. The 1U space between devices must be maintained.
4. **Label Integrity**:
   - Ensure that labels on screws, fiber connectors, and ground cables are clear and legible.
5. **Wire and Fiber Checks**:
   - Verify that fiber optic cables are unblocked, have proper protection, and are within the correct bend radius.
   - Ensure wiring is organized and free from obstructions.

## File Locations
- **Dataset Example**: `data/ds_v2_full/all_samples.jsonl`
- **Taxonomy File**: `data_conversion/attribute_taxonomy.json`
- **Hierarchical Mapping File**: `data_conversion/hierarchical_attribute_mapping.json`
- **Conversion Documentation**: `data_conversion/README.md`

## Conclusion
This dataset provides a detailed representation of BBU installation components, covering various quality control checks for each component. The vision-language model is expected to:
  - Detect components like BBUs, shields, connectors, fiber, and wires.
  - Infer attributes such as brand, visibility, compliance, and condition for each component.
  - Provide quality control judgments for each installation, ensuring that standards and regulations are met.

By training the model with these data points and checks, it will be capable of automating the BBU installation review process, ensuring consistency and compliance in telecom cabinet installations.
