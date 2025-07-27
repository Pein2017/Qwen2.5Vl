# API Reference - Complete Guide

**Comprehensive API documentation for all BBU Detection System components**

## 🏋️ Training System APIs

### BBUTrainer (`src.training.trainer`)

```python
from src.training.trainer import BBUTrainer

class BBUTrainer(Trainer):
    """Enhanced HuggingFace Trainer with coordinate token support"""
    
    def __init__(
        self,
        model,                    # Qwen25VLWithDetection instance
        args,                     # HuggingFace TrainingArguments
        train_dataset,            # BBUDataset for training
        eval_dataset=None,        # BBUDataset for evaluation
        data_collator=None,       # BBU-specific data collator
        cfg=None,                 # DirectConfig instance
        image_processor=None,     # Image processor for VLM
        training_coordinator=None # Optional TrainingCoordinator
    )
    
    # Key Methods
    def compute_loss(self, model, inputs, return_outputs=False)
    def log(self, logs: Dict[str, float]) -> None
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval")
```

**Usage Example**:
```python
trainer = BBUTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    cfg=config,
    image_processor=image_processor
)
trainer.train()
```

### TrainingCoordinator (`src.training.training_coordinator`)

```python
from src.training.training_coordinator import TrainingCoordinator

class TrainingCoordinator:
    """Training orchestration for multi-task learning"""
    
    def __init__(
        self,
        model,           # Qwen25VLWithDetection instance
        tokenizer,       # Tokenizer with coordinate tokens
        config_obj       # DirectConfig instance
    )
    
    # Key Methods
    def setup_training(self) -> None
    def get_training_state(self) -> Dict[str, Any]
    def cleanup(self) -> None
    
    # Properties
    @property
    def loss_manager(self) -> LossManager
```

**Usage Example**:
```python
coordinator = TrainingCoordinator(
    model=model,
    tokenizer=tokenizer,
    config_obj=config
)
coordinator.setup_training()
```

### LossManager (`src.training.loss_manager`)

```python
from src.training.loss_manager import LossManager

class LossManager:
    """Multi-task loss computation with teacher-student splitting"""
    
    def __init__(
        self,
        tokenizer,              # Tokenizer with coordinate tokens
        model,                  # Model for loss computation
        teacher_loss_weight=0.3, # Weight for teacher samples
        student_loss_weight=1.0  # Weight for student samples
    )
    
    # Key Methods
    def compute_total_loss(
        self, 
        inputs: Dict[str, torch.Tensor], 
        model_outputs
    ) -> Tuple[torch.Tensor, Dict[str, float]]
    
    def split_teacher_student_loss(
        self,
        llm_loss: torch.Tensor,
        coordinate_loss: torch.Tensor,
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]
```

**Usage Example**:
```python
loss_manager = LossManager(
    tokenizer=tokenizer,
    model=model,
    teacher_loss_weight=0.3,
    student_loss_weight=1.0
)

total_loss, loss_components = loss_manager.compute_total_loss(inputs, outputs)
```

### TrainerFactory (`src.training.trainer_factory`)

```python
from src.training.trainer_factory import create_trainer_with_coordinator

def create_trainer_with_coordinator(
    training_args,              # HuggingFace TrainingArguments
    use_coordinator=True        # Whether to use TrainingCoordinator
) -> BBUTrainer:
    """Factory function for creating fully configured trainer"""
```

**Usage Example**:
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2
)

trainer = create_trainer_with_coordinator(training_args)
trainer.train()
```

## 🤖 Model System APIs

### ModelLoader (`src.models.model_loader`)

```python
from src.models.model_loader import load_model_and_processor_unified

def load_model_and_processor_unified(
    model_path: str,                    # Path to model
    for_inference: bool = False,        # Training vs inference mode
    attn_implementation: str = "eager", # Attention implementation
    torch_dtype: str = "auto"          # Model precision
) -> Tuple[Qwen25VLWithDetection, AutoTokenizer, AutoImageProcessor]:
    """Unified model loading for training and inference"""
```

**Usage Example**:
```python
# For training
model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path="/path/to/qwen2.5-vl-7b-instruct",
    for_inference=False,
    attn_implementation="flash_attention_2"
)

# For inference
model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path="output/final_model",
    for_inference=True
)
```

### Qwen25VLWithDetection (`src.models.wrapper`)

```python
from src.models.wrapper import Qwen25VLWithDetection

class Qwen25VLWithDetection(Qwen2VLForConditionalGeneration):
    """Main model wrapper with coordinate token support"""
    
    # Key Methods
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]
    
    def generate(
        self,
        input_ids,
        pixel_values=None,
        **generation_kwargs
    ) -> torch.LongTensor
    
    def parse_coordinate_tokens(
        self,
        generated_ids: torch.LongTensor
    ) -> List[Dict[str, Any]]
```

## 🏭 Core System APIs

### DataProcessor (`src.core.data_processor`)

```python
from src.core.data_processor import DataProcessor

class DataProcessor:
    """Unified data processing pipeline"""
    
    def __init__(
        self,
        tokenizer,        # Tokenizer with coordinate tokens
        image_processor,  # Image processor for VLM
        model=None       # Optional model for advanced processing
    )
    
    # Key Methods
    def create_datasets(self) -> Tuple[Dataset, Dataset]:
        """Create training and evaluation datasets"""
    
    def create_data_collator(self) -> Callable:
        """Create data collator for batching"""
    
    def create_datasets_and_collator(self) -> Tuple[Dataset, Dataset, Callable]:
        """One-liner for complete data setup"""
```

**Usage Example**:
```python
processor = DataProcessor(tokenizer, image_processor, model)
train_dataset, eval_dataset = processor.create_datasets()
data_collator = processor.create_data_collator()

# Or one-liner
train_ds, eval_ds, collator = processor.create_datasets_and_collator()
```

### CheckpointManager (`src.core.checkpoint_manager`)

```python
from src.core.checkpoint_manager import CheckpointManager

class CheckpointManager:
    """Model saving and loading utilities"""
    
    # Key Methods
    def save_model_safely(
        self,
        trainer,           # BBUTrainer instance
        checkpoint_path: str
    ) -> bool:
        """Save model with integrity checking"""
    
    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """Validate checkpoint integrity"""
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """Get checkpoint metadata"""
```

**Usage Example**:
```python
manager = CheckpointManager()
success = manager.save_model_safely(trainer, "output/checkpoint")
is_valid = manager.validate_checkpoint("output/checkpoint")
```

## ⚙️ Configuration APIs

### GlobalConfig (`src.config.global_config`)

```python
from src.config import get_config

def get_config() -> DirectConfig:
    """Get global configuration instance"""

class DirectConfig:
    """Direct configuration access with 149+ parameters"""
    
    # Model parameters
    model_path: str
    model_max_length: int
    torch_dtype: str
    attn_implementation: str
    
    # Training parameters
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    
    # Detection parameters
    detection_enabled: bool
    coordinate_tokens_enabled: bool
    max_coord_value: int
    
    # Data parameters
    train_data_path: str
    val_data_path: str
    teacher_ratio: float
    
    # ... and 130+ more parameters
```

**Usage Example**:
```python
config = get_config()

# Access any parameter directly
print(f"Model path: {config.model_path}")
print(f"Learning rate: {config.learning_rate}")
print(f"Detection enabled: {config.detection_enabled}")

# Use in components
model_path = config.model_path
batch_size = config.per_device_train_batch_size
```

## 🛠️ Utility APIs

### SimpleTokenManager (`src.utils.simple_token_manager`)

```python
from src.utils.simple_token_manager import create_simple_token_manager

def create_simple_token_manager(
    tokenizer,    # Base tokenizer
    model        # Model to extend
) -> SimpleTokenManager:
    """Create token manager for coordinate tokens"""

class SimpleTokenManager:
    """Lightweight token management"""
    
    # Key Methods
    def add_coordinate_tokens(self) -> None
    def add_geometry_tokens(self) -> None
    def resize_model_embeddings(self) -> None
    def get_coordinate_token_ids(self) -> List[int]
```

**Usage Example**:
```python
token_manager = create_simple_token_manager(tokenizer, model)
coord_token_ids = token_manager.get_coordinate_token_ids()
```

### ResponseParser (`src.utils.response_parser`)

```python
from src.utils.response_parser import ResponseParser

class ResponseParser:
    """Parse model responses for structured data"""
    
    def parse_response(
        self,
        response: str
    ) -> Dict[str, Any]:
        """Parse coordinate tokens and descriptions"""
    
    def extract_coordinates(
        self,
        response: str
    ) -> List[List[int]]:
        """Extract coordinate sequences"""
    
    def extract_descriptions(
        self,
        response: str
    ) -> List[str]:
        """Extract object descriptions"""
```

**Usage Example**:
```python
parser = ResponseParser()
parsed = parser.parse_response(model_response)

# Access structured data
objects = parsed["objects"]
for obj in objects:
    coords = obj["coordinates"]
    desc = obj["description"]
```

## 🔄 Data Conversion APIs

### PipelineManager (`data_conversion.pipeline_manager`)

```python
from data_conversion.pipeline_manager import PipelineManager
from data_conversion.config import DataConversionConfig

class PipelineManager:
    """5-stage data processing pipeline"""
    
    def __init__(self, config: DataConversionConfig)
    
    # Key Methods
    def run_pipeline(self) -> bool:
        """Execute complete 5-stage pipeline"""
    
    def run_stage(self, stage_num: int) -> bool:
        """Execute specific pipeline stage"""
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get processing status and metrics"""

class DataConversionConfig:
    """Configuration for data conversion"""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        object_types: List[str] = ["full"],
        resize: bool = True,
        val_ratio: float = 0.1,
        teacher_ratio: float = 0.1
    )
```

**Usage Example**:
```python
config = DataConversionConfig(
    input_dir="ds_v2",
    output_dir="data",
    object_types=["bbu", "label", "fiber"],
    resize=True
)

manager = PipelineManager(config)
success = manager.run_pipeline()
```

## 📊 Common Usage Patterns

### Complete Training Setup
```python
# 1. Load configuration
from src.config import get_config
config = get_config()

# 2. Load model and processors
from src.models.model_loader import load_model_and_processor_unified
model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path=config.model_path,
    for_inference=False
)

# 3. Create datasets
from src.core.data_processor import DataProcessor
processor = DataProcessor(tokenizer, image_processor, model)
train_dataset, eval_dataset = processor.create_datasets()
data_collator = processor.create_data_collator()

# 4. Setup training coordinator
from src.training.training_coordinator import TrainingCoordinator
coordinator = TrainingCoordinator(model, tokenizer, config)
coordinator.setup_training()

# 5. Create and run trainer
from src.training.trainer import BBUTrainer
trainer = BBUTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    cfg=config,
    image_processor=image_processor,
    training_coordinator=coordinator
)
trainer.train()
```

### Complete Inference Setup
```python
# 1. Load trained model
from src.models.model_loader import load_model_and_processor_unified
model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path="output/final_model",
    for_inference=True
)

# 2. Process image
from PIL import Image
image = Image.open("test_image.jpg")
pixel_values = image_processor(image, return_tensors="pt").pixel_values

# 3. Generate response
input_ids = tokenizer.encode("<image>\n请描述图像中的BBU设备。", return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=512
    )

# 4. Parse results
from src.utils.response_parser import ResponseParser
response = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
parser = ResponseParser()
results = parser.parse_response(response)
```

### Factory Pattern Usage (Recommended)
```python
# Training with factory pattern
from src.training.trainer_factory import create_trainer_with_coordinator
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2
)

trainer = create_trainer_with_coordinator(training_args)
trainer.train()
```

---

**Next Steps**:
- **Commands Reference**: [commands.md](commands.md)
- **Troubleshooting**: [troubleshooting.md](troubleshooting.md)
- **Component Details**: [../components/](../components/)
