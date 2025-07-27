# BBU Training System Documentation

Welcome to the BBU (Bounding Box Understanding) training system documentation. This system provides advanced vision-language model training with coordinate token support.

## 🚀 **Quick Navigation**

### **Getting Started**
- **[Getting Started Guide](getting-started.md)** - Setup, installation, and first training run
- **[Configuration Guide](configuration.md)** - Complete configuration reference
- **[Coordinate Tokens](coordinate-tokens.md)** - Coordinate token system guide

### **Training & Usage**
- **[Training Guide](training.md)** - Training workflows and best practices
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

### **Advanced Topics**
- **[System Architecture](architecture.md)** - Technical architecture overview
- **[Migration Guide](migration.md)** - Upgrading and migration instructions

### **Reference**
- **[Archive](archive/)** - Historical documentation and deprecated content

---

## 🎯 **What is BBU Training System?**

The BBU Training System is a comprehensive solution for training vision-language models with advanced coordinate understanding capabilities. It supports:

- **Dual-mode coordinate processing**: Standard integers and coordinate tokens
- **Advanced loss functions**: Multi-component loss with coordinate-aware training
- **Flexible data pipeline**: Support for various data formats and preprocessing
- **Production-ready training**: Optimized for both research and production use

## 🏗️ **System Overview**

### **Core Components**
- **Model System**: Qwen2.5-VL based architecture with coordinate token extensions
- **Data Pipeline**: Flexible data loading and preprocessing with coordinate support
- **Training System**: Advanced trainer with multi-component loss logging
- **Configuration**: Explicit, fail-fast configuration system

### **Key Features**
- ✅ **Coordinate Token System**: Both standard and token-based coordinate processing
- ✅ **Multi-component Loss**: Language modeling, bounding box, caption, and objectness losses
- ✅ **Flexible Training**: Support for various training strategies and optimizations
- ✅ **Production Ready**: Comprehensive testing and validation

## 📋 **Quick Start**

1. **Setup Environment**
   ```bash
   # Clone and setup
   git clone <repository>
   cd Qwen2.5-VL-main
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   ```bash
   # Prepare your training data
   python scripts/prepare_data.py --input data/raw --output data/processed
   ```

3. **Configure Training**
   ```yaml
   # configs/my_config.yaml
   coordinate_tokens_enabled: false  # Start with standard mode
   remove_unused_columns: false     # Required for proper data processing
   ```

4. **Start Training**
   ```bash
   python scripts/train.py --config configs/my_config.yaml
   ```

For detailed instructions, see the [Getting Started Guide](getting-started.md).

## 🔧 **Configuration Modes**

### **Standard Mode** (Recommended for Production)
- Integer coordinates: `[150,10,211,35]`
- Minimal vocabulary extension (+4 tokens)
- Stable and well-tested

### **Coordinate Mode** (Advanced Features)
- Token coordinates: `[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]`
- Extended vocabulary (+2052 tokens)
- Advanced sequence-based coordinate prediction

Both modes are production-ready. See [Coordinate Tokens Guide](coordinate-tokens.md) for details.

## 🆘 **Need Help?**

- **Common Issues**: Check [Troubleshooting Guide](troubleshooting.md)
- **Configuration Problems**: See [Configuration Guide](configuration.md)
- **Training Issues**: Refer to [Training Guide](training.md)
- **API Questions**: Check [API Reference](api-reference.md)

## 📚 **Documentation Structure**

This documentation is organized for easy navigation:

- **Linear Learning Path**: Follow the guides in order for comprehensive understanding
- **Reference Material**: Jump to specific topics as needed
- **Troubleshooting**: Quick solutions to common problems
- **Archive**: Historical information and migration notes

---

*Last updated: 2025-01-27 | Version: 2.0 (Simplified Structure)*
