# CAD Draft Reviewing System

## Overview

The **CAD Draft Reviewing System** is designed to automate the analysis and validation of CAD drafts in the petroleum sector. It identifies signs and connections, detects rule violations, and suggests optimizations based on predefined rules.

## Features

- **PDF to PNG Conversion**: Efficiently converts large CAD PDFs into high-resolution PNG images.
- **Image Tiling**: Splits oversized images into manageable tiles to facilitate processing.
- **Open-World Object Detection**: Utilizes transformer-based models capable of detecting both known and unseen sign categories.
- **OCR Integration**: Extracts codes differentiating similar signs to assist in classification.
- **Knowledge Base**: Parses and structures rule files into a graph database for validation.
- **Interactive Dashboard**: Visualizes detections, rule violations, and accepts user feedback for continuous improvement.
- **Active Learning**: Focuses on uncertain detections to optimize labeling efforts.
- **Contextual Reasoning**: Enhances understanding of sign relationships using Graph Neural Networks.

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/AilvenLiu/cad_draft_review.git
   cd cad_draft_review
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv .venv   
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Neo4j**

   - Download and install [Neo4j](https://neo4j.com/download/).
   - Start the Neo4j server and create a database.
   - Update the `uri`, `user`, and `password` in `modules/knowledge_base.py` as needed.

## Usage

### 1. Convert PDFs to PNGs

```bash
python scripts/convert_pdfs.py --input_dir ./data/raw/pdfs --output_dir ./data/processed/images
```

### 2. Tile Images

```bash
python scripts/tile_images.py --input_dir ./data/processed/images --output_dir ./data/processed/tiles
```

### 3. Train the Model

```bash
python scripts/train.py --train_dir ./data/processed/tiles --val_dir ./data/processed/tiles
```

### 4. Launch the Interactive Dashboard

```bash
streamlit run app/streamlit_app.py -- --data_dir ./data/processed/tiles
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [DETR (Facebook Research)](https://github.com/facebookresearch/detr)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Neo4j](https://neo4j.com/)
- [Streamlit](https://streamlit.io/)

## Conclusion

This comprehensive implementation integrates all components necessary for the CAD Draft Reviewing project, emphasizing:

- Open-World Capabilities: By leveraging transformer-based models like DETR and incorporating incremental learning mechanisms, the system can handle unseen categories of CAD signs effectively.
- Efficient Data Handling: Utilization of fitz for PDF conversion and image tiling ensures that large CAD drafts are processed efficiently.
- Knowledge Integration: Parsing rule files and constructing a knowledge graph using Neo4j allows for sophisticated validation and optimization based on predefined rules.
- Innovative Design: Features like active learning, GNN-based reasoning, and an interactive dashboard introduce novel elements that enhance the system's effectiveness and user experience.
- Scalability and Maintainability: A modular project structure facilitates easy maintenance, scalability, and future enhancements, including TensorRT integration for inference optimization.

By following this structured approach, you can develop a robust and innovative system tailored to the specific challenges of reviewing CAD drafts in the petroleum sector. Each module is designed to address distinct aspects of the problem, ensuring a cohesive and efficient workflow.
