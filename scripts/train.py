import torch
from models.multimodal_transformer import MultimodalTransformer
from utils.device_utils import get_device
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.transforms import get_train_transform
from utils.vocab import Vocab
from modules.object_detection import OpenImagesDataset
from sklearn.metrics import average_precision_score

def train_object_detection_refined():
    device = get_device()
    print(f"Using device: {device}")

    # Model parameters
    vocab_size = 10000  # Adjust as needed
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    max_seq_length = 50
    dropout = 0.1
    img_channels = 3
    embed_dim = 512
    num_classes = 600  # Adjust based on dataset

    # Initialize the Multimodal Transformer
    model = MultimodalTransformer(
        num_layers=num_layers,
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout,
        img_channels=img_channels,
        embed_dim=embed_dim,
        num_classes=num_classes
    ).to(device)

    # Optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion_cls = nn.CrossEntropyLoss(ignore_index=0)
    criterion_bbox = nn.SmoothL1Loss()

    # Define transformations for the images
    transform = get_train_transform()

    # Build Vocabulary for object categories
    annotations_file = 'data/annotations/open_images_annotations.json'
    images_dir = 'data/processed/tiles/'
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    from utils.vocab import Vocab
    vocab = Vocab(freq_threshold=5)
    all_categories = [anno['category'] for anno in annotations['annotations']]
    vocab.build_vocabulary(all_categories)

    # Initialize dataset and dataloader
    dataset = OpenImagesDataset(images_dir, annotations_file, vocab, transform=transform, max_seq_length=max_seq_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Training loop with evaluation
    model.train()
    for epoch in range(10):  # Number of epochs
        total_loss = 0
        for batch_idx, (images, bboxes, labels) in enumerate(dataloader):
            images = images.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs_cls, outputs_bbox, predicted_latent = model(images, images, actions=None)  # Adjust inputs accordingly

            # Classification loss
            loss_cls = criterion_cls(outputs_cls.view(-1, num_classes + 1), labels.view(-1))

            # Bounding box loss
            loss_bbox = criterion_bbox(outputs_bbox, bboxes)

            # Total loss
            loss = loss_cls + loss_bbox
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/10], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/10], Average Loss: {avg_loss:.4f}")

        # Save model checkpoint after each epoch
        torch.save(model.state_dict(), f'models/multimodal_world_model_object_detection_epoch_{epoch+1}.pth')

    # Evaluation (simplistic example)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, bboxes, labels in dataloader:
            images = images.to(device)
            outputs_cls, outputs_bbox, _ = model(images, images, actions=None)
            
            # Example: Compute average precision for classification
            preds = torch.argmax(outputs_cls, dim=-1).cpu().numpy()
            targets = labels.cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())

    ap_score = average_precision_score(all_targets, all_preds, average='macro')
    print(f"Average Precision (AP) Score: {ap_score:.4f}")

if __name__ == "__main__":
    train_object_detection_refined()