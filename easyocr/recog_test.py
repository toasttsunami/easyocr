import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import os
import json
import time
from collections import OrderedDict
import math
import argparse
from pathlib import Path

# Import necessary components from EasyOCR (adjust imports based on your setup)
try:
    from easyocr.utils import CTCLabelConverter
    from easyocr.recognition import NormalizePAD, AlignCollate, ListDataset, recognizer_predict, custom_mean
except ImportError:
    print("Please ensure EasyOCR is installed and accessible")

class ModelTester:
    def __init__(self, model_path, character_set, network_params, 
                 recog_network='generation2', device='cuda'):
        """
        Initialize the model tester
        
        Args:
            model_path: Path to your fine-tuned model (.pth file)
            character_set: String containing all characters your model can recognize
            network_params: Dictionary with network parameters (input_channel, output_channel, hidden_size)
            recog_network: 'generation1' or 'generation2' or custom network name
            device: 'cuda', 'cpu', or 'mps'
        """
        self.device = device
        self.character = character_set
        self.network_params = network_params
        self.recog_network = recog_network
        
        # Initialize converter
        self.converter = CTCLabelConverter(character_set, {}, {})
        
        # Load model
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load the fine-tuned model"""
        # Import the appropriate model architecture
        if self.recog_network == 'generation1':
            from easyocr.model.model import Model
        elif self.recog_network == 'generation2':
            from easyocr.model.vgg_model import Model
        else:
            # For custom networks, adjust this import
            import importlib
            model_pkg = importlib.import_module(self.recog_network)
            Model = model_pkg.Model
        
        # Initialize model
        num_class = len(self.converter.character)
        model = Model(num_class=num_class, **self.network_params)
        
        # Load weights
        if self.device == 'cpu':
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            # Handle DataParallel wrapper if present
            if list(state_dict.keys())[0].startswith('module.'):
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    new_key = key[7:]  # Remove 'module.' prefix
                    new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
        else:
            model = torch.nn.DataParallel(model).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(state_dict)
        
        model.eval()
        return model
    
    def preprocess_image(self, image_path, imgH=32, imgW=100):
        """
        Preprocess a single image for recognition
        
        Args:
            image_path: Path to image file
            imgH: Target height
            imgW: Target width
        """
        # Load image
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('L')
        else:
            img = Image.fromarray(image_path, 'L')
        
        w, h = img.size
        ratio = w / float(h)
        
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = math.ceil(imgH * ratio)
        
        # Resize image
        resized_image = img.resize((resized_w, imgH), Image.BICUBIC)
        
        # Normalize and pad
        transform = NormalizePAD((1, imgH, imgW))
        image_tensor = transform(resized_image).unsqueeze(0)
        
        return image_tensor
    
    def predict_single_image(self, image_path, decoder='greedy', beamWidth=5):
        """
        Predict text from a single image
        
        Args:
            image_path: Path to image file or numpy array
            decoder: 'greedy', 'beamsearch', or 'wordbeamsearch'
            beamWidth: Beam width for beam search
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            batch_size = 1
            batch_max_length = 25  # Adjust based on your needs
            
            # Prepare input
            length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(self.device)
            text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(self.device)
            
            # Forward pass
            preds = self.model(image_tensor, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            
            # Apply softmax
            preds_prob = F.softmax(preds, dim=2)
            
            # Decode based on method
            if decoder == 'greedy':
                _, preds_index = preds_prob.max(2)
                preds_index = preds_index.view(-1)
                pred_str = self.converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)[0]
            elif decoder == 'beamsearch':
                k = preds_prob.cpu().detach().numpy()
                pred_str = self.converter.decode_beamsearch(k, beamWidth=beamWidth)[0]
            else:
                k = preds_prob.cpu().detach().numpy()
                pred_str = self.converter.decode_wordbeamsearch(k, beamWidth=beamWidth)[0]
            
            # Calculate confidence
            preds_prob_np = preds_prob.cpu().detach().numpy()
            values = preds_prob_np.max(axis=2)
            indices = preds_prob_np.argmax(axis=2)
            
            max_probs = values[0][indices[0] != 0]
            if len(max_probs) > 0:
                confidence = custom_mean(max_probs)
            else:
                confidence = 0.0
            
            return pred_str, confidence
    
    def test_on_dataset(self, test_dir, ground_truth_file=None, output_file='results.json'):
        """
        Test the model on a dataset
        
        Args:
            test_dir: Directory containing test images
            ground_truth_file: JSON file with ground truth labels (optional)
            output_file: File to save results
        """
        results = []
        total_images = 0
        correct_predictions = 0
        
        # Load ground truth if provided
        ground_truth = {}
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in os.listdir(test_dir) 
                      if Path(f).suffix.lower() in image_extensions]
        
        print(f"Testing on {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(test_dir, image_file)
            
            try:
                # Predict
                start_time = time.time()
                prediction, confidence = self.predict_single_image(image_path)
                inference_time = time.time() - start_time
                
                # Get ground truth if available
                gt_text = ground_truth.get(image_file, None)
                
                # Check accuracy
                is_correct = False
                if gt_text is not None:
                    is_correct = prediction.strip().lower() == gt_text.strip().lower()
                    if is_correct:
                        correct_predictions += 1
                
                total_images += 1
                
                result = {
                    'image': image_file,
                    'prediction': prediction,
                    'confidence': float(confidence),
                    'ground_truth': gt_text,
                    'correct': is_correct,
                    'inference_time': inference_time
                }
                results.append(result)
                
                # Print progress
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images")
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue
        
        # Calculate metrics
        accuracy = correct_predictions / total_images if ground_truth else None
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_inference_time = np.mean([r['inference_time'] for r in results])
        
        # Summary
        summary = {
            'total_images': total_images,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'average_confidence': float(avg_confidence),
            'average_inference_time': float(avg_inference_time)
        }
        
        # Save results
        output = {
            'summary': summary,
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nTesting complete!")
        print(f"Total images: {total_images}")
        if accuracy is not None:
            print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_images})")
        print(f"Average confidence: {avg_confidence:.4f}")
        print(f"Average inference time: {avg_inference_time:.4f}s")
        print(f"Results saved to: {output_file}")
        
        return summary, results
    
    def visualize_predictions(self, test_dir, num_samples=10, save_dir='visualizations'):
        """
        Visualize predictions on sample images
        
        Args:
            test_dir: Directory containing test images
            num_samples: Number of samples to visualize
            save_dir: Directory to save visualization images
        """
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get sample images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in os.listdir(test_dir) 
                      if Path(f).suffix.lower() in image_extensions]
        
        sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        for image_file in sample_files:
            image_path = os.path.join(test_dir, image_file)
            
            # Load and display image
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get prediction
            prediction, confidence = self.predict_single_image(image_path)
            
            # Create visualization
            plt.figure(figsize=(10, 4))
            plt.imshow(img_rgb)
            plt.title(f'Prediction: "{prediction}"\nConfidence: {confidence:.4f}')
            plt.axis('off')
            
            # Save
            output_path = os.path.join(save_dir, f'pred_{image_file}')
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
        
        print(f"Visualizations saved to: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Test fine-tuned text recognition model')
    parser.add_argument('--model_path', required=True, help='Path to fine-tuned model')
    parser.add_argument('--test_dir', required=True, help='Directory containing test images')
    parser.add_argument('--character_file', required=True, help='File containing character set')
    parser.add_argument('--ground_truth', help='JSON file with ground truth labels')
    parser.add_argument('--network', default='generation2', help='Network architecture')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output', default='results.json', help='Output file for results')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    # Load character set
    with open(args.character_file, 'r', encoding='utf-8') as f:
        character_set = f.read().strip()
    
    # Network parameters (adjust based on your architecture)
    if args.network == 'generation1':
        network_params = {
            'input_channel': 1,
            'output_channel': 512,
            'hidden_size': 512
        }
    else:  # generation2
        network_params = {
            'input_channel': 1,
            'output_channel': 256,
            'hidden_size': 256
        }
    
    # Initialize tester
    tester = ModelTester(
        model_path=args.model_path,
        character_set=character_set,
        network_params=network_params,
        recog_network=args.network,
        device=args.device
    )
    
    # Run tests
    summary, results = tester.test_on_dataset(
        test_dir=args.test_dir,
        ground_truth_file=args.ground_truth,
        output_file=args.output
    )
    
    # Create visualizations if requested
    if args.visualize:
        tester.visualize_predictions(args.test_dir)

if __name__ == '__main__':
    main()