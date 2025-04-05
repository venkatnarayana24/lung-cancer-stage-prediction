# lung-cancer-stage-prediction
🫁 Lung Cancer Stage Prediction using EfficientNet and Grad-CAM
This is a Streamlit-based web application that allows users to upload chest CT scan images and receive predictions on the lung cancer stage (Normal, Stage I, Stage II). The app also visualizes predictions using Grad-CAM to highlight regions of interest.

🔧 Features
Upload multiple CT scan images (.jpg, .jpeg, .png).

Classifies images into:

Normal

Stage I

Stage II

Visualize:

Model prediction with confidence

Grad-CAM heatmap overlays

Model confidence graph

Alternative colormaps

Performance metrics

🧠 Model
Base model: EfficientNet-B0 (Pretrained on ImageNet)

Classifier Head: Fine-tuned to classify 3 classes

Explainability: Grad-CAM for heatmap visualization

📁 Project Structure

.
├── efficientnet_lung_cancer.pth        # Trained model weights (Place this in root)
├── app.py                                # Streamlit app (main file)
├── README.md                           # Project description
🚀 Running the App
Install dependencies:

pip install streamlit torch torchvision pillow matplotlib opencv-python
Ensure the model file is present: Place efficientnet_lung_cancer.pth in the same directory as f.py.

Start the app:
streamlit run app.py

📊 Model Performance (on validation set)
Accuracy: 92.5%

Precision: 91.3%

Recall: 90.7%

F1-Score: 91.0%

AUC-ROC: 94.2%

⚠️ Disclaimer
This tool is intended for educational and research purposes only. It is not a medical device. Please consult medical professionals for actual diagnosis or treatment.
