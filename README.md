# Flower Classification

This web app, powered by Flask and PyTorch, serves as a powerful tool for identifying flower species. The core components of the CNN classifier include:

- **Custom Dataset Handling**: Utilized the Custom Dataset class in PyTorch to efficiently load and prepare the data.
- **Data Transformation**: Employed the torchvision.transforms module for seamless data transformation.
- **Flower Classifier Class**: Designed a specialized class, FlowerClassifier, inheriting essential functions from nn.Module.
- **Loss and Optimization**: Implemented the nn.CrossEntropyLoss() for loss calculation and the torch.optim.Adam() optimizer for model optimization.
- **Data Visualization**: Leveraged Matplotlib for visualizing data and utilized the scikit-learn library for generating accuracy and confusion matrices.

![Confusion Matrix](https://github.com/Raz200327/Flower-Classification/assets/115984448/bb93ec26-b4cb-40f8-8fb8-3a45f371a791)

### Training and Validation Metrics

- **Loss**: Training and validation loss can be visualized using the following graph:

    ![Loss Graph](https://github.com/Raz200327/Flower-Classification/assets/115984448/398bb09b-e9c1-4b52-87be-4ea6b882a9f1)

- **Accuracy**: Training and validation accuracy can be seen in the graph below:

    ![Accuracy Graph](https://github.com/Raz200327/Flower-Classification/assets/115984448/021514ec-50b5-4fef-921a-adb7bbfb6263)

### Example

Here's an example of the classification results:

![Flower Classification Example](https://github.com/Raz200327/Flower-Classification/assets/115984448/880befb0-1c70-44f2-8da2-b877a3f883bd)

Explore this app to classify flower species with ease and accuracy!
