# Flower Classification

This web app, powered by Flask and PyTorch, serves as a powerful tool for identifying flower species. The core components of the CNN classifier include:

- **Custom Dataset Handling**: Utilized the Custom Dataset class in PyTorch to efficiently load and prepare the data.
- **Data Transformation**: Employed the torchvision.transforms module for seamless data transformation.
- **Flower Classifier Class**: Designed a specialized class, FlowerClassifier, inheriting essential functions from nn.Module.
- **Loss and Optimization**: Implemented the nn.CrossEntropyLoss() for loss calculation and the torch.optim.Adam() optimizer for model optimization.
- **Data Visualization**: Leveraged Matplotlib for visualizing data and utilized the scikit-learn library for generating accuracy and confusion matrices.

![Confusion Matrix]<img width="176" alt="image" src="https://github.com/Raz200327/Flower-Classification/assets/115984448/0df5f164-69a1-4da3-a92f-bdb333121ea8">


### Training and Validation Metrics

- **Loss**: Training and validation loss can be visualized using the following graph:

    ![Loss Graph](https://github.com/Raz200327/Flower-Classification/assets/115984448/14cf8318-fca0-4919-ad88-e558a64e9715")


- **Accuracy**: Training and validation accuracy can be seen in the graph below:

    ![Accuracy Graph](https://github.com/Raz200327/Flower-Classification/assets/115984448/0d5e3833-c0d8-4662-8f02-2ed937b0a9e2")


### Example

Here's an example of the classification results:

![Flower Classification Example](https://github.com/Raz200327/Flower-Classification/assets/115984448/cbf83364-41b1-47d4-9006-2a272bea5e7b)

Explore this app to classify flower species with ease and accuracy!
