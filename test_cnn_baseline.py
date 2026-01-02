from baselines.cnn_baseline import SimpleCNN,  ImprovedCNN

if __name__ == "__main__":
    print("Testing CNN Baseline...")
    model = SimpleCNN()
    model = ImprovedCNN()
    print(model)
