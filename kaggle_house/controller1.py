from datas import train_features, test_features, train_labels


if __name__ == "__main__":
    print(train_features.shape) # torch.Size([1460, 330])
    print(test_features.shape) # torch.Size([1459, 330])
    print(train_labels.shape) # torch.Size([1460, 1])
    