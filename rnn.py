from data_processor import load_and_preprocess_data, Vocabulary

def main():
    train_loader, val_loader, vocab = load_and_preprocess_data('./data/train_data-text_and_labels.csv', data_type='train_val')
    test_loader = load_and_preprocess_data('./data/test01_text_only.csv', data_type='test', shared_vocab=vocab)

if __name__ == "__main__":
    main()