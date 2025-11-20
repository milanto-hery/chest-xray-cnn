from src.model import build_cnn
from src.dataset import get_dataloaders
from src.utils import plot_history

def train_model(data_dir="data", img_size=224, batch_size=32, epochs=15):
    train_ds, val_ds, test_ds = get_dataloaders(data_dir, img_size, batch_size)

    model = build_cnn(input_shape=(img_size, img_size, 3))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Save model
    model.save("chest_xray_cnn.keras")  

    plot_history(history)
    return model
