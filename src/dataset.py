import tensorflow as tf

def get_dataloaders(data_dir, img_size=224, batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int"
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        label_mode="int"
    )

    # Prefetching for performance
    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
        test_ds.prefetch(tf.data.AUTOTUNE),
    )
