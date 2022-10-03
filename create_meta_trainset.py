import numpy as np


def create_meta_trainset(
    all_image_batches, all_label_batches,
):
    num_batches, B, K, N, d = all_image_batches.shape

    labels_for_supportset = all_label_batches[:, :, :-1]  # (B, K- 1, N , N)
    masking_for_queryset = np.zeros_like(all_label_batches[:, :, -1:])  # (B, 1, N, N)
    train_labels = np.concatenate(
        [labels_for_supportset, masking_for_queryset], axis=2,
    )

    meta_labels = np.argmax(all_label_batches[:, :, -1, :], axis=-1)  # (B, 1, N)

    images_for_supportset = all_image_batches[:, :, :-1]

    images_for_queryset = all_image_batches[:, :, -1:]
    images_for_queryset = images_for_queryset.reshape(-1, N, d)
    meta_labels = meta_labels.reshape(-1, N)

    temp_images = []
    temp_labels = []

    for _images, _labels in zip(images_for_queryset, meta_labels):
        # _images.shape == 4 x 784
        # _labels.shape == 4
        index = np.arange(N)
        np.random.shuffle(index)
        index
        temp_images.append(_images[index])
        temp_labels.append(_labels[index])

    images_for_queryset = np.stack(temp_images).reshape(-1, B, 1, N, d)
    meta_labels = np.stack(temp_labels).reshape(-1, B, N)
    all_image_batches = np.concatenate(
        [images_for_supportset, images_for_queryset], axis=2
    )

    meta_inputs = np.concatenate([all_image_batches, train_labels], axis=-1)
    meta_inputs = meta_inputs.reshape(num_batches, B, -1, d + N)

    return (
        meta_inputs,
        meta_labels,
    )

