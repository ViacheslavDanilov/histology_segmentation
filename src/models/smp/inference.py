import os
import cv2
import hydra
import torch
import torchvision
from glob import glob
import numpy as np
from omegaconf import DictConfig, OmegaConf
from typing import Tuple
from src.models.smp.model import OCTSegmentationModel


def get_img_mask_union(
        img_0: np.ndarray,
        alpha_0: float,
        img_1: np.ndarray,
        alpha_1: float,
        color: Tuple[int, int, int],
) -> np.ndarray:
    return cv2.addWeighted(
        np.array(img_0).astype('uint8'),
        alpha_0,
        (cv2.cvtColor(np.array(img_1).astype('uint8'), cv2.COLOR_GRAY2RGB) * color).astype(
            np.uint8,
        ),
        alpha_1,
        0,
    )


def get_img_color_mask(
        img_0: np.ndarray,
        alpha_0: float,
        img_1: np.ndarray,
        alpha_1: float,
        color: Tuple[int, int, int],
) -> np.ndarray:
    return cv2.addWeighted(
        np.array(img_0).astype('uint8'),
        alpha_0,
        (cv2.cvtColor(np.array(img_1).astype('uint8'), cv2.COLOR_GRAY2BGR) * color).astype(
            np.uint8,
        ),
        alpha_1,
        0,
    )


def to_tensor(
        x: np.ndarray,
) -> np.ndarray:
    return x.transpose([2, 0, 1]).astype('float32')


get_tensor = torchvision.transforms.ToTensor()


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='inference_smp_model',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Module = OCTSegmentationModel(
    #     cfg.architecture,
    #     cfg.encoder,
    #     in_channels=3,
    #     classes=cfg.classes,
    #     colors=cfg.classes_color,
    # )
    model = OCTSegmentationModel.load_from_checkpoint(
        "models/Histology segmentation/models_epoch=161.ckpt",
        arch=cfg.architecture,
        encoder_name=cfg.encoder,
        in_channels=3,
        classes=cfg.classes,
        colors=cfg.classes_color,
    )
    # disable randomness, dropout, etc...
    model.eval()
    # class_values = [idx + 1 for idx, _ in enumerate(cfg.classes)]
    # predict with the model
    # y_hat = model(x)

    for idy, img_path in enumerate(glob(f'{cfg.data_dir}/*.[pj][np][g]')):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (cfg.input_size, cfg.input_size))
        image_input = image.copy()
        image_gr = image.copy()
        image_pred = image.copy()
        color_mask_gr = np.zeros(image_input.shape)
        color_mask_pred = np.zeros(image_input.shape)
        mask = cv2.imread(img_path.replace('img', 'mask'), 0)
        mask = cv2.resize(mask, (cfg.input_size, cfg.input_size), interpolation=cv2.INTER_NEAREST)

        # masks = [(mask == v) for v in class_values]
        # mask = np.stack(masks, axis=-1).astype('float')
        # image, mask = to_tensor(np.array(image)), to_tensor(np.array(mask))
        image = to_tensor(np.array(image))
        # image = get_tensor(image)

        # tensored_imgs = torch.stack([image]).to('cpu')

        y_hat = model(torch.Tensor([image]))
        mask_pred = y_hat.sigmoid()
        mask_pred = (mask_pred > 0.5).float()
        mask_pred = mask_pred.permute(0, 2, 3, 1)
        mask_pred = mask_pred.squeeze().cpu().numpy().round()
        # for idy, (img_, mask_, pr_mask) in enumerate(zip(img, mask, pred_mask)):

        color_mask_pred[:, :] = (128, 128, 128)
        color_mask_gr[:, :] = (128, 128, 128)

        for id, cl in enumerate(zip(cfg.classes)):
            # Groundtruth
            m = np.zeros(mask.shape)
            # print(cfg.classes_idx[cl[0]] + 1)
            # m[mask != (cfg.classes_idx[cl[0]] + 1)] = 0
            m[mask == (cfg.classes_idx[cl[0]] + 1)] = 1
            image_gr = get_img_mask_union(
                img_0=image_gr,
                alpha_0=1,
                img_1=m,
                alpha_1=0.5,
                color=cfg.classes_color[cl[0]],
            )
            color_mask_gr[m == 1] = cfg.classes_color[cl[0]]

            image_pred = get_img_mask_union(
                img_0=image_pred,
                alpha_0=1,
                img_1=mask_pred[:, :, id],
                alpha_1=0.5,
                color=cfg.classes_color[cl[0]],
            )

            # Color_mask
            # img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            # ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)

            color_mask_pred[mask_pred[:, :, id] == 1] = cfg.classes_color[cl[0]]

        res = np.hstack((image_input, image_gr))
        res = np.hstack((res, image_pred))

        res_c = np.hstack((image_input, color_mask_gr))
        res_c = np.hstack((res_c, color_mask_pred))

        cv2.imwrite(f'data/experiment/test/union_example_{idy}.png', res)
        cv2.imwrite(f'data/experiment/test//color_mask_example_{idy}.png', res_c)

    pass


if __name__ == '__main__':
    main()
