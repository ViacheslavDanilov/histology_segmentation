import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.utils import CLASS_COLOR, CLASS_ID

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_mask(
    img_path: str,
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    image_width = int(df.image_width.unique())
    image_height = int(df.image_height.unique())
    mask = np.zeros((image_height, image_width), dtype='uint8')
    mask_color = np.zeros((image_height, image_width, 3), dtype='uint8')
    mask_color[:, :] = (128, 128, 128)
    for _, row in df.iterrows():
        obj_mask = sly.Bitmap.base64_2_data(row.encoded_mask).astype(int)
        mask = build_mask(
            mask=mask,
            obj_mask=obj_mask,
            class_id=CLASS_ID[row.class_name],  # type: ignore
            origin=[row['x1'], row['y1']],
        )
        mask_color[mask == CLASS_ID[row.class_name]] = CLASS_COLOR[row.class_name]

    img_name = Path(img_path).name
    new_img_path = os.path.join(save_dir, 'img', img_name)
    mask_path = os.path.join(save_dir, 'mask', img_name)
    color_mask_path = os.path.join(save_dir, 'mask_color', img_name)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(color_mask_path, mask_color)
    shutil.copy(img_path, new_img_path)


def build_mask(
    mask: np.ndarray,
    obj_mask: np.ndarray,
    class_id: int,
    origin: List[int],
) -> np.ndarray:
    obj_mask[obj_mask == 1] = class_id
    obj_height, obj_width = obj_mask.shape
    mask[
        origin[1] : origin[1] + obj_height,
        origin[0] : origin[0] + obj_width,
    ] = obj_mask[:, :]
    return mask


def process_metadata(
    df: pd.DataFrame,
    classes: List[str],
) -> pd.DataFrame:
    df = df[df['class_name'].isin(classes)]
    df = df[df['area'] > 0]
    return df


def split_dataset(
    df: pd.DataFrame,
    train_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    slides = df.slide.unique()
    train_slides, test_slides = train_test_split(
        slides,
        train_size=train_size,
        shuffle=True,
        random_state=seed,
    )

    df_train = df[df['slide'].isin(train_slides)]
    df_test = df[df['slide'].isin(test_slides)]

    return df_train, df_test


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_int_to_final',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    for subset in ['train', 'test']:
        for dir_type in ['img', 'mask', 'mask_color']:
            os.makedirs(f'{cfg.save_dir}/{subset}/{dir_type}', exist_ok=True)

    # Read and process metadata
    df_path = os.path.join(cfg.data_dir, 'metadata.csv')
    df = pd.read_csv(df_path)
    df_filtered = process_metadata(df=df, classes=cfg.class_names)

    # Split dataset
    df_train, df_test = split_dataset(
        df=df_filtered,
        train_size=cfg.train_size,
        seed=cfg.seed,
    )
    gb_train = df_train.groupby(['image_path'])
    gb_test = df_test.groupby(['image_path'])
    log.info(f'Train images...: {len(gb_train)}')
    log.info(f'Test images....: {len(gb_test)}')

    # Process train and test subsets
    Parallel(n_jobs=-1, backend='threading')(
        delayed(process_mask)(
            img_path=img_path,
            df=df,
            save_dir=f'{cfg.save_dir}/train',
        )
        for img_path, df in tqdm(gb_train, desc='Process train subset')
    )

    Parallel(n_jobs=-1, backend='threading')(
        delayed(process_mask)(
            img_path=img_path,
            df=df,
            save_dir=f'{cfg.save_dir}/test',
        )
        for img_path, df in tqdm(gb_test, desc='Process test subset')
    )


if __name__ == '__main__':
    main()
