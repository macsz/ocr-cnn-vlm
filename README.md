# ocr-cnn-vlm

## Datasets

The dataset have been augmented and made publicly available to allow for easy reproducibility of this work. Datasets are available on HuggingFace:

| Dataset | |
| --- |
| [The Street View Text Weather]([http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset](https://huggingface.co/datasets/macsz/street-view-text-weather)) |
| [ICDAR15-Weather]([http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset](https://huggingface.co/datasets/macsz/icdar-2015-weather)) |


## How To's

### Create augmented dataset

#### The Street View Text Dataset

Run:

```bash
python data/augment.py
```

This will create `data/svt1_augmented/img` directory containing light, medium and heavy weather conditions (rain, fog and combined), with short description appended to the filename, e.g.:

* 19_13_combined_3x3.jpg - a 3x3 grid of all augmented images
* 19_13_combined.jpg - 3x1 grid of medium conditions
* 19_13_fog_heavy.jpg - Heavy fog
* 19_13_fog_light.jpg - Light fog
* 19_13_fog_medium.jpg - Medium fog
* 19_13_original.jpg - original image, unmodified
* 19_13_rain_fog_heavy.jpg - Heavy rain & heafy fog
* 19_13_rain_fog_light.jpg - Light rain & light fog
* 19_13_rain_fog_medium.jpg - Medium rain & medium fog
* 19_13_rain_heavy.jpg - Heavy rain
* 19_13_rain_light.jpg - Light rain
* 19_13_rain_medium.jpg - Medium rain

The script will only augment files listed in the `test.xml` file (249 files).
