import os
import shutil
import zipfile
from cog import BasePredictor, Input, Path

from caption import Captioner
from train import extract_zip

class Predictor(BasePredictor):
    def setup(self):
        self.model = Captioner()
        self.model.load_models()

    def predict(self, 
                input_images: Path = Input(
                    description="A zip file containing the images that will be used for training. We recommend a minimum of 10 images. If you include captions, include them as one .txt file per image, e.g. my-photo.jpg should have a caption file named my-photo.txt. If you don't include captions, you can use autocaptioning (enabled by default).",
                    default=None,
                ),
                autocaption_prefix: str = Input(description="text that you want to appear at the beginning of all your generated captions; for example, ‘a photo of TOK, ’. You can include your trigger word in the prefix.", default=None), 
                autocaption_suffix: str = Input(description="text that you want to appear at the end of all your generated captions; for example, ‘ in the style of TOK’. You can include your trigger word in suffixes.", default=None)    ) -> Path:
        input_dir = Path("./input_imgs")
        # deletes the input_dir if it exists
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        # creates the input_dir if it doesn't exist
        os.makedirs(input_dir, exist_ok=True)

        extract_zip(input_images, input_dir)
        self.model.caption_images(input_dir, autocaption_prefix, autocaption_suffix)

        # zips up captions, which are .txt files in input_dir, into a zip file
        # unnecessary but takes zero time and space
        output_dir = Path("./output_imgs")
        # deletes the output_dir if it exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        # creates the output_dir if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        for _, caption in self.model.iter_images_captions(input_dir):
            shutil.copy(caption, output_dir / caption.name)

        output_zip = "captions.zip"
        if os.path.exists(output_zip):
            os.remove(output_zip)

        with zipfile.ZipFile(output_zip, 'w') as zipf:
            for file in os.listdir(output_dir):
                if file.endswith(".txt"):
                    zipf.write(os.path.join(output_dir, file), file)

        return Path(output_zip)
        