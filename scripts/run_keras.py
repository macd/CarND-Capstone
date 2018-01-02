import glob
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json, load_model
import numpy as np
from PIL import Image

model = load_model("full_model.hd5")

lmap = {"green": 0, "red": 1, "unknown": 2, "yellow": 3}


def get_prediction(fn, model):
    img = Image.open(fn)    
    x = np.array(img.resize((400, 300), resample=3), dtype=np.float64)
    x = np.expand_dims(x, axis=0) / 255.

    preds = model.predict(x)
    return preds


def test_light(clr):
    fls = glob.glob("data/train/" + clr + "/*.jpg")
    total = len(fls)
    num = 0
    for f in fls:
        preds = get_prediction(f, model)
        #print(preds[0], preds[0].sum())
        if preds[0][lmap[clr]] > 0.5:
            num += 1
        
    print("Number %s found: %d out of a total: %d" % (clr, num, total))
    return num, total


if __name__ == "__main__":
    num = 0
    tot = 0
    for light in lmap:
        n, t = test_light(light)
        num += n
        tot += t
    
    print("Number correct %d out of a total %d.  Accuracy: %f" % ( num, tot, float(num) / tot))
