import os
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import ezkl


# ==========================
# CONFIG
# ==========================

IMAGE_SIZE = 32
THRESHOLD = 0.5

ENROLLED_IMAGE = "data/enrolled.bmp"
TEST_IMAGE = "data/test.bmp"


ONNX_MODEL = "models/siamese_fingerprint_zkml.onnx"

INPUT_JSON = "models/auth_input.json"

COMPILED_MODEL = "zkml_data/network.compiled"

WITNESS = "zkml_data/auth_witness.json"

PK = "zkml_data/proving.key"
VK = "zkml_data/verification.key"
SRS = "zkml_data/kzg.srs"

PROOF = "zkml_data/auth_proof.json"



# ==========================
# MODEL
# ==========================


class EmbeddingNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1,4,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(4,8,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8,16,3,padding=1),
            nn.ReLU(),
             nn.AdaptiveAvgPool2d((4,4))
        )


        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                16*4*4,
                32
            )
        )


    def forward(self,x):

        return self.fc(
            self.conv(x)
        )



class SiameseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = EmbeddingNet()



    def forward(self,x1,x2):

        return (
            self.embedding(x1),
            self.embedding(x2)
        )



# ==========================
# IMAGE LOAD
# ==========================


def load_image(path):

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise Exception(
            f"Cannot read {path}"
        )


    img=cv2.resize(
        img,
        (IMAGE_SIZE,IMAGE_SIZE)
    )


    img = img.astype(
        "float32"
    ) / 255.0


    return torch.tensor(img).reshape(
        1,1,IMAGE_SIZE,IMAGE_SIZE
    )



# ==========================
# MAIN
# ==========================


def main():

    print("\nLoading fingerprints...")


    img1 = load_image(
        ENROLLED_IMAGE
    )


    img2 = load_image(
        TEST_IMAGE
    )



    # ----------------------
    # Prediction
    # ----------------------

    model = SiameseNetwork()


    model.load_state_dict(
        torch.load(
            "models/siamese_fingerprint_zkml.pth",
            map_location="cpu"
        )
    )


    model.eval()



    with torch.no_grad():

        e1,e2 = model(
            img1,
            img2
        )


        distance = F.pairwise_distance(
            e1,e2
        )


    score = distance.item()


    print(
        "\nDistance:",
        score
    )


    if score < THRESHOLD:

        prediction = 1

        print(
            "Prediction: Genuine fingerprint"
        )

    else:

        prediction = 0

        print(
            "Prediction: Altered fingerprint"
        )



    # ----------------------
    # Create EZKL input
    # ----------------------


    input_data = {

        "input_data":
        [

            img1.numpy().flatten().tolist(),

            img2.numpy().flatten().tolist()

        ]
    }



    with open(
        INPUT_JSON,
        "w"
    ) as f:

        json.dump(
            input_data,
            f,
            indent=2
        )


    print(
        "\nEZKL input created"
    )



    # ----------------------
    # Generate witness
    # ----------------------

    print(
        "Generating witness..."
    )


    ezkl.gen_witness(
        INPUT_JSON,
        COMPILED_MODEL,
        WITNESS
    )


    print(
        "Generating proof..."
    )


    ezkl.prove(
        WITNESS,
        COMPILED_MODEL,
        PK,
        PROOF,
        srs_path=SRS
    )


    print(
        "Proof generated:"
    )

    print(PROOF)



    # ----------------------
    # Verify
    # ----------------------

    result = ezkl.verify(
        PROOF,
        "zkml_data/settings.json",
        VK,
        srs_path=SRS
    )


    print(
        "\nProof verified:",
        result
    )



if __name__=="__main__":
    main()
