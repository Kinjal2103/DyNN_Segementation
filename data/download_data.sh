#!/usr/bin/env bash
# Script to download and prepare the Polyp segmentation dataset and pretrained Res2Net backbone
set -e

DATA_DIR="data"
PRETRAINED_DIR="pretrained"
mkdir -p "$DATA_DIR"
mkdir -p "$PRETRAINED_DIR"

echo "=== 1. Downloading Pretrained Res2Net-50 Backbone ==="
BACKBONE_FILE="$PRETRAINED_DIR/res2net50_v1b_26w_4s-3cf99910.pth"
if [ ! -f "$BACKBONE_FILE" ]; then
    echo "Downloading backbone weights..."
    wget -c -O "$BACKBONE_FILE" "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth"
else
    echo "Backbone file already exists."
fi

echo "=== 2. Downloading PolypDataset.zip from Zenodo ==="
ZIP_FILE="PolypDataset.zip"
if [ ! -d "$DATA_DIR/TrainDataset/image" ] || [ ! -d "$DATA_DIR/TestDataset/Kvasir" ]; then
    if [ ! -f "$ZIP_FILE" ]; then
        wget -c --progress=dot:giga -O "$ZIP_FILE" "https://zenodo.org/records/5579392/files/PolypDataset.zip?download=1"
    fi

    echo "Extracting datasets..."
    mkdir -p polyp_raw
    unzip -q -o "$ZIP_FILE" "TestDataset/*" "NewTRimage/*" "NewTRmask/*" -d polyp_raw

    echo "Organizing directory structure..."
    mkdir -p "$DATA_DIR/TrainDataset"
    cp -r polyp_raw/TestDataset/TestDataset "$DATA_DIR/TestDataset" || cp -r polyp_raw/TestDataset "$DATA_DIR/TestDataset"
    cp -r polyp_raw/NewTRimage "$DATA_DIR/TrainDataset/image"
    cp -r polyp_raw/NewTRmask "$DATA_DIR/TrainDataset/masks"
    
    # Symlink for PraNet compatibility
    ln -sf image "$DATA_DIR/TrainDataset/images"

    # Cleanup raw extraction
    rm -rf polyp_raw "$ZIP_FILE"
    echo "Data preparation complete!"
else
    echo "Dataset already present in $DATA_DIR. Nothing to download."
fi
