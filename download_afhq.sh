#URL=https://www.dropbox.com/s/scckftx13grwmiv/afhq_v2.zip?dl=0
URL=https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0
ZIP_FILE=./data/afhq_v2.zip
mkdir -p ./data
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data
rm $ZIP_FILE

# Arrange files from data/train/* and data/test/* into ./data/afhq/{cat,dog,wild,animal}/
mkdir -p ./data/afhq/cat/0/
mkdir -p ./data/afhq/dog/0/
mkdir -p ./data/afhq/wild/0/

# Move cat images
if [ -d ./data/train/cat ]; then
    mv ./data/train/cat/* ./data/afhq/cat/0/ 2>/dev/null || true
fi
if [ -d ./data/test/cat ]; then
    mv ./data/test/cat/* ./data/afhq/cat/0/ 2>/dev/null || true
fi

# Move dog images
if [ -d ./data/train/dog ]; then
    mv ./data/train/dog/* ./data/afhq/dog/0/ 2>/dev/null || true
fi
if [ -d ./data/test/dog ]; then
    mv ./data/test/dog/* ./data/afhq/dog/0/ 2>/dev/null || true
fi

# Move wild images
if [ -d ./data/train/wild ]; then
    mv ./data/train/wild/* ./data/afhq/wild/0/ 2>/dev/null || true
fi
if [ -d ./data/test/wild ]; then
    mv ./data/test/wild/* ./data/afhq/wild/0/ 2>/dev/null || true
fi

rm -rf ./data/train
rm -rf ./data/test