# setup LIBERO-PLUS environment based on https://github.com/sylvestf/LIBERO-plus
bash ./gr00t/eval/sim/LIBEROPLUS/setup_libero_plus.sh
wget "https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip?download=true" -O ./assets.zip
mkdir -p ./temp/
unzip ./assets.zip -d ./temp/
mv  ./temp/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets ./external_dependencies/LIBERO-plus/libero/libero/
rm -r ./temp/
rm ./assets.zip

# Download the dataset from Hugging Face
huggingface-cli download \
  Sylvest/libero_plus_lerobot \
  --repo-type dataset \
  --local-dir /vla/users/luojingzhou/data/datasets/libero_plus_lerobot_4suite \
  --local-dir-use-symlinks False