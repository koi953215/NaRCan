NAME=bear

python infer.py \
  --model_path pth_file/${NAME}-model\
  --validation_image data/${NAME}/target/target.png \
  --validation_mask data/${NAME}/target/mask.png \
  --output_dir output/${NAME}-test/