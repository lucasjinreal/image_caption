# this is a sample to generate tfrecords
python build_mscoco_data.py --train_image_dir /media/jintian/Netac/Datasets/Images/Caption/coco/train2014 \
 --val_image_dir /media/jintian/Netac/Datasets/Images/Caption/coco/val2014 \
 --train_captions_file /media/jintian/Netac/Datasets/Images/Caption/coco/annotations/captions_train2014.json \
  --val_captions_file /media/jintian/Netac/Datasets/Images/Caption/coco/annotations/captions_val2014.json \
  --output_dir ./tfrecords
